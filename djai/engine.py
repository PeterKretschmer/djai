"""The real-time audio engine: two decks, a master clock, and the callback.

THREADING CONTEXT
-----------------
:meth:`Engine.callback` runs on the **audio thread**, driven by PortAudio via
sounddevice. It is the only genuinely real-time code in the program and it
obeys these rules absolutely:

* no LLM, no network, no file I/O, no logging, no printing
* no locks and no blocking calls
* no large or unbounded allocation
* no exception may escape (an escaping exception kills the stream)

The only channel into it is :attr:`Engine.queue`, a ``queue.Queue`` of
:mod:`djai.commands` drained at the top of every callback. ``get_nowait`` on a
``queue.Queue`` does take a lock, but it is an uncontended CPython mutex held
for a few instructions by producers that are never themselves real-time; the
spec mandates this channel and it is the pragmatic choice in Python.

Everything else here -- ``start``, ``stop``, state snapshots -- is main-thread
API and is safe to call while the stream runs, because it only reads floats
and swaps immutable references.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd

from djai import config
from djai import transition as tr
from djai.commands import (
    Command,
    Cut,
    KillBass,
    LoadTrack,
    Resync,
    BeatJump,
    ExitLoop,
    SetEQ,
    SetFilter,
    SetGain,
    SetKeyLock,
    SwapStems,
    SetLoop,
    SetPitch,
    SetRate,
    StartTransition,
    Stop,
    SyncDeck,
)
from djai.deck import (
    CHANNELS,
    DTYPE,
    MAX_BLOCK,
    SAMPLE_RATE,
    Deck,
    LoadedTrack,
    time_stretch,
)

log = logging.getLogger(__name__)

#: Reverb delay lengths as multiples of the shortest, which is at least one
#: callback block. Mutually far from integer ratios, so echoes do not stack.
_REVERB_RATIOS: tuple[float, ...] = (
    1.0, 1.1347, 1.2703, 1.4119, 1.5553, 1.7041, 1.8517, 1.9973,
)
#: Wet level of the reverb into the mix, before the per-line normalisation.
REVERB_WET: float = 0.35

#: Frames per callback, from djai.config (AUDIO_BLOCKSIZE, default 2048).
#:
#: 2048 at 44.1 kHz is ~46 ms per block. That is deliberately generous: the LLM
#: now runs locally on this same machine, and an 8B model spikes CPU and GPU
#: hard enough to starve a 512-frame (11.6 ms) callback. A bigger buffer buys
#: the stream time to ride the spike out. It still leaves a panic `cut` at
#: roughly 46 ms, inside the 100 ms budget.
DEFAULT_BLOCKSIZE: int = config.AUDIO_BLOCKSIZE

#: Hard limiter ceiling. Two decks at full gain can exceed 0dBFS during a
#: blend; clipping there is preferable to letting the device wrap.
MASTER_CEILING: float = 0.98

#: Where the master limiter starts working. See :mod:`djai.config`.
MASTER_KNEE: float = config.LIMITER_KNEE
MASTER_SUBBLOCK: int = config.LIMITER_SUBBLOCK

BEATS_PER_BAR: int = 4

#: Host APIs to search when no device is given, best first. PortAudio's default
#: host API is not always the one with working devices -- on some Windows
#: setups MME, DirectSound and WASAPI all enumerate zero devices and only
#: WDM-KS is usable -- so falling back by host API is what makes `play` start.
_HOSTAPI_PREFERENCE = ("WASAPI", "DirectSound", "MME", "WDM-KS")


def list_output_devices() -> list[tuple[int, str, str, int]]:
    """``(index, name, host_api, max_output_channels)`` for every output device."""
    hostapis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] >= CHANNELS:
            out.append(
                (i, d["name"], hostapis[d["hostapi"]]["name"], d["max_output_channels"])
            )
    return out


class _Ring:
    """Single-producer single-consumer float ring. Lock-free by construction.

    THREADING CONTEXT: :meth:`write` is called from the **audio thread** and is
    the only writer; :meth:`read_into` is called from one consumer thread (the
    cue callback, or the recorder's writer thread) and is the only reader.

    Both indices are plain Python ints that only ever increase, and each is
    written by exactly one thread. Under the GIL an int rebind is atomic, so the
    other side may read a stale value but never a torn one -- a stale read makes
    the ring look emptier or fuller than it is by at most one block, which costs
    nothing here. There is no lock, so the audio thread can never be blocked by
    a slow consumer; it overwrites and moves on, and the consumer notices via
    :attr:`dropped`.
    """

    def __init__(self, frames: int, channels: int = CHANNELS) -> None:
        self._buf = np.zeros((frames, channels), dtype=DTYPE)
        self._frames = frames
        self.written: int = 0   # audio thread only
        self.read: int = 0      # consumer thread only
        self.dropped: int = 0   # audio thread only

    @property
    def available(self) -> int:
        """Frames the consumer can take. May undercount by a block; never over."""
        return max(0, self.written - self.read)

    def write(self, block: np.ndarray, n: int) -> None:
        """Append ``n`` frames. AUDIO THREAD. No allocation, never blocks.

        If the consumer has fallen behind by a whole buffer the oldest frames
        are overwritten and counted. Dropping the recording's tail is always
        preferable to stalling the callback.
        """
        if self.written - self.read > self._frames - n:
            self.dropped += n
        start = self.written % self._frames
        end = start + n
        if end <= self._frames:
            np.copyto(self._buf[start:end], block[:n])
        else:
            split = self._frames - start
            np.copyto(self._buf[start:], block[:split])
            np.copyto(self._buf[: end - self._frames], block[split:n])
        self.written += n

    def read_into(self, out: np.ndarray, n: int) -> int:
        """Take up to ``n`` frames into ``out``. CONSUMER THREAD.

        Returns how many frames were actually available; the remainder of
        ``out`` is left untouched for the caller to fill (with silence, for the
        cue stream) or ignore.
        """
        # If the producer has lapped us, the frames our index points at have
        # already been overwritten. Skip to the oldest one that is still really
        # there: returning whatever now occupies those slots would hand back
        # audio spliced from two different points in the set. Only the consumer
        # writes `read`, so moving it here stays single-writer.
        oldest = self.written - self._frames
        if self.read < oldest:
            self.read = oldest

        have = min(n, self.available)
        if have <= 0:
            return 0
        start = self.read % self._frames
        end = start + have
        if end <= self._frames:
            np.copyto(out[:have], self._buf[start:end])
        else:
            split = self._frames - start
            np.copyto(out[:split], self._buf[start:])
            np.copyto(out[split:have], self._buf[: end - self._frames])
        self.read += have
        return have


class SessionRecorder:
    """Writes the master output to a file continuously, off the audio thread.

    THREADING CONTEXT: :meth:`capture` is called from the **audio thread** and
    does nothing but copy into a ring. All file I/O happens on the writer
    thread, which is a plain daemon thread doing blocking writes -- exactly the
    work that must never appear in the callback.

    The format follows the file extension: ``.flac`` for session recordings,
    which is what `play --record` writes, or ``.wav``. Both are 16-bit PCM.

    If the process is killed, a WAV is complete to its last write. A FLAC is
    not quite: its length is never written and the encoder's buffer is lost.
    Measured: an unclosed FLAC reads back in chunks up to its last encoded
    frame and then stops with a seek error, so ffmpeg or any tolerant reader
    recovers it minus about a fifth of a second. Starting a FLAC recording
    applies the retention cap in :data:`djai.config.RECORD_RETENTION`.
    """

    def __init__(self, path: Path, ring_seconds: float | None = None) -> None:
        import soundfile as sf  # local: only a recording session needs it

        self._sf = sf
        self.path = path
        seconds = (
            config.RECORD_RING_SECONDS if ring_seconds is None else ring_seconds
        )
        self._ring = _Ring(max(MAX_BLOCK * 2, int(SAMPLE_RATE * seconds)))
        self._chunk = np.zeros((MAX_BLOCK * 4, CHANNELS), dtype=DTYPE)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.frames_written: int = 0
        self.error: str | None = None

    @property
    def dropped_frames(self) -> int:
        """Frames the writer thread was too slow to take. Audio is unaffected."""
        return self._ring.dropped

    @property
    def seconds_written(self) -> float:
        return self.frames_written / SAMPLE_RATE

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._sf.SoundFile(
            str(self.path),
            mode="w",
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            subtype="PCM_16",
        )
        if self.path.suffix.lower() == ".flac":
            from djai.housekeeping import enforce_recording_retention

            for removed in enforce_recording_retention(
                self.path.parent, config.RECORD_RETENTION, protect=self.path
            ):
                log.info("recording retention: removed %s", removed)
        self._thread = threading.Thread(
            target=self._run, name="djai-recorder", daemon=True
        )
        self._thread.start()

    def capture(self, block: np.ndarray, n: int) -> None:
        """AUDIO THREAD. One ring write; no I/O, no allocation, never blocks."""
        self._ring.write(block, n)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._drain():
                    # Nothing waiting. Sleeping a fraction of the ring keeps the
                    # thread cheap without ever letting it fall a ring behind.
                    time.sleep(0.05)
            self._drain()  # final flush
        except Exception as exc:  # a full disk must not take the set down
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("session recording stopped: %s", self.error)
        finally:
            try:
                self._file.close()
            except Exception:
                pass

    def _drain(self) -> bool:
        took = self._ring.read_into(self._chunk, self._chunk.shape[0])
        if took <= 0:
            return False
        self._file.write(self._chunk[:took])
        self.frames_written += took
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


def _device_channels(device: int | str | None) -> int:
    """Output channels a device offers, or 0 if it cannot be queried."""
    try:
        info = sd.query_devices(device, "output")
    except Exception:
        return 0
    return int(info.get("max_output_channels", 0))


def _can_open(device: int | None, blocksize: int) -> bool:
    try:
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=blocksize,
            device=device,
            channels=CHANNELS,
            dtype="float32",
            callback=lambda o, f, t, s: o.fill(0.0),
        )
    except Exception:
        return False
    try:
        stream.start()
        return True
    except Exception:
        return False
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


def pick_output_device(blocksize: int = DEFAULT_BLOCKSIZE) -> int | None:
    """Find an output device that actually opens, preferring the system default.

    Returns a device index, or None to let PortAudio choose (only when the
    default already works). Raises RuntimeError if nothing opens at all.
    """
    if _can_open(None, blocksize):
        return None

    candidates = list_output_devices()
    ranked = sorted(
        candidates,
        key=lambda c: (
            _HOSTAPI_PREFERENCE.index(c[2].replace("Windows ", ""))
            if c[2].replace("Windows ", "") in _HOSTAPI_PREFERENCE
            else len(_HOSTAPI_PREFERENCE)
        ),
    )
    for index, _name, _api, _ch in ranked:
        if _can_open(index, blocksize):
            return index

    raise RuntimeError(
        "No usable audio output device found. Devices seen: "
        + (", ".join(f"{i}:{n} [{a}]" for i, n, a, _ in candidates) or "none")
    )


def render_riser(n_frames: int, seed: int = 20260915) -> np.ndarray:
    """A noise riser: white noise through a band-pass swept from 300 Hz to 6 kHz.

    CONTROL THREAD, once per armed transition: all of the riser's DSP happens
    here, and the callback only plays the finished buffer back at the level
    the envelope asks for. Normalised to a peak of 1; the envelope sets level.
    """
    from scipy.signal import butter, sosfilt

    n = max(1, int(n_frames))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, CHANNELS))
    out = np.empty((n, CHANNELS), dtype=np.float64)
    chunk = 1024
    zi = None
    for i in range(0, n, chunk):
        progress = i / n
        low = 300.0 * (6000.0 / 300.0) ** progress
        high = min(low * 3.0, 0.45 * SAMPLE_RATE)
        sos = butter(2, [low, high], btype="band", fs=SAMPLE_RATE, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2, CHANNELS))
        out[i:i + chunk], zi = sosfilt(sos, noise[i:i + chunk], axis=0, zi=zi)
    peak = float(np.max(np.abs(out))) or 1.0
    return np.ascontiguousarray(out / peak, dtype=DTYPE)


class _StretchWorker:
    """Pre-stretches cued tracks on a background thread.

    THREADING CONTEXT: owns one daemon thread. :meth:`request` is called from
    whichever control thread submitted the cue; :meth:`get` is called from the
    **audio thread** and must therefore never block or allocate.

    ``get`` is a plain dict lookup and the worker only ever rebinds whole
    entries, so no lock is needed on either side -- the same reasoning that lets
    the control thread swap ``deck.track`` while the callback reads it.
    """

    def __init__(self, cache_size: int = 4) -> None:
        self._queue: queue.Queue[tuple[str, LoadedTrack, float] | None] = queue.Queue()
        self._ready: dict[str, LoadedTrack] = {}
        self._order: list[str] = []
        self._inflight: set[str] = set()
        #: Called with the finished copy, on this worker's thread. Key lock
        #: uses it to swap a copy in once it exists.
        self._callbacks: dict[str, list] = {}
        self._cache_size = cache_size
        self._lock = threading.Lock()  # control side only; never the callback
        self.requested = 0
        self.completed = 0
        self.failures = 0
        self._thread = threading.Thread(
            target=self._run, name="djai-stretch", daemon=True
        )
        self._thread.start()

    @staticmethod
    def key(track: LoadedTrack, rate: float) -> str:
        return f"{track.analysis.track_id}@{rate:.6f}"

    def request(self, track: LoadedTrack, rate: float, on_ready=None) -> None:
        """Queue a stretch. Returns immediately; never blocks the caller.

        ``on_ready(copy)``, if given, is called once the copy exists -- at once
        if it already does, otherwise from the worker thread when it is done.
        A failed stretch never calls it.
        """
        k = self.key(track, rate)
        with self._lock:
            ready = self._ready.get(k)
            if ready is None and on_ready is not None:
                self._callbacks.setdefault(k, []).append(on_ready)
            queue_it = ready is None and k not in self._inflight
            if queue_it:
                self._inflight.add(k)
                self.requested += 1
        if ready is not None and on_ready is not None:
            on_ready(ready)
        if queue_it:
            self._queue.put((k, track, rate))

    def get(self, track: LoadedTrack, rate: float) -> LoadedTrack | None:
        """AUDIO THREAD. The stretched copy if it is ready, else None."""
        return self._ready.get(f"{track.analysis.track_id}@{rate:.6f}")

    def _run(self) -> None:
        while True:
            # Drop the previous iteration's references before blocking. Found
            # by a Phase 0 test: these locals outlived their iteration, so the
            # most recent stretched copy stayed pinned by this frame until the
            # next request arrived, and `retain` could never actually free it.
            item = track = stretched = waiting = None
            item = self._queue.get()
            if item is None:
                return
            k, track, rate = item
            try:
                started = time.time()
                stretched = time_stretch(track, rate)
                with self._lock:
                    self._ready[k] = stretched
                    self._order.append(k)
                    while len(self._order) > self._cache_size:
                        self._ready.pop(self._order.pop(0), None)
                    self._inflight.discard(k)
                    waiting = self._callbacks.pop(k, [])
                self.completed += 1
                log.info(
                    "stretched %r to %.4f in %.1fs",
                    track.title, rate, time.time() - started,
                )
                for callback in waiting:
                    try:
                        callback(stretched)
                    except Exception as exc:  # a caller's bug must not kill the worker
                        log.warning("stretch callback failed: %s", exc)
            except Exception as exc:
                with self._lock:
                    self._inflight.discard(k)
                    self._callbacks.pop(k, None)
                self.failures += 1
                # Not fatal: the deck falls back to resampling, which is the
                # behaviour that shipped before stretching existed.
                log.warning(
                    "stretch failed for %r at %.4f (%s: %s); "
                    "falling back to resampling",
                    track.title, rate, type(exc).__name__, exc,
                )

    def retain(self, track_ids: set[str]) -> int:
        """Drop every stretched copy whose source track is not in ``track_ids``.

        CONTROL THREAD. Returns how many were dropped. The caller passes every
        track a deck holds or is about to be given, so a copy is only released
        once no deck references it -- which keeps the large free off the audio
        thread: the dict's reference is dropped here, and a deck swapping
        tracks on the audio thread never holds the last one.
        """
        dropped = 0
        with self._lock:
            for key in list(self._ready):
                if key.split("@", 1)[0] in track_ids:
                    continue
                self._ready.pop(key, None)
                if key in self._order:
                    self._order.remove(key)
                dropped += 1
        return dropped

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=2.0)


@dataclass
class DeckState:
    """A flat, JSON-safe snapshot of one deck. Built on the main thread."""

    name: str
    title: str | None
    bpm: float | None
    native_bpm: float | None
    key: str | None
    camelot: str | None
    playing: bool
    ended: bool
    #: :class:`djai.deck.TransportState` value, as a plain string for the wire.
    transport: str
    position_bars: float
    remaining_seconds: float
    gain: float
    eq: tuple[float, float, float]
    rate: float
    key_lock: bool = True
    #: Filter knob, -1..1, and its resonance, 0..1.
    filter: float = 0.0
    filter_resonance: float = 0.0
    #: Performance state: a loop engaged and its length in beats, the pitch
    #: fader as a percentage from native tempo, and whether it was moved by hand.
    loop_active: bool = False
    loop_beats: float = 0.0
    pitch_percent: float = 0.0
    manual_pitch: bool = False


class Engine:
    """Owns the two decks, the output stream and the master clock."""

    def __init__(
        self,
        device: int | str | None = None,
        blocksize: int = DEFAULT_BLOCKSIZE,
        queue_size: int = 256,
        cue_device: int | str | None = None,
        cue_channels: tuple[int, int] | None = None,
        recorder: SessionRecorder | None = None,
    ) -> None:
        if blocksize > MAX_BLOCK:
            raise ValueError(f"blocksize {blocksize} exceeds deck MAX_BLOCK {MAX_BLOCK}")
        if cue_device is not None and cue_channels is not None:
            raise ValueError(
                "cue_device and cue_channels are alternatives: the first opens a "
                "second stream, the second routes within one device"
            )

        self.blocksize = blocksize
        self.device = device
        self.queue: queue.Queue[Command] = queue.Queue(maxsize=queue_size)

        self.deck_a = Deck("a")
        self.deck_b = Deck("b")
        self._decks: dict[str, Deck] = {"a": self.deck_a, "b": self.deck_b}

        # --- master clock (audio thread writes, everyone else reads) ---
        #: Frames since the stream started. The authoritative engine-time clock.
        self.frames_played: int = 0
        #: Free-running beat counter. Advanced by a single multiply-add per
        #: block; the grid-accurate mapping lives in djai.phrase.
        self.master_beat: float = 0.0
        #: A consistent snapshot of the whole clock, published as one tuple at
        #: the end of each callback: (frames_played, master_beat, pos_a, pos_b).
        #:
        #: Reading those four attributes individually from another thread is a
        #: race -- the callback advances deck positions before the beat counter,
        #: so a monitor that reads between them sees a full block of phantom
        #: skew (11.6 ms at 512 frames, which is squarely inside the drift
        #: monitor's nudge band). Rebinding one tuple is atomic under the GIL,
        #: so a reader either sees the whole previous block or the whole next
        #: one, never a mixture.
        self._clock: tuple[int, float, float, float] = (0, 0.0, 0.0, 0.0)
        self._beats_per_frame: float = 0.0
        self.master_deck: str = "a"

        # --- diagnostics (audio thread writes, monitor thread reads) ---
        self.underruns: int = 0
        self.callbacks: int = 0
        #: How many loads got a pitch-preserving stretch, and how many fell back
        #: to resampling because the stretch was not ready or not attempted.
        self.stretched_loads: int = 0
        self.resampled_loads: int = 0
        self.max_callback_load: float = 0.0
        self.stop_requested: bool = False

        # --- transition state (audio thread only) ---
        self._trans_active: bool = False
        self._trans_from: Deck | None = None
        self._trans_to: Deck | None = None
        self._trans_frames: float = 0.0
        self._trans_total: float = 1.0
        self._trans_bars_per_frame: float = 0.0

        #: The precomputed envelope for the running transition, one row per
        #: block, or None to fall back to :func:`djai.transition.gains_at`.
        #: The fallback exists for a bare StartTransition submitted without a
        #: plan -- the supervisor's paths and older tests -- and keeps them
        #: behaving exactly as before.
        self._trans_env = None
        self._trans_style: str = "bass_swap"
        #: Loop length currently applied to the outgoing deck, in frames. Kept
        #: so a steady loop is not re-set every block.
        self._trans_loop_len: float = 0.0
        #: The outgoing deck's position when the transition started. The
        #: envelope's loop starts are offsets from here, so applying one is a
        #: single addition.
        self._trans_src_origin: float = 0.0
        #: The outgoing deck's rate when the transition started, and the last
        #: RATE_A multiplier applied to it. RATE_A scales the rate the deck
        #: already had -- its beatmatch -- and is only applied when its value
        #: changes, so a steady envelope never overwrites a supervisor nudge.
        self._trans_src_rate: float = 1.0
        self._trans_rate_mult: float = 1.0
        #: Block size the envelope rows were built against.
        self._trans_block: int = blocksize
        #: Resonance a filter knob returns to after automation has used it.
        self._knob_resonance: float = float(config.FILTER_KNOB_RESONANCE)
        #: Set by the control thread before StartTransition is queued, consumed
        #: when it is applied. A single reference assignment, and only one
        #: transition is ever armed at a time. It lives here rather than on the
        #: command because `commands.py` is outside this phase's scope.
        self._armed_plan: tuple | None = None

        # --- echo send (audio thread only) --------------------------------------
        #: Preallocated delay line for `echo_out`. Sized for the slowest tempo
        #: worth mixing, so the ring never has to be reallocated for a BPM.
        self._echo_len: int = int(SAMPLE_RATE * config.ECHO_MAX_SECONDS)
        self._echo_ring = np.zeros((self._echo_len, CHANNELS), dtype=DTYPE)
        self._echo_read_buf = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._echo_work = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._echo_write: int = 0
        self._echo_delay: int = 1
        #: How much of deck A is going to the send this block. Zero for every
        #: style but `echo_out`, and the callback skips the delay entirely then.
        self._echo_send: float = 0.0
        #: Frames the echo keeps running once its send has closed, so repeats
        #: already in the delay line ring out -- past the hand-over, too -- and
        #: are not cut off. Counted down per block; zero means silent.
        self._echo_ring_left: int = 0
        #: Repeats until the tail is ECHO_RING_OUT_DB down. Multiplied by the
        #: delay time on the audio thread; worked out once here.
        self._echo_ring_repeats: int = int(
            math.ceil((tr.ECHO_RING_OUT_DB / 20.0) / math.log10(tr.ECHO_FEEDBACK))
        ) + 1
        #: The echo's tail for this block, kept so the reverb can take it in.
        self._echo_tail = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._echo_tail_live: bool = False

        # --- reverb (audio thread only) -----------------------------------------
        #: A feedback delay network: eight lines mixed through a Householder
        #: matrix, each averaged with the sample before it for damping. Every
        #: delay is at least one callback block long, so a block only ever reads
        #: what earlier blocks wrote -- the network runs block-wise, with every
        #: buffer preallocated here and no per-sample loop in the callback.
        shortest = max(int(blocksize), int(0.030 * SAMPLE_RATE))
        self._rv_delays: tuple[int, ...] = tuple(
            int(round(shortest * r)) for r in _REVERB_RATIOS
        )
        self._rv_len: int = max(self._rv_delays) + MAX_BLOCK + 2
        n_lines = len(self._rv_delays)
        self._rv_ring = np.zeros((n_lines, self._rv_len, CHANNELS), dtype=DTYPE)
        self._rv_read = np.zeros((n_lines, MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._rv_prev = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._rv_sum = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._rv_in = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._rv_work = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._rv_write: int = 0
        mean_delay = sum(self._rv_delays) / n_lines
        #: Per-pass feedback for a tail of REVERB_TAIL_SECONDS to -60 dB.
        self._rv_feedback: float = 10.0 ** (
            -3.0 * mean_delay / (tr.REVERB_TAIL_SECONDS * SAMPLE_RATE)
        )
        self._rv_mix: float = 2.0 / n_lines
        self._rv_wet: float = REVERB_WET / math.sqrt(n_lines)
        #: This block's send level, from the envelope, and how many frames of
        #: tail are left to run once it has closed.
        self._rv_send: float = 0.0
        self._rv_tail_frames: int = int(tr.REVERB_TAIL_SECONDS * SAMPLE_RATE)
        self._rv_tail: int = 0

        # --- noise riser (audio thread plays it; control thread renders it) -----
        self._riser: np.ndarray | None = None
        self._riser_pos: int = 0
        self._riser_gain: float = 0.0
        self._riser_prev: float = 0.0
        self._rs_work = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._rs_ramp = np.zeros((MAX_BLOCK, 1), dtype=DTYPE)
        self._rs_unit = np.minimum(
            np.arange(1, MAX_BLOCK + 1, dtype=np.float64) / max(1, blocksize), 1.0
        ).astype(DTYPE).reshape(-1, 1)

        # --- master limiter (audio thread only) ---------------------------------
        self._limiter_gain: float = 1.0
        self._abs_buf = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        #: Fraction of the way the gain travels back toward unity per
        #: sub-block. Derived once; the callback only multiplies by it.
        release_frames = max(1.0, config.LIMITER_RELEASE_MS / 1000.0 * SAMPLE_RATE)
        self._limiter_release = 1.0 - math.exp(
            -config.LIMITER_SUBBLOCK / release_frames
        )
        #: Highest sample the master bus produced this block, before limiting.
        #: This is the number that says whether the mix is running hot.
        self.master_peak_in: float = 0.0
        #: And after. Never above :data:`MASTER_CEILING` by construction.
        self.master_peak: float = 0.0
        #: Current limiter gain, 1.0 when it is doing nothing.
        self.limiter_gain: float = 1.0
        #: Blocks where the backstop clamp had to engage. Should stay zero:
        #: if it ever moves, the limiter has a hole in it.
        self.limiter_clips: int = 0
        #: Blocks the limiter pulled down by more than LIMITER_LOG_REDUCTION_DB,
        #: the lowest gain in the last block, and the lowest since the
        #: supervisor last reported. Counted here, never logged here: the
        #: supervisor reads them and writes the log off the audio thread.
        self.limiter_min_gain: float = 1.0
        self.limiter_heavy_blocks: int = 0
        self.limiter_deepest_gain: float = 1.0
        self._log_reduction_gain: float = 10.0 ** (
            -config.LIMITER_LOG_REDUCTION_DB / 20.0
        )

        # --- cue / pre-listen ---------------------------------------------------
        #: Which deck the cue output carries, or None for no pre-listen. Set
        #: from any thread; the callback reads it once per block.
        self.cue_deck: str | None = None
        self.cue_device = cue_device
        self.cue_channels = cue_channels
        #: Set once start() knows what the hardware actually gave us, so callers
        #: can say plainly at startup whether pre-listen is available.
        self.cue_mode: str = "none"
        self._cue_ring: _Ring | None = (
            _Ring(max(MAX_BLOCK * 2, blocksize * config.CUE_RING_BLOCKS))
            if cue_device is not None
            else None
        )
        self._cue_stream: sd.OutputStream | None = None
        self._cue_out = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self.cue_underruns: int = 0

        # --- session recording --------------------------------------------------
        self.recorder = recorder

        # --- fallback / watchdog ------------------------------------------------
        #: Monotonic time the last callback finished. The watchdog in
        #: djai.fallback compares this against now; a stalled callback is how a
        #: dead engine looks from outside.
        self.last_callback_at: float = time.monotonic()
        #: When set, the callback stops mixing and fills from this object's
        #: ``fill(outdata, frames)`` instead. Rebinding one reference is atomic
        #: under the GIL, so takeover happens on the very next block.
        self.fallback: Any = None

        self._master_gain: float = 1.0
        self._mix = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._stream: sd.OutputStream | None = None
        self._stretcher: _StretchWorker | None = (
            _StretchWorker(config.STRETCH_CACHE_SIZE)
            if config.TIME_STRETCH_ENABLED
            else None
        )

    # --- lifecycle (main thread) --------------------------------------------

    def start(self) -> None:
        if self.device is None:
            self.device = pick_output_device(self.blocksize)

        # Cue on channels 3/4 of one device: one stream, one clock, nothing to
        # drift. Falls back to master-only rather than refusing to start, since
        # no pre-listen is survivable and no audio is not.
        channels = CHANNELS
        if self.cue_channels is not None:
            need = max(self.cue_channels)
            if _device_channels(self.device) >= need:
                channels = need
                self.cue_mode = f"channels {self.cue_channels[0]}/{self.cue_channels[1]}"
            else:
                self.cue_channels = None
                self.cue_mode = "none"

        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=self.blocksize,
            device=self.device,
            channels=channels,
            dtype="float32",
            callback=self.callback,
        )
        self._stream.start()

        if self.cue_device is not None:
            try:
                self._cue_stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=self.blocksize,
                    device=self.cue_device,
                    channels=CHANNELS,
                    dtype="float32",
                    callback=self._cue_callback,
                )
                self._cue_stream.start()
                self.cue_mode = f"device {self.cue_device}"
            except Exception as exc:
                # Master is already running and must stay that way.
                log.warning("cue device %s unavailable: %s", self.cue_device, exc)
                self._cue_stream = None
                self.cue_device = None
                self._cue_ring = None
                self.cue_mode = "none"

    def stop(self) -> None:
        for attr in ("_cue_stream", "_stream"):
            stream = getattr(self, attr)
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        if self._stretcher is not None:
            self._stretcher.stop()
            self._stretcher = None
        if self.recorder is not None:
            self.recorder.stop()

    @property
    def running(self) -> bool:
        return self._stream is not None and self._stream.active

    # --- command submission (any non-audio thread) ---------------------------

    def submit(self, cmd: Command) -> bool:
        """Enqueue a command for the audio thread. False if the queue is full.

        Never blocks: dropping a command and reporting it is strictly better
        than stalling a producer that might be the scheduler.
        """
        # A cue is the signal that this track will be needed at a tempo, and it
        # arrives a lead-time before the transition -- exactly the pre-roll the
        # stretch has to fit inside. Requesting here is non-blocking; the work
        # happens on the stretch worker, and if it is not finished when the
        # deck actually loads, _apply falls back to resampling.
        if self._stretcher is not None and isinstance(cmd, LoadTrack):
            self._maybe_request_stretch(cmd)
        try:
            self.queue.put_nowait(cmd)
            return True
        except queue.Full:
            return False

    def _maybe_request_stretch(self, cmd: LoadTrack) -> None:
        """Control thread. Ask for a stretched copy if one would help."""
        if cmd.track is None or self._stretcher is None:
            return
        rate = cmd.rate
        if not np.isfinite(rate) or rate <= 0:
            return
        if abs(rate - 1.0) < config.STRETCH_DEADBAND:
            return  # under ~4 cents; not worth the CPU
        if abs(rate - 1.0) > config.MAX_STRETCH_RATIO + 1e-9:
            return  # selector should not have offered this; resampling it is
        if cmd.track.is_stretched:
            return
        deck = self._decks.get(cmd.deck)
        if deck is not None and not deck.key_lock:
            return  # key lock off: this deck resamples, so no copy is wanted
        self._stretcher.request(cmd.track, rate)

    def set_key_lock(self, name: str, on: bool) -> str:
        """CONTROL THREAD. Turn one deck's key lock on or off.

        Returns what happened: ``"off"``; ``"on"`` when the deck is already at
        original pitch or has nothing to stretch; ``"stretching"`` when a copy
        is being made on the worker and will swap in when it exists;
        ``"out of range"``; or ``"unavailable"`` when time-stretch is disabled.
        """
        deck = self._decks.get(name)
        if deck is None:
            raise ValueError(f"unknown deck {name!r}")
        on = bool(on)
        self.submit(SetKeyLock(deck=name, on=on, origin="key_lock"))
        if not on:
            return "off"
        if self._stretcher is None:
            return "unavailable"
        track = deck.track
        rate = deck.load_rate
        if track is None or track.is_stretched or abs(rate - 1.0) < config.STRETCH_DEADBAND:
            return "on"
        if abs(rate - 1.0) > config.MAX_STRETCH_RATIO + 1e-9:
            return "out of range"

        def swap_in(copy: LoadedTrack) -> None:
            # Worker thread. Only a submit: the audio thread checks the copy is
            # still of the track under the playhead before it swaps.
            self.submit(SetKeyLock(deck=name, on=True, track=copy, origin="key_lock"))

        self._stretcher.request(track, rate, on_ready=swap_in)
        return "stretching"

    def retune_master(self, name: str) -> None:
        """CONTROL THREAD. Re-derive the master tempo after a grid correction.

        Only for the master deck, and only its beat clock: the audio is not
        touched and plays at exactly the speed it did. One float assignment,
        atomic under the GIL. The supervisor re-baselines both decks after.
        """
        deck = self._decks.get(name)
        if deck is None or deck.track is None or self.master_deck != name:
            return
        self._beats_per_frame = deck.track.analysis.bpm * deck.rate / 60.0 / SAMPLE_RATE

    def release_stretches(self, keep_track_ids: set[str]) -> int:
        """CONTROL THREAD. Free stretched copies no deck needs; returns count.

        Measured need: the stretch cache held up to four full-length copies
        whether or not anything still used them. A deck that still holds a
        copy keeps it -- the outgoing deck after a transition, for instance --
        and that copy is released at the next call after the deck moves on.
        """
        if self._stretcher is None:
            return 0
        return self._stretcher.retain(set(keep_track_ids))

    def _set_master(self, name: str, effective_bpm: float) -> None:
        """Point the master beat clock at a deck. AUDIO THREAD only."""
        self.master_deck = name
        self._beats_per_frame = effective_bpm / 60.0 / SAMPLE_RATE

    # --- introspection (main / monitor thread) -------------------------------

    def deck(self, name: str) -> Deck:
        return self._decks[name]

    @property
    def transition_active(self) -> bool:
        return self._trans_active

    @property
    def clock(self) -> tuple[int, float, float, float]:
        """``(frames_played, master_beat, position_a, position_b)`` as of one
        callback. Use this, not the individual attributes, whenever two of them
        are compared -- see :attr:`_clock`."""
        return self._clock

    @property
    def master_bpm(self) -> float:
        """Tempo of the master beat clock, in BPM."""
        return self._beats_per_frame * 60.0 * SAMPLE_RATE

    def deck_in_transition(self, name: str) -> bool:
        if not self._trans_active:
            return False
        deck = self._decks.get(name)
        return deck is not None and (
            deck is self._trans_from or deck is self._trans_to
        )

    @property
    def transition_progress_bars(self) -> float:
        if not self._trans_active:
            return 0.0
        return self._trans_frames * self._trans_bars_per_frame

    def deck_state(self, name: str) -> DeckState:
        from djai import phrase  # local import: phrase imports deck, not engine

        deck = self._decks[name]
        track = deck.track
        analysis = track.analysis if track else None
        return DeckState(
            name=name,
            title=analysis.title if analysis else None,
            bpm=round(analysis.bpm * deck.rate, 2) if analysis else None,
            native_bpm=round(analysis.bpm, 2) if analysis else None,
            key=analysis.key_name if analysis else None,
            camelot=analysis.camelot if analysis else None,
            playing=deck.playing,
            ended=deck.ended,
            transport=deck.transport.value,
            position_bars=(
                round(phrase.bar_at_frame(analysis, deck.position), 2)
                if analysis
                else 0.0
            ),
            remaining_seconds=(
                round(deck.remaining_frames / SAMPLE_RATE / max(deck.rate, 1e-6), 1)
                if analysis
                else 0.0
            ),
            gain=round(deck.gain.target, 3),
            eq=(
                round(deck.eq_low.target, 3),
                round(deck.eq_mid.target, 3),
                round(deck.eq_high.target, 3),
            ),
            rate=round(deck.rate, 5),
            key_lock=bool(deck.key_lock),
            filter=round(deck.filter_pos.target, 3),
            filter_resonance=round(deck.filter_res, 3),
            loop_active=bool(deck.loop_active),
            loop_beats=(
                round(deck.loop_region[1] / (analysis.beat_period * SAMPLE_RATE), 3)
                if analysis and deck.loop_active
                else 0.0
            ),
            pitch_percent=round((deck.rate - 1.0) * 100.0, 2),
            manual_pitch=bool(deck.manual_pitch),
        )

    # --- command application (AUDIO THREAD) ----------------------------------

    def _apply(self, cmd: Command) -> None:
        """Apply one command. Assignments and float math only."""
        if isinstance(cmd, LoadTrack):
            deck = self._decks.get(cmd.deck)
            if deck is not None:
                start = cmd.start_frame
                if not cmd.is_immediate and self.frames_played > cmd.execute_at:
                    # The command was released late -- the scheduler polls every
                    # 2 ms and commands are applied at the top of a callback, so
                    # a cue lands up to a block after its musical moment. Start
                    # the deck where it *would* have been had it fired on time,
                    # instead of at the cue point a block too late: that lateness
                    # is otherwise a systematic ~12-24 ms phase error at the
                    # start of every transition, which the drift monitor then
                    # has to hard-resync away.
                    late = self.frames_played - cmd.execute_at
                    start += int(round(late * cmd.rate))

                # Use the pre-stretched copy if the worker finished in time.
                # A dict lookup and a reference swap -- no blocking, and if it
                # is not ready we simply resample as before.
                track = cmd.track
                if (
                    self._stretcher is not None
                    and track is not None
                    and not track.is_stretched
                    and deck.key_lock
                ):
                    ready = self._stretcher.get(track, cmd.rate)
                    if ready is not None:
                        track = ready
                        self.stretched_loads += 1
                    elif abs(cmd.rate - 1.0) >= config.STRETCH_DEADBAND:
                        self.resampled_loads += 1

                # A pause is a re-load of the audio the deck is already
                # running, stopped at the current playhead. Nothing about the
                # resulting position distinguishes it from a fresh cue, so
                # record it here, where both the before and after are visible.
                # Compared against cmd.track, not the possibly stretch-swapped
                # `track` above: the caller pauses by handing back the object
                # the deck already holds, and the swap would break that match.
                was_running_this = deck.playing and deck.track is cmd.track
                same_track = deck.track is not None and cmd.track is not None and (
                    deck.track.analysis is cmd.track.analysis
                )
                if not same_track:
                    # A different record: the old pitch fader position does
                    # not come with it. A pause or re-cue of the same one keeps it.
                    deck.manual_pitch = False

                deck.attach(track, start)
                deck.set_rate(cmd.rate)
                deck.load_rate = cmd.rate
                deck.playing = cmd.play
                deck.paused = bool(was_running_this and not cmd.play)
                if cmd.play:
                    # Starting a deck is what ends a stop. Without this the
                    # flag latched for the life of the process and the
                    # autopilot never ran again after a panic.
                    self.stop_requested = False
                if cmd.master and cmd.track is not None:
                    self._set_master(cmd.deck, cmd.track.analysis.bpm * cmd.rate)
                if self._trans_from is deck or self._trans_to is deck:
                    self._end_transition()

        elif isinstance(cmd, StartTransition):
            src = self._decks.get(cmd.from_deck)
            dst = self._decks.get(cmd.to_deck)
            if src is not None and dst is not None and cmd.total_frames > 0:
                self._trans_from = src
                self._trans_to = dst
                self._trans_frames = 0.0
                self._trans_total = float(cmd.total_frames)
                self._trans_bars_per_frame = tr.TRANSITION_BARS / self._trans_total
                self._trans_active = True
                dst.playing = True

                # Take the envelope the control thread built for this one.
                # Nothing is computed here: it is a reference swap and three
                # scalar reads.
                plan = self._armed_plan
                self._armed_plan = None
                if plan is None:
                    self._trans_env = None
                    self._trans_style = "bass_swap"
                    self._echo_delay = 1
                    self._echo_send = 0.0
                else:
                    env, style, block, delay, needs_echo, riser, needs_reverb = plan
                    self._trans_env = env
                    self._trans_style = style
                    self._trans_block = block
                    self._echo_delay = delay
                    self._echo_send = 0.0
                    self._riser = riser
                    self._riser_pos = 0
                    self._riser_gain = 0.0
                    self._riser_prev = 0.0
                    self._rv_send = 0.0
                    if needs_reverb and self._rv_tail <= 0:
                        # A fresh network, unless a previous tail is still
                        # ringing -- that is allowed to carry on. One memset.
                        self._rv_ring.fill(0.0)
                    self._trans_loop_len = 0.0
                    src.clear_loop()
                    # Loop offsets count from the downbeat the transition was
                    # placed on. A late release has already carried the deck
                    # past it, so step back by the lateness -- the same once-
                    # per-command correction LoadTrack applies above, and
                    # nothing per block.
                    origin = src.position
                    if not cmd.is_immediate and self.frames_played > cmd.execute_at:
                        origin -= (self.frames_played - cmd.execute_at) * src.rate
                    self._trans_src_origin = origin
                    self._trans_src_rate = src.rate
                    self._trans_rate_mult = 1.0
                    if needs_echo and self._echo_ring_left <= 0:
                        # Start from silence so a dead tail's leftovers cannot
                        # sound again. One memset, once. A tail still ringing
                        # from the last transition is left to finish.
                        self._echo_ring.fill(0.0)
                        self._echo_write = 0

        elif isinstance(cmd, SetEQ):
            deck = self._decks.get(cmd.deck)
            if deck is not None:
                deck.set_eq(cmd.low, cmd.mid, cmd.high)

        elif isinstance(cmd, SetGain):
            deck = self._decks.get(cmd.deck)
            if deck is not None:
                deck.set_gain(cmd.gain)
                # A manual level move outranks the crossfade's gain automation,
                # for the same reason a manual bass kill outranks its EQ
                # automation: the transition rewrites these every block, so a
                # running fade would undo the operator on the next one.
                if self._trans_from is deck or self._trans_to is deck:
                    self._end_transition()

        elif isinstance(cmd, SetFilter):
            deck = self._decks.get(cmd.deck)
            if deck is not None:
                deck.set_filter(cmd.position, cmd.resonance)

        elif isinstance(cmd, SetLoop):
            deck = self._decks.get(cmd.deck)
            if deck is not None and deck.track is not None:
                deck.set_loop(cmd.start_frame, cmd.length_frames)

        elif isinstance(cmd, ExitLoop):
            deck = self._decks.get(cmd.deck)
            if deck is not None:
                deck.clear_loop()

        elif isinstance(cmd, BeatJump):
            deck = self._decks.get(cmd.deck)
            if deck is not None and deck.track is not None:
                deck.jump(cmd.frames)

        elif isinstance(cmd, SetPitch):
            deck = self._decks.get(cmd.deck)
            if deck is not None and deck.track is not None:
                deck.set_rate(cmd.rate)
                deck.manual_pitch = True
                if self.master_deck == cmd.deck:
                    # The master's fader moves the master clock with it, so the
                    # other deck is corrected toward the new tempo.
                    self._beats_per_frame = (
                        deck.track.analysis.bpm * cmd.rate / 60.0 / SAMPLE_RATE
                    )

        elif isinstance(cmd, SyncDeck):
            deck = self._decks.get(cmd.deck)
            if deck is not None and deck.track is not None:
                deck.set_rate(cmd.rate)
                deck.manual_pitch = False
                # Always a new timeline for the supervisor, shifted or not.
                deck.jump(cmd.shift_frames)

        elif isinstance(cmd, SetKeyLock):
            deck = self._decks.get(cmd.deck)
            if deck is not None:
                deck.key_lock = cmd.on
                current = deck.track
                if current is not None:
                    if not cmd.on and current.source is not None:
                        # Back to resampling: the original under the playhead.
                        deck.swap_audio(current.source)
                    elif cmd.on and cmd.track is not None and cmd.track.source is current:
                        deck.swap_audio(cmd.track)

        elif isinstance(cmd, SwapStems):
            deck = self._decks.get(cmd.deck)
            if deck is not None and cmd.track is not None:
                # swap_audio refuses anything that is not this deck's own
                # track, so a stale mix from a superseded transition cannot
                # land on the wrong music.
                deck.swap_audio(cmd.track)

        elif isinstance(cmd, Cut):
            if cmd.deck == "master":
                for deck in self._decks.values():
                    deck.set_gain(0.0)
                self._end_transition()
                self._silence_effects()
            else:
                deck = self._decks.get(cmd.deck)
                if deck is not None:
                    deck.set_gain(0.0)
                    if self._trans_from is deck or self._trans_to is deck:
                        self._end_transition()

        elif isinstance(cmd, KillBass):
            level = 0.0 if cmd.killed else 1.0
            if cmd.deck == "master":
                for deck in self._decks.values():
                    deck.set_eq(low=level)
                # A manual bass kill outranks the transition's bass automation.
                self._end_transition()
            else:
                deck = self._decks.get(cmd.deck)
                if deck is not None:
                    deck.set_eq(low=level)
                    if self._trans_from is deck or self._trans_to is deck:
                        self._end_transition()

        elif isinstance(cmd, SetRate):
            deck = self._decks.get(cmd.deck)
            if deck is not None:
                # Deliberately does NOT retune the master clock, even when this
                # is the master deck. SetRate exists for drift correction, and
                # a correction measured *against* the master clock must not
                # move it -- otherwise the reference chases the deck, drift can
                # never read as zero, and the nudge is never taken back off.
                # The master tempo changes only on LoadTrack(master=True) and
                # at transition handover.
                deck.set_rate(cmd.rate)

        elif isinstance(cmd, Resync):
            deck = self._decks.get(cmd.deck)
            if deck is not None:
                deck.resync(cmd.frame)

        elif isinstance(cmd, Stop):
            # STOP has a defined post-state, and every deck lands in it.
            #
            # It used to zero the gains, clear `playing`, and stop there. That
            # left `paused` untouched, so what a deck reported afterwards
            # depended on what it had been doing before -- and it left the
            # gains at zero with nothing to restore them, so a later PLAY was
            # silent for no visible reason.
            #
            # What it deliberately does NOT do: tear down the stream, or
            # detach the tracks. Both decks stay loaded and ready.
            for deck in self._decks.values():
                # `jump`, not `set`: a stopped deck is never read, so its
                # parameter ramps stop advancing. A ramped target would leave
                # the deck reporting a level of 1.0 that it no longer has, and
                # the next PLAY would spend a block fading down from it.
                # Nothing is audible either way -- `playing` goes false in this
                # same block -- but the state has to mean what it says.
                deck.gain.jump(0.0)
                deck.playing = False
                deck.paused = False
                deck.set_eq(low=1.0, mid=1.0, high=1.0)
                deck.filter_pos.jump(0.0)
                if deck.track is not None:
                    deck.position = 0.0
                    deck.ended = False
            self._end_transition()
            self._echo_send = 0.0
            self._silence_effects()
            self.stop_requested = True

    def _end_transition(self) -> None:
        self._trans_active = False
        self._trans_from = None
        self._trans_to = None
        self._trans_env = None
        # Dropping the send is what stops the delay line being read at all.
        self._echo_send = 0.0
        # The reverb's send closes; its tail rings on by itself. The riser is
        # over the moment the transition is.
        self._rv_send = 0.0
        self._riser_gain = 0.0

    def _silence_effects(self) -> None:
        """Panic: no reverb tail and no riser keep sounding. AUDIO THREAD."""
        self._rv_send = 0.0
        self._rv_tail = 0
        self._echo_send = 0.0
        self._echo_ring_left = 0
        self._riser_gain = 0.0
        self._riser_prev = 0.0

    def arm_transition_plan(
        self,
        style: str,
        total_frames: int,
        bpm: float,
    ) -> str:
        """Build the envelope for the next transition. **Control thread only.**

        Every allocation this transition will ever need happens here: the
        envelope array and the delay time. The audio thread then only reads
        them. Returns the style actually armed.
        """
        style = style if style in tr.STYLES else "bass_swap"
        env = tr.build_envelope(
            style, int(total_frames), self.blocksize, bpm, SAMPLE_RATE
        )
        delay = max(1, min(self._echo_len - 1,
                           tr.echo_delay_frames(bpm, SAMPLE_RATE)))
        self._armed_plan = (env, style, self.blocksize, delay) + self._effects_for(env)
        return style

    def _effects_for(self, env) -> tuple[bool, "np.ndarray | None", bool]:
        """CONTROL THREAD. What an envelope needs: echo, a rendered riser, reverb."""
        needs_echo = bool(env[:, tr.ECHO_SEND].max() > 0.0)
        needs_reverb = bool(env[:, tr.REVERB_SEND].max() > 0.0)
        riser = None
        active = np.nonzero(env[:, tr.RISER_GAIN] > 0.0)[0]
        if active.size:
            # Long enough for every block the gain is up, plus the block its
            # ramp back down runs over.
            riser = render_riser((int(active.size) + 2) * self.blocksize)
        return needs_echo, riser, needs_reverb

    def arm_transition_params(
        self, params, total_frames: int, bpm: float
    ) -> str:
        """Build the envelope for a designed transition. **Control thread.**

        The named presets deliberately do NOT come through here: they keep
        their own builders, so a preset is bit-identical to what it has always
        produced and the Phase 1 criteria still measure the same shape. This
        path is for points in the parameter space that no preset names.
        """
        env = tr.build_envelope_from_params(
            params, int(total_frames), self.blocksize, bpm, SAMPLE_RATE
        )
        delay = max(1, min(self._echo_len - 1,
                           tr.echo_delay_frames(bpm, SAMPLE_RATE)))
        self._armed_plan = (env, params.name, self.blocksize, delay) + self._effects_for(env)
        return params.name

    @property
    def transition_style(self) -> str:
        """Style of the running transition, for the UI and the logs."""
        return self._trans_style

    # --- the master limiter (AUDIO THREAD) -----------------------------------

    def _limiter_target(self, peak: float) -> float:
        """Gain that maps ``peak`` onto the soft-knee curve. Pure float maths.

        Below the knee the limiter is transparent. Above it the output
        approaches the ceiling asymptotically, so however hot the input gets
        the output never reaches full scale and the curve has no corner in it
        for the ear to catch.
        """
        knee = MASTER_KNEE
        span = MASTER_CEILING - knee
        if peak <= knee or span <= 0.0:
            return 1.0
        over = (peak - knee) / span
        allowed = knee + span * (1.0 - math.exp(-over))
        return allowed / peak

    def _limit(self, mix: np.ndarray, frames: int) -> None:
        """Soft-knee limiter, last stage before the output. AUDIO THREAD.

        Works in short sub-blocks with an instantaneous attack: the gain for a
        sub-block is computed from that sub-block's own peak, so the ceiling is
        arithmetic rather than aspiration -- there is no window in which a
        transient can get out before the gain catches up. Release is a one-pole
        glide back toward unity.

        Every buffer is preallocated and every operation writes into an
        existing one. Slices are views, not copies.
        """
        step = MASTER_SUBBLOCK
        gain = self._limiter_gain
        peak_in = 0.0
        peak_out = 0.0
        min_gain = gain

        i = 0
        while i < frames:
            n = step if i + step <= frames else frames - i
            sub = mix[i:i + n]
            scratch = self._abs_buf[:n]
            np.abs(sub, out=scratch)
            peak = float(scratch.max())
            if peak > peak_in:
                peak_in = peak

            target = self._limiter_target(peak)
            if target < gain:
                gain = target          # attack: immediate, so nothing escapes
            else:
                gain += (target - gain) * self._limiter_release
            if gain < min_gain:
                min_gain = gain

            if gain < 1.0:
                np.multiply(sub, gain, out=sub)
            out = peak * gain
            if out > peak_out:
                peak_out = out
            i += n

        self._limiter_gain = gain
        self.limiter_gain = gain
        self.limiter_min_gain = min_gain
        if min_gain < self._log_reduction_gain:
            self.limiter_heavy_blocks += 1
            if min_gain < self.limiter_deepest_gain:
                self.limiter_deepest_gain = min_gain
        self.master_peak_in = peak_in
        self.master_peak = peak_out

        # Backstop. By construction it cannot fire; counted rather than
        # trusted, because "cannot" and "does not" are different claims.
        if peak_out > MASTER_CEILING + 1e-4:
            self.limiter_clips += 1
            np.clip(mix, -MASTER_CEILING, MASTER_CEILING, out=mix)

    # --- the echo send (AUDIO THREAD) ----------------------------------------

    def _apply_echo(self, src: Deck | None, frames: int, mix: np.ndarray) -> None:
        """Add deck A's beat-synced echo tail to the mix. AUDIO THREAD.

        The send decides what goes in; the repeats come back out at the fixed
        ECHO_RETURN level. That split is what lets a tail ring out: once the
        send closes -- or the transition hands over and there is no deck A to
        feed it -- nothing new goes in, and what is already in the delay line
        keeps sounding, each repeat ECHO_FEEDBACK quieter than the last.

        Every buffer here is preallocated. The ring is written and read with
        two slice copies each, so a wrap costs no more than a straight run.
        """
        n = self._echo_len
        d = self._echo_delay
        send = self._echo_send

        delayed = self._echo_read_buf[:frames]
        r = (self._echo_write - d) % n
        first = min(frames, n - r)
        delayed[:first] = self._echo_ring[r:r + first]
        if first < frames:
            delayed[first:] = self._echo_ring[:frames - first]

        work = self._echo_work[:frames]

        # The tail into the mix, at the return level -- not the send level.
        np.multiply(delayed, tr.ECHO_RETURN, out=work)
        np.add(mix, work, out=mix)
        if self._rv_send > 0.0:
            # And into the reverb, which takes the echo as part of its input.
            self._echo_tail[:frames] = work
            self._echo_tail_live = True

        # What goes back in: this block's source at the send level, plus the
        # decayed tail. Feedback below unity is what makes the repeats die.
        # Fed from after the deck's EQ and filter, before its fader. It used to
        # take the raw block, so `filter_echo` echoed back the full-range track
        # its filter was removing: measured 1.091 into the limiter, with deck A
        # at gain 0.05 and deck B at full.
        if send > 0.0 and src is not None:
            np.multiply(src._pre_gain[:frames], send, out=work)
        else:
            work.fill(0.0)
        np.multiply(delayed, tr.ECHO_FEEDBACK, out=delayed)
        np.add(work, delayed, out=work)

        w = self._echo_write
        first = min(frames, n - w)
        self._echo_ring[w:w + first] = work[:first]
        if first < frames:
            self._echo_ring[:frames - first] = work[first:]
        self._echo_write = (w + frames) % n

    def _apply_reverb(self, frames: int, mix: np.ndarray) -> None:
        """The reverb: read every line, mix, write back. AUDIO THREAD.

        Block-wise, which is only correct because every delay is at least a
        block long: nothing read here was written in this block. Every buffer
        is preallocated; slices are views, and each ring is written and read
        with at most two slice copies.
        """
        delays = self._rv_delays
        if frames > delays[0]:
            return  # a block longer than the shortest line: cannot run block-wise
        n = self._rv_len
        w = self._rv_write
        ring = self._rv_ring
        total = self._rv_sum[:frames]
        total.fill(0.0)
        prev = self._rv_prev[:frames]
        for k in range(len(delays)):
            buf = self._rv_read[k, :frames]
            r = (w - delays[k]) % n
            first = min(frames, n - r)
            buf[:first] = ring[k, r:r + first]
            if first < frames:
                buf[first:] = ring[k, :frames - first]
            # Damping: average with the sample before, a gentle high cut that
            # compounds on every pass, which is what makes a tail darken.
            r1 = (r - 1) % n
            first = min(frames, n - r1)
            prev[:first] = ring[k, r1:r1 + first]
            if first < frames:
                prev[first:] = ring[k, :frames - first]
            np.add(buf, prev, out=buf)
            np.multiply(buf, 0.5, out=buf)
            np.add(total, buf, out=total)

        # Output: the lines with alternating signs, into the mix.
        out = self._rv_work[:frames]
        out.fill(0.0)
        for k in range(len(delays)):
            if k % 2:
                np.subtract(out, self._rv_read[k, :frames], out=out)
            else:
                np.add(out, self._rv_read[k, :frames], out=out)
        np.multiply(out, self._rv_wet, out=out)
        np.add(mix, out, out=mix)

        # Input: deck A pre-fader at the send level, plus the echo's tail.
        x = self._rv_in[:frames]
        src = self._trans_from
        if self._rv_send > 0.0 and src is not None:
            np.multiply(src._pre_gain[:frames], self._rv_send, out=x)
            if self._echo_tail_live:
                np.add(x, self._echo_tail[:frames], out=x)
        else:
            x.fill(0.0)
        self._echo_tail_live = False

        # Householder feedback: each line gets itself minus the mean of all,
        # scaled below unity so the tail decays.
        np.multiply(total, self._rv_mix, out=total)
        work = self._rv_work[:frames]
        for k in range(len(delays)):
            np.subtract(self._rv_read[k, :frames], total, out=work)
            np.multiply(work, self._rv_feedback, out=work)
            np.add(work, x, out=work)
            first = min(frames, n - w)
            ring[k, w:w + first] = work[:first]
            if first < frames:
                ring[k, :frames - first] = work[first:]
        self._rv_write = (w + frames) % n

    def _apply_riser(self, frames: int, mix: np.ndarray) -> None:
        """Play the prerendered riser into the mix, gain ramped. AUDIO THREAD."""
        buf = self._riser
        if buf is None:
            self._riser_prev = 0.0
            return
        pos = self._riser_pos
        m = min(frames, buf.shape[0] - pos)
        gain, prev = self._riser_gain, self._riser_prev
        self._riser_prev = gain
        if m <= 0:
            return
        ramp = self._rs_ramp[:m]
        np.multiply(self._rs_unit[:m], gain - prev, out=ramp)
        np.add(ramp, prev, out=ramp)
        work = self._rs_work[:m]
        np.multiply(buf[pos:pos + m], ramp, out=work)
        np.add(mix[:m], work, out=mix[:m])
        self._riser_pos = pos + m

    def _drain(self) -> None:
        """Pull every pending command. Bounded so a flood cannot stall a block."""
        q = self.queue
        for _ in range(32):
            if q.empty():
                return
            try:
                self._apply(q.get_nowait())
            except queue.Empty:
                return

    def _abort_transition(self) -> None:
        """Stop a transition part-way and leave both decks usable. AUDIO THREAD.

        Whatever the reason, an aborted crossfade must not strand a deck
        half-faded with its bass swapped out. Every deck still playing goes
        back to unity and a neutral EQ; the one that stopped gets its EQ
        neutralised too, so the next track to land on it does not inherit a
        filtered channel.
        """
        for deck in (self._trans_from, self._trans_to):
            if deck is None:
                continue
            if deck.playing:
                deck.set_gain(1.0)
            deck.set_eq(low=1.0, mid=1.0, high=1.0)
            deck.set_filter(0.0, self._knob_resonance)
            # A style may have left the deck looping or running backwards.
            deck.clear_loop()
            if deck is self._trans_from and self._trans_rate_mult != 1.0:
                # Back to the beatmatch it had, not to 1.0.
                deck.set_rate(self._trans_src_rate)
            elif deck.rate <= 0.0:
                deck.set_rate(1.0)
        self._trans_rate_mult = 1.0
        self._end_transition()

    def _advance_transition(self, frames: int) -> None:
        """Step the crossfade automation. Pure float math."""
        src = self._trans_from
        dst = self._trans_to
        if src is None or dst is None:
            self._end_transition()
            return

        # Gated on deck STATE, not on a flag and a block count alone.
        #
        # `_trans_active` says a transition was started; it says nothing about
        # whether both decks are still running. A crossfade that keeps
        # advancing across a paused deck writes an envelope for a deck that is
        # not playing, and strands the one that is at whatever gain and EQ the
        # curve had reached. The state of the decks is the real precondition,
        # so it is the thing tested.
        if not (src.playing and dst.playing):
            self._abort_transition()
            return

        self._trans_frames += frames
        env = self._trans_env
        if env is None:
            # No plan was armed: the original per-block shape, unchanged.
            g = tr.gains_at(self._trans_frames * self._trans_bars_per_frame)
            src.gain.set(g.from_gain)
            src.eq_low.set(g.from_low)
            dst.gain.set(g.to_gain)
            dst.eq_low.set(g.to_low)
            self._echo_send = 0.0
        else:
            # One clamped row lookup and eight assignments. No curve maths, no
            # branch on style: the style is already baked into the array.
            i = int(self._trans_frames) // self._trans_block
            if i >= env.shape[0]:
                i = env.shape[0] - 1
            row = env[i]
            src.gain.set(float(row[tr.FROM_GAIN]))
            src.eq_low.set(float(row[tr.FROM_LOW]))
            src.eq_mid.set(float(row[tr.FROM_MID]))
            src.eq_high.set(float(row[tr.FROM_HIGH]))
            dst.gain.set(float(row[tr.TO_GAIN]))
            dst.eq_low.set(float(row[tr.TO_LOW]))
            dst.eq_mid.set(float(row[tr.TO_MID]))
            dst.eq_high.set(float(row[tr.TO_HIGH]))
            self._echo_send = float(row[tr.ECHO_SEND])
            # Filter knobs, as positions: the decks' tables hold the maths.
            src.filter_pos.set(float(row[tr.FROM_FILTER]))
            dst.filter_pos.set(float(row[tr.TO_FILTER]))
            res = float(row[tr.FILTER_RES])
            src.filter_res = res
            dst.filter_res = res
            self._rv_send = float(row[tr.REVERB_SEND])
            self._riser_gain = float(row[tr.RISER_GAIN])

            # Deck control, still just table lookups. The loop region was
            # resolved to frames on the control thread; here it is one
            # comparison per block, and one addition when it changes.
            length = float(row[tr.LOOP_LEN])
            if length != self._trans_loop_len:
                self._trans_loop_len = length
                if length > 0.0:
                    src.set_loop(
                        self._trans_src_origin + float(row[tr.LOOP_START]), length
                    )
                else:
                    src.clear_loop()
            # RATE_A is a multiplier on the rate the deck was beatmatched at.
            # It was once assigned as an absolute rate, which reset every
            # outgoing deck to 1.0 mid-blend -- a 2.9% error the supervisor then
            # hard-resynced every bar. Applied only when it changes, so a steady
            # 1.0 leaves drift correction alone.
            mult = float(row[tr.RATE_A])
            if mult != self._trans_rate_mult:
                self._trans_rate_mult = mult
                src.set_rate(self._trans_src_rate * mult)

        if self._trans_frames >= self._trans_total:
            src.playing = False
            src.gain.set(0.0)
            dst.gain.set(1.0)
            dst.eq_low.set(1.0)
            # Hand back a neutral EQ on both decks. A filter sweep leaves the
            # outgoing deck high-passed, and the next track to land on it must
            # not inherit that.
            dst.eq_mid.set(1.0)
            dst.eq_high.set(1.0)
            src.eq_low.set(1.0)
            src.eq_mid.set(1.0)
            src.eq_high.set(1.0)
            # And open filters. The outgoing deck has stopped, so its knob
            # jumps rather than ramping into the next track's first block.
            src.filter_pos.jump(0.0)
            dst.filter_pos.set(0.0)
            src.filter_res = self._knob_resonance
            dst.filter_res = self._knob_resonance
            self._echo_send = 0.0
            # The incoming deck is now the mix, so it takes over the beat clock.
            if dst.track is not None:
                self._set_master(dst.name, dst.track.analysis.bpm * dst.rate)
            self._end_transition()

    # --- the callback (AUDIO THREAD) -----------------------------------------

    def callback(
        self, outdata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        try:
            if status:
                # `status` is truthy on underflow/overflow. Counting is the only
                # reporting allowed here; the monitor thread does the logging.
                self.underruns += 1

            # A single reference read. Once the watchdog has decided this engine
            # is dead, the room hears the fallback from the very next block --
            # no stream teardown, no gap.
            fallback = self.fallback
            if fallback is not None:
                fallback.fill(outdata, frames)
                self.last_callback_at = time.monotonic()
                self.callbacks += 1
                return

            self._drain()

            if self._trans_active:
                self._advance_transition(frames)

            a = self.deck_a.read(frames)
            b = self.deck_b.read(frames)

            mix = self._mix[:frames]
            np.add(a, b, out=mix)

            # Only `echo_out` ever sets a send, so every other style pays one
            # float comparison for this.
            # The echo runs while its send is open and until its repeats have
            # died away after that, including past the hand-over.
            if (self._echo_send > 0.0 and self._trans_from is not None) or (
                self._echo_ring_left > 0
            ):
                self._apply_echo(self._trans_from, frames, mix)
                if self._echo_send > 0.0:
                    self._echo_ring_left = self._echo_ring_repeats * self._echo_delay
                else:
                    self._echo_ring_left -= frames

            # The reverb runs while its send is open and for its tail after;
            # the riser while its gain, or its last ramp down, is non-zero.
            # Every other block pays two float comparisons.
            if self._rv_send > 0.0 or self._rv_tail > 0:
                self._apply_reverb(frames, mix)
                if self._rv_send > 0.0:
                    self._rv_tail = self._rv_tail_frames
                else:
                    self._rv_tail -= frames
            if self._riser_gain > 0.0 or self._riser_prev > 0.0:
                self._apply_riser(frames, mix)

            if self._master_gain != 1.0:
                np.multiply(mix, self._master_gain, out=mix)

            # Last stage before the output, and before the cue tap and the
            # recorder, so all three carry the same audio the room hears.
            self._limit(mix, frames)
            if outdata.shape[1] == CHANNELS:
                outdata[:] = mix
            else:
                # A wider stream: master on 1/2, cue on its own pair. Anything
                # we do not drive has to be zeroed or the card repeats the last
                # block into it.
                outdata.fill(0.0)
                outdata[:, 0:CHANNELS] = mix

            self._tap_cue(frames, outdata)

            if self.recorder is not None:
                self.recorder.capture(mix, frames)

            self.frames_played += frames
            self.master_beat += frames * self._beats_per_frame
            self._clock = (
                self.frames_played,
                self.master_beat,
                self.deck_a.position,
                self.deck_b.position,
            )
            self.callbacks += 1
            self.last_callback_at = time.monotonic()
        except Exception:
            # An exception escaping the callback tears down the stream. Silence
            # is survivable; a dead stream is not. The monitor thread notices
            # via the underrun/callback counters, and the watchdog promotes the
            # fallback if this keeps happening.
            outdata.fill(0.0)
            self.underruns += 1

    def _tap_cue(self, frames: int, outdata: np.ndarray) -> None:
        """Send the cued deck's pre-fader signal to the cue output. AUDIO THREAD.

        Takes ``Deck._raw``, the interpolated source block the deck has just
        rendered. That is deliberate and it is why cue costs nothing: the deck
        is not read a second time, so its position is not advanced twice, and
        no buffer is allocated. It is the signal *before* the crossfader, which
        is the whole point -- you need to hear the incoming track while its
        fader is still down.

        ``deck.py`` is outside this phase's scope, so this reaches into that
        buffer rather than adding an accessor for it. It is valid until that
        deck's next ``read``, which is the next block.
        """
        name = self.cue_deck
        if name is None:
            return
        deck = self._decks.get(name)
        if deck is None:
            return
        raw = deck._raw[:frames]
        if self.cue_channels is not None:
            lo = self.cue_channels[0] - 1
            outdata[:, lo : lo + CHANNELS] = raw
        elif self._cue_ring is not None:
            self._cue_ring.write(raw, frames)

    def _cue_callback(
        self, outdata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        """The second device's callback. AUDIO THREAD (a different one).

        Drains whatever the master callback has produced. The two streams have
        independent clocks, so this deliberately does not try to stay in step:
        it plays what is there and fills the rest with silence. Pre-listen is
        allowed to be a block late; it is not allowed to block the master.
        """
        try:
            out = self._cue_out[:frames]
            ring = self._cue_ring
            took = 0 if ring is None else ring.read_into(out, frames)
            if took < frames:
                out[took:].fill(0.0)
                self.cue_underruns += 1
            outdata[:] = out
        except Exception:
            outdata.fill(0.0)
            self.cue_underruns += 1
