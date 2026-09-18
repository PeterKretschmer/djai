"""One deck: a loaded track, a playhead, a gain, a 3-band EQ and a playback rate.

THREADING CONTEXT
-----------------
* :meth:`Deck.read` runs on the **audio thread**. It must never allocate large
  buffers, block, log, or touch the filesystem. Every working buffer is
  preallocated in ``__init__``; the only allocations left in the hot path are
  the two output arrays scipy's ``sosfilt`` returns (it has no ``out=``
  parameter) and small numpy view objects.
* :func:`load_track` runs on a **worker thread** -- it decodes and resamples a
  whole file. Never call it from the audio thread.
* Setter methods (``set_gain``, ``set_eq``, ``set_rate`` ...) are called from
  the audio thread only, by :class:`djai.engine.Engine` while draining its
  command queue. They only assign floats.

Tempo matching: with key lock on (the default) a deck plays a copy stretched by
Rubber Band on a worker thread, at original pitch. With key lock off, or until
that copy is ready, the deck resamples: 124 BPM at rate 1.016 becomes 126 BPM
and its pitch rises ~28 cents.

After the EQ each deck has a resonant filter on one centre-detented knob: left
low-passes, right high-passes. Its coefficients are a table built once off the
audio thread; the callback only indexes that table and runs the filter.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from djai import config
from djai.analysis import HOT_CUE_SR, TrackAnalysis

# --- audio format ------------------------------------------------------------

SAMPLE_RATE: int = 44100

#: Frames of crossfade at a loop seam. ~1.5 ms: long enough to remove the
#: waveform discontinuity that would otherwise click on every lap, short
#: enough that the loop still lands on the beat it was cut to.
_LOOP_XFADE: int = 64

# Hot cue positions are stored in deck frames. analysis.py cannot import this
# module (it would be circular), so the agreement is asserted from this side.
assert HOT_CUE_SR == SAMPLE_RATE, (
    f"hot cue rate {HOT_CUE_SR} != deck rate {SAMPLE_RATE}"
)
CHANNELS: int = 2
DTYPE = np.float32

#: Largest block the audio callback may ask for. Buffers are sized to this.
MAX_BLOCK: int = 8192

# --- EQ ----------------------------------------------------------------------

#: Crossover frequencies for the 3-band EQ, in Hz.
EQ_LOW_XOVER: float = 250.0
EQ_HIGH_XOVER: float = 2500.0

#: The bands are split by Linkwitz-Riley 4th-order crossovers: a pair of
#: cascaded 2nd-order Butterworths, which is what real DJ mixers use.
#:
#: The obvious cheaper alternative -- one lowpass per crossover and subtraction
#: for the upper bands -- is wrong, and wrong in a way that is easy to miss. It
#: reconstructs *perfectly* at unity gain (the bands telescope back to the
#: input algebraically), so a flat-EQ test passes. But a Butterworth's phase
#: shift means `raw - low` does not cancel low frequencies: at 60 Hz through a
#: 250 Hz lowpass the magnitude is 1.0 yet the phase is ~-54 deg, so the
#: "kill" leaves 62% of the bass amplitude. Measured, not theorised.
#:
#: LR4 branches are in phase at their crossover and sum to an allpass, so flat
#: EQ preserves the magnitude spectrum (not the exact samples) and killing a
#: band genuinely removes it (-49 dB at 60 Hz with the low band cut).
EQ_ORDER: int = 2  # per Butterworth stage; cascaded twice for LR4

# --- resonant filter ---------------------------------------------------------

#: Cutoff with the knob fully left (low-pass) and fully right (high-pass), Hz.
FILTER_LP_MIN_HZ: float = 60.0
FILTER_HP_MAX_HZ: float = 12000.0
#: Where each side starts, just past the detent: effectively open.
FILTER_LP_OPEN_HZ: float = 20000.0
FILTER_HP_OPEN_HZ: float = 20.0
#: Resonance 0 is a Butterworth response, no peak; resonance 1 is Q 4.
FILTER_Q_MIN: float = 0.7071
FILTER_Q_MAX: float = 4.0
#: The knob's centre detent. Inside it the filter is bypassed outright.
FILTER_DETENT: float = 0.02
#: Past the detent the filtered signal fades in over this much knob travel, so
#: engaging the filter never switches it in with a step.
FILTER_FADE: float = 0.06
#: Table resolution. 1024 steps per side moves the cutoff 0.57% per step across
#: the low-pass range -- finer than a hand or an envelope row moves it.
FILTER_STEPS: int = 1024
FILTER_RES_STEPS: int = 17
#: While the knob moves, coefficients step every this many frames (1.45 ms).
FILTER_SUBBLOCK: int = 64

# --- parameter smoothing -----------------------------------------------------

#: Every audible parameter is ramped linearly across one block instead of
#: jumping, which is what keeps EQ kills and cuts from clicking. One block at
#: 512 frames is ~11.6 ms.
_EPS = 1e-12


@dataclass
class LoadedTrack:
    """Decoded audio plus its analysis. Immutable once built; safe to publish
    to the audio thread by a single reference assignment."""

    analysis: TrackAnalysis
    audio: np.ndarray  # (n_frames, CHANNELS) float32, C-contiguous

    #: Factor this audio has already been time-compressed by, at original
    #: pitch. 1.0 means raw audio that the deck must resample to change tempo.
    stretch_rate: float = 1.0

    #: The unstretched track this copy was made from, or None for raw audio.
    #: Kept so turning key lock off can put the original back under the
    #: playhead without decoding anything.
    source: LoadedTrack | None = None

    @property
    def n_frames(self) -> int:
        """Frames in the buffer we hold."""
        return int(self.audio.shape[0])

    @property
    def source_frames(self) -> float:
        """Length in ORIGINAL track frames, which is what positions count in."""
        return self.audio.shape[0] * self.stretch_rate

    @property
    def is_stretched(self) -> bool:
        return self.stretch_rate != 1.0

    @property
    def title(self) -> str:
        return self.analysis.title


def load_track(analysis: TrackAnalysis) -> LoadedTrack:
    """Decode ``analysis.path`` to (n, 2) float32 at :data:`SAMPLE_RATE`.

    Blocking and slow. Worker-thread only.
    """
    audio, file_sr = sf.read(str(Path(analysis.path)), dtype="float32", always_2d=True)

    if audio.shape[1] == 1:
        audio = np.repeat(audio, CHANNELS, axis=1)
    elif audio.shape[1] > CHANNELS:
        audio = audio[:, :CHANNELS]

    if file_sr != SAMPLE_RATE:
        import librosa  # local: keeps the audio-thread import surface small

        audio = librosa.resample(
            audio.T.astype(np.float32), orig_sr=file_sr, target_sr=SAMPLE_RATE
        ).T

    # Loudness normalisation, applied here at load and never in the callback:
    # one scalar over the decoded buffer, on the thread that decoded it, so
    # everything downstream -- stretched copies, the fallback, the channel
    # fader -- sees the normalised audio.
    gain_db = float(getattr(analysis, "track_gain_db", 0.0) or 0.0)
    if gain_db:
        audio = audio * np.float32(10.0 ** (gain_db / 20.0))

    # One trailing frame of silence so linear interpolation can always read
    # index+1 without a bounds check in the hot path.
    audio = np.ascontiguousarray(audio, dtype=DTYPE)
    audio = np.vstack([audio, np.zeros((2, CHANNELS), dtype=DTYPE)])
    return LoadedTrack(analysis=analysis, audio=audio)


class StretchError(RuntimeError):
    """A stretch could not be produced. Callers fall back to resampling."""


#: The Rubber Band command-line tool, vendored under ``djai/bin/rubberband``
#: (GPL; COPYING.txt sits beside it). A ``rubberband`` on PATH is used if the
#: vendored copy is absent.
RUBBERBAND_BIN: Path = (
    Path(__file__).resolve().parent / "bin" / "rubberband"
    / ("rubberband.exe" if os.name == "nt" else "rubberband")
)

#: The R3 ("fine") engine. Measured at +6% against R2 and against the librosa
#: phase vocoder it replaced, on a 440 Hz tone and a click train:
#:
#:   pitch shift         R3 0.00 cents   R2 +25.8 cents   vocoder 0.00 cents
#:   energy in 5 ms      R3 0.72         R2 0.58          vocoder 0.65
#:   seconds, 209 s trk  R3 16.6         R2 3.7           vocoder 2.9
#:
#: R3 is the one that keeps both pitch and attack. Its cost is time, which the
#: cue's pre-roll absorbs: a cue arrives a couple of phrases ahead of its blend.
RUBBERBAND_ENGINE_ARGS: tuple[str, ...] = ("--fine",)

#: Where the tool's input and output files sit while it runs: inside the
#: project, and each pair is deleted as soon as it has been read back.
STRETCH_TEMP_DIR: Path = Path(__file__).resolve().parent.parent / "cache" / "_stretch"

#: Frames written to the tool per write call: bounds the handover's memory.
_STRETCH_WRITE_CHUNK: int = SAMPLE_RATE * 10


def _rubberband_exe() -> str:
    return str(RUBBERBAND_BIN) if RUBBERBAND_BIN.exists() else "rubberband"


def time_stretch(track: LoadedTrack, rate: float) -> LoadedTrack:
    """Time-compress ``track`` by ``rate`` at its original pitch.

    THREADING CONTEXT: **worker thread only.** Rubber Band over a whole decoded
    track is seconds of CPU in a child process. It must never be called from
    the audio thread, and :mod:`djai.engine` only ever calls it from its
    stretch worker.

    ``rate`` >1 makes the track shorter and faster. So a 124 BPM track played at
    a 128 BPM master needs ``rate = 128/124``.

    The tool is run directly rather than through ``pyrubberband``, which hands
    audio over as 16-bit WAV (clipping anything the stretch pushes past full
    scale) and keeps two extra full-length copies in memory. Here the input is
    written as float in chunks straight from the decoded buffer, and the output
    is read into the one array the deck will play.

    Raises :class:`StretchError` on any failure; the caller resamples instead.
    """
    if not np.isfinite(rate) or rate <= 0:
        raise StretchError(f"nonsense rate {rate!r}")
    # Epsilon because the limit is inclusive and floats are not: 1.08 - 1.0 is
    # 0.08000000000000007, so a track needing exactly the maximum would be
    # refused and fall back to resampling -- a full semitone -- at the boundary.
    if abs(rate - 1.0) > config.MAX_STRETCH_RATIO + 1e-9:
        raise StretchError(
            f"rate {rate:.4f} exceeds the +-{config.MAX_STRETCH_RATIO:.0%} limit"
        )

    # Drop the two interpolation guard frames before stretching, then put them
    # back, so they stay silent rather than being smeared into the audio.
    source = track.audio[:-2] if track.audio.shape[0] > 2 else track.audio
    n_in = int(source.shape[0])

    stem = f"{os.getpid()}_{id(track)}_{time.monotonic_ns()}"
    infile = STRETCH_TEMP_DIR / f"{stem}_in.wav"
    outfile = STRETCH_TEMP_DIR / f"{stem}_out.wav"
    try:
        STRETCH_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(
            str(infile), "w", SAMPLE_RATE, CHANNELS, subtype="FLOAT"
        ) as fh:
            for i in range(0, n_in, _STRETCH_WRITE_CHUNK):
                fh.write(source[i:i + _STRETCH_WRITE_CHUNK])

        command = [
            _rubberband_exe(), "-q", *RUBBERBAND_ENGINE_ARGS,
            "--tempo", f"{float(rate):.8f}", str(infile), str(outfile),
        ]
        subprocess.run(
            command,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            # Far beyond the ~0.08x realtime R3 measured, so only a hung tool
            # ever reaches it.
            timeout=60.0 + 2.0 * n_in / SAMPLE_RATE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        with sf.SoundFile(str(outfile)) as fh:
            frames = int(fh.frames)
            if fh.channels != CHANNELS or fh.samplerate != SAMPLE_RATE or frames <= 0:
                raise StretchError(
                    f"rubberband wrote {fh.channels} ch at {fh.samplerate} Hz, "
                    f"{frames} frames"
                )
            out = np.zeros((frames + 2, CHANNELS), dtype=DTYPE)
            got = fh.read(frames, dtype="float32", out=out[:frames])
            if got.shape[0] != frames:
                raise StretchError(f"rubberband output short: {got.shape[0]} of {frames}")
    except StretchError:
        raise
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip()[-200:]
        raise StretchError(f"rubberband exited {exc.returncode}: {detail}") from exc
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        raise StretchError(f"rubberband failed: {type(exc).__name__}: {exc}") from exc
    finally:
        for path in (infile, outfile):
            try:
                path.unlink()
            except OSError:
                pass

    return LoadedTrack(
        analysis=track.analysis, audio=out, stretch_rate=float(rate), source=track
    )


def _lr4(cutoff: float, btype: str) -> np.ndarray:
    """A Linkwitz-Riley 4th-order section: one Butterworth-2 stage, cascaded."""
    stage = butter(EQ_ORDER, cutoff, btype=btype, fs=SAMPLE_RATE, output="sos")
    return np.vstack([stage, stage])


def filter_cutoff_hz(position: float) -> float:
    """Cutoff for a knob position in -1..1. Negative low-passes, positive high-passes.

    Exponential in travel, so equal turns of the knob are equal musical
    intervals rather than equal numbers of hertz.
    """
    m = min(1.0, abs(float(position)))
    if position < 0:
        return FILTER_LP_OPEN_HZ * (FILTER_LP_MIN_HZ / FILTER_LP_OPEN_HZ) ** m
    return FILTER_HP_OPEN_HZ * (FILTER_HP_MAX_HZ / FILTER_HP_OPEN_HZ) ** m


def filter_q(resonance: float) -> float:
    """Q for a resonance in 0..1."""
    r = min(1.0, max(0.0, float(resonance)))
    return FILTER_Q_MIN * (FILTER_Q_MAX / FILTER_Q_MIN) ** r


_FILTER_TABLE: np.ndarray | None = None
_FILTER_WET: list[float] | None = None


def filter_tables() -> tuple[np.ndarray, list[float]]:
    """Every coefficient the filter knob can select, and its wet level.

    THREADING CONTEXT: built once, on whichever control thread constructs the
    first :class:`Deck`; read-only afterwards. This is where all of the
    filter's maths happens, so the callback never computes a coefficient.

    Returns ``(table, wet)``. ``table`` is float64, shaped
    ``(2, FILTER_STEPS, FILTER_RES_STEPS, 1, 6)``: side 0 low-pass, side 1
    high-pass, one biquad per knob step and resonance step, in scipy's
    second-order-section layout. ``wet`` is a plain list of floats, one per
    knob step, so reading it in the callback creates no objects.

    RBJ-cookbook biquads. Each is scaled down by sqrt(Q_min / Q): a resonant
    peak adds up to Q in level at the cutoff, and taking some of that off the
    whole response keeps a high-resonance sweep from driving the limiter.
    """
    global _FILTER_TABLE, _FILTER_WET
    if _FILTER_TABLE is not None and _FILTER_WET is not None:
        return _FILTER_TABLE, _FILTER_WET

    travel = np.arange(FILTER_STEPS, dtype=np.float64) / (FILTER_STEPS - 1)
    q = filter_q(0.0) * (FILTER_Q_MAX / FILTER_Q_MIN) ** (
        np.arange(FILTER_RES_STEPS, dtype=np.float64) / (FILTER_RES_STEPS - 1)
    )
    table = np.zeros((2, FILTER_STEPS, FILTER_RES_STEPS, 1, 6), dtype=np.float64)
    comp = np.sqrt(FILTER_Q_MIN / q)[None, :]
    for side, (open_hz, end_hz) in enumerate(
        ((FILTER_LP_OPEN_HZ, FILTER_LP_MIN_HZ), (FILTER_HP_OPEN_HZ, FILTER_HP_MAX_HZ))
    ):
        cutoff = open_hz * (end_hz / open_hz) ** travel
        w0 = (2.0 * np.pi * cutoff / SAMPLE_RATE)[:, None]
        cos_w = np.cos(w0)
        alpha = np.sin(w0) / (2.0 * q[None, :])
        a0 = 1.0 + alpha
        if side == 0:
            b0 = (1.0 - cos_w) / 2.0
            b1 = 1.0 - cos_w
        else:
            b0 = (1.0 + cos_w) / 2.0
            b1 = -(1.0 + cos_w)
        table[side, :, :, 0, 0] = b0 * comp / a0
        table[side, :, :, 0, 1] = b1 * comp / a0
        table[side, :, :, 0, 2] = b0 * comp / a0
        table[side, :, :, 0, 3] = 1.0
        table[side, :, :, 0, 4] = (-2.0 * cos_w) / a0
        table[side, :, :, 0, 5] = (1.0 - alpha) / a0

    wet = np.clip((travel - FILTER_DETENT) / FILTER_FADE, 0.0, 1.0)
    _FILTER_WET = [float(v) for v in wet]
    _FILTER_TABLE = table
    return _FILTER_TABLE, _FILTER_WET


class _Smoothed:
    """A float parameter that ramps to its target over exactly one block."""

    __slots__ = ("cur", "target")

    def __init__(self, value: float) -> None:
        self.cur = float(value)
        self.target = float(value)

    def set(self, value: float) -> None:
        self.target = float(value)

    def jump(self, value: float) -> None:
        self.cur = self.target = float(value)

    @property
    def moving(self) -> bool:
        return abs(self.target - self.cur) > _EPS


class TransportState(Enum):
    """What a deck is doing, as one value instead of three booleans.

    Deliberately NOT named ``DeckState``: :class:`djai.engine.DeckState` is
    already the flat JSON snapshot of a deck, and two types with one name is
    how the caller ends up reading the wrong one.

    The distinction that matters is :attr:`PAUSED` versus a deck whose track
    ran out. Both leave the deck stopped, and code that tests "is it advancing"
    cannot tell them apart -- which is exactly how a pause came to be read as a
    hand-off cue. Only a deck that is :attr:`PLAYING` may be a transition
    candidate.
    """

    #: No track on the deck at all.
    EMPTY = "empty"
    #: A track is loaded and parked -- freshly cued, or played to its end.
    LOADED_STOPPED = "loaded_stopped"
    #: Advancing.
    PLAYING = "playing"
    #: Stopped part-way through by an operator. Triggers nothing.
    PAUSED = "paused"


class Deck:
    """A single playback deck. See the module docstring for threading rules."""

    def __init__(self, name: str) -> None:
        self.name = name

        # --- state read/written by the audio thread ---
        self.track: LoadedTrack | None = None
        self.position: float = 0.0  # fractional frame index into track.audio
        self.playing: bool = False
        self.ended: bool = False
        # --- loop (audio thread reads; control thread sets) ---
        #: Loop region in ORIGINAL track frames, matching `position`. Zero
        #: length means no loop. Set as one pair so the audio thread cannot
        #: observe a half-updated region.
        self._loop: tuple[float, float] = (0.0, 0.0)

        #: Set by the engine when a deck is stopped while it was running the
        #: same audio: an operator's pause, as opposed to a fresh cue or a
        #: track that played out. Cleared by :meth:`attach`, so a new load is
        #: never inherited as a pause.
        self.paused: bool = False
        #: Incremented on every attach. Anything caching per-load state (the
        #: supervisor's phase baseline) must key on this, not on the track id:
        #: re-cueing the *same* track is still a new playhead, and treating it
        #: as continuous makes stale state look like enormous drift.
        self.load_seq: int = 0

        # Rate is not ramped: it retunes the resampler, and a step change in
        # pitch is inaudible as a click. Everything else is ramped per block.
        self.rate: float = 1.0
        self.gain = _Smoothed(0.0)
        self.eq_low = _Smoothed(1.0)
        self.eq_mid = _Smoothed(1.0)
        self.eq_high = _Smoothed(1.0)

        # --- filter knob and key lock ---
        #: -1..1. Negative sweeps a low-pass down, positive a high-pass up, and
        #: 0 (the detent) is no filter. Ramped per block like the EQ.
        self.filter_pos = _Smoothed(0.0)
        #: 0..1. Stepped rather than ramped: it changes when a transition starts
        #: or an operator sets it, not continuously.
        self.filter_res: float = float(config.FILTER_KNOB_RESONANCE)
        #: Play the stretched copy, at original pitch, when there is one.
        self.key_lock: bool = bool(config.KEY_LOCK_DEFAULT)
        #: The rate this deck's track was loaded at: the rate a stretch is made
        #: for, as opposed to `rate`, which drift correction nudges.
        self.load_rate: float = 1.0
        #: The operator moved this deck's pitch fader. Drift correction leaves
        #: such a deck alone until it is synced or a new track loads.
        self.manual_pitch: bool = False
        #: Set by swap_audio: the buffer the playhead just left, crossfaded
        #: against on the next block so the swap does not click.
        self._swap_from: LoadedTrack | None = None

        # --- slip: where the deck would be had it never looped ---
        #: Advanced every block while a loop is engaged. Leaving the loop jumps
        #: the playhead here, which is what keeps the deck in phase with the
        #: master clock whatever the loop's length or however many laps it ran.
        self._slip: float = 0.0
        #: Set by clear_loop: the looped position the playhead left from, so
        #: the next block can crossfade the jump instead of clicking.
        self._exit_pos: float = 0.0
        self._exit_pending: bool = False

        # --- loop seam crossfade (preallocated) ---
        self._xf_pos = np.zeros(_LOOP_XFADE, dtype=np.float64)
        self._xf_floor = np.zeros(_LOOP_XFADE, dtype=np.float64)
        self._xf_i0 = np.zeros(_LOOP_XFADE, dtype=np.int64)
        self._xf_i1 = np.zeros(_LOOP_XFADE, dtype=np.int64)
        # (K, 1), broadcast across channels, matching `_frac`.
        self._xf_frac = np.zeros((_LOOP_XFADE, 1), dtype=DTYPE)
        self._xf_a = np.zeros((_LOOP_XFADE, CHANNELS), dtype=DTYPE)
        self._xf_b = np.zeros((_LOOP_XFADE, CHANNELS), dtype=DTYPE)
        self._xf_cont = np.zeros((_LOOP_XFADE, CHANNELS), dtype=DTYPE)
        ramp = (np.arange(1, _LOOP_XFADE + 1, dtype=np.float64) / _LOOP_XFADE)
        self._xf_w = ramp.astype(DTYPE).reshape(-1, 1)
        self._xf_iw = (1.0 - ramp).astype(DTYPE).reshape(-1, 1)

        # --- preallocated hot-path buffers ---
        self._out = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._raw = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        #: This block after EQ and filter, before the channel gain: what an
        #: effect send takes, so an echo of a filtered deck is filtered too.
        self._pre_gain = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._a = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._b = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._acc = np.zeros((MAX_BLOCK, CHANNELS), dtype=DTYPE)
        self._pos = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._floor = np.zeros(MAX_BLOCK, dtype=np.float64)
        self._i0 = np.zeros(MAX_BLOCK, dtype=np.int64)
        self._i1 = np.zeros(MAX_BLOCK, dtype=np.int64)
        self._frac = np.zeros((MAX_BLOCK, 1), dtype=DTYPE)
        self._steps = np.arange(MAX_BLOCK, dtype=np.float64)

        # Ramp buffers, shaped (MAX_BLOCK, 1) so they broadcast over channels.
        self._ramp_gain = np.zeros((MAX_BLOCK, 1), dtype=DTYPE)
        self._ramp_low = np.zeros((MAX_BLOCK, 1), dtype=DTYPE)
        self._ramp_mid = np.zeros((MAX_BLOCK, 1), dtype=DTYPE)
        self._ramp_high = np.zeros((MAX_BLOCK, 1), dtype=DTYPE)
        self._unit = np.zeros((MAX_BLOCK, 1), dtype=DTYPE)
        self._unit_n = -1

        # --- EQ filters (Linkwitz-Riley 4th order = Butterworth 2nd, twice) ---
        # Coefficients and state stay float64 on purpose: a low-frequency
        # crossover at 250 Hz / 44.1 kHz is at a normalised frequency of 0.011,
        # close enough to DC that float32 coefficients visibly degrade it.
        self._sos_lp_low = _lr4(EQ_LOW_XOVER, "low")
        self._sos_hp_low = _lr4(EQ_LOW_XOVER, "high")
        self._sos_lp_high = _lr4(EQ_HIGH_XOVER, "low")
        self._sos_hp_high = _lr4(EQ_HIGH_XOVER, "high")
        n_sec = self._sos_lp_low.shape[0]
        self._zi_lp_low = np.zeros((n_sec, 2, CHANNELS), dtype=np.float64)
        self._zi_hp_low = np.zeros((n_sec, 2, CHANNELS), dtype=np.float64)
        self._zi_lp_high = np.zeros((n_sec, 2, CHANNELS), dtype=np.float64)
        self._zi_hp_high = np.zeros((n_sec, 2, CHANNELS), dtype=np.float64)

        # --- resonant filter: coefficients come from the prebuilt table ---
        self._flt_table, self._flt_wet = filter_tables()
        self._flt_zi = np.zeros((1, 2, CHANNELS), dtype=np.float64)
        #: Which response the filter state belongs to: 0 low-pass, 1 high-pass,
        #: -1 bypassed with its state cleared.
        self._flt_side: int = -1

    # --- control surface (audio thread, called while draining the queue) -----

    def attach(self, track: LoadedTrack | None, start_frame: int = 0) -> None:
        """Publish a preloaded track to this deck. The decode already happened."""
        self.track = track
        self.position = float(start_frame)
        # A new playhead is a new straight-through timeline.
        self._slip = self.position
        self._exit_pending = False
        self.paused = False
        self.ended = track is None
        self.load_seq += 1
        self._swap_from = None
        self._flt_zi.fill(0.0)
        self._flt_side = -1
        for zi in (
            self._zi_lp_low,
            self._zi_hp_low,
            self._zi_lp_high,
            self._zi_hp_high,
        ):
            zi.fill(0.0)

    def set_gain(self, g: float, immediate: bool = False) -> None:
        self.gain.jump(g) if immediate else self.gain.set(g)

    def set_eq(
        self,
        low: float | None = None,
        mid: float | None = None,
        high: float | None = None,
    ) -> None:
        if low is not None:
            self.eq_low.set(low)
        if mid is not None:
            self.eq_mid.set(mid)
        if high is not None:
            self.eq_high.set(high)

    def set_rate(self, rate: float) -> None:
        self.rate = float(rate)

    def set_filter(
        self, position: float | None = None, resonance: float | None = None
    ) -> None:
        """Turn the filter knob (-1..1) and/or set its resonance (0..1)."""
        if position is not None:
            self.filter_pos.set(max(-1.0, min(1.0, float(position))))
        if resonance is not None:
            self.filter_res = max(0.0, min(1.0, float(resonance)))

    def swap_audio(self, track: LoadedTrack | None) -> None:
        """Put another rendering of the SAME track under the playhead.

        **Audio thread.** Key lock swaps between a stretched copy and its
        source. ``position`` counts original frames, so it does not move; the
        next block crossfades from the buffer being left, so the change does
        not click. Refuses anything that is not the same track.
        """
        current = self.track
        if current is None or track is None or track is current:
            return
        if track.analysis is not current.analysis:
            return
        self._swap_from = current
        self.track = track

    def jump(self, frames: float) -> None:
        """Move the playhead by ``frames`` original frames. AUDIO THREAD.

        The slip position and any engaged loop move with it, so a jump inside
        a loop moves the loop. A whole number of beats keeps the deck's phase;
        ``load_seq`` is bumped because the supervisor's phase baseline for the
        old timeline no longer applies to the new one.
        """
        self.position += frames
        self._slip += frames
        start, length = self._loop
        if length > 0.0:
            self._loop = (start + frames, length)
        self.load_seq += 1

    def resync(self, frame: float) -> None:
        """Hard-set the playhead. Supervisor intervention only."""
        self.position = float(frame)
        self._slip = self.position

    # --- looping ---------------------------------------------------------------

    def set_loop(self, start_frame: float, length_frames: float) -> None:
        """Loop between ``start_frame`` and ``start_frame + length_frames``.

        Both in original track frames. **Audio thread**: the engine calls this
        while applying an envelope row. The region is published as a single
        tuple assignment, so a read never sees half of an update.

        Loops are slip loops. Engaging one starts a straight-through shadow of
        the playhead; changing the region of a loop already engaged (halving it,
        say) keeps that shadow running. Clearing the loop resumes from the
        shadow, so the deck comes out exactly where it would have been -- in
        phase with the master clock for any length, whole beats or not.
        """
        if length_frames <= 0:
            self.clear_loop()
            return
        if self._loop[1] <= 0.0:
            self._slip = self.position
        self._loop = (float(start_frame), float(length_frames))

    def clear_loop(self) -> None:
        """Leave the loop, resuming where straight playback would be.

        A no-op when no loop is engaged, so clearing defensively costs nothing
        and never moves a deck that was playing straight.
        """
        if self._loop[1] <= 0.0:
            return
        self._loop = (0.0, 0.0)
        self._exit_pos = self.position
        self._exit_pending = True
        self.position = self._slip

    @property
    def loop_active(self) -> bool:
        return self._loop[1] > 0.0

    @property
    def loop_region(self) -> tuple[float, float]:
        """``(start_frame, length_frames)`` in original track frames."""
        return self._loop

    @property
    def transport(self) -> TransportState:
        """This deck's transport state, derived from what the engine set.

        Derived from the flags the audio thread actually acts on, so it can
        never drift out of step with them. The one thing that cannot be
        inferred is PAUSED: a pause re-loads the deck's own audio at its
        current playhead, which is indistinguishable from a fresh cue by
        position alone, so the engine records it explicitly.
        """
        if self.track is None:
            return TransportState.EMPTY
        if self.playing:
            return TransportState.PLAYING
        if self.ended:
            return TransportState.LOADED_STOPPED
        if self.paused:
            return TransportState.PAUSED
        return TransportState.LOADED_STOPPED

    # --- introspection (monitor / main thread; single float reads) -----------

    @property
    def remaining_frames(self) -> int:
        if self.track is None:
            return 0
        # In original frames, to match `position`: a stretched buffer is shorter
        # (or longer) than the track it came from.
        return max(0, int(self.track.source_frames - self.position))

    @property
    def position_seconds(self) -> float:
        return self.position / SAMPLE_RATE

    # --- hot path ------------------------------------------------------------

    def _prepare_unit(self, n: int) -> None:
        """Rebuild the 0..1 ramp basis. Only allocates when the block size
        changes, which in practice is once (plus any short final block)."""
        if self._unit_n == n:
            return
        self._unit[:n, 0] = (np.arange(1, n + 1, dtype=np.float64) / n).astype(DTYPE)
        self._unit_n = n

    def _ramp(self, param: _Smoothed, buf: np.ndarray, n: int) -> np.ndarray:
        """Fill ``buf[:n]`` with a linear ramp from ``param.cur`` to its target."""
        view = buf[:n]
        if param.moving:
            np.multiply(self._unit[:n], DTYPE(param.target - param.cur), out=view)
            np.add(view, DTYPE(param.cur), out=view)
            param.cur = param.target
        else:
            view.fill(param.cur)
        return view

    def _declick_seam(
        self,
        raw: np.ndarray,
        seam: int,
        seam_pos0: float,
        step: float,
        span: float,
        data: np.ndarray,
        last: int,
        n: int,
    ) -> None:
        """Crossfade a loop seam. AUDIO THREAD, preallocated throughout.

        A loop lands the playhead back at the region's start mid-waveform, and
        the step from wherever the audio had got to down to wherever it starts
        is a discontinuity -- a click, once per lap. This blends the first few
        samples after the wrap against the audio that would have played had the
        loop not happened, so the two waveforms meet instead of colliding.

        Truncated when the seam falls near the end of a block, which leaves a
        smaller step rather than none. At 64 frames against a 2048-frame block
        that is one lap in thirty-two, and the residual is a fraction of the
        original.
        """
        k = min(_LOOP_XFADE, n - seam)
        if k <= 1:
            return

        cont_pos = self._xf_pos[:k]
        np.multiply(self._steps[:k], step, out=cont_pos)
        np.add(cont_pos, seam_pos0 + span, out=cont_pos)
        np.clip(cont_pos, 0.0, float(last), out=cont_pos)

        floor = self._xf_floor[:k]
        np.floor(cont_pos, out=floor)
        i0 = self._xf_i0[:k]
        np.copyto(i0, floor, casting="unsafe")
        i1 = self._xf_i1[:k]
        np.add(i0, 1, out=i1)

        frac = self._xf_frac[:k]
        np.subtract(cont_pos, floor, out=cont_pos)
        np.copyto(frac[:, 0], cont_pos, casting="unsafe")

        a = np.take(data, i0, axis=0, out=self._xf_a[:k])
        b = np.take(data, i1, axis=0, out=self._xf_b[:k])
        cont = self._xf_cont[:k]
        np.subtract(b, a, out=cont)
        np.multiply(cont, frac, out=cont)
        np.add(cont, a, out=cont)

        # raw = looped * w + continuation * (1 - w), w rising from 0 to 1.
        seg = raw[seam:seam + k]
        np.multiply(seg, self._xf_w[:k], out=seg)
        np.multiply(cont, self._xf_iw[:k], out=cont)
        np.add(seg, cont, out=seg)

    def _run_filter(self, out: np.ndarray, n: int) -> None:
        """The deck's resonant filter, in place on ``out``. AUDIO THREAD.

        Coefficients are looked up, never computed: :func:`filter_tables` built
        them off the audio thread. A moving knob steps them every
        :data:`FILTER_SUBBLOCK` frames with the filter state carried across,
        which is what keeps a sweep free of zipper noise; a still knob runs the
        whole block in one pass. Crossing the detent clears the state, at a
        point where the wet level is already zero.

        Allocation: like the EQ above, ``sosfilt`` returns a new output and
        state array per call (it has no ``out=``). Nothing else here allocates.
        """
        fp = self.filter_pos
        start = fp.cur
        span = fp.target - start
        fp.cur = fp.target
        table = self._flt_table
        wet_levels = self._flt_wet
        top = FILTER_STEPS - 1
        res = int(self.filter_res * (FILTER_RES_STEPS - 1) + 0.5)
        sub = FILTER_SUBBLOCK if span != 0.0 else n

        i = 0
        while i < n:
            m = sub if i + sub <= n else n - i
            x = start + span * ((i + m) / n)
            idx = int((x if x >= 0.0 else -x) * top + 0.5)
            wet = wet_levels[idx]
            seg = out[i:i + m]
            if wet <= 0.0:
                if self._flt_side >= 0:
                    self._flt_zi.fill(0.0)
                    self._flt_side = -1
            else:
                side = 1 if x > 0.0 else 0
                if side != self._flt_side:
                    self._flt_zi.fill(0.0)
                    self._flt_side = side
                y, self._flt_zi = sosfilt(
                    table[side, idx, res], seg, axis=0, zi=self._flt_zi
                )
                if wet >= 1.0:
                    np.copyto(seg, y, casting="same_kind")
                else:
                    np.subtract(y, seg, out=y)
                    np.multiply(y, wet, out=y)
                    np.add(seg, y, out=seg)
            i += m

    def read(self, n: int) -> np.ndarray:
        """Render ``n`` frames. Returns a view of a preallocated (n, 2) buffer.

        The returned array is overwritten on the next call -- consume it before
        calling ``read`` again.
        """
        out = self._out[:n]
        track = self.track  # single read; the control thread swaps this atomically

        if track is None or not self.playing or self.ended:
            out.fill(0.0)
            return out

        self._prepare_unit(n)
        data = track.audio
        last = data.shape[0] - 2

        # --- read the source at the right speed ------------------------------
        # `position` is always in ORIGINAL track frames, so every beat-grid and
        # phrase calculation elsewhere is unaffected by stretching. What changes
        # is how that maps onto the buffer we actually hold:
        #
        #   stretch_rate == 1   raw audio, stepped at `rate` (resampling, which
        #                       shifts pitch by the same factor)
        #   stretch_rate == r   audio already time-compressed by r at original
        #                       pitch, so it is stepped at 1.0
        #
        # Writing the step as rate/stretch_rate covers both, and keeps working
        # when the supervisor nudges `rate` a fraction of a percent off the
        # ratio the buffer was stretched at.
        rate = self.rate
        start_position = self.position
        stretch = track.stretch_rate
        step = rate / stretch if stretch != 1.0 else rate
        base = self.position / stretch if stretch != 1.0 else self.position

        pos = self._pos[:n]
        np.multiply(self._steps[:n], step, out=pos)
        np.add(pos, base, out=pos)

        end_pos = self.position + rate * n
        end_base = base + step * n

        # --- loop ------------------------------------------------------------
        # Applied to the position array itself, so every sample is wrapped
        # exactly rather than the block being nudged. `seam` is where the
        # playhead jumped back; there is at most one per block for any loop
        # longer than a block, and the shortest loop this is used for -- half a
        # bar at 174 BPM -- is still 30k frames against a 2048-frame block.
        loop_start, loop_len = self._loop
        seam = -1
        seam_pos0 = 0.0
        span = 0.0
        if loop_len > 0.0:
            lo = loop_start / stretch if stretch != 1.0 else loop_start
            span = loop_len / stretch if stretch != 1.0 else loop_len
            np.subtract(pos, lo, out=pos)
            np.mod(pos, span, out=pos)
            np.add(pos, lo, out=pos)
            if n > 1:
                # np.diff would allocate; compare the two shifted views.
                back = np.less(pos[1:], pos[:-1])
                if back.any():
                    seam = int(np.argmax(back)) + 1
                    # Captured now: `pos` is reused for the interpolation
                    # fraction a few lines below and no longer holds positions.
                    seam_pos0 = float(pos[seam])
            end_pos = loop_start + (end_pos - loop_start) % loop_len
            end_base = lo + (end_base - lo) % span
            self._slip += rate * n

        np.floor(pos, out=self._floor[:n])
        np.clip(self._floor[:n], 0.0, float(last), out=self._floor[:n])
        i0 = self._i0[:n]
        np.copyto(i0, self._floor[:n], casting="unsafe")
        i1 = self._i1[:n]
        np.add(i0, 1, out=i1)

        frac = self._frac[:n]
        np.subtract(pos, self._floor[:n], out=pos)
        np.copyto(frac[:, 0], pos, casting="unsafe")

        a = np.take(data, i0, axis=0, out=self._a[:n])
        b = np.take(data, i1, axis=0, out=self._b[:n])
        raw = self._raw[:n]
        np.subtract(b, a, out=raw)
        np.multiply(raw, frac, out=raw)
        np.add(raw, a, out=raw)

        if seam >= 0:
            self._declick_seam(raw, seam, seam_pos0, step, span, data, last, n)
        if self._exit_pending:
            # The first block after leaving a loop jumped from the looped
            # position to the slip position. Same crossfade as a seam, with the
            # audio the loop would have played next as the continuation.
            self._exit_pending = False
            exit_base = self._exit_pos / stretch if stretch != 1.0 else self._exit_pos
            self._declick_seam(raw, 0, exit_base, step, 0.0, data, last, n)
        swapped = self._swap_from
        if swapped is not None:
            # Key lock changed which buffer is under the playhead. Blend in
            # from the one it left, read at the same musical position.
            self._swap_from = None
            old_stretch = swapped.stretch_rate
            old_data = swapped.audio
            self._declick_seam(
                raw, 0,
                start_position / old_stretch if old_stretch != 1.0 else start_position,
                rate / old_stretch if old_stretch != 1.0 else rate,
                0.0, old_data, old_data.shape[0] - 2, n,
            )

        # Past the end: silence the tail rather than looping the last sample.
        # Compared in buffer indices, which is what `last` and `_floor` are in.
        # A looping deck never gets here: its positions are wrapped inside the
        # region, which is by construction inside the track.
        if loop_len <= 0.0 and end_base >= last:
            over = int(np.searchsorted(self._floor[:n], float(last)))
            if over < n:
                raw[over:].fill(0.0)
            self.ended = True

        self.position = end_pos

        # --- 3-band EQ (Linkwitz-Riley crossovers) ---------------------------
        low, self._zi_lp_low = sosfilt(
            self._sos_lp_low, raw, axis=0, zi=self._zi_lp_low
        )
        rest, self._zi_hp_low = sosfilt(
            self._sos_hp_low, raw, axis=0, zi=self._zi_hp_low
        )
        mid, self._zi_lp_high = sosfilt(
            self._sos_lp_high, rest, axis=0, zi=self._zi_lp_high
        )
        high, self._zi_hp_high = sosfilt(
            self._sos_hp_high, rest, axis=0, zi=self._zi_hp_high
        )

        acc = self._acc[:n]
        np.multiply(low, self._ramp(self.eq_low, self._ramp_low, n), out=out)
        np.multiply(mid, self._ramp(self.eq_mid, self._ramp_mid, n), out=acc)
        np.add(out, acc, out=out)
        np.multiply(high, self._ramp(self.eq_high, self._ramp_high, n), out=acc)
        np.add(out, acc, out=out)

        # --- resonant filter ---------------------------------------------------
        # A knob at rest in the detent with its state cleared costs one
        # comparison.
        fp = self.filter_pos
        if fp.moving or fp.cur != 0.0 or self._flt_side >= 0:
            self._run_filter(out, n)

        # The effect sends' tap: post-EQ and filter, pre-fader. One copy into a
        # preallocated buffer.
        np.copyto(self._pre_gain[:n], out)

        # --- gain -------------------------------------------------------------
        np.multiply(out, self._ramp(self.gain, self._ramp_gain, n), out=out)
        return out
