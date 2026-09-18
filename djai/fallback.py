"""The last line of defence: something to play when the engine stops playing.

Dead air is the only failure that matters in front of a room, so this module is
deliberately the least clever code in the program. It holds one decoded track in
memory and loops it. It does not beat-match, does not read the crate, does not
touch the scheduler, and never asks the engine a question -- every one of those
is a thing that could be broken at the moment it is needed.

THREADING CONTEXT
-----------------
:meth:`FallbackPlayer.fill` runs on an **audio thread** -- either the engine's,
once the watchdog has handed it over, or the fallback's own stream if the
engine's stream is gone entirely. It obeys the same rules as the engine
callback: no allocation, no locks, no I/O, no logging.

:class:`Watchdog` runs a plain daemon thread and does the deciding. Takeover is
a single reference assignment to ``engine.fallback``, which is atomic under the
GIL, so the room hears audio again on the very next block rather than after a
stream teardown.

Two distinct failures are covered, because they look completely different from
outside:

* **The engine is being called but is broken** -- the scheduler thread died, or
  the callback is throwing every block. The stream is fine, so the fix is to
  route the existing callback to this player.
* **The engine is not being called at all** -- the stream died, or the callback
  is wedged. Nothing routes anywhere, so the watchdog opens its own stream.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import numpy as np
import sounddevice as sd

from djai import config
from djai.deck import CHANNELS, DTYPE, MAX_BLOCK, SAMPLE_RATE, LoadedTrack

log = logging.getLogger(__name__)

#: How long the engine has to look healthy again -- called on time, no burst of
#: underruns, every thread alive -- before the output is handed back to it.
#:
#: Automatic recovery exists because a takeover is not only a loudness event: it
#: freezes the mix. The engine callback returns the fallback's audio and never
#: drains the command queue or advances a deck, so nothing is ever cued again
#: and the set becomes one looping track. Measured on this machine, a single
#: 453-547 ms stall under memory pressure is enough to trip it, which is far
#: too small a fault to end a night's mixing.
RECOVER_AFTER_S: float = 5.0

#: How many times one session may hand the output back. Past this the fallback
#: keeps it: an engine that has failed this often is not one to keep returning
#: to, and flapping between two sources in front of a room is its own fault.
MAX_RECOVERIES: int = 3


class FallbackPlayer:
    """One preloaded track, looped. That is the entire feature set."""

    def __init__(self, audio: np.ndarray, title: str = "fallback", gain: float = 0.9):
        if audio.ndim == 1:
            audio = np.repeat(audio[:, None], CHANNELS, axis=1)
        self.audio = np.ascontiguousarray(audio[:, :CHANNELS], dtype=DTYPE)
        if self.audio.shape[0] < MAX_BLOCK:
            # Anything shorter than a block would need wrap logic per call for
            # no reason; pad once, here, off the audio thread.
            reps = int(np.ceil(MAX_BLOCK / max(1, self.audio.shape[0]))) + 1
            self.audio = np.tile(self.audio, (reps, 1))
        self.title = title
        self.gain = float(gain)
        self.position: int = 0
        self.frames_played: int = 0
        self.loops: int = 0

    @classmethod
    def from_track(cls, track: LoadedTrack, gain: float = 0.9) -> FallbackPlayer:
        return cls(track.audio, title=track.analysis.title, gain=gain)

    def fill(self, outdata: np.ndarray, frames: int) -> None:
        """Write ``frames`` of looped audio. AUDIO THREAD, allocation-free."""
        try:
            n = self.audio.shape[0]
            start = self.position
            end = start + frames
            target = outdata[:, :CHANNELS] if outdata.shape[1] > CHANNELS else outdata
            if outdata.shape[1] > CHANNELS:
                outdata.fill(0.0)
            if end <= n:
                np.copyto(target, self.audio[start:end])
                self.position = end if end < n else 0
            else:
                split = n - start
                np.copyto(target[:split], self.audio[start:])
                np.copyto(target[split:frames], self.audio[: frames - split])
                self.position = frames - split
                self.loops += 1
            if self.gain != 1.0:
                np.multiply(target, self.gain, out=target)
            self.frames_played += frames
        except Exception:
            # Even here. This is the code that runs when everything else failed.
            outdata.fill(0.0)


class Watchdog:
    """Promotes the fallback when the engine stops producing audio.

    THREADING CONTEXT: one daemon thread. Reads engine counters (plain ints
    written by the audio thread), never writes engine state except the single
    ``engine.fallback`` reference that performs the takeover.
    """

    def __init__(
        self,
        engine: Any,
        player: FallbackPlayer,
        threads: dict[str, threading.Thread] | None = None,
        on_takeover: Callable[[str], None] | None = None,
        stall_ms: float | None = None,
        underrun_burst: int | None = None,
        poll_hz: float | None = None,
        on_release: Callable[[str], None] | None = None,
        recover_after_s: float | None = None,
        max_recoveries: int | None = None,
    ) -> None:
        self.engine = engine
        self.player = player
        #: Threads whose death means the engine can no longer be steered. The
        #: scheduler is the one that matters: without it nothing is ever queued
        #: again, so the current track plays out and then there is silence.
        self.threads = threads or {}
        self.on_takeover = on_takeover
        self.on_release = on_release
        self.recover_after_s = (
            RECOVER_AFTER_S if recover_after_s is None else recover_after_s
        )
        self.max_recoveries = (
            MAX_RECOVERIES if max_recoveries is None else max_recoveries
        )
        self.stall_s = (
            config.FALLBACK_STALL_MS if stall_ms is None else stall_ms
        ) / 1000.0
        self.underrun_burst = (
            config.FALLBACK_UNDERRUN_BURST if underrun_burst is None else underrun_burst
        )
        self.poll_s = 1.0 / (
            config.FALLBACK_WATCHDOG_HZ if poll_hz is None else poll_hz
        )

        self.tripped: bool = False
        self.reason: str | None = None
        self.tripped_at: float | None = None
        self.checks: int = 0
        #: How many times the output has been handed back to a healthy engine.
        self.recoveries: int = 0
        #: Monotonic time the engine started looking healthy again, or None.
        self._healthy_since: float | None = None
        self._last_callbacks: int = 0
        self._own_stream: sd.OutputStream | None = None
        self._last_underruns: int = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        #: Set once the engine has genuinely started, so a watchdog armed before
        #: the first block does not trip on a stream that has yet to begin.
        self._armed_at: float = time.monotonic()

    # --- lifecycle (main thread) ---------------------------------------------

    def start(self) -> None:
        self._last_underruns = self.engine.underruns
        self._armed_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name="djai-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close_own_stream()

    # --- the decision ---------------------------------------------------------

    def check(self) -> str | None:
        """One pass. Returns the reason to take over, or None. Pure of effects."""
        engine = self.engine
        now = time.monotonic()

        since = now - engine.last_callback_at
        # Only meaningful once the engine has actually produced a block; before
        # that, last_callback_at is just construction time.
        if engine.callbacks > 0 and since > self.stall_s:
            return f"audio callback stalled for {since * 1000:.0f} ms"

        burst = engine.underruns - self._last_underruns
        if burst >= self.underrun_burst:
            return f"{burst} underruns in {self.poll_s * 1000:.0f} ms"

        for name, thread in self.threads.items():
            if thread is not None and not thread.is_alive():
                return f"{name} thread died"

        # Only a stream that existed can have died. An engine driven by
        # something other than PortAudio -- an offline render, a test -- has no
        # stream and has not failed.
        if engine._stream is not None and not engine.running:
            return "audio stream is no longer running"

        return None

    def healthy(self) -> bool:
        """Is the engine producing blocks on time again? Pure of effects.

        Called while the fallback holds the output, where the engine callback
        still runs -- it is what plays the fallback -- so "being called on
        time" is exactly what can be checked, plus the thread and stream
        conditions that made the takeover necessary in the first place.
        """
        engine = self.engine
        if engine.callbacks <= self._last_callbacks:
            return False  # not being called at all
        if time.monotonic() - engine.last_callback_at > self.stall_s:
            return False
        if engine.underruns - self._last_underruns >= self.underrun_burst:
            return False
        for thread in self.threads.values():
            if thread is not None and not thread.is_alive():
                return False
        if engine._stream is not None and not engine.running:
            return False
        return True

    def _consider_recovery(self) -> None:
        """Hand the output back once the engine has been healthy long enough."""
        if self.recoveries >= self.max_recoveries or self._own_stream is not None:
            return
        if not self.healthy():
            self._healthy_since = None
            return
        now = time.monotonic()
        if self._healthy_since is None:
            self._healthy_since = now
            return
        if now - self._healthy_since < self.recover_after_s:
            return
        held_s = now - (self.tripped_at or now)
        was = self.reason
        self.release()
        self.recoveries += 1
        detail = (
            f"engine healthy for {self.recover_after_s:.0f}s after {held_s:.0f}s on "
            f"the fallback ({was})"
        )
        log.error("FALLBACK RELEASED: %s", detail)
        if self.on_release is not None:
            try:
                self.on_release(detail)
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self.checks += 1
            try:
                if not self.tripped:
                    reason = self.check()
                    if reason is not None:
                        self.take_over(reason)
                else:
                    self._consider_recovery()
                self._last_underruns = self.engine.underruns
                self._last_callbacks = self.engine.callbacks
            except Exception as exc:
                log.error("watchdog check failed: %s", exc)
            self._stop.wait(self.poll_s)

    def take_over(self, reason: str) -> None:
        """Hand the master output to the fallback player. Idempotent."""
        if self.tripped:
            return
        self.tripped = True
        self.reason = reason
        self.tripped_at = time.monotonic()
        self._healthy_since = None
        # The assignment IS the takeover: the engine callback reads this
        # reference at the top of every block.
        self.engine.fallback = self.player
        log.error("FALLBACK ENGAGED: %s", reason)

        # If the stream itself is what died, routing the callback achieves
        # nothing, because the callback is not being called. Open our own.
        if self._should_open_own_stream():
            self._start_own_stream()

        if self.on_takeover is not None:
            try:
                self.on_takeover(reason)
            except Exception:
                pass

    def release(self) -> None:
        """Give the output back to the engine.

        Called by hand, and by :meth:`_consider_recovery` once the engine has
        been healthy for :data:`RECOVER_AFTER_S` -- bounded by
        :data:`MAX_RECOVERIES` so it cannot flap between two sources all night
        on the strength of a counter.

        The hand-back is a splice: the decks are where they were when the
        takeover froze them, so the room hears the mix resume mid-phrase. That
        is the price of the alternative, which is a single looping track and an
        autopilot that can never cue again.
        """
        self.engine.fallback = None
        self._close_own_stream()
        self.tripped = False
        self.reason = None
        self._healthy_since = None
        self._last_underruns = self.engine.underruns
        self._last_callbacks = self.engine.callbacks

    # --- our own stream, for when the engine's is gone -------------------------

    def _should_open_own_stream(self) -> bool:
        """Only when the engine HAD a device and has lost it.

        The distinction matters. An engine that never opened a stream is not a
        failed engine -- it is an offline render, a test, or a session whose
        callback is driven by something else, and grabbing the sound card out
        from under it would be a side effect nobody asked for. Worse, opening a
        device the engine still holds can abort the process inside PortAudio
        rather than raising, so this cannot be left to a try/except.
        """
        try:
            if self.engine._stream is None:
                return False
            return not bool(self.engine.running)
        except Exception:
            return False

    def _start_own_stream(self) -> None:
        if self._own_stream is not None:
            return
        try:
            self._own_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                blocksize=self.engine.blocksize,
                device=self.engine.device,
                channels=CHANNELS,
                dtype="float32",
                callback=self._own_callback,
            )
            self._own_stream.start()
            log.error("fallback opened its own stream on device %s", self.engine.device)
        except Exception as exc:
            self._own_stream = None
            log.error("fallback could not open a stream: %s", exc)

    def _own_callback(
        self, outdata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        """AUDIO THREAD (the fallback's own stream)."""
        self.player.fill(outdata, frames)

    def _close_own_stream(self) -> None:
        if self._own_stream is not None:
            try:
                self._own_stream.stop()
                self._own_stream.close()
            except Exception:
                pass
            self._own_stream = None
