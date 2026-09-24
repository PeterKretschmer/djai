"""Pending commands, ordered by musical position, released to the engine.

THREADING CONTEXT: a dedicated **scheduler thread** (plus main-thread calls to
:meth:`Scheduler.submit` and the introspection helpers). Never the audio
thread. Because it is not real-time, it may take locks freely -- and it does,
around the pending list.

This is the layer that hides LLM latency. A user's sentence takes 1-3 s to come
back as a structured command; that command is then given an ``execute_at`` at
the next 32-bar phrase boundary, which at 124 BPM is up to 62 s away. The
scheduler holds it until the engine's playhead arrives.

Release timing is *not* sample accurate: the thread polls at
:data:`TICK_SECONDS` and pushes a command when the engine's frame counter has
reached its deadline, so a command fires within roughly one tick plus one
audio block of its musical position (~15 ms). That is inaudible against a
32-bar fade, and it is the price of keeping the audio callback free of
scheduling logic.

**Priority bands (Phase 2.1).** Each command is placed, when it is submitted,
in one of three bands by how far ahead of the playhead it is: *immediate*
(0-2 bars), *short* (up to 8) and *medium* (the 16-32+ bar horizon: glides,
rides, phrase-aligned transitions). Two commands due on the same frame release
in band order, so a correction lands before the plan it corrects. A command
carrying a :class:`~djai.commands.CancelToken` can be withdrawn with its whole
group by :meth:`Scheduler.cancel`; a token cancelled in the instant between
the tick taking a command and pushing it is caught by a second check.
"""

from __future__ import annotations

import bisect
import logging
import queue
import threading
from collections.abc import Callable

from djai.commands import CancelToken, Command
from djai.deck import SAMPLE_RATE
from djai.engine import Engine

log = logging.getLogger(__name__)

#: Scheduler poll interval. 2 ms keeps release jitter well under one audio block.
TICK_SECONDS: float = 0.002

#: The three priority bands, most urgent first.
BANDS: tuple[str, ...] = ("immediate", "short", "medium")
#: Upper edges, in bars of lead time, of the immediate and short bands.
IMMEDIATE_MAX_BARS: float = 2.0
SHORT_MAX_BARS: float = 8.0


class Scheduler:
    """Holds future commands and releases them to the engine on time."""

    def __init__(
        self,
        engine: Engine,
        tick_seconds: float = TICK_SECONDS,
        on_dropped: Callable[[Command], None] | None = None,
        on_pre_roll: Callable[[Command], None] | None = None,
        pre_roll_frames: int = 0,
    ) -> None:
        self._engine = engine
        self._tick = tick_seconds
        self._on_dropped = on_dropped

        self._lock = threading.Lock()
        # (execute_at, band rank), kept sorted: same-frame commands release
        # most urgent band first.
        self._keys: list[tuple[int, int]] = []
        self._pending: list[Command] = []  # parallel to _keys

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # --- the pre-roll hook ---
        # A command that is still some way off has a window in front of it in
        # which work can be done about it. `on_pre_roll` is called once per
        # command, that many frames before it is due, ON A WORKER THREAD -- the
        # transition preview is what this exists for, and it renders audio and
        # talks to a language model, neither of which may happen anywhere near
        # the tick loop. All the tick itself does is a non-blocking put.
        self._on_pre_roll = on_pre_roll
        self._pre_roll_frames = int(pre_roll_frames)
        # Keyed by id(), with the command itself as the value: holding a strong
        # reference is what keeps that id from being reused by a later object.
        self._announced: dict[int, Command] = {}
        # Bounded: every queued command can pin a whole decoded track, so a
        # dead or starved pre-roll worker must not grow memory without limit.
        # A full queue drops the announcement (see _announce_pre_roll), which
        # only costs a preview -- the command still fires on time.
        self._pre_roll_q: queue.Queue[Command | None] = queue.Queue(maxsize=16)
        self._pre_roll_thread: threading.Thread | None = None

    # --- submission ----------------------------------------------------------

    def submit(self, cmd: Command) -> None:
        """Queue a command. Immediate ones go straight to the engine.

        Panic commands (``cut``, ``killbass``, ``stop``) are constructed with
        ``execute_at=IMMEDIATE`` and therefore never wait here.
        """
        if cmd.token is not None and cmd.token.cancelled:
            return
        if cmd.is_immediate:
            self._push(cmd)
            return
        key = (cmd.execute_at, BANDS.index(self.band_of(cmd)))
        with self._lock:
            i = bisect.bisect_right(self._keys, key)
            self._keys.insert(i, key)
            self._pending.insert(i, cmd)

    def band_of(self, cmd: Command, now_frames: int | None = None) -> str:
        """Which priority band ``cmd`` falls in, from its lead time in bars.

        Bars at the master clock's tempo; before any clock exists (no track
        yet) 120 BPM stands in, which only matters for a queue built before
        the first note plays.
        """
        if cmd.is_immediate:
            return "immediate"
        now = self._engine.frames_played if now_frames is None else now_frames
        bpm = getattr(self._engine, "master_bpm", 0.0) or 120.0
        frames_per_bar = SAMPLE_RATE * 240.0 / bpm
        lead_bars = (cmd.execute_at - now) / frames_per_bar
        if lead_bars <= IMMEDIATE_MAX_BARS:
            return "immediate"
        if lead_bars <= SHORT_MAX_BARS:
            return "short"
        return "medium"

    def _push(self, cmd: Command) -> None:
        if not self._engine.submit(cmd) and self._on_dropped is not None:
            self._on_dropped(cmd)

    # --- introspection -------------------------------------------------------

    def pending(self) -> list[Command]:
        with self._lock:
            return list(self._pending)

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)

    def cancel_all(self) -> list[Command]:
        """Drop every pending command. Backs the ``skip_queued`` action."""
        with self._lock:
            dropped = list(self._pending)
            self._keys.clear()
            self._pending.clear()
        # Nothing is pending, so nothing is awaiting its pre-roll either.
        self._announced.clear()
        return dropped

    def pending_by_band(self) -> dict[str, list[Command]]:
        """Pending commands grouped by the band they were submitted in."""
        with self._lock:
            out: dict[str, list[Command]] = {b: [] for b in BANDS}
            for (_, rank), cmd in zip(self._keys, self._pending):
                out[BANDS[rank]].append(cmd)
            return out

    def cancel(self, token: CancelToken) -> list[Command]:
        """Withdraw every pending command in ``token``'s group.

        The token stays cancelled, so a command submitted under it later, or
        one the tick has already taken, never reaches the engine either.
        """
        token.cancel()
        return self.cancel_matching(lambda c: c.token is token)

    def cancel_matching(self, predicate: Callable[[Command], bool]) -> list[Command]:
        with self._lock:
            keep_keys: list[int] = []
            keep_cmds: list[Command] = []
            dropped: list[Command] = []
            for key, cmd in zip(self._keys, self._pending):
                if predicate(cmd):
                    dropped.append(cmd)
                else:
                    keep_keys.append(key)
                    keep_cmds.append(cmd)
            self._keys, self._pending = keep_keys, keep_cmds
        for cmd in dropped:
            self._announced.pop(id(cmd), None)
        return dropped

    # --- the tick ------------------------------------------------------------

    def tick(self, now_frames: int) -> list[Command]:
        """Release everything due at ``now_frames``. Returns what was released.

        Separated from the thread loop so it can be driven deterministically in
        tests.
        """
        if self._on_pre_roll is not None and self._pre_roll_frames > 0:
            self._announce_pre_roll(now_frames)
        with self._lock:
            cut = bisect.bisect_right(self._keys, (now_frames, len(BANDS)))
            if cut == 0:
                return []
            due = self._pending[:cut]
            del self._keys[:cut]
            del self._pending[:cut]
        released = []
        for cmd in due:
            self._announced.pop(id(cmd), None)
            if cmd.token is not None and cmd.token.cancelled:
                continue  # cancelled between the cut and here
            self._push(cmd)
            released.append(cmd)
        return released

    def _announce_pre_roll(self, now_frames: int) -> None:
        """Hand any newly-in-range command to the pre-roll worker. Never blocks.

        The keys are sorted, so the commands inside the horizon are a prefix and
        finding them is a bisect rather than a scan. Everything expensive
        happens on the other end of the queue.
        """
        horizon = now_frames + self._pre_roll_frames
        with self._lock:
            cut = bisect.bisect_right(self._keys, (horizon, len(BANDS)))
            soon = [c for c in self._pending[:cut] if id(c) not in self._announced]
            for cmd in soon:
                self._announced[id(cmd)] = cmd
        for cmd in soon:
            try:
                self._pre_roll_q.put_nowait(cmd)
            except queue.Full:  # worker behind: skip the preview, not the command
                pass

    def _pre_roll_run(self) -> None:
        """Drain the pre-roll queue. A worker thread, with all the time it needs."""
        while True:
            cmd = self._pre_roll_q.get()
            if cmd is None:
                return
            try:
                self._on_pre_roll(cmd)
            except Exception:
                # Pre-roll work is an optimisation. Whatever it was trying to
                # do, the command it was about is still going to fire on time.
                log.warning("pre-roll hook failed", exc_info=True)

    # --- thread lifecycle ----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        if self._on_pre_roll is not None and self._pre_roll_thread is None:
            self._pre_roll_thread = threading.Thread(
                target=self._pre_roll_run, name="djai-preroll", daemon=True
            )
            self._pre_roll_thread.start()
        self._thread = threading.Thread(
            target=self._run, name="djai-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._pre_roll_thread is not None:
            # A preview in flight is worth waiting a moment for, but never worth
            # hanging a shutdown over: the thread is a daemon either way.
            self._pre_roll_q.put(None)
            self._pre_roll_thread.join(timeout=1.0)
            self._pre_roll_thread = None

    @property
    def thread(self) -> threading.Thread | None:
        """The ticking thread, for the fallback watchdog to check is alive."""
        return self._thread

    def _run(self) -> None:
        while not self._stop.wait(self._tick):
            try:
                self.tick(self._engine.frames_played)
            except Exception:
                # The scheduler thread must outlive any single bad command.
                pass
