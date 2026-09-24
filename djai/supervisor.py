"""The supervisor: makes failures legible, and keeps LLM output off the decks.

THREADING CONTEXT: a dedicated **monitor thread** running at
:data:`MONITOR_INTERVAL`, plus main-thread calls to :meth:`Supervisor.validate`
from the intent layer. Never the audio thread -- it logs to disk, which is
exactly why it cannot live there. It influences playback only by submitting
commands through the scheduler, like every other control-layer component.

Its purpose is not to prevent a bad mix. It is to make what happened
inspectable afterwards, so every intervention lands in
``./logs/session_<timestamp>.jsonl`` with the track pair, the position in bars,
what triggered it and what was done.

Two checks in v1:

**Drift monitor.** Every 100 ms, compare each deck's actual beat position
against where the master clock says it should be. Under 5 ms is ignored;
5-20 ms gets a fractional rate nudge; over 20 ms is hard-resynced at the next
downbeat.

**Command validator.** Nothing produced by the LLM reaches the scheduler
without passing :meth:`Supervisor.validate` first.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from djai import phrase
from djai import transition as tr
from djai.analysis import TrackAnalysis
from djai.commands import (
    BeatJump,
    Command,
    ExitLoop,
    LoadTrack,
    Resync,
    SetEQ,
    SetGain,
    SetLoop,
    SetPitch,
    SetRate,
    StartTransition,
    SyncDeck,
)
from djai.deck import SAMPLE_RATE, Deck
from djai.engine import Engine
from djai.scheduler import Scheduler

log = logging.getLogger(__name__)

# --- drift thresholds --------------------------------------------------------

#: How often the monitor thread checks drift.
MONITOR_INTERVAL: float = 0.1

#: Below this, drift is inaudible and ignored.
DRIFT_IGNORE_MS: float = 5.0

#: Above this, a rate nudge cannot catch up in reasonable time; hard resync.
DRIFT_RESYNC_MS: float = 20.0

#: Above this, the reading is not believable as drift at all -- roughly two bars
#: at any usable tempo. It means our own phase baseline is stale, so the right
#: response is to recapture it rather than to fling the playhead at it.
DRIFT_INSANE_MS: float = 4000.0

#: A nudge aims to erase the measured drift over this many seconds. 2 s means a
#: 10 ms error is corrected by bending the rate 0.5%, which is inaudible.
NUDGE_HORIZON_S: float = 2.0

#: Ceiling on how far the rate may bend away from nominal while correcting.
MAX_RATE_NUDGE: float = 0.005

#: Once drift falls back under DRIFT_IGNORE_MS the rate is restored to nominal.
#: Without this the correction would stay applied and the deck would sail past
#: the target and oscillate -- a proportional controller with no way home.
RATE_RESTORE_EPS: float = 1e-6

#: Widest tempo change the decks may be asked to make, as a fraction. Beyond
#: this the resampling pitch shift is too obvious to use.
MAX_RATE_DELTA: float = 0.08

#: A transition may not start before this fraction of the outgoing track has
#: played. Purely a regression guard: transitions were being scheduled at the
#: first phrase boundary after track *selection* rather than back from the
#: track's mix-out point, which landed hand-offs mid-track. Placement now runs
#: through phrase.plan_transition; this catches it if that ever stops being true.
#:
#: 0.40, not 0.50. A blend is a fixed number of *bars*, so on a short track it
#: is a large fraction of the running time: 24 bars is 52 s, which on a 103 s
#: track has to start at 45% however correctly it is placed. At 0.50 the two
#: shortest tracks in a 96-track crate had their transitions rejected and fell
#: back to a hard cut. The guard is about placement being anchored to mix-out,
#: not about a track's absolute length, and a start at 45% that still ends
#: exactly on mix-out is not the fault this was written to catch.
#:
#: It remains tight enough to catch the original bug, which placed hand-offs at
#: the first phrase boundary after selection -- bar 32 of a 130-bar track is 25%.
MIN_TRANSITION_START_FRACTION: float = 0.40

#: The guard for drop-aligned transitions, and the floor nothing goes under.
#: A drop is where it is in the track -- often a third of the way in -- and
#: lining two of them up cannot be moved later without no longer being a drop
#: swap. So drop-aligned placement may start from 25%; everything else keeps
#: 40%. Below 25% is the original "first boundary after selection" bug, and
#: is rejected whatever the placement.
DROP_ALIGNED_MIN_START_FRACTION: float = 0.25

#: A deck below this gain is not heard, so its vocal cannot clash.
VOCAL_AUDIBLE_GAIN: float = 0.1
#: Overlapping vocals shorter than this are a pickup, not a clash.
VOCAL_CLASH_MIN_BARS: float = 1.0

#: If the engine's frame counter stops moving for this long while the stream
#: still claims to be active, the audio device has stopped calling us.
#:
#: This is a third check, beyond the two the v1 spec asks for, and it is here
#: because a 10-minute run on this machine's default output device produced
#: 481 s of audio in 630 s of wall clock: PortAudio stopped delivering
#: callbacks, reported no error, and `underruns` stayed at 0. Nothing else in
#: the program could tell that the music had stopped. Making exactly that kind
#: of failure legible is what this module is for. It reports; it does not try
#: to restart the stream.
STALL_TIMEOUT_S: float = 1.0

DEFAULT_LOG_DIR = Path("logs")


@dataclass(frozen=True)
class Rejection:
    """Why a command was refused."""

    command: Command
    reason: str


#: Schema version of the session log, written as its first line. A reader six
#: months from now needs to know which shape it is holding, and the eval
#: harness (SPEC §7) refuses a log whose version it does not understand rather
#: than quietly misreading it.
#:
#: 1 -- events only.
#: 2 -- adds `decision` and `outcome` records: a decision carries the feature-bus
#:      snapshot that triggered it, and an outcome closes the window over it.
LOG_SCHEMA_VERSION: int = 2

#: A session log rotates once it reaches this size and keeps this many gzipped
#: predecessors. At the measured rate a 30-minute log is well under a megabyte;
#: this bounds the pathological case, not the normal one.
LOG_ROTATE_BYTES: int = 10 * 1024 * 1024
LOG_ROTATE_KEEP: int = 5

#: Supervisor interventions of one type on one deck, repeated within this many
#: seconds of the first, become a single log entry carrying a ``count``.
#: Measured across 32 session logs: drift nudges alone were 47% of all bytes,
#: and the three drift event types together 64%.
COLLAPSE_WINDOW_S: float = 1.0
COLLAPSIBLE_EVENTS: frozenset[str] = frozenset(
    {"drift_nudge", "drift_settled", "drift_resync", "baseline_reset"}
)


class SessionLog:
    """Append-only JSONL session log. Thread-safe; monitor/main threads only.

    Two things keep it small. Repeated interventions collapse: the first is
    written at once, and each repeat inside :data:`COLLAPSE_WINDOW_S` rewrites
    that same last line with a ``count``. A reader mid-run therefore never
    misses the event, and a reader afterwards sees one line per burst. And the
    file rotates at :data:`LOG_ROTATE_BYTES`, gzipping what it rotates out.
    """

    def __init__(self, log_dir: Path = DEFAULT_LOG_DIR) -> None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"session_{stamp}.jsonl"
        self._lock = threading.Lock()
        self._file = self.path.open("ab")
        # The collapsible run currently open, if any: where its line starts in
        # the file, which (event, deck) it is, when it began and how many it
        # holds. Any other write closes it, so its line is always the last one.
        self._run_offset: int | None = None
        self._run_key: tuple[str, Any] | None = None
        self._run_started: float = 0.0
        self._run_first_ts: str = ""
        self._run_count: int = 0
        #: Decisions issued so far; the id a decision is joined to its outcome
        #: by. Monotonic within a session, which is all a replay needs.
        self._decisions: int = 0
        self.write(
            "log_opened",
            schema_version=LOG_SCHEMA_VERSION,
            trigger="session log opened",
            action=f"schema v{LOG_SCHEMA_VERSION}",
        )

    def write(self, event: str, **fields: Any) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            **fields,
        }
        now = time.monotonic()
        with self._lock:
            if self._file.closed:
                return record
            key = (event, fields.get("deck"))
            collapsible = event in COLLAPSIBLE_EVENTS
            if (
                collapsible
                and self._run_offset is not None
                and key == self._run_key
                and now - self._run_started <= COLLAPSE_WINDOW_S
            ):
                self._run_count += 1
                record["count"] = self._run_count
                record["first_ts"] = self._run_first_ts
                self._file.flush()
                self._file.truncate(self._run_offset)
                self._emit(record)
                return record

            offset = self._file.tell()
            self._emit(record)
            if collapsible:
                self._run_offset = offset
                self._run_key = key
                self._run_started = now
                self._run_first_ts = record["ts"]
                self._run_count = 1
            else:
                self._run_offset = None
            if self._file.tell() >= LOG_ROTATE_BYTES:
                self._rotate()
        return record

    def decision(
        self, kind: str, snapshot: Any, action: str, **fields: Any
    ) -> str:
        """Log a decision together with the state that caused it (SPEC §7).

        ``snapshot`` is a :class:`djai.telemetry.Snapshot` -- the feature bus
        row the decision was taken on -- or None where a decision genuinely had
        no audio context (before the first block). Storing it beside the action
        is what makes the log replayable: prose says *that* something was
        decided, the snapshot says *what it was decided on*.

        Returns the decision id, to be passed to :meth:`outcome` when the
        window over it closes.
        """
        self._decisions += 1
        did = f"d{self._decisions}"
        # The event keeps its own name, and gains `decision_id` and the
        # snapshot. Additive on purpose: every existing reader of this log --
        # the UI, the feedback learner, the tests -- goes on seeing the event
        # it already knows, and the eval harness finds decisions by the
        # presence of `decision_id`.
        self.write(
            kind,
            decision_id=did,
            action=action,
            snapshot=snapshot.as_dict() if snapshot is not None else None,
            **fields,
        )
        return did

    def outcome(self, decision_id: str, snapshot: Any, **fields: Any) -> dict:
        """Close the outcome window over a decision.

        The pair is what an evaluation compares: what the system saw, what it
        did, and what the room was doing a while later.
        """
        return self.write(
            "outcome",
            decision_id=decision_id,
            snapshot=snapshot.as_dict() if snapshot is not None else None,
            **fields,
        )

    def _emit(self, record: dict) -> None:
        """Write one line and flush it. Lock held."""
        self._file.write((json.dumps(record, default=str) + "\n").encode("utf-8"))
        self._file.flush()

    def _rotate(self) -> None:
        """Gzip the full log out of the way and start an empty one. Lock held."""
        import gzip
        import shutil

        self._file.close()
        name = self.path.name
        oldest = self.path.with_name(f"{name}.{LOG_ROTATE_KEEP}.gz")
        if oldest.exists():
            oldest.unlink()
        for i in range(LOG_ROTATE_KEEP - 1, 0, -1):
            older = self.path.with_name(f"{name}.{i}.gz")
            if older.exists():
                older.replace(self.path.with_name(f"{name}.{i + 1}.gz"))
        with self.path.open("rb") as raw, gzip.open(
            self.path.with_name(f"{name}.1.gz"), "wb"
        ) as packed:
            shutil.copyfileobj(raw, packed)
        self._file = self.path.open("wb")
        self._run_offset = None

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


class Supervisor:
    """Drift correction and command validation, with everything logged."""

    def __init__(
        self,
        engine: Engine,
        scheduler: Scheduler,
        crate: list[TrackAnalysis],
        session_log: SessionLog | None = None,
        notify: Callable[[str], None] | None = None,
        interval: float = MONITOR_INTERVAL,
    ) -> None:
        self.engine = engine
        self.scheduler = scheduler
        self.crate = crate
        self.log = session_log or SessionLog()
        self._notify = notify
        self._interval = interval

        #: deck name -> beat offset from the master clock, captured at load
        self._beat_offset: dict[str, float] = {}
        #: deck name -> the Deck.load_seq the offset was captured for
        self._offset_seq: dict[str, int] = {}
        #: deck name -> engine frame of a resync already in flight
        self._pending_resync: dict[str, int] = {}

        self.interventions: int = 0
        self._last_underruns: int = 0
        self._last_limiter_heavy: int = 0
        self._last_frames: int = -1
        self._last_progress: float = time.monotonic()
        self.stalled: bool = False

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="djai-supervisor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.log.close()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.check_stalled()
                self.check_drift()
                self.check_underruns()
                self.check_limiter()
            except Exception as exc:  # the monitor must outlive its own bugs
                log.exception("supervisor tick failed: %s", exc)

    def _emit(self, message: str) -> None:
        if self._notify is not None:
            self._notify(message)

    # --- context for the log -------------------------------------------------

    def _track_pair(self) -> list[str | None]:
        return [
            self.engine.deck_a.track.title if self.engine.deck_a.track else None,
            self.engine.deck_b.track.title if self.engine.deck_b.track else None,
        ]

    def _position_bars(self, deck: Deck) -> float:
        if deck.track is None:
            return 0.0
        return round(phrase.bar_at_frame(deck.track.analysis, deck.position), 3)

    # --- drift monitor -------------------------------------------------------

    def _expected_beat(
        self,
        deck: Deck,
        position: float,
        master_beat: float,
        master: tuple[Deck, float] | None = None,
    ) -> float | None:
        """Where the master clock says this deck should be, in its own beats."""
        if deck.track is None or not deck.playing or deck.ended:
            return None
        if deck.rate <= 0.0:
            # A backspin: the deck is deliberately running backwards, which
            # reads as enormous drift. It is not drift, and correcting it would
            # fling the playhead forward in the middle of the gesture.
            return None
        if getattr(deck, "manual_pitch", False) and getattr(
            self.engine, "master_deck", None
        ) != deck.name:
            # The operator has this deck's pitch fader. Correcting it back to
            # the master is exactly what they took control to stop; `sync`
            # hands it back.
            return None
        if (
            deck is getattr(self.engine, "_trans_from", None)
            and getattr(self.engine, "_trans_rate_mult", 1.0) != 1.0
        ):
            # A brake or a spin is moving this deck's rate on purpose. It is
            # leaving the mix; correcting its phase would fight the gesture.
            return None
        if getattr(deck, "reversing", False):
            # A slip reverse: off the clock on purpose for a beat or two, and
            # it resumes from its slip position, in phase, by construction.
            return None
        if deck.loop_active:
            # A looping deck is deliberately not advancing with the master
            # clock, so every lap reads as growing drift. Correcting it would
            # mean the supervisor fighting the loop and hard-resyncing out of
            # it. The loop's length is a whole number of beats, which is what
            # keeps the phase right without any correction.
            return None
        key = deck.name
        actual = phrase.beat_at_frame(deck.track.analysis, position)
        # In this deck's own beats: a half-time deck advances one beat for
        # every two of the master's.
        ratio = float(getattr(deck, "metric_ratio", 1.0)) or 1.0
        scaled = master_beat / ratio

        if self._offset_seq.get(key) != deck.load_seq:
            self._offset_seq[key] = deck.load_seq
            self._beat_offset[key] = self._baseline_offset(
                deck, actual, scaled, master, ratio
            )
            return None
        return scaled + self._beat_offset[key]

    def _baseline_offset(
        self,
        deck: Deck,
        actual: float,
        master_beat: float,
        master: tuple[Deck, float] | None,
        ratio: float = 1.0,
    ) -> float:
        """The beat offset to hold this deck at, captured once per load.

        For the master deck this is just its phase against the free-running
        counter -- exact by definition.

        For the other deck we want its **bar lines to coincide with the master
        deck's**, so the baseline is the bar difference between the two decks,
        rounded to a whole bar. Rounding the offset against ``master_beat``
        instead -- which is what this used to do -- is wrong: that counter runs
        free from the moment a track was loaded and has no particular phase
        relative to the master deck's bars, so snapping to it could land a beat
        out. Measured: a +469 ms baseline error at the start of a transition,
        which then drove a resync every bar that never converged.
        """
        raw = actual - master_beat
        if deck.name == self.engine.master_deck or master is None:
            return raw
        if abs(ratio - 1.0) > 1e-6:
            # Counted half or double: its bars are not the master's bars, so
            # "how many bars apart" has no answer to round to. The phase it
            # was cued at is the phase to hold, which is what the raw offset
            # says, and the entry was placed on a downbeat to begin with.
            return raw

        master_deck, _master_position = master
        master_offset = self._beat_offset.get(master_deck.name)
        if master_deck.track is None or master_deck is deck or master_offset is None:
            return raw

        a = master_deck.track.analysis
        b = deck.track.analysis
        dbo_a = phrase.downbeat_offset_beats(a)
        dbo_b = phrase.downbeat_offset_beats(b)
        bpb = phrase.BEATS_PER_BAR

        # Where the master deck is *supposed* to be, in its own bars. That grid,
        # not the free counter, is what the other deck has to line up with.
        master_bar = (master_beat + master_offset - dbo_a) / bpb
        own_bar = (actual - dbo_b) / bpb
        bars_apart = round(own_bar - master_bar)

        # Solve expected_beat = master_beat + offset for the offset that makes
        # own_bar == master_bar + bars_apart.
        return master_offset - dbo_a + dbo_b + bpb * bars_apart

    def nominal_rate(self, deck: Deck) -> float:
        """The beatmatched rate this deck should sit at, ignoring any nudge.

        Derived from the master clock rather than remembered, so it stays right
        across master handovers and tempo changes.
        """
        if deck.track is None:
            return deck.rate
        master_bpm = self.engine.master_bpm
        native = deck.track.analysis.bpm
        if master_bpm <= 0 or native <= 0:
            return deck.rate
        # A deck counted half or double against the master is already at the
        # right speed: matching it to the master's own number would double it.
        ratio = float(getattr(deck, "metric_ratio", 1.0)) or 1.0
        return master_bpm / (native * ratio)

    def check_drift(self) -> None:
        # One consistent snapshot: comparing a deck position against the beat
        # clock requires both to come from the same callback (see Engine._clock).
        frames, master_beat, pos_a, pos_b = self.engine.clock
        pairs = [(self.engine.deck_a, pos_a), (self.engine.deck_b, pos_b)]
        # Master deck first: the other deck's baseline is expressed relative to
        # the master's grid, so the master's own offset has to exist by then.
        pairs.sort(key=lambda dp: dp[0].name != self.engine.master_deck)
        master = next(
            (dp for dp in pairs if dp[0].name == self.engine.master_deck), None
        )

        for deck, position in pairs:
            expected = self._expected_beat(deck, position, master_beat, master)
            if expected is None or deck.track is None:
                continue

            analysis = deck.track.analysis
            actual = phrase.beat_at_frame(analysis, position)
            drift_beats = actual - expected
            drift_ms = drift_beats * analysis.beat_period * 1000.0
            magnitude = abs(drift_ms)

            if magnitude < DRIFT_IGNORE_MS:
                self._restore_rate(deck)
                continue

            if magnitude > DRIFT_INSANE_MS:
                # Drift this large is not drift. It means our own baseline is
                # stale (a deck re-cued behind our back, a clock handover we
                # missed). Resyncing to it would fling the playhead somewhere
                # absurd, so re-baseline and record that we did -- the point of
                # this module is that such moments are visible afterwards.
                self._rebaseline(deck, drift_ms)
                continue

            if magnitude < DRIFT_RESYNC_MS:
                self._nudge(deck, drift_ms)
            else:
                self._hard_resync(deck, drift_ms, expected, position, frames)

    def _rebaseline(self, deck: Deck, drift_ms: float) -> None:
        self._offset_seq.pop(deck.name, None)
        self._pending_resync.pop(deck.name, None)
        self.interventions += 1
        self.log.write(
            "baseline_reset",
            deck=deck.name,
            track_pair=self._track_pair(),
            position_bars=self._position_bars(deck),
            trigger=f"implausible drift {drift_ms:+.0f} ms",
            action="phase baseline discarded and recaptured",
            drift_ms=round(drift_ms, 1),
        )
        self._emit(
            f"[supervisor] deck {deck.name} implausible drift "
            f"({drift_ms / 1000:+.1f} s) -> re-baselined"
        )

    def forget_baseline(self, deck_name: str) -> None:
        """Drop a deck's phase baseline so the next check recaptures it.

        For when the grid it was measured against has been corrected: the old
        offset describes a grid that no longer exists, and measuring against it
        would read a correction as drift.
        """
        self._offset_seq.pop(deck_name, None)
        self._pending_resync.pop(deck_name, None)

    def _restore_rate(self, deck: Deck) -> None:
        """Drift is back in tolerance: take the correction off."""
        nominal = self.nominal_rate(deck)
        if abs(deck.rate - nominal) <= RATE_RESTORE_EPS:
            return
        self.scheduler.submit(
            SetRate(deck=deck.name, rate=nominal, origin="supervisor")
        )
        self.log.write(
            "drift_settled",
            deck=deck.name,
            track_pair=self._track_pair(),
            position_bars=self._position_bars(deck),
            trigger="drift back under threshold",
            action=f"rate {deck.rate:.6f} -> nominal {nominal:.6f}",
        )

    def _nudge(self, deck: Deck, drift_ms: float) -> None:
        """Bend the rate against the drift. Running ahead -> slow down.

        The correction is applied relative to the *nominal* rate, never to the
        current one, so nudges cannot accumulate.
        """
        nominal = self.nominal_rate(deck)
        correction = -(drift_ms / 1000.0) / NUDGE_HORIZON_S
        correction = max(-MAX_RATE_NUDGE, min(MAX_RATE_NUDGE, correction))
        new_rate = nominal * (1.0 + correction)

        self.scheduler.submit(
            SetRate(deck=deck.name, rate=new_rate, origin="supervisor")
        )
        self.interventions += 1
        self.log.write(
            "drift_nudge",
            deck=deck.name,
            track_pair=self._track_pair(),
            position_bars=self._position_bars(deck),
            trigger=f"drift {drift_ms:+.2f} ms",
            action=f"rate {deck.rate:.6f} -> {new_rate:.6f}",
            drift_ms=round(drift_ms, 3),
        )
        self._emit(
            f"[supervisor] deck {deck.name} drift {drift_ms:+.1f} ms "
            f"-> rate nudge to {new_rate:.5f}"
        )

    def _hard_resync(
        self,
        deck: Deck,
        drift_ms: float,
        expected_beat: float,
        position: float,
        frames: int,
    ) -> None:
        """Snap the deck onto the grid at the next downbeat.

        ``position`` and ``frames`` come from the same clock snapshot the drift
        was measured from. Reading ``deck.position`` and ``engine.frames_played``
        live here instead -- which is what this used to do -- mixes two different
        callbacks and lands the target one audio block out. Measured: resyncs
        that repeated every bar at -22 to -42 ms and never converged, which is
        exactly one 2048-frame block at 44.1 kHz.
        """
        analysis = deck.track.analysis if deck.track else None
        if analysis is None:
            return
        boundary = phrase.next_downbeat(deck, after_frame=position)
        rate = deck.rate if deck.rate > 0 else 1.0
        execute_at = int(round(frames + (boundary - position) / rate))

        # A resync is already on its way; re-issuing one every 100 ms until it
        # lands just floods the queue and the log with the same intervention.
        in_flight = self._pending_resync.get(deck.name)
        if in_flight is not None and frames < in_flight:
            return
        self._pending_resync[deck.name] = execute_at
        # The snap happens in the future, so aim at where the deck *should* be
        # then, not where it should be now: advance the expected beat by however
        # much the master clock will move between now and the boundary.
        engine_frames_ahead = max(0, execute_at - frames)
        expected_at_boundary = expected_beat + engine_frames_ahead * (
            self.engine.master_bpm / 60.0 / SAMPLE_RATE
        )
        target_frame = phrase.frame_at_beat(analysis, expected_at_boundary)
        self.scheduler.submit(
            Resync(
                deck=deck.name,
                frame=int(round(target_frame)),
                execute_at=execute_at,
                origin="supervisor",
            )
        )
        self.interventions += 1
        self.log.write(
            "drift_resync",
            deck=deck.name,
            track_pair=self._track_pair(),
            position_bars=self._position_bars(deck),
            trigger=f"drift {drift_ms:+.2f} ms",
            action=f"hard resync at next downbeat (frame {boundary})",
            drift_ms=round(drift_ms, 3),
        )
        self._emit(
            f"[supervisor] deck {deck.name} drift {drift_ms:+.1f} ms "
            f"-> hard resync at next downbeat"
        )

    def check_stalled(self) -> None:
        """Notice when the audio device stops calling the callback at all.

        Reported once per stall and once again on recovery, so a dead stream
        does not fill the log with one line every 100 ms.
        """
        frames = self.engine.frames_played
        now = time.monotonic()

        if frames != self._last_frames:
            if self.stalled:
                self.stalled = False
                self.log.write(
                    "stream_resumed",
                    track_pair=self._track_pair(),
                    position_bars=self._position_bars(self.engine.deck_a),
                    trigger="frame counter moving again",
                    action="none (reported)",
                )
                self._emit("[supervisor] audio stream resumed")
            self._last_frames = frames
            self._last_progress = now
            return

        if self.stalled or not self.engine.running:
            return
        silent_for = now - self._last_progress
        if silent_for > STALL_TIMEOUT_S:
            self.stalled = True
            self.interventions += 1
            self.log.write(
                "stream_stalled",
                track_pair=self._track_pair(),
                position_bars=self._position_bars(self.engine.deck_a),
                trigger=f"no audio callback for {silent_for:.1f}s",
                action="none (reported) - the output device stopped calling back",
                frames_played=frames,
            )
            self._emit(
                f"[supervisor] AUDIO STALLED: no callback for {silent_for:.1f}s. "
                "The output device stopped. Try `djai devices` and --device."
            )

    def check_limiter(self) -> None:
        """Log any limiting deeper than LIMITER_LOG_REDUCTION_DB.

        With every track loudness-normalised the limiter is true-peak safety
        only, so reduction past that bar indicates a bug, not a hot master.
        The callback counts; this reports, off the audio thread.
        """
        import math

        from djai import config

        heavy = self.engine.limiter_heavy_blocks
        if heavy <= self._last_limiter_heavy:
            return
        blocks = heavy - self._last_limiter_heavy
        self._last_limiter_heavy = heavy
        deepest = self.engine.limiter_deepest_gain
        # Reporting only: a lost update from the audio thread costs one reading.
        self.engine.limiter_deepest_gain = 1.0
        reduction_db = -20.0 * math.log10(max(deepest, 1e-9))
        self.log.write(
            "limiter_reduction",
            track_pair=self._track_pair(),
            trigger=(
                f"{blocks} block(s) limited by more than "
                f"{config.LIMITER_LOG_REDUCTION_DB:g} dB"
            ),
            action="none (reported) - with loudness normalisation this is a bug",
            blocks=blocks,
            deepest_reduction_db=round(reduction_db, 2),
        )
        self._emit(
            f"[supervisor] limiter reduced gain by {reduction_db:.1f} dB "
            f"({blocks} block(s)) - check loudness normalisation"
        )

    def check_underruns(self) -> None:
        """Surface buffer underruns. The callback can only count them."""
        current = self.engine.underruns
        if current > self._last_underruns:
            delta = current - self._last_underruns
            self._last_underruns = current
            self.log.write(
                "underrun",
                track_pair=self._track_pair(),
                position_bars=self._position_bars(self.engine.deck_a),
                trigger=f"{delta} new underrun(s)",
                action="none (reported)",
                total=current,
            )
            self._emit(f"[supervisor] {delta} buffer underrun(s) (total {current})")

    # --- command validator ---------------------------------------------------

    def validate(self, cmd: Command) -> Rejection | None:
        """Check a command before it may enter the scheduler.

        Returns None if it is acceptable, otherwise a :class:`Rejection` that
        has already been logged. Every command originating from the LLM must
        pass through here -- that is the whole point of the layer.
        """
        reason = self._reject_reason(cmd)
        if reason is None:
            return None
        rejection = Rejection(command=cmd, reason=reason)
        self.log.write(
            "command_rejected",
            track_pair=self._track_pair(),
            position_bars=self._position_bars(self.engine.deck_a),
            trigger=cmd.describe(),
            action=f"rejected: {reason}",
            origin=cmd.origin,
        )
        self._emit(f"[supervisor] rejected {cmd.describe()}: {reason}")
        return rejection

    def validate_transition_style(self, name: str) -> str | None:
        """Vet a requested transition style. None if acceptable.

        The model is allowed to ask for a style by name, which means a
        hallucinated name is a thing that can reach this layer. It is checked
        against the styles the engine actually implements, and anything else is
        rejected and logged rather than silently falling back to a default --
        a style that quietly did not happen is a style the operator will keep
        asking for.
        """
        if name in tr.STYLE_CHOICES:
            return None
        reason = (
            f"unknown transition style {name!r}; known styles are "
            f"{', '.join(tr.STYLE_CHOICES)}"
        )
        self.log.write(
            "transition_style_rejected",
            trigger=str(name),
            action=f"rejected: {reason}",
        )
        self._emit(f"[supervisor] rejected transition style {name!r}")
        return reason

    def validate_transition_params(
        self,
        raw: object,
        track_b: TrackAnalysis | None = None,
        track_a: TrackAnalysis | None = None,
    ) -> tuple["tr.TransitionParams | None", str]:
        """Vet a designed transition. ``(params, "")`` or ``(None, reason)``.

        This is the layer that matters most once a model is designing
        transitions rather than picking from a list of four. Everything it
        returns is checked: every field's type and range, the swap bar against
        the transition's own length, the echo against the same, and a named
        hot cue against the cues the incoming track actually has.

        A rejection is never a silent downgrade -- the caller falls back to the
        rule-based preset, and both the parameters and the reason are logged,
        because a transition that sounded wrong is only debuggable if you can
        see what was asked for as well as what ran.
        """
        if not isinstance(raw, dict):
            return None, f"expected an object, got {type(raw).__name__}"

        def number(field: str, lo: float, hi: float):
            v = raw.get(field)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{field} must be a number, got {v!r}")
            if not lo <= float(v) <= hi:
                raise ValueError(f"{field} {v!r} is outside {lo}..{hi}")
            return float(v)

        def whole(field: str, lo: int, hi: int) -> int:
            v = raw.get(field)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{field} must be a whole number, got {v!r}")
            if not lo <= v <= hi:
                raise ValueError(f"{field} {v} is outside {lo}..{hi}")
            return v

        def require(field: str):
            """Absent is not the same as a default.

            Every field is in the decode schema's `required` list, so a model
            that leaves one out has not answered -- and quietly substituting a
            default turns an omission into a decision nobody made. Each of
            these defaults happened to point at a plain bass swap.
            """
            if field not in raw:
                raise ValueError(f"{field} is required and was not given")
            return raw[field]

        try:
            length = number("length_bars", *tr.LENGTH_BARS_RANGE)

            curve = require("curve")
            if curve not in tr.CURVES:
                raise ValueError(f"curve {curve!r} is not one of {tr.CURVES}")

            sweep = require("filter_sweep")
            if sweep not in tr.FILTER_SWEEPS:
                raise ValueError(f"filter_sweep {sweep!r} is not known")
            resonance = number("filter_resonance", *tr.FILTER_RESONANCE_RANGE)

            rolloff = require("deck_a_high_rolloff")
            if not isinstance(rolloff, bool):
                raise ValueError(f"deck_a_high_rolloff must be true or false, "
                                 f"got {rolloff!r}")

            low_swap_bars = whole("low_swap_bars", *tr.LOW_SWAP_BARS_RANGE)

            # Present-but-null and absent are different things. Null means
            # "no managed swap", which is a real choice; absent means the model
            # did not answer, and defaulting that to the same thing is how an
            # omission turns into a decision nobody made.
            low_swap_bar = require("low_swap_bar")
            if low_swap_bar is not None:
                if isinstance(low_swap_bar, bool) or not isinstance(low_swap_bar, int):
                    raise ValueError(
                        f"low_swap_bar must be a whole number or null, "
                        f"got {low_swap_bar!r}"
                    )
                if not 0 <= low_swap_bar <= length:
                    raise ValueError(
                        f"low_swap_bar {low_swap_bar} is outside the "
                        f"transition's own {length:.0f} bars"
                    )

            delay = whole("deck_b_low_delay_bars", *tr.DECK_B_LOW_DELAY_RANGE)
            echo = whole("echo_bars", *tr.ECHO_BARS_RANGE)
            if echo > length:
                raise ValueError(
                    f"echo_bars {echo} exceeds the transition's {length:.0f} bars"
                )

            intensity = number("intensity", *tr.INTENSITY_RANGE)

            # --- the high-energy set ---------------------------------------
            loop_out = number("loop_out_bars", *tr.LOOP_OUT_BARS_RANGE)
            loop_halving = require("loop_halving")
            if not isinstance(loop_halving, bool):
                raise ValueError(
                    f"loop_halving must be true or false, got {loop_halving!r}"
                )
            division = whole("beat_repeat_division", 0, 16)
            if division not in tr.BEAT_REPEAT_DIVISIONS:
                raise ValueError(
                    f"beat_repeat_division {division} is not one of "
                    f"{tr.BEAT_REPEAT_DIVISIONS}"
                )
            backspin = number("backspin_bars", *tr.BACKSPIN_BARS_RANGE)
            double_drop = number("double_drop_bars", *tr.DOUBLE_DROP_BARS_RANGE)
            if double_drop > length:
                raise ValueError(
                    f"double_drop_bars {double_drop:.0f} exceeds the "
                    f"transition's {length:.0f} bars"
                )
            if backspin > length:
                raise ValueError(
                    f"backspin_bars {backspin:.0f} exceeds the transition's "
                    f"{length:.0f} bars"
                )
            if loop_out > length:
                raise ValueError(
                    f"loop_out_bars {loop_out:.0f} exceeds the transition's "
                    f"{length:.0f} bars"
                )

            # --- effects ---------------------------------------------------
            reverb = whole("reverb_bars", *tr.REVERB_BARS_RANGE)
            if reverb > length:
                raise ValueError(
                    f"reverb_bars {reverb} exceeds the transition's {length:.0f} bars"
                )
            brake = whole("brake_beats", 0, max(tr.BRAKE_BEATS_CHOICES))
            if brake not in tr.BRAKE_BEATS_CHOICES:
                raise ValueError(
                    f"brake_beats {brake} is not one of {tr.BRAKE_BEATS_CHOICES}"
                )
            riser = whole("riser_bars", 0, max(tr.RISER_BARS_CHOICES))
            if riser not in tr.RISER_BARS_CHOICES:
                raise ValueError(f"riser_bars {riser} must be 0, or 4 to 8")
            if riser > length:
                raise ValueError(
                    f"riser_bars {riser} exceeds the transition's {length:.0f} bars"
                )
            if brake and backspin > 0:
                raise ValueError(
                    "brake_beats and backspin_bars both move the outgoing record; "
                    "ask for one"
                )
            vocal_aware = require("vocal_aware")
            if not isinstance(vocal_aware, bool):
                raise ValueError(
                    f"vocal_aware must be true or false, got {vocal_aware!r}"
                )

            align = require("align_mode")
            if align not in tr.ALIGN_MODES:
                raise ValueError(f"align_mode {align!r} is not known")

            # Lining two drops up needs a drop on both records. Without one on
            # either side there is nothing to align, and the transition would
            # land on an arbitrary bar and sound like a mistake.
            if align == "drop" or double_drop > 0:
                which = "double_drop" if double_drop > 0 else "drop_swap"
                for label, track in (("incoming", track_b), ("outgoing", track_a)):
                    if track is None:
                        continue
                    if tr.drop_cue(track) is None:
                        raise ValueError(
                            f"{which} needs a drop hot cue on both tracks; the "
                            f"{label} track has none"
                        )

            entry = require("entry_point")
            if not isinstance(entry, str):
                raise ValueError(f"entry_point must be a string, got {entry!r}")
            if entry != "mix_in":
                if not entry.startswith("hot_cue_"):
                    raise ValueError(f"entry_point {entry!r} is not recognised")
                try:
                    index = int(entry.split("_")[-1])
                except ValueError:
                    raise ValueError(f"entry_point {entry!r} has no cue number")
                cues = list(getattr(track_b, "hot_cues", None) or [])
                cue = next((c for c in cues if c.get("index") == index), None)
                if cue is None:
                    have = sorted(c.get("index") for c in cues)
                    raise ValueError(
                        f"entry_point {entry!r}: the incoming track has no hot "
                        f"cue {index} (it has {have})"
                    )
                # Existing is not the same as usable. A cue near the track's
                # own mix-out leaves nothing playing after the blend hands
                # over, and the silence recovery that follows is dead air.
                # Measured in a soak: a designed hot_cue_3 entry emptied the
                # deck ninety seconds later.
                seconds = float(cue["sample_position"]) / 44100.0
                if track_b is not None and not tr.entry_has_runway(track_b, seconds):
                    raise ValueError(
                        f"entry_point {entry!r} leaves under "
                        f"{tr.MIN_ENTRY_RUNWAY_BARS:.0f} bars before mix-out"
                    )
        except ValueError as exc:
            return None, str(exc)

        return (
            tr.TransitionParams(
                length_bars=length,
                curve=curve,
                low_swap_bar=low_swap_bar,
                low_swap_bars=low_swap_bars,
                deck_a_high_rolloff=rolloff,
                deck_b_low_delay_bars=delay,
                echo_bars=echo,
                filter_sweep=sweep,
                filter_resonance=resonance,
                entry_point=entry,
                intensity=intensity,
                loop_out_bars=loop_out,
                loop_halving=loop_halving,
                beat_repeat_division=division,
                backspin_bars=backspin,
                double_drop_bars=double_drop,
                align_mode=align,
                reverb_bars=reverb,
                brake_beats=brake,
                riser_bars=riser,
                vocal_aware=vocal_aware,
                name="llm",
            ),
            "",
        )

    def check_vocal_clash(
        self,
        track_a: TrackAnalysis,
        track_b: TrackAnalysis,
        start_bar_a: float,
        entry_bar_b: float,
        bars: float,
        envelope,
        style: str = "",
        log: bool = True,
    ) -> str | None:
        """Refuse a transition that plays two vocals over each other.

        Walks the envelope the engine will run, row by row. A row clashes when
        both decks are audible (gain above :data:`VOCAL_AUDIBLE_GAIN`), both
        are inside one of their vocal bar ranges at that moment, and the
        incoming deck's low band is open. The exception is the brief's: while
        the incoming low band is still cut, the incoming vocal is not yet
        carrying the mix, and it is allowed.

        A clash of at least :data:`VOCAL_CLASH_MIN_BARS` is rejected and logged.
        Returns the reason, or None.
        """
        from djai import transition as tr

        if bars <= 0 or envelope is None or len(envelope) < 2:
            return None
        if not track_a.vocal_bars or not track_b.vocal_bars:
            return None
        rows = len(envelope) - 1
        clash_rows = 0
        first_clash: float | None = None

        def vocal_at(track: TrackAnalysis, bar: float) -> bool:
            return any(a <= bar < b for a, b in track.vocal_bars)

        for i in range(rows):
            t = i / rows
            row = envelope[i]
            if row[tr.FROM_GAIN] <= VOCAL_AUDIBLE_GAIN or row[tr.TO_GAIN] <= VOCAL_AUDIBLE_GAIN:
                continue
            if row[tr.TO_LOW] < 0.5:
                continue  # incoming bass still cut: the exception
            if vocal_at(track_a, start_bar_a + t * bars) and vocal_at(
                track_b, entry_bar_b + t * bars
            ):
                clash_rows += 1
                if first_clash is None:
                    first_clash = t * bars
        clash_bars = clash_rows / rows * bars
        if clash_bars < VOCAL_CLASH_MIN_BARS:
            return None
        reason = (
            f"vocal clash: both decks carry vocals over {clash_bars:.1f} bar(s) of the "
            f"{bars:.0f}-bar {style or 'transition'}, from bar {first_clash:.1f} of it, "
            f"with the incoming low band open (deck A from bar {start_bar_a:.1f}, "
            f"deck B from bar {entry_bar_b:.1f})"
        )
        if not log:
            return reason  # a probe for a better placement, not a rejection
        self.log.write(
            "vocal_clash_rejected",
            trigger=style or "transition",
            action=f"rejected: {reason}",
            track_pair=[track_a.title, track_b.title],
            clash_bars=round(clash_bars, 2),
            start_bar_a=round(start_bar_a, 2),
            entry_bar_b=round(entry_bar_b, 2),
        )
        self.interventions += 1
        self._emit(f"[supervisor] rejected a vocal clash ({clash_bars:.1f} bars)")
        return reason

    def _reject_reason(self, cmd: Command) -> str | None:
        known_ids = {t.track_id for t in self.crate}

        if isinstance(cmd, LoadTrack):
            if cmd.track is None:
                return "no track payload"
            track_id = cmd.track.analysis.track_id
            if track_id not in known_ids:
                return f"track {track_id!r} is not in the analysis cache"
            if not Path(cmd.track.analysis.path).exists():
                return f"audio file is missing: {cmd.track.analysis.path}"
            if cmd.rate <= 0 or abs(cmd.rate - 1.0) > MAX_RATE_DELTA:
                return (
                    f"rate {cmd.rate:.4f} is outside the +-"
                    f"{MAX_RATE_DELTA:.0%} stretch range"
                )
            deck = self.engine.deck(cmd.deck) if cmd.deck in ("a", "b") else None
            if deck is None:
                return f"unknown deck {cmd.deck!r}"
            if self.engine.deck_in_transition(cmd.deck):
                return f"deck {cmd.deck} is mid-transition"
            return None

        if isinstance(cmd, StartTransition):
            if cmd.from_deck == cmd.to_deck:
                return "transition source and destination are the same deck"
            for name in (cmd.from_deck, cmd.to_deck):
                if name not in ("a", "b"):
                    return f"unknown deck {name!r}"
            if self.engine.transition_active:
                return "a transition is already running"
            src = self.engine.deck(cmd.from_deck)
            dst = self.engine.deck(cmd.to_deck)
            if src.track is None or dst.track is None:
                return "both decks must have a track loaded"
            if cmd.total_frames <= 0:
                return "transition length must be positive"
            # Compare the decks as they are counted, not as they are clocked:
            # 87 BPM under 174 is a half-time mix, not a 50% tempo error.
            dst_ratio = float(getattr(dst, "metric_ratio", 1.0)) or 1.0
            bpm_delta = abs(
                (dst.track.analysis.bpm * dst.rate * dst_ratio)
                - (src.track.analysis.bpm * src.rate)
            )
            reference = src.track.analysis.bpm * src.rate
            # A cut is one block long and overlaps nothing, so there is no
            # beat-matching for the stretch range to protect. Refusing it left
            # a track whose tempo has no partner in the crate with no way out
            # at all: the automation could not cue, and the room went quiet.
            # Every longer shape still has to be inside the range.
            is_cut = cmd.total_frames <= max(self.engine.blocksize, 1)
            if not is_cut and reference > 0 and bpm_delta / reference > MAX_RATE_DELTA:
                return (
                    f"decks are {bpm_delta:.1f} BPM apart, beyond the "
                    f"{MAX_RATE_DELTA:.0%} stretch range"
                )

            # Where will the outgoing deck actually be when this fires?
            #
            # Skipped for origin="user": this guard exists to catch automatic
            # placement regressing to "first boundary after selection". A human
            # asking to mix out now has decided to, and blocking that would be
            # the tool overruling the DJ.
            if cmd.origin == "user":
                return None
            analysis = src.track.analysis
            # Measured against the audio the deck actually has: a truncated
            # file ends long before its analysed duration, and its last bar is
            # not "17% of the track".
            total = min(analysis.duration_s * SAMPLE_RATE,
                        float(getattr(src.track, "source_frames", 0) or float("inf")))
            frames_ahead = (
                0
                if cmd.is_immediate
                else max(0, cmd.execute_at - self.engine.frames_played)
            )
            start_frame = src.position + frames_ahead * max(src.rate, 1e-9)
            if total > 0:
                fraction = start_frame / total
                minimum = (
                    DROP_ALIGNED_MIN_START_FRACTION
                    if getattr(cmd, "drop_aligned", False)
                    else MIN_TRANSITION_START_FRACTION
                )
                if fraction < minimum:
                    return (
                        f"transition would start at {fraction:.0%} of "
                        f"{analysis.title!r} "
                        f"({start_frame / SAMPLE_RATE:.1f}s of "
                        f"{analysis.duration_s:.1f}s, bar "
                        f"{phrase.bar_at_frame(analysis, start_frame):.1f}); "
                        f"mix-out is at bar {analysis.mix_out_bar:.1f}. "
                        f"Minimum is {minimum:.0%} -- "
                        f"placement must come from phrase.plan_transition"
                    )
            return None

        if isinstance(cmd, SetEQ):
            if cmd.deck not in ("a", "b"):
                return f"unknown deck {cmd.deck!r}"
            for band, value in (("low", cmd.low), ("mid", cmd.mid), ("high", cmd.high)):
                if value is None:
                    continue
                if not isinstance(value, (int, float)) or not 0.0 <= value <= 2.0:
                    return f"{band} gain {value!r} is out of range 0..2"
            return None

        if isinstance(cmd, SetGain):
            if cmd.deck not in ("a", "b"):
                return f"unknown deck {cmd.deck!r}"
            # Unity is the ceiling. The band gains allow up to 2.0 because an
            # EQ cut elsewhere leaves headroom to make up; channel level has no
            # such headroom, and boosting it is how a mix clips.
            if not isinstance(cmd.gain, (int, float)) or not 0.0 <= cmd.gain <= 1.0:
                return f"gain {cmd.gain!r} is out of range 0..1"
            return None

        if isinstance(cmd, SetRate):
            if cmd.rate <= 0 or abs(cmd.rate - 1.0) > MAX_RATE_DELTA:
                return f"rate {cmd.rate:.4f} is outside the stretch range"
            return None

        # --- performance controls -------------------------------------------
        if isinstance(cmd, (SetPitch, SyncDeck)):
            if cmd.deck not in ("a", "b"):
                return f"unknown deck {cmd.deck!r}"
            if not isinstance(cmd.rate, (int, float)) or not (
                0 < cmd.rate and abs(cmd.rate - 1.0) <= MAX_RATE_DELTA + 1e-9
            ):
                return f"rate {cmd.rate!r} is outside the +-{MAX_RATE_DELTA:.0%} pitch range"
            return None

        if isinstance(cmd, (SetLoop, ExitLoop, BeatJump)):
            if cmd.deck not in ("a", "b"):
                return f"unknown deck {cmd.deck!r}"
            deck = self.engine.deck(cmd.deck)
            if deck.track is None:
                return f"deck {cmd.deck} is empty"
            total = deck.track.analysis.duration_s * SAMPLE_RATE
            if isinstance(cmd, SetLoop):
                if not cmd.length_frames > 0:
                    return "a loop must have a positive length"
                if not 0 <= cmd.start_frame < total:
                    return f"loop start {cmd.start_frame:.0f} is outside the track"
            if isinstance(cmd, BeatJump):
                # Where the deck will be once the jump fires, allowing for the
                # time until then.
                ahead = 0 if cmd.is_immediate else max(0, cmd.execute_at - self.engine.frames_played)
                landing = deck.position + ahead * max(deck.rate, 0.0) + cmd.frames
                if not 0 <= landing < total:
                    return f"a {cmd.frames:+.0f}-frame jump would leave the track"
            return None

        return None

    # --- pass-through submission --------------------------------------------

    def submit_validated(self, cmd: Command) -> Rejection | None:
        """Validate, then schedule. The only route LLM commands may take."""
        rejection = self.validate(cmd)
        if rejection is not None:
            return rejection
        self.scheduler.submit(cmd)
        self.log.write(
            "command_scheduled",
            track_pair=self._track_pair(),
            position_bars=self._position_bars(self.engine.deck_a),
            trigger=cmd.describe(),
            action=(
                "immediate"
                if cmd.is_immediate
                else f"queued for engine frame {cmd.execute_at}"
            ),
            origin=cmd.origin,
        )
        return None

