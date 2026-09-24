"""The terminal REPL and the session that runs behind it.

THREADING CONTEXT: this module owns the **main thread** (argument parsing, the
input loop) and starts the three background threads that make up the control
layer:

* the **scheduler thread** (:mod:`djai.scheduler`) releases queued commands
* the **monitor thread** (:mod:`djai.supervisor`) watches drift and logs
* the **autopilot thread** (below) decodes the next track and arms transitions

Typed text goes to :mod:`djai.intent`, which blocks for 1-3 s on the LLM. That
is why it runs on the REPL thread and not on any of the above: a slow model must
never delay a queued command or a drift correction, and it can never delay
audio, which runs on a thread none of this code touches.

The literal words ``cut``, ``killbass`` and ``stop`` never reach the LLM. They
are matched before anything else and dispatched straight to the engine queue.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import re
import queue
import json
import sys
import threading
from collections import Counter

import numpy as np
import time
from pathlib import Path
from typing import Any

from djai import analysis as an
from djai import actions as actions_mod
from djai import config, glue, phrase, transition, understanding
from djai import preview as preview_mod
from djai import stems as stems_mod
from djai import revise as revise_mod
from djai.analysis import (
    DEFAULT_CACHE_DIR,
    TrackAnalysis,
    analyze_folder,
    load_crate,
)
from djai.commands import (
    IMMEDIATE,
    BeatJump,
    CancelToken,
    CancelTransition,
    Cut,
    ExitLoop,
    KillBass,
    LoadTrack,
    SetEQ,
    SetFilter,
    SetLoop,
    SetMasterGain,
    SetPitch,
    StartTransition,
    Stop,
    SwapStems,
    SyncDeck,
)
from djai.deck import SAMPLE_RATE, LoadedTrack, TransportState, load_track
from djai.engine import (
    DEFAULT_BLOCKSIZE,
    Engine,
    SessionRecorder,
    list_output_devices,
)
from djai.fallback import FallbackPlayer, Watchdog
from djai.preflight import run_preflight
from djai.intent import Intent, IntentEngine
from djai.scheduler import Scheduler
from djai.selector import nearest_tempo as select_nearest_tempo
from djai.selector import plan_journey, rank_candidates, select_next
from djai.selector import RIDE_MARGIN_BARS
from djai import selector as selector_mod
from djai import planner, room, songs
from djai.setstate import SetState
from djai.selector import tempo_path as selector_tempo_path
from djai.supervisor import DEFAULT_LOG_DIR, SessionLog, Supervisor

#: Safety margin on top of the computed autopilot lead, for decode time.
AUTOPILOT_MARGIN_SECONDS: float = 15.0

#: The lead has to cover *two* transitions' worth of time, not one: the
#: autopilot arms at the next 32-bar boundary, which can itself be a full
#: phrase away, and only then does the 32-bar transition start. A fixed 90 s
#: lead lets a 63 s transition begin with 27 s of the outgoing track left,
#: which runs the outgoing deck off its end mid-swap.
AUTOPILOT_LEAD_PHRASES: float = 2.0

#: Autopilot poll interval.
AUTOPILOT_TICK: float = 0.5

#: One unresolved "no next track is cued" condition is reported again at most
#: this often. The tick runs twice a second, and a warning that floods the log
#: is one nobody reads mid-set.
NO_CUE_REPEAT_S: float = 30.0

#: A transition armed with nothing queued and nothing running must stay that
#: way this long before it is reported. Shorter would catch the few blocks
#: between the scheduler releasing the swap and the engine starting it.
STUCK_ARM_S: float = 2.0

log = logging.getLogger(__name__)

#: A tap more than this many seconds after the previous one starts a new tap
#: tempo, rather than dragging a stale sequence into the average.
TAP_RESET_S: float = 2.5

PANIC_WORDS = {"cut", "killbass", "kill bass", "stop", "quit", "exit"}

BANNER = """\
djai - type plain English to steer the mix.
  cut / killbass / stop   panic commands, immediate, never sent to the LLM
  state                   print the current state without calling the LLM
  help                    this message

Manual override - matched before the model, so these work when it is down:
  freeze / resume         take over from the autopilot / hand it back
  go                      blend into the cued track at the next phrase
  mode assisted|autonomous  co-pilot: I propose, you say `go` (or I blend
                          at the last call); autonomous: I blend on my own
  mc on|off               dip the music 12 dB for the mic, hold the blend
  explain                 why: the last decisions, read back from the log
  force <track>           pin the next track, overriding the selector
  hotcue <n>              jump to a stored marker (set / clear also work)
  style <name|auto>       pick the transition style for the next blend
  phase <name|auto>       set arc: warmup, build, peak or cooldown
  loop [a|b] in|out|exit|halve|double|<beats>    loops (slip: exit keeps phase)
  roll [a|b] <1/8..4>|off loop roll, in bars
  jump [a|b] <+-1|4|8|16> beat jump, in bars
  pitch [a|b] <+-pct>|reset   pitch fader, +-8%
  sync [a|b]              match tempo and phase to the master
  quantize on|off         snap loops and jumps to the beat grid (default on)
  (every manual control holds the automation; `resume` hands it back)
  cue a | cue b | cue off route a deck to the pre-listen output
  keylock [a|b] on|off    original pitch at any tempo (on), or resample (off)
  filter [a|b] <-1..1>    filter knob: -1 low-pass, 0 off, 1 high-pass;
                          add `res <0..1>` for resonance, or say `filter off`

Anything else is interpreted by the local model and queued to the next 32-bar
phrase. If the model is slow or down, keyword matching takes over.
"""


class Session:
    """Everything that is running: engine, scheduler, supervisor, autopilot."""

    def __init__(
        self,
        crate: list[TrackAnalysis],
        device: int | None = None,
        blocksize: int = DEFAULT_BLOCKSIZE,
        log_dir: Path = DEFAULT_LOG_DIR,
        cue_device: int | None = None,
        cue_channels: tuple[int, int] | None = None,
        record_path: Path | None = None,
        fallback_enabled: bool | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.crate = crate
        #: Where sidecars live, so an edited hot cue can be written back.
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.played: set[str] = set()
        #: Tracks cued this session, oldest first. The selector's repetition
        #: penalties (artist, key, back-to-back vocals) look back through it.
        self.history: list[TrackAnalysis] = []
        self._notices: queue.Queue[str] = queue.Queue()
        self._notice_listeners: list[Any] = []
        #: Tap tempo timestamps per deck, from time.monotonic().
        self._taps: dict[str, list[float]] = {}

        self.recorder = (
            SessionRecorder(Path(record_path)) if record_path is not None else None
        )
        self.engine = Engine(
            device=device,
            blocksize=blocksize,
            cue_device=cue_device,
            cue_channels=cue_channels,
            recorder=self.recorder,
        )
        # The pre-roll hook is what previews a transition before it plays. It
        # runs on the scheduler's own worker thread, never on the tick; when the
        # preview is disabled no hook is passed and no worker thread exists.
        self.scheduler = Scheduler(
            self.engine,
            on_dropped=self._on_dropped,
            on_pre_roll=self._on_pre_roll if config.PREVIEW_ENABLED else None,
            pre_roll_frames=int(config.PREVIEW_PRE_ROLL_S * SAMPLE_RATE),
        )
        #: What the preview needs about the transition arm_transition last
        #: queued: its two commands, the shape, and where it was placed. Replaced
        #: wholesale on every arm, so a preview of a superseded transition can
        #: tell it is stale by identity and leave the new one alone.
        self._armed_preview: dict | None = None
        #: Serialises the preview's check-cancel-resubmit of the queued commands.
        self._preview_lock = threading.Lock()
        self._preview_listeners: list[Any] = []
        #: The most recent preview outcome, for the REPL's `state`.
        self.last_preview: Any = None
        self.session_log = SessionLog(log_dir)
        self.supervisor = Supervisor(
            self.engine,
            self.scheduler,
            crate,
            session_log=self.session_log,
            notify=self.notify,
        )

        #: The closed loop (SPEC §5): the master output measured bar by bar,
        #: the optional room mic, and the energy loop. None when disabled.
        self.room: room.RoomMonitor | None = (
            room.RoomMonitor(self.engine) if config.CLOSED_LOOP_ENABLED else None
        )
        #: Bars since the last room_reading line, so the log gets one per
        #: :data:`ROOM_LOG_BARS` rather than one per bar.
        self._room_bars: int = 0
        #: Whether the loop last reported a deviation past its reach, so that
        #: is logged once per episode rather than once per bar.
        self._room_saturated: bool = False
        self._room_deferred: bool = False

        #: Which deck is currently the one the audience hears.
        self.live_deck: str = "a"
        #: The decoded track waiting on the idle deck. Held here rather than
        #: read back from the deck because a LoadTrack is applied by the audio
        #: thread on its next callback -- code that cues and then immediately
        #: reads ``deck.track`` sees None and silently does nothing.
        self._cued: LoadedTrack | None = None
        #: The opening track, kept so the fallback can be armed from it.
        self._first_track: LoadedTrack | None = None
        #: Set while the autopilot has already armed a transition.
        self._transition_armed: bool = False
        #: How the last cue's tempo gets from the playing track to it.
        self.tempo_plan: Any = None
        #: Ride steps queued but not yet landed, as (engine frame, rate). The
        #: beat clock follows them -- see _follow_ride.
        #: Tracks whose audio would not decode this session. A file can rot,
        #: move onto a disconnected drive or be re-encoded between analysis and
        #: the gig, and the failure only shows at load. Remembering them is
        #: what stops the selector reaching for the same broken file every
        #: tick: without it one unreadable track stalls the whole set, because
        #: a track that never cues is never marked played.
        self._unplayable: set[str] = set()
        #: The cue decision whose outcome window is still open. Closed when
        #: the transition it led to finishes, which is the first moment there
        #: is anything to say about how it went.
        self._cue_decision: str | None = None
        self._ride_steps: list[tuple[int, float]] = []
        self._ride_deck: str = "a"
        #: Every ride and glide step carries this token, so dropping the ride
        #: withdraws all of them, whatever their origin. Replaced once used.
        self._ride_token = CancelToken("tempo")
        #: The energy correction in flight; a new request replaces it.
        self._energy_token = CancelToken("energy")
        #: The action language's plan in flight (djai.actions); superseded by
        #: the next plan.
        self._action_token: CancelToken | None = None
        #: One-shot: bring the next cued track in at this ranked mix-in region
        #: (1 = best) instead of its default mix-in. Set by a `cue` action.
        self.cue_mix_point: int | None = None
        #: Hold the beat clock where it is instead of gliding to each
        #: incoming track's own tempo. Off by default; `lock bpm` turns
        #: it on for operators who want a fixed-tempo set.
        self.master_tempo_locked: bool = bool(config.MASTER_TEMPO_LOCKED)
        self._clock_at_arm: float = 0.0
        #: Where the set is taking the tempo, in BPM, or None to stay where it
        #: is. The journey rides toward it a track at a time; nothing jumps.
        self.tempo_target: float | None = None
        #: The plan the last cue made: its first step is what was cued, the
        #: rest is where the set was heading. Shown in the UI, re-made at
        #: every cue.
        self.journey: Any = None
        #: Selector and critic weights, learned from the feedback log. Empty
        #: until :mod:`djai.feedback` has something to say.
        self.selector_weights: dict[str, float] = {}
        self.critic_weights: dict[str, float] = {}
        #: What the logs said when the weights were last learned, for the UI.
        self.feedback_summary: dict[str, Any] = {}
        #: Bars added to (or taken off) every blend's planned length, learned
        #: from "too early" / "too late" (djai.feedback). 0 until taught.
        self.timing_bias_bars: int = 0
        #: The stable model in use (djai.feedback), loaded once at start and
        #: changed only by `model learn` or `model rollback`.
        self._stable_model: dict | None = None
        #: Where tonight's layer starts reading this session's log: after
        #: whatever a mid-set `model learn` already folded into the stable one.
        self._night_from: int = 0
        self._model_checked: bool = False
        #: The suggestion rows last written to the log, so a UI polling the
        #: panel writes one line per change rather than one per poll.
        self._shown_ids: tuple[str, ...] = ()
        #: Stem moves the armed transition will make: one dict per move, for
        #: the log and the UI. See :meth:`_start_stem_moves`.
        self.stem_moves: list[dict] = []
        #: A stem move decided while the transition was still being planned,
        #: carried until the commands are queued.
        self._pending_stem_moves: list[dict] = []
        #: Stems are an improvement to reach for, never a dependency: with no
        #: separated stems, or with this off, transitions run on EQ alone.
        self.stems_enabled: bool = True
        #: Each deck's own audio before a stem move, so it can be put back --
        #: the original, not a sum of stems, which is ~19 dB off the master.
        self._stem_origin: dict[str, LoadedTrack] = {}
        #: When each no-cue warning was last reported, keyed by (reason, live
        #: track title). See :meth:`_report_no_cue`.
        self._no_cue_reported: dict[tuple[str, str | None], float] = {}
        #: Monotonic time a transition was first seen armed with nothing queued
        #: and nothing running, or None.
        self._stuck_arm_since: float | None = None
        #: Extra bars the user asked to hold the blend for.
        self.hold_extra_bars: int = 0

        #: Manual override. While frozen the autopilot cues and arms nothing;
        #: the current track plays and the operator decides what happens next.
        #: Deliberately does NOT stop the supervisor -- drift correction is not
        #: automation the DJ is overriding, it is the mix staying beat-matched --
        #: and does not stop recovery from silence, because the whole point of
        #: this program is that the room never hears nothing.
        self.frozen: bool = False
        #: "autonomous": the autopilot cues and blends on its own. "assisted"
        #: (co-pilot): it cues and proposes, and blends on the operator's `go`
        #: -- or at the last call, because the room never hears nothing.
        self.mode: str = "autonomous"
        #: The MC has the mic: the master is dipped, arming waits for the last
        #: call, and the closed loop does not read the dip as a fault.
        self.mc: bool = False
        self._mc_token = CancelToken("mc")
        #: The cued track a hold (co-pilot or MC) was last announced for, so
        #: the proposal and the last call are each said once.
        self._held_for: tuple[str, str] | None = None
        #: A track the operator has forced to be next, overriding the selector.
        self._forced_next: TrackAnalysis | None = None
        #: Operator's transition-style preference: a name from
        #: :data:`djai.transition.STYLES`, or "auto" to let the rules decide.
        #: The model may set this; it never computes timing or curves.
        self.transition_style: str = "auto"
        #: Where the set is: one of selector.SET_PHASES, or "auto" for no
        #: intensity target at all.
        self.set_phase: str = "auto"
        #: Snap manual loops, rolls and jumps to the beat grid.
        self.quantize: bool = True
        #: A manual performance control is in charge. Shown as HELD; `resume`
        #: (or RESUME in the UI) clears it along with the freeze.
        self.manual_held: bool = False
        #: Pending loop-in point per deck, in original track frames.
        self._loop_in: dict[str, float] = {}
        #: What the rules picked last time, and why. Shown in the UI so the
        #: operator can see the reasoning rather than guess at it.
        self.last_transition_choice: tuple[str, str] = ("", "")
        #: Energy direction the operator last asked for, in -1..1. Passed to
        #: the transition designer as context.
        self.energy_direction: float = 0.0
        #: An in-flight transition design: {"thread", "track_id", "raw",
        #: "reason"}. Filled during the pre-roll window by a worker thread and
        #: read without blocking when the transition is armed.
        self._design: dict | None = None
        #: The intent engine, when one is wired up. Only ever used off the
        #: scheduler and audio paths.
        self.intent_engine = None
        #: Per-session tally: where each transition's shape came from, and
        #: which style ran. A silent fallback looks exactly like a model with
        #: no imagination, so the counts are on screen rather than in a log
        #: nobody reads mid-set.
        self.design_counts: Counter = Counter()
        self.style_counts: Counter = Counter()
        #: Transition mode for generated shapes (SPEC §4): "invisible",
        #: "showy", or "auto" to decide per pair from energy and set phase.
        self.transition_mode: str = "auto"
        #: Shape signatures of the transitions armed so far, oldest first. The
        #: generator's diversity penalty looks back through it.
        self.shape_history: list[str] = []
        #: Seed for generation. With the track pair and the transition's index
        #: it fixes every generated transition, so a set replays exactly.
        self.set_seed: int = 0
        #: Persisted set state: the cue queue, history and plan. In memory
        #: only unless given a path (cmd_play gives it one). See djai.setstate.
        self.set_state: SetState = SetState()
        #: Track id -> why the next transition into it must be conservative
        #: (an echo out into its mix-in, no long beatmatched blend): a
        #: quarantined grid, or a tempo bridge the operator chose.
        self._conservative: dict[str, str] = {}
        #: Track id cued by "play now", waiting for the next tick to arm it.
        self._play_now: str | None = None
        #: The set plan (SPEC §6), or None: with none, selection is the journey
        #: planner alone, exactly as before Phase 3. `plan` or cmd_play makes one.
        self.set_plan: "planner.Plan | None" = None
        self.persona: str = "default"
        self.set_minutes: float = planner.SET_MINUTES

        self.fallback_enabled = (
            config.FALLBACK_ENABLED if fallback_enabled is None else fallback_enabled
        )
        self.watchdog: Watchdog | None = None
        #: Let the autopilot arm the fallback the first time a deck reaches
        #: PLAYING. Only an idle start needs this -- there is nothing decoded
        #: at launch to hand the safety net -- and it stays off by default so
        #: that constructing a Session never starts a watchdog thread on its
        #: own. `cmd_play` turns it on for the idle path.
        self.autoarm_fallback: bool = False

        self._autopilot: threading.Thread | None = None
        self._stop = threading.Event()

    # --- notices -------------------------------------------------------------

    def add_notice_listener(self, fn: Any) -> None:
        """Also hand notices to ``fn``. Used by the web UI.

        A fan-out rather than a second consumer of the queue: if the UI drained
        the same queue the REPL does, each line would go to whichever happened
        to ask first and the other would never see it.
        """
        self._notice_listeners.append(fn)

    def notify(self, message: str) -> None:
        """Queue a line for the REPL to print. Called from background threads."""
        self._notices.put(message)
        for fn in self._notice_listeners:
            try:
                fn(message)
            except Exception:
                pass  # a broken listener must not silence the REPL

    def add_preview_listener(self, fn: Any) -> None:
        """Call ``fn(kind, payload)`` as previews start and finish.

        ``kind`` is "started" (payload: the style being previewed) or
        "finished" (payload: the :class:`djai.revise.PreviewOutcome`). Called
        from the pre-roll worker thread.
        """
        self._preview_listeners.append(fn)

    def _preview_event(self, kind: str, payload: Any) -> None:
        for fn in self._preview_listeners:
            try:
                fn(kind, payload)
            except Exception:
                pass  # a broken listener must not stop a transition

    def drain_notices(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self._notices.get_nowait())
            except queue.Empty:
                return out

    def _on_dropped(self, cmd: Any) -> None:
        self.notify(f"[engine] command queue full, dropped {cmd.describe()}")

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        # What the logs already taught, before a note is played: weights for
        # the critic's measures and the selector's similarities.
        try:
            self.reload_feedback()
        except Exception as exc:  # noqa: BLE001 - a set starts without it
            log.warning("could not load feedback weights: %s", exc)
        if self.recorder is not None:
            self.recorder.start()
        self.engine.start()
        self.scheduler.start()
        self.supervisor.start()
        self._autopilot = threading.Thread(
            target=self._autopilot_loop, name="djai-autopilot", daemon=True
        )
        self._autopilot.start()
        self.session_log.write(
            "session_start",
            device=str(self.engine.device),
            blocksize=self.engine.blocksize,
            crate_size=len(self.crate),
            cue=self.engine.cue_mode,
            recording=str(self.recorder.path) if self.recorder else None,
        )

    def arm_fallback(self, track: LoadedTrack | None = None) -> bool:
        """Load the safety net and start watching. Call once, after start().

        Defaults to the opening track, which is already decoded and in memory --
        so the safety net needs no disk access at the moment it is needed, and
        the room hears something it has already heard rather than silence.

        The watchdog watches the scheduler and autopilot threads because those
        are what stop the music being replaced: with either dead, the current
        track plays out and then the room hears nothing.
        """
        if not self.fallback_enabled:
            return False
        if track is None:
            track = self._first_track
        if track is None:
            self.notify("[fallback] nothing decoded yet - safety net NOT armed")
            return False
        player = FallbackPlayer.from_track(track)
        # The player holds the audio array it needs, so the session's reference
        # to the opening track has done its job. Keeping it pinned a whole
        # decoded track for the length of the night.
        if track is self._first_track:
            self._first_track = None
        self.watchdog = Watchdog(
            self.engine,
            player,
            threads={
                "scheduler": self.scheduler.thread,
                "autopilot": self._autopilot,
            },
            on_takeover=self._on_fallback,
            on_release=self._on_fallback_released,
        )
        self.watchdog.start()
        return True

    def _on_fallback_released(self, detail: str) -> None:
        """The watchdog handed the output back to the engine. Watchdog thread.

        Worth a line of its own: while the fallback held the output the engine
        drained no commands and no deck advanced, so this is the moment the
        autopilot can cue again.
        """
        self.notify(f"[FALLBACK] released - {detail}")
        try:
            self.session_log.write(
                "fallback_released",
                trigger=detail,
                action="master output handed back to the engine; cueing resumes",
                track=(
                    self.engine.deck(self.live_deck).track.analysis.title
                    if self.engine.deck(self.live_deck).track is not None
                    else None
                ),
            )
        except Exception:
            pass

    def _on_fallback(self, reason: str) -> None:
        self.notify(f"[FALLBACK] {reason} - playing the safety track")
        try:
            self.session_log.write(
                "fallback_engaged",
                trigger=reason,
                action="master output handed to the fallback player",
                track=self.watchdog.player.title if self.watchdog else None,
            )
        except Exception:
            pass

    def attach_set_state(self, path: Path) -> str:
        """Persist the set to ``path``, resuming it if the last run was killed.

        A file marked ``ended`` belongs to a set that finished cleanly, and a
        fresh set starts over it. Anything else is a set that was killed
        mid-flight: its cue queue, what had played, its seed and its plan come
        back, so a restart carries on rather than starting the night again.
        """
        import random

        state = SetState.load(Path(path))
        if Path(path).exists() and not state.ended:
            by_id = {t.track_id: t for t in self.crate}
            for tid in state.history:
                if tid in by_id and tid not in self.played:
                    self.played.add(tid)
                    self.history.append(by_id[tid])
            self.set_state = state
            self.set_seed = state.set_seed
            self.persona = state.persona
            self.session_log.write(
                "set_resumed", trigger=str(path),
                action=f"{len(state.cue_queue)} cue(s), {len(state.history)} played",
            )
            return (f"Resumed the set: {len(state.cue_queue)} cue(s) queued, "
                    f"{len(state.history)} track(s) already played.")
        self.set_seed = random.randrange(1 << 31)
        self.set_state = SetState(
            path=Path(path), set_seed=self.set_seed,
            cue_queue=self.set_state.cue_queue,
            history=[t.track_id for t in self.history],
        )
        self.set_state.save()
        return "New set."

    def shutdown(self) -> None:
        if self.set_state.path is not None:
            self.set_state.ended = True
            self.set_state.save()
        self._stop.set()
        if self.watchdog is not None:
            self.watchdog.stop()
        if self._autopilot is not None:
            self._autopilot.join(timeout=2.0)
        self.scheduler.stop()
        try:
            self.session_log.write(
                "session_end",
                frames_played=self.engine.frames_played,
                underruns=self.engine.underruns,
                interventions=self.supervisor.interventions,
                recorded_s=(
                    round(self.recorder.seconds_written, 1) if self.recorder else None
                ),
            )
        except ValueError:
            pass  # log already closed
        self.supervisor.stop()
        self.engine.stop()  # also stops the recorder, flushing its ring

    # --- manual override -----------------------------------------------------

    def freeze(self, frozen: bool = True) -> str:
        """Stop or resume the autopilot. The current track keeps playing."""
        self.frozen = frozen
        if not frozen:
            self.manual_held = False
        elif self._transition_armed and not self.engine.transition_active:
            # Taking over means the automation makes no further move: a blend
            # armed 16 bars out would otherwise fire into the operator's hands.
            # The cued track stays cued (an operator's own cue is never
            # dropped), so `resume` re-arms the same track. A blend already
            # running is the fader's and `cancel`'s to take.
            cued = self._cued
            self.abort_armed_transition(trigger="operator took over")
            self._drop_unlanded_ride()
            self._cued = cued
        self.session_log.write(
            "automation_frozen" if frozen else "automation_resumed",
            trigger="manual override",
            action="autopilot will not cue or arm" if frozen else "autopilot resumed",
        )
        if frozen:
            return (
                "Automation frozen. The current track plays on; nothing will be "
                "cued or blended until you say `resume`."
            )
        return "Automation resumed."

    def force_next(self, needle: str) -> str:
        """Pin the next track, overriding the selector. Re-cues if idle."""
        track = _resolve_track(self.crate, needle)
        if track is None:
            return f"No track in the crate matches {needle!r}."
        reason = self._set_aside(track)
        if reason is not None:
            return f"Cannot play {track.title}: {reason}. Re-analyse it first."
        self._forced_next = track
        self.session_log.write(
            "forced_next",
            track=track.title,
            trigger=f"manual override: {needle!r}",
            action="next cue will use this track instead of the selector's pick",
        )
        # If a track is already sitting cued, replace it now rather than at the
        # next tick -- the operator asked for this one, not the one after.
        if self.has_cued_track() and not self.engine.transition_active:
            self._cued = None
            self.engine.deck(self.cued_deck()).set_gain(0.0)
            if self.cue_next(origin="manual"):
                return f"Next up: {track.title} (cued now)."
        return f"Next up: {track.title}."

    MODES: tuple[str, ...] = ("autonomous", "assisted")

    def set_mode(self, name: str) -> str:
        """Autonomous, or assisted (co-pilot): the AI proposes, you say `go`."""
        name = {"auto": "autonomous", "copilot": "assisted", "co-pilot": "assisted"}.get(name, name)
        if name not in self.MODES:
            return f"Mode is {self.mode}. Say `mode autonomous` or `mode assisted`."
        self.mode = name
        self._held_for = None
        self.session_log.write("mode_changed", trigger="operator", action=name)
        if name == "assisted":
            return ("Co-pilot: I cue and propose the next track; say `go` to blend. "
                    "With no answer by the last call I blend anyway.")
        return "Autonomous: I cue and blend on my own."

    #: The MC's dip: -12 dB, in over a beat, back out over a bar.
    MC_DUCK: float = 0.25
    MC_STEPS: int = 8

    def set_mc(self, on: bool) -> str:
        """Dip the master under the MC, or bring it back. On the next beat."""
        if on == self.mc:
            return "The MC already has the mic." if on else "No MC on the mic."
        self.mc = on
        self._held_for = None
        # A new dip supersedes one still ramping; start from where it got to.
        self.scheduler.cancel(self._mc_token)
        token = self._mc_token = CancelToken("mc")
        start, beat = glue.beat_grid(self.engine)
        span = beat if on else 4.0 * beat
        cur = float(self.engine._master_gain)
        to = self.MC_DUCK if on else 1.0
        for i in range(1, self.MC_STEPS + 1):
            self.scheduler.submit(SetMasterGain(
                gain=cur + (to - cur) * i / self.MC_STEPS,
                execute_at=int(start + span * (i - 1) / (self.MC_STEPS - 1)),
                origin="mc", token=token,
            ))
        self.session_log.write(
            "mc_on" if on else "mc_off", trigger="operator",
            action=(f"master to {20 * math.log10(to):+.0f} dB over "
                    f"{'a beat' if on else 'a bar'}; arming "
                    f"{'waits for the last call' if on else 'free'}"),
        )
        if on:
            return "Mic's yours: music dipped 12 dB, nothing blends until `mc off` or the last call."
        return "Music back up over a bar."

    def last_call_seconds(self) -> float:
        """The latest a held blend can wait: one phrase and the margin."""
        phrase_s = (self.autopilot_lead_seconds() - AUTOPILOT_MARGIN_SECONDS) / AUTOPILOT_LEAD_PHRASES
        return phrase_s + AUTOPILOT_MARGIN_SECONDS

    def _arm_held(self, live) -> bool:
        """Whether arming waits: co-pilot for `go`, or the MC. Until the last call."""
        why = ("the MC has the mic" if self.mc
               else "co-pilot: waiting for `go`" if self.mode == "assisted" else None)
        if why is None or self._cued is None:
            return False
        title = self._cued.analysis.title
        waiting = self._seconds_to_mix_out(live) > self.last_call_seconds()
        said = (title, "wait" if waiting else "last call")
        if self._held_for != said:
            self._held_for = said
            if waiting:
                self.session_log.write("arm_held", track=title, trigger=why,
                                       action="proposed; waiting")
                self.notify(f"[copilot] next up: {title}. Say `go` to blend, or cue "
                            "something else." if not self.mc else
                            f"[mc] holding the blend into {title} while the mic is live")
            else:
                self.session_log.write("arm_last_call", track=title, trigger=why,
                                       action="no answer by the last call; blending")
                self.notify(f"[autopilot] last call: blending into {title} now")
        return waiting

    def explain(self) -> str:
        """Why the set is doing what it is doing, read back from the log.

        Only fields as they were logged when each decision was taken -- never
        recomputed now, so the answer cannot be a story told after the fact.
        """
        wanted = ("track_cued", "transition_armed", "closed_loop", "arm_held",
                  "run_dry", "track_unplayable", "cue_missing")
        last: dict[str, dict] = {}
        try:
            lines = self.session_log.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return f"The session log cannot be read: {exc}"
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") in wanted:
                last[row["event"]] = row
        if not last:
            return "Nothing decided yet: the log has no cue, blend or correction."
        out = []
        for event in wanted:
            row = last.get(event)
            if row is None:
                continue
            when = str(row.get("ts", ""))[11:19]
            did = f" {row['decision_id']}" if row.get("decision_id") else ""
            head = f"[{when}{did}] {event}"
            if row.get("track"):
                head += f" {row['track']}"
            out.append(f"{head}: {row.get('action')} (because: {row.get('trigger')})")
            if event == "transition_armed":
                out.append(f"    rule: {row.get('style_rule')}; placement: "
                           f"{row.get('plan_reason')}; design: {row.get('design_source')}")
            snap = row.get("snapshot")
            if snap:
                out.append(f"    seen: master {snap.get('master_bpm')} BPM, "
                           f"peak {snap.get('master_peak')}, blend "
                           f"{'running' if snap.get('transition_active') else 'idle'}")
        return "\n".join(out)

    # --- song suggest and cue (Phase 3.2) -------------------------------------

    #: Cue modes. "after" waits for N more tracks first.
    CUE_MODES: tuple[str, ...] = ("next", "after", "now")

    @property
    def cue_queue(self) -> list[dict]:
        return self.set_state.cue_queue

    def suggest(self, n: int = 5) -> list["songs.Suggestion"]:
        """Top ``n`` next tracks from the playing one, each with its reasons.

        A list that differs from the last one shown is logged with every row's
        reason terms: which of them the operator then cues, and which they pass
        over, is what the preference learner reads (djai.feedback.choices).
        """
        current = self.analysis_for(self.live_deck)
        live_bpm = self._playing_bpm()
        rows = songs.suggest(
            self.crate, current, self._selectable(), n=n,
            plan_fit=self.plan_fit,
            energy_direction=self.energy_direction, set_phase=self._phase_arg(),
            history=self.history, cache_dir=self.cache_dir,
            weights=self.selector_weights, playing_bpm=live_bpm or None,
            travel=self.tempo_target is not None,
        )
        ids = tuple(r.track.track_id for r in rows)
        if ids and ids != self._shown_ids:
            self._shown_ids = ids
            self.session_log.write(
                "suggestions_shown", trigger="suggest", persona=self.persona,
                action=f"{len(rows)} shown",
                rows=[{"track_id": r.track.track_id, "rank": r.rank,
                       "score": round(r.score, 4), "terms": r.terms} for r in rows],
            )
        return rows

    def plan_fit(self, track: TrackAnalysis) -> float:
        """How well a track fits the set plan, 0..1 (0 with no plan)."""
        return planner.plan_fit(self.set_plan, track, self.crate)

    def replan(self, reason: str) -> "planner.Plan | None":
        """Re-make the set plan from where the set is now. Control thread.

        Soft by design: called after every cue, on every change to the cue
        queue, and when the set goes off the plan. Only its first slot is ever
        acted on. Logged with its critic score and its three horizons.
        """
        if self.set_plan is None and reason != "planned":
            return None
        persona = planner.PERSONAS.get(self.persona, planner.PERSONAS["default"])
        current = self.analysis_for(self.live_deck) if self.history else None
        elapsed = self.engine.frames_played / SAMPLE_RATE / 60.0
        plan = planner.plan_set(
            self.crate, current, list(self.history), list(self.cue_queue), persona,
            elapsed_min=elapsed, minutes=self.set_minutes,
            playing_bpm=self._playing_bpm() or None, excluded=self._unplayable,
            weights=self.selector_weights, cache_dir=self.cache_dir,
        )
        plan.reason = reason
        self.set_plan = plan
        self.set_state.plan = plan.as_dict()
        self.set_state.persona = self.persona
        self.set_state.save()
        c = plan.critique
        self.session_log.write(
            "set_plan", trigger=reason,
            action=(f"{len(plan.slots)} tracks over {plan.minutes:.0f} min, "
                    f"critic {c['score']:.1f}/10"),
            persona=self.persona, critic=c,
            near=[s.track.title for s in plan.near],
            mid=[s.track.title for s in plan.mid],
            full=[s.track.title for s in plan.slots],
            slots=[s.as_dict() for s in plan.slots],
        )
        self._ask_plan_note(plan)
        return plan

    def _ask_plan_note(self, plan: "planner.Plan") -> None:
        """An advisory LLM note on the plan, on its own thread. Never waited on.

        The rules-based critic's score is the one that is logged and used;
        the model only adds a sentence. With no model, nothing happens.
        """
        engine = self.intent_engine
        if engine is None or not getattr(engine, "available", False) \
                or not hasattr(engine, "critique_plan"):
            return
        summary = plan.summary()

        def work() -> None:
            note, why = engine.critique_plan(summary)
            self.session_log.write(
                "set_plan_note", trigger="llm narrative critic",
                action=(note or {}).get("note", "") if note else f"none ({why})",
                llm_score=(note or {}).get("score"),
            )

        threading.Thread(target=work, name="djai-plan-note", daemon=True).start()

    def set_persona(self, name: str) -> str:
        if name not in planner.PERSONAS:
            return f"Personas: {', '.join(planner.PERSONAS)}."
        self.persona = name
        self.set_state.persona = name
        self.set_state.save()
        # Preferences are learned per persona: the weights change with it.
        self.reload_feedback()
        self.replan(f"persona {name}")
        return f"Persona: {name}."

    def _cap_path(self, track: TrackAnalysis, from_bpm: float | None = None):
        """How the playing tempo meets ``track`` inside the stretch cap."""
        bpm = from_bpm if from_bpm else self._playing_bpm()
        if not bpm:
            return selector_mod.TempoPath("direct")
        return selector_mod.tempo_path(bpm, track.bpm, travel=True)

    def cue_song(self, query: str, mode: str = "next", after: int = 0) -> str:
        """Find a track by name and queue it. Never guesses, never drops.

        Deterministic: the lookup is djai.songs.lookup. Several close matches
        are returned to the operator and nothing is queued; no match says so.
        """
        mode = mode if mode in self.CUE_MODES else "next"
        found = songs.lookup(self.crate, query)
        if found.status == "none":
            self.session_log.write("cue_lookup", trigger=query, action="no match")
            return f"No track matches {query!r}. Nothing was queued."
        if found.status == "ambiguous":
            self.session_log.write(
                "cue_lookup", trigger=query, action="ambiguous",
                candidates=[t.title for t in found.tracks],
            )
            lines = "\n".join(f"  - {t.title}" for t in found.tracks)
            return (f"{query!r} matches more than one track:\n{lines}\n"
                    "Say `cue <title>` with the one you mean.")
        return self.cue_track(found.track, mode, after, query=query)

    def cue_track(self, track: TrackAnalysis, mode: str = "next", after: int = 0,
                  query: str = "") -> str:
        """Queue a known track. The one entry point for every surface."""
        reason = self._set_aside(track)
        if reason is not None:
            return f"Cannot cue {track.title}: {reason}. Re-analyse it first."
        warning = None
        if getattr(track, "quarantined", False):
            warning = (f"low grid confidence ({track.grid_confidence:.2f}): it will "
                       "come in on an echo out, not a long blend")
        entry = self.set_state.add_cue({
            "track_id": track.track_id, "title": track.title, "mode": mode,
            "after": max(0, int(after)) if mode == "after" else 0,
            "status": "queued", "bridge": None, "warning": warning,
            "added_at": time.time(),
        }, front=(mode == "now"))
        self.session_log.write(
            "cue_added", track=track.title, trigger=f"operator: {query or track.title}",
            action=f"queued ({mode}{' ' + str(entry['after']) if mode == 'after' else ''})",
            cue_id=entry["id"], warning=warning, track_id=track.track_id,
            persona=self.persona,
            from_suggestion=(self._shown_ids.index(track.track_id) + 1
                             if track.track_id in self._shown_ids else None),
        )
        reply = f"Queued {track.title} ({self._mode_text(entry)})."
        bridge = self._check_bridge(entry)
        if bridge:
            reply += " " + bridge
        if warning:
            reply += f" Warning: {warning}."
        self.replan("cue added")
        if mode == "now" and entry["status"] == "queued":
            reply += " " + self._play_now_entry(entry)
        return reply

    @staticmethod
    def _mode_text(entry: dict) -> str:
        if entry["mode"] == "after":
            return f"after {entry['after']} more track(s)"
        return {"next": "plays next", "now": "plays now"}[entry["mode"]]

    def _check_bridge(self, entry: dict) -> str:
        """Mark a cue the stretch cap cannot reach, and ask. '' if it can."""
        track = self._track_by_id(entry["track_id"])
        if track is None or entry.get("bridge"):
            return ""
        path = self._cap_path(track)
        if path.blendable:
            entry["status"] = "queued"
            self.set_state.save()
            return ""
        entry["status"] = "needs_bridge"
        self.set_state.save()
        msg = (f"{track.title} ({track.bpm:.1f} BPM) is beyond the "
               f"+-{config.MAX_STRETCH_RATIO:.0%} stretch cap from {self._playing_bpm():.1f} "
               "BPM, even half or double time. Choose: `bridge tempo` (echo out, it "
               "comes in at its own tempo) or `bridge track` (a bridge track first). "
               "It stays queued until you choose.")
        self.session_log.write("cue_needs_bridge", track=track.title,
                               trigger="stretch cap", action="asked the operator",
                               cue_id=entry["id"])
        self.notify(f"[cue] {msg}")
        return msg

    def _track_by_id(self, track_id: str) -> TrackAnalysis | None:
        return next((t for t in self.crate if t.track_id == track_id), None)

    def resolve_bridge(self, choice: str) -> str:
        """Answer the bridge question for the first cue waiting on one."""
        entry = next((e for e in self.cue_queue if e["status"] == "needs_bridge"), None)
        if entry is None:
            return "No cue is waiting on a bridge."
        track = self._track_by_id(entry["track_id"])
        if choice == "tempo":
            entry.update(status="queued", bridge="tempo")
            self.set_state.save()
            self.session_log.write("cue_bridge", track=entry["title"], trigger="operator",
                                   action="tempo bridge (echo out at its own tempo)")
            return f"{entry['title']} will come in on a tempo bridge."
        if choice == "track":
            bridge = self._bridge_track(track) if track else None
            if bridge is None:
                return ("No track in the crate bridges that gap inside the cap. "
                        "Say `bridge tempo` instead.")
            entry.update(status="queued", bridge="track")
            i = self.cue_queue.index(entry)
            self.cue_queue.insert(i, {
                "id": self.set_state.next_cue_id, "track_id": bridge.track_id,
                "title": bridge.title, "mode": "next", "after": 0, "status": "queued",
                "bridge": None, "warning": None, "added_at": time.time(),
                "bridge_for": entry["id"],
            })
            self.set_state.next_cue_id += 1
            self.set_state.save()
            self.session_log.write("cue_bridge", track=entry["title"], trigger="operator",
                                   action=f"bridge track {bridge.title} inserted before it")
            self.replan("bridge track inserted")
            return f"{bridge.title} ({bridge.bpm:.1f} BPM) goes first, then {entry['title']}."
        return "Say `bridge tempo` or `bridge track`."

    def _bridge_track(self, target: TrackAnalysis) -> TrackAnalysis | None:
        """The best-ranked track reachable now from which ``target`` is reachable."""
        current = self.analysis_for(self.live_deck)
        queued = {e["track_id"] for e in self.cue_queue}
        for c in rank_candidates(
            self.crate, current, self._selectable() | queued, self.energy_direction,
            history=self.history, cache_dir=self.cache_dir,
            weights=self.selector_weights, playing_bpm=self._playing_bpm() or None,
            travel=True,
        ):
            played_at = c.track.bpm * (c.tempo_path.rate_b or 1.0)
            if self._cap_path(target, from_bpm=played_at).blendable:
                return c.track
        return None

    def queue_text(self) -> str:
        if not self.cue_queue:
            return "The cue queue is empty."
        rows = []
        for i, e in enumerate(self.cue_queue, start=1):
            flag = " [NEEDS BRIDGE]" if e["status"] == "needs_bridge" else ""
            warn = f" [warning: {e['warning']}]" if e.get("warning") else ""
            rows.append(f"{i}. {e['title']} - {self._mode_text(e)}{flag}{warn}")
        return "\n".join(rows)

    def queue_remove(self, index: int) -> str:
        if not 1 <= index <= len(self.cue_queue):
            return f"No cue {index}; the queue has {len(self.cue_queue)}."
        entry = self.cue_queue.pop(index - 1)
        self.set_state.save()
        self.session_log.write("cue_removed", track=entry["title"], trigger="operator",
                               action=f"removed from position {index}")
        self.replan("cue removed")
        return f"Removed {entry['title']}."

    def queue_move(self, index: int, to: int) -> str:
        n = len(self.cue_queue)
        if not (1 <= index <= n and 1 <= to <= n):
            return f"Positions run 1 to {n}."
        entry = self.cue_queue.pop(index - 1)
        self.cue_queue.insert(to - 1, entry)
        self.set_state.save()
        self.session_log.write("cue_moved", track=entry["title"], trigger="operator",
                               action=f"moved {index} -> {to}")
        self.replan("cue moved")
        return f"Moved {entry['title']} to {to}."

    def _due_cue(self) -> dict | None:
        """The cue to play next, or None to let the selector choose.

        Queue order is never changed: an "after N" entry still waiting lets
        the entries behind it through, but one waiting on the operator's
        bridge choice holds everything behind it, and the selector fills the
        gap meanwhile rather than letting the room go quiet.
        """
        for entry in self.cue_queue:
            if entry["status"] == "needs_bridge":
                return None
            if entry["mode"] == "after" and entry["after"] > 0:
                continue
            return entry
        return None

    def _count_cue(self, cued_id: str) -> None:
        """A track was cued: every "after N" cue waits one track less."""
        changed = False
        for e in self.cue_queue:
            if e["mode"] == "after" and e["after"] > 0 and e["track_id"] != cued_id:
                e["after"] -= 1
                changed = True
        self.set_state.history.append(cued_id)
        if changed or self.set_state.path is not None:
            self.set_state.save()

    def _play_now_entry(self, entry: dict) -> str:
        """Bring a "now" cue in at the next 4-bar point, through the Short band."""
        self._complete_handover()
        if self.engine.transition_active:
            entry["mode"] = "next"
            self.set_state.save()
            return "A transition is running; it plays straight after."
        self.abort_armed_transition()
        if self.has_cued_track() and (self._cued is None or
                                      self._cued.analysis.track_id != entry["track_id"]):
            self._cued = None
            self.engine.deck(self.cued_deck()).set_gain(0.0)
        if not self.has_cued_track() and not self.cue_next(origin="play now"):
            return "It could not be cued; it stays at the head of the queue."
        self._play_now = entry["track_id"]
        return "It comes in at the next 4-bar line."

    #: Play now lands on the next line of this many bars.
    PLAY_NOW_BARS: int = 4

    def _arm_play_now(self, live) -> None:
        track_id, self._play_now = self._play_now, None
        if self._cued is None or self._cued.analysis.track_id != track_id:
            return
        # origin "user": the supervisor's 40% placement floor exists to catch
        # automatic placement regressing, and exempts a human asking to go now.
        execute_at = self.arm_transition(origin="user", at_bars=self.PLAY_NOW_BARS)
        if execute_at is None:
            self.notify("[cue] play now could not be armed; it plays next instead")
            return
        # A filter swell on the outgoing deck over the beat before the line
        # masks the change of direction (SPEC §1 glue). It lands back on the
        # detent before the transition's own filter columns take over.
        beat = SAMPLE_RATE * 60.0 / max(self.engine.master_bpm or 120.0, 1.0)
        start = int(execute_at - 4 * beat)
        if start > self.engine.frames_played:
            for cmd in glue.filter_swell(self.live_deck, start, beat, beats=4.0,
                                         peak=0.45, origin="play now"):
                self.scheduler.submit(cmd)
        bars = self.bars_until(execute_at)
        self.session_log.write("play_now", track=self._cued.title, trigger="operator",
                               action=f"armed {bars:.1f} bars out, filter swell glue")
        self.notify(f"[cue] {self._cued.title} in {bars:.0f} bars")

    def force_transition(self) -> str:
        """Blend now, on the next phrase boundary. The 'go' button."""
        self._complete_handover()
        if self.engine.transition_active:
            return "A transition is already running."
        if not self.has_cued_track():
            if not self.cue_next(origin="manual"):
                return "Nothing to blend into: no compatible track could be cued."
            return (
                "Cued a track; it lands on the deck next block. Say `go` again to "
                "start the blend."
            )
        execute_at = self.arm_transition(at_next_phrase=True)
        if execute_at is None:
            return "Could not arm a transition; the supervisor rejected it."
        bars = self.bars_until(execute_at)
        title = self._cued.title if self._cued else "?"
        self.session_log.write(
            "forced_transition",
            track=title,
            trigger="manual override",
            action=f"armed at the next phrase, {bars:.0f} bars away",
        )
        return f"Blending into {title} in {bars:.0f} bars."

    #: "Harder" and "calmer" on the live deck, at full strength: high-band and
    #: mid-band gain for harder (+3.5 dB / +1.2 dB), high-band gain and a
    #: low-pass knob position for calmer. Absolute targets from neutral, so a
    #: repeated request does not stack.
    ENERGY_HIGH_LIFT: float = 0.5
    ENERGY_MID_LIFT: float = 0.15
    ENERGY_HIGH_CUT: float = 0.3
    ENERGY_LOWPASS: float = 0.35
    #: The move starts on the next beat and ramps in over this many bars, in
    #: this many steps: the Short band, audible long before the next track.
    ENERGY_RAMP_BARS: float = 2.0
    ENERGY_RAMP_STEPS: int = 4

    def energy_correction(
        self, direction: float, lead_bars: float = 0.0, origin: str = "operator",
    ) -> str | None:
        """Answer "harder" / "calmer" on the deck playing now, within bars.

        Re-cueing the next track is the long answer, and it can be a phrase
        or more away. This is the immediate one: a tempo-relative EQ and
        filter trajectory on the live deck. It lasts until the hand-over,
        which returns both decks to neutral. Inside a running blend the
        envelope owns these controls, and the incoming track *is* the
        change, so nothing is scheduled.
        """
        d = max(-1.0, min(1.0, float(direction)))
        # The closed loop may ask for neutral (0) to take a correction back;
        # an operator's "a little harder" of nothing is not a request.
        if (abs(d) < 0.05 and origin == "operator") or self.engine.transition_active:
            return None
        name = self.live_deck
        deck = self.engine.deck(name)
        if deck.track is None:
            return None
        if origin == "operator" and self.room is not None:
            self.room.operator(d)
        self.scheduler.cancel(self._energy_token)
        self._energy_token = token = CancelToken("energy")
        if d > 0:
            high, mid, pos = 1.0 + self.ENERGY_HIGH_LIFT * d, 1.0 + self.ENERGY_MID_LIFT * d, 0.0
        else:
            high, mid, pos = 1.0 - self.ENERGY_HIGH_CUT * -d, 1.0, -self.ENERGY_LOWPASS * -d
        h0, m0, p0 = deck.eq_high.target, deck.eq_mid.target, deck.filter_pos.target
        l0 = deck.eq_low.target
        low = None
        if origin != "operator":
            # The closed loop's correction is the same shape plus a level
            # trim across all three bands: a pushed fader wants trimming, not
            # a low-pass that dulls the record. It owns the low band too, so
            # putting drifted controls back covers a boosted or killed bass.
            trim = 10.0 ** (self.ROOM_TRIM_DB * d / 20.0)
            # Inside the supervisor's 0..2 band range, or the step is refused.
            low, mid, high = (round(min(2.0, v), 4) for v in (trim, mid * trim, high * trim))
        start, beat = glue.beat_grid(self.engine)
        if lead_bars > 0:
            # The Short band: on the next 4-bar line at least `lead_bars` out,
            # so the move lands on the music's own boundaries.
            ta = deck.track.analysis
            bar = phrase.bar_at_frame(ta, deck.position)
            line = (math.floor((bar + lead_bars) / 4.0) + 1) * 4
            start = phrase.deck_frame_to_engine_frame(
                deck, self.engine.frames_played, phrase.frame_at_bar(ta, line))
        steps = max(1, self.ENERGY_RAMP_STEPS)
        span = self.ENERGY_RAMP_BARS * 4 * beat
        if origin != "operator" and any(
            isinstance(c, StartTransition) and c.execute_at <= start + span + 4 * beat
            for c in self.scheduler.pending()
        ):
            # The loop's ramp would still be moving when an armed blend takes
            # the decks over. The blend is the bigger change; leave it be.
            return None
        queued = 0
        tag = "energy" if origin == "operator" else origin
        for i in range(steps):
            f = (i + 1) / steps
            at = int(round(start + span * i / steps))
            for cmd in (
                SetEQ(deck=name, mid=round(m0 + (mid - m0) * f, 4),
                      high=round(h0 + (high - h0) * f, 4),
                      low=None if low is None else round(l0 + (low - l0) * f, 4),
                      execute_at=at, origin=tag, token=token),
                SetFilter(deck=name, position=round(p0 + (pos - p0) * f, 4),
                          execute_at=at, origin=tag, token=token),
            ):
                if self.supervisor.validate(cmd) is None:
                    self.scheduler.submit(cmd)
                    queued += 1
        if self.room is not None:
            self.room.controls = (l0 if low is None else low, mid, high, pos)
        way = "harder" if d > 0 else ("calmer" if d < 0 else "neutral")
        self.session_log.write(
            "energy_correction",
            deck=name,
            trigger=f"{origin}: {way} ({d:+.2f})",
            action=(
                f"high x{high:.2f}, mid x{mid:.2f}, filter {pos:+.2f} over "
                f"{self.ENERGY_RAMP_BARS:g} bars from "
                + (f"the next 4-bar line {lead_bars:g}+ bars out" if lead_bars > 0
                   else "the next beat")
                + f"; {queued} command(s)"
            ),
        )
        return f"Pushing deck {name} {way} now."

    #: The closed loop's level trim at full correction, in dB either way.
    ROOM_TRIM_DB: float = 3.0
    #: A room_reading line every this many bars (about 15 s at 128 BPM).
    ROOM_LOG_BARS: int = 8
    #: The closed loop's corrections start on a 4-bar line at least this many
    #: bars out: past the Immediate band, inside the Short one.
    ROOM_LEAD_BARS: float = 2.0

    def _room_tick(self) -> None:
        """Measure the bars the room heard since the last tick, and answer a
        drop or a spike the playing track does not explain. Autopilot thread.

        Paused through a blend (the blend *is* the change), while the operator
        holds manual control, and while frozen: the loop never fights a
        person. Each correction goes through :meth:`energy_correction` in the
        Short band, and is a logged decision with its snapshot.
        """
        monitor = self.room
        if monitor is None:
            return
        paused = (self.engine.transition_active or self.manual_held or self.frozen
                  or self.mc)
        for reading, verdict in monitor.poll(self.live_deck, paused):
            self._room_bars += 1
            if self._room_bars >= self.ROOM_LOG_BARS:
                self._room_bars = 0
                self.session_log.write(
                    "room_reading", trigger="closed loop", action="measured",
                    bar=reading.bar, **monitor.state(),
                )
            if verdict is not None and verdict.direction is not None and monitor.operator_owns:
                if not self._room_deferred:
                    self._room_deferred = True
                    self.session_log.write(
                        "closed_loop_deferred", trigger=verdict.reason,
                        action="the operator has the EQ and filter; watching only",
                        bar=verdict.bar, deviation_db=round(verdict.deviation_db, 2),
                    )
                continue
            if verdict is None or verdict.direction is None:
                saturated = verdict is not None and verdict.status == "saturated"
                if saturated and not self._room_saturated:
                    self.session_log.write(
                        "closed_loop_saturated", trigger="closed loop",
                        action=verdict.reason, bar=verdict.bar,
                        deviation_db=round(verdict.deviation_db, 2),
                    )
                if verdict is not None:
                    self._room_saturated = saturated
                continue
            self._room_saturated = False
            reply = self.energy_correction(
                verdict.direction, lead_bars=self.ROOM_LEAD_BARS, origin="closed loop")
            self.session_log.decision(
                "closed_loop", self.engine.features.latest(),
                action=(f"correction level {verdict.direction:+.2f}"
                        + ("" if reply else " (nothing scheduled)")),
                trigger=verdict.reason, status=verdict.status, bar=verdict.bar,
                deviation_db=round(verdict.deviation_db, 2),
                direction=verdict.direction, heuristic=True,
                proxies=monitor.master.summary(),
            )

    def operator_touched(self, deck_name: str) -> None:
        """A hand on a deck's EQ or filter. Any control thread.

        On the live deck, those knobs are the operator's until the next
        hand-over: the closed loop keeps measuring and never writes them.
        """
        if self.room is not None and deck_name == self.live_deck:
            self.room.operator_touched()

    def cancel_transition(self, fade_beats: float = 1.0) -> str:
        """Back out of the blend in flight, or drop one armed but not started.

        A running blend is reverted in the engine over ``fade_beats``: the
        incoming deck fades out and stops and the outgoing one returns to
        unity. Anything still queued for the incoming deck (a stem swap, a
        loop) goes too, or it would fire into a deck that is no longer coming
        in. Clearing ``_transition_armed`` is what stops the autopilot reading
        the stopped deck as a finished hand-over.
        """
        if not self.engine.transition_active:
            dropped = self.abort_armed_transition()
            return (
                f"Dropped the armed transition ({dropped} command(s))."
                if dropped else "No transition to cancel."
            )
        incoming = self.cued_deck()
        dropped = self.scheduler.cancel_matching(
            lambda c: getattr(c, "deck", None) == incoming
            or isinstance(c, (StartTransition, SwapStems))
        )
        fade = int(round(
            fade_beats * SAMPLE_RATE * 60.0 / (self.engine.master_bpm or 120.0)
        ))
        self.engine.submit(CancelTransition(fade_frames=fade, origin="operator"))
        self._transition_armed = False
        self._cued = None
        self.hold_extra_bars = 0
        if self._cue_decision is not None:
            self.session_log.outcome(
                self._cue_decision,
                self.engine.features.latest(),
                trigger="transition cancelled",
                action=f"deck {incoming} faded out over {fade_beats:g} beat(s)",
            )
            self._cue_decision = None
        self.session_log.write(
            "transition_cancelled",
            trigger="operator cancel",
            action=(
                f"deck {incoming} fades out over {fade_beats:g} beat(s); "
                f"{len(dropped)} queued command(s) withdrawn"
            ),
        )
        return f"Cancelled: back to deck {self.live_deck}, deck {incoming} fading out."

    def set_cue(self, deck_name: str | None) -> str:
        """Route a deck to the pre-listen output."""
        if self.engine.cue_mode == "none":
            return (
                "No cue output configured. Start with --cue-device <n> or "
                "--cue-channels 3,4."
            )
        if deck_name is not None and deck_name not in ("a", "b"):
            return f"Unknown deck {deck_name!r}."
        self.engine.cue_deck = deck_name
        if deck_name is None:
            return "Cue off."
        title = self.engine.deck_state(deck_name).title or "nothing loaded"
        return f"Cue: deck {deck_name} ({title}) on {self.engine.cue_mode}."

    def set_master_tempo_lock(self, on: bool) -> str:
        """Hold the beat clock still, or let it follow each track's own tempo.

        Off by default. On, the set runs at a fixed tempo and every track is
        matched back to it -- which is what the tempo glide exists to end, so
        this is an operator's deliberate choice and never a default.
        """
        self.master_tempo_locked = bool(on)
        if not on:
            return (
                f"Master BPM unlocked: the clock will glide to each track's "
                f"own tempo over {config.MASTER_GLIDE_BARS} bars."
            )
        # Locking mid-glide would leave the steps in flight to finish the move
        # the operator just asked to stop.
        self._drop_unlanded_ride()
        return (
            f"Master BPM locked at {self.engine.master_bpm:.1f}. Every track "
            f"will be matched to it, within +-{config.MAX_STRETCH_RATIO:.0%}."
        )

    def set_key_lock(self, deck_name: str, on: bool) -> str:
        """Key lock for one deck: original pitch at any tempo, or resampling."""
        if deck_name not in ("a", "b"):
            return f"Unknown deck {deck_name!r}."
        outcome = self.engine.set_key_lock(deck_name, on)
        self.session_log.write(
            "key_lock",
            trigger="manual override",
            action=f"deck {deck_name} {'on' if on else 'off'}: {outcome}",
        )
        label = f"Deck {deck_name.upper()} key lock"
        return {
            "off": f"{label} off: resampling, pitch follows tempo.",
            "on": f"{label} on.",
            "stretching": f"{label} on: stretching now, original pitch in a few seconds.",
            "out of range": f"{label} on, but this tempo change is past the stretch limit, so it resamples.",
            "unavailable": f"{label} on, but time-stretch is disabled (TIME_STRETCH_ENABLED=0).",
        }.get(outcome, f"{label}: {outcome}.")

    def set_filter(
        self, deck_name: str, position: float, resonance: float | None = None
    ) -> str:
        """Turn one deck's filter knob: -1 low-pass, 0 open, 1 high-pass."""
        from djai.commands import SetFilter
        from djai.deck import FILTER_DETENT, filter_cutoff_hz

        if deck_name not in ("a", "b"):
            return f"Unknown deck {deck_name!r}."
        # Written so NaN fails too.
        if not -1.0 <= position <= 1.0:
            return "Refused: the filter runs from -1 (low-pass) through 0 (off) to 1 (high-pass)."
        if resonance is not None and not 0.0 <= resonance <= 1.0:
            return "Refused: filter resonance runs from 0 to 1."
        self.scheduler.submit(
            SetFilter(
                deck=deck_name, position=position, resonance=resonance,
                origin="manual:filter",
            )
        )
        detail = f"position {position:+.2f}"
        if resonance is not None:
            detail += f", resonance {resonance:.2f}"
        self.session_log.write(
            "filter", trigger="manual override", action=f"deck {deck_name} {detail}"
        )
        self.operator_touched(deck_name)
        if abs(position) <= FILTER_DETENT:
            return f"Deck {deck_name.upper()} filter off."
        kind = "low-pass" if position < 0 else "high-pass"
        return (
            f"Deck {deck_name.upper()} filter: {kind} at "
            f"{filter_cutoff_hz(position):.0f} Hz."
        )

    # --- performance controls -------------------------------------------------
    #
    # Loops, rolls, beat jumps, the pitch fader, sync and quantize. The same
    # methods back the REPL and the web UI. Every one of them:
    #   * refuses, and logs, on a deck that is part of a running transition --
    #     the crossfade is driving that deck against a fixed length;
    #   * otherwise takes manual control: automation is held (HELD / RESUME);
    #   * resolves every position from the beat grid here, so the audio thread
    #     only assigns, and with quantize on waits for the next beat line.
    # Loops are slip loops: leaving one resumes where straight playback would
    # be, which keeps phase and bar alignment to the master clock for any
    # length.

    #: Loop-roll sizes, in bars: 1/8 to 4.
    ROLL_BARS: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
    #: Beat-jump sizes, in bars, either direction.
    JUMP_BARS: tuple[int, ...] = (1, 4, 8, 16)
    #: Pitch fader range, percent either side of native tempo.
    MAX_PITCH_PERCENT: float = 8.0
    #: Loop lengths a manual loop may take, in beats.
    MIN_LOOP_BEATS: float = 0.5
    MAX_LOOP_BEATS: float = 64.0

    def _perform(self, deck_name: str, action: str):
        """Common gate. ``(deck, None)``, or ``(None, refusal)``."""
        if deck_name not in ("a", "b"):
            return None, f"Refused: unknown deck {deck_name!r}."
        deck = self.engine.deck(deck_name)
        if deck.track is None:
            return None, f"Refused: deck {deck_name.upper()} is empty."
        if self.engine.deck_in_transition(deck_name):
            self.session_log.write(
                "manual_refused", deck=deck_name, trigger=action,
                action="refused: the deck is mid-transition",
            )
            self.notify(f"[{action}] refused - deck {deck_name.upper()} is mid-transition")
            return None, f"Refused: deck {deck_name.upper()} is mid-transition."
        return deck, None

    def _take_manual(self, deck_name: str, action: str, detail: str) -> None:
        """Hold the automation and log the gesture."""
        if not self.frozen:
            self.freeze(True)
        self.manual_held = True
        self.operator_touched(deck_name)
        self.session_log.write(
            "manual_control", deck=deck_name, trigger=action, action=detail,
            quantized=self.quantize,
        )

    def _submit_manual(self, cmd) -> str | None:
        """Validate and schedule. A refusal message, or None."""
        rejection = self.supervisor.validate(cmd)
        if rejection is not None:
            return f"Refused: {rejection.reason}."
        self.scheduler.submit(cmd)
        return None

    @staticmethod
    def _frames_per_beat(deck) -> float:
        return deck.track.analysis.beat_period * SAMPLE_RATE

    def _next_beat(self, deck) -> tuple[float, int]:
        """``(deck frame, engine frame)`` of the next beat line.

        The playhead and "now" when quantize is off or the deck is stopped.
        """
        if not self.quantize or not deck.playing:
            return float(deck.position), IMMEDIATE
        analysis = deck.track.analysis
        beat = math.floor(phrase.beat_at_frame(analysis, deck.position)) + 1
        frame = phrase.frame_at_beat(analysis, beat)
        return frame, phrase.deck_frame_to_engine_frame(
            deck, self.engine.frames_played, frame
        )

    def _beat_line_at_or_before(self, deck, frame: float) -> float:
        analysis = deck.track.analysis
        return phrase.frame_at_beat(
            analysis, math.floor(phrase.beat_at_frame(analysis, frame) + 1e-9)
        )

    def set_quantize(self, on: bool) -> str:
        """Quantize on or off. Not a deck gesture, so it holds nothing."""
        self.quantize = bool(on)
        self.session_log.write(
            "quantize", trigger=f"quantize {'on' if on else 'off'}",
            action="manual gestures snap to the beat grid" if on else "manual gestures act immediately",
        )
        return f"Quantize {'on' if on else 'off'}."

    def loop_in(self, deck_name: str) -> str:
        deck, refused = self._perform(deck_name, "loop in")
        if refused:
            return refused
        position = float(deck.position)
        if self.quantize:
            analysis = deck.track.analysis
            position = phrase.frame_at_beat(
                analysis, round(phrase.beat_at_frame(analysis, position))
            )
        self._loop_in[deck_name] = position
        self._take_manual(deck_name, "loop in", f"loop in at {position / SAMPLE_RATE:.3f}s")
        return f"Deck {deck_name.upper()} loop in at {position / SAMPLE_RATE:.2f}s."

    def loop_out(self, deck_name: str) -> str:
        deck, refused = self._perform(deck_name, "loop out")
        if refused:
            return refused
        start = self._loop_in.get(deck_name)
        if start is None:
            return f"Refused: set loop in on deck {deck_name.upper()} first."
        fpb = self._frames_per_beat(deck)
        beats = (float(deck.position) - start) / fpb
        if self.quantize:
            beats = round(beats)
        if not self.MIN_LOOP_BEATS <= beats <= self.MAX_LOOP_BEATS:
            return (
                f"Refused: a {beats:g}-beat loop is outside "
                f"{self.MIN_LOOP_BEATS:g}-{self.MAX_LOOP_BEATS:g} beats."
            )
        refused = self._submit_manual(SetLoop(
            deck=deck_name, start_frame=start, length_frames=beats * fpb,
            origin="manual:loop",
        ))
        if refused:
            return refused
        self._loop_in.pop(deck_name, None)
        self._take_manual(deck_name, "loop out", f"{beats:g}-beat loop")
        return f"Deck {deck_name.upper()} looping {beats:g} beats."

    def auto_loop(self, deck_name: str, beats: float) -> str:
        deck, refused = self._perform(deck_name, "loop")
        if refused:
            return refused
        if not self.MIN_LOOP_BEATS <= beats <= self.MAX_LOOP_BEATS:
            return (
                f"Refused: a {beats:g}-beat loop is outside "
                f"{self.MIN_LOOP_BEATS:g}-{self.MAX_LOOP_BEATS:g} beats."
            )
        start, execute_at = self._next_beat(deck)
        if execute_at == IMMEDIATE and self.quantize:
            start = self._beat_line_at_or_before(deck, start)
        refused = self._submit_manual(SetLoop(
            deck=deck_name, start_frame=start,
            length_frames=beats * self._frames_per_beat(deck),
            execute_at=execute_at, origin="manual:loop",
        ))
        if refused:
            return refused
        self._take_manual(deck_name, "loop", f"{beats:g}-beat loop")
        when = "on the next beat" if execute_at != IMMEDIATE else "now"
        return f"Deck {deck_name.upper()} loops {beats:g} beats {when}."

    def loop_resize(self, deck_name: str, factor: float) -> str:
        deck, refused = self._perform(deck_name, "loop resize")
        if refused:
            return refused
        if not deck.loop_active:
            return f"Refused: deck {deck_name.upper()} is not looping."
        start, length = deck.loop_region
        fpb = self._frames_per_beat(deck)
        beats = length * factor / fpb
        if not self.MIN_LOOP_BEATS - 1e-9 <= beats <= self.MAX_LOOP_BEATS + 1e-9:
            return f"Refused: a {beats:g}-beat loop is outside the loop range."
        # The start stays put: halving moves the end, as on a CDJ, and the
        # slip position keeps running throughout.
        refused = self._submit_manual(SetLoop(
            deck=deck_name, start_frame=start, length_frames=length * factor,
            origin="manual:loop",
        ))
        if refused:
            return refused
        word = "halved" if factor < 1 else "doubled"
        self._take_manual(deck_name, f"loop {word}", f"{beats:g} beats")
        return f"Deck {deck_name.upper()} loop {word}: {beats:g} beats."

    def loop_exit(self, deck_name: str) -> str:
        deck, refused = self._perform(deck_name, "loop exit")
        if refused:
            return refused
        if not deck.loop_active:
            return f"Deck {deck_name.upper()} is not looping."
        _frame, execute_at = self._next_beat(deck)
        refused = self._submit_manual(ExitLoop(
            deck=deck_name, execute_at=execute_at, origin="manual:loop",
        ))
        if refused:
            return refused
        self._take_manual(deck_name, "loop exit", "slip exit, phase kept")
        return f"Deck {deck_name.upper()} leaves its loop, in phase."

    def roll(self, deck_name: str, bars: float) -> str:
        deck, refused = self._perform(deck_name, "roll")
        if refused:
            return refused
        if not any(abs(bars - b) < 1e-9 for b in self.ROLL_BARS):
            return "Refused: a roll is 1/8, 1/4, 1/2, 1, 2 or 4 bars."
        # From the beat line just passed, so the roll repeats what was playing.
        start = (
            self._beat_line_at_or_before(deck, float(deck.position))
            if self.quantize else float(deck.position)
        )
        refused = self._submit_manual(SetLoop(
            deck=deck_name, start_frame=max(0.0, start),
            length_frames=bars * 4 * self._frames_per_beat(deck),
            origin="manual:roll",
        ))
        if refused:
            return refused
        self._take_manual(deck_name, "roll", f"{bars:g}-bar roll")
        return f"Deck {deck_name.upper()} rolling {_bars_text(bars)}."

    def roll_off(self, deck_name: str) -> str:
        deck, refused = self._perform(deck_name, "roll off")
        if refused:
            return refused
        if not deck.loop_active:
            return f"Deck {deck_name.upper()} is not rolling."
        # A slip exit is in phase whenever it happens, so a roll lets go at once.
        refused = self._submit_manual(ExitLoop(deck=deck_name, origin="manual:roll"))
        if refused:
            return refused
        self._take_manual(deck_name, "roll off", "slip exit, phase kept")
        return f"Deck {deck_name.upper()} roll released, in phase."

    def beat_jump(self, deck_name: str, bars: int) -> str:
        deck, refused = self._perform(deck_name, "beat jump")
        if refused:
            return refused
        if abs(int(bars)) not in self.JUMP_BARS or int(bars) != bars:
            return "Refused: a beat jump is 1, 4, 8 or 16 bars, forward or back."
        frames = int(bars) * 4 * self._frames_per_beat(deck)
        _frame, execute_at = self._next_beat(deck)
        refused = self._submit_manual(BeatJump(
            deck=deck_name, frames=frames, execute_at=execute_at, origin="manual:jump",
        ))
        if refused:
            return refused
        self._take_manual(deck_name, "beat jump", f"{int(bars):+d} bars")
        when = "on the next beat" if execute_at != IMMEDIATE else "now"
        return f"Deck {deck_name.upper()} jumps {int(bars):+d} bars {when}."

    def set_pitch(self, deck_name: str, percent: float) -> str:
        deck, refused = self._perform(deck_name, "pitch")
        if refused:
            return refused
        if not -self.MAX_PITCH_PERCENT <= percent <= self.MAX_PITCH_PERCENT:
            return f"Refused: the pitch fader runs +-{self.MAX_PITCH_PERCENT:g}%."
        rate = 1.0 + percent / 100.0
        refused = self._submit_manual(SetPitch(deck=deck_name, rate=rate, origin="manual:pitch"))
        if refused:
            return refused
        self._take_manual(deck_name, "pitch", f"{percent:+.2f}%")
        bpm = deck.track.analysis.bpm * rate
        return f"Deck {deck_name.upper()} pitch {percent:+.2f}% ({bpm:.1f} BPM)."

    def sync(self, deck_name: str) -> str:
        """Match this deck to the master clock: tempo, then phase."""
        deck, refused = self._perform(deck_name, "sync")
        if refused:
            return refused
        master_name = self.engine.master_deck
        if master_name == deck_name:
            return f"Refused: deck {deck_name.upper()} is the master; sync the other deck to it."
        master = self.engine.deck(master_name)
        if master.track is None or self.engine.master_bpm <= 0:
            return "Refused: there is no master tempo to sync to."
        analysis = deck.track.analysis
        rate = self.engine.master_bpm / analysis.bpm
        if abs(rate - 1.0) > self.MAX_PITCH_PERCENT / 100.0 + 1e-9:
            return (
                f"Refused: matching {self.engine.master_bpm:.1f} BPM needs "
                f"{(rate - 1) * 100:+.1f}%, past the pitch range."
            )
        # Phase from one consistent snapshot of both playheads.
        _frames, _beat, pos_a, pos_b = self.engine.clock
        master_pos = pos_a if master_name == "a" else pos_b
        own_pos = pos_a if deck_name == "a" else pos_b
        master_frac = phrase.beat_at_frame(master.track.analysis, master_pos) % 1.0
        own_frac = phrase.beat_at_frame(analysis, own_pos) % 1.0
        diff = master_frac - own_frac
        if diff > 0.5:
            diff -= 1.0
        elif diff <= -0.5:
            diff += 1.0
        shift = diff * self._frames_per_beat(deck)
        refused = self._submit_manual(SyncDeck(
            deck=deck_name, rate=rate, shift_frames=shift, origin="manual:sync",
        ))
        if refused:
            return refused
        self._take_manual(
            deck_name, "sync", f"rate {rate:.4f}, phase shifted {diff:+.3f} beats"
        )
        return (
            f"Deck {deck_name.upper()} synced to {self.engine.master_bpm:.1f} BPM "
            f"({(rate - 1) * 100:+.2f}%), phase moved {diff:+.2f} beats."
        )

    # --- track loading -------------------------------------------------------

    def analysis_for(self, deck_name: str) -> TrackAnalysis | None:
        track = self.engine.deck(deck_name).track
        return track.analysis if track else None

    def _playing_bpm(self) -> float:
        """The tempo the mix is actually running at, ridden and stretched."""
        playing = self.engine.master_bpm
        if playing > 0:
            return playing
        live = self.engine.deck(self.live_deck)
        if live.track is None:
            return 0.0
        return live.track.analysis.bpm * max(live.rate, 1e-6)

    def tempo_path_to(self, incoming: TrackAnalysis) -> Any:
        """How the mix gets from what is playing to this track.

        Measured against the master clock rather than the playing track's
        printed tempo, so a deck that has already been ridden is where the
        next path starts from.
        """
        playing = self._playing_bpm()
        return selector_tempo_path(playing, incoming.bpm, travel=self._travelling(incoming))

    def _travelling(self, incoming: TrackAnalysis) -> bool:
        """Is this hand-over meant to move the set's tempo, or hold it?

        Only when a target is set and the set has not arrived. A travelling
        path lets the new track keep its own tempo and rides the playing deck
        to meet it; holding does the opposite, which is what a set with no
        destination wants. The condition is deliberately the same one
        :meth:`cue_next` plans under, so the path the candidate was chosen
        with is the path the transition is armed with.
        """
        if self.tempo_target is None:
            return False
        playing = self.engine.master_bpm
        if playing <= 0:
            return False
        return abs(playing - self.tempo_target) >= 0.5

    def _schedule_tempo_ride(self, path: Any) -> None:
        """Walk the playing deck toward the incoming tempo, a step per bar line.

        Scheduled commands on the control thread -- the same pitch fader an
        operator has, moved in small steps over `path.bars` bars. Nothing in
        the audio callback changes, and the ride is over before the transition
        it was planned for begins.
        """
        live = self.engine.deck(self.live_deck)
        if live.track is None or not path.ride_percent:
            return
        analysis = live.track.analysis
        # A ride still moving when the blend starts would pitch this deck while
        # the incoming one is held at the rate it was armed with: the two would
        # walk apart mid-transition. So it is cut to fit the time left before
        # mix-out, with a phrase to spare, and skipped when there is no room.
        seconds = self._seconds_to_mix_out(live)
        # Measured at the fastest this deck will be running, not the slowest:
        # a ride that speeds it up brings its mix-out forward, and an estimate
        # made at today's rate leaves the last steps stranded past the blend.
        playing_bpm = analysis.bpm * max(live.rate, 1e-6)
        ridden_bpm = playing_bpm * (1.0 + max(path.ride_percent, 0.0))
        room_bars = seconds * playing_bpm / 60.0 / 4.0 * (
            playing_bpm / ridden_bpm
        ) - RIDE_MARGIN_BARS
        bars = min(float(path.bars), room_bars)
        if bars < 2.0:
            self.session_log.write(
                "tempo_ride",
                deck=self.live_deck,
                track=analysis.title,
                trigger=path.describe(),
                action=(
                    f"not ridden: only {max(room_bars, 0.0):.1f} bar(s) before "
                    "mix-out, the deck stays where it is"
                ),
            )
            return
        start_bar = math.floor(phrase.bar_at_frame(analysis, live.position)) + 1
        steps = max(2, min(8, int(bars // 2) or 2))
        base = float(live.rate)
        queued = 0
        for i in range(1, steps + 1):
            bar = start_bar + bars * i / steps
            frame = phrase.frame_at_bar(analysis, bar)
            cmd = SetPitch(
                deck=self.live_deck,
                rate=round(base * (1.0 + path.ride_percent * i / steps), 6),
                execute_at=phrase.deck_frame_to_engine_frame(
                    live, self.engine.frames_played, frame
                ),
                origin="tempo_ride",
                token=self._ride_token,
            )
            if self.supervisor.validate(cmd) is None:
                self.scheduler.submit(cmd)
                self._ride_steps.append((cmd.execute_at, cmd.rate))
                queued += 1
        self._ride_deck = self.live_deck
        self.session_log.write(
            "tempo_ride",
            deck=self.live_deck,
            track=analysis.title,
            trigger=f"{path.describe()}",
            action=(
                f"{queued} pitch step(s) over {bars:.0f} bars to "
                f"{base * (1.0 + path.ride_percent):.4f}x"
            ),
        )

    def _schedule_master_glide(self, deck_name: str) -> None:
        """Glide the beat clock to the deck's own BPM after a hand-over.

        The tempo lock this ends, in one line of arithmetic: an incoming track
        is matched to master before the blend, so its rate is
        ``master / native``; :meth:`Engine.hand_master_to` then adopts
        ``native * rate`` -- which is ``master`` again. The hand-over cannot
        move the clock, so without this the set stays at whatever tempo it
        opened at and every later track is stretched back to it for the night.

        So the glide walks the deck that is now the mix from the rate it was
        matched at back to 1.0 -- its own tempo -- one pitch step per bar line
        over :data:`config.MASTER_GLIDE_BARS`. The engine moves the clock with
        the master deck's fader, so the clock arrives with it, and the last
        step puts the deck back on its unstretched source.

        Scheduled commands on the control thread, like :meth:`_schedule_tempo_ride`
        it borrows from. Nothing in the audio callback changes.
        """
        if self.master_tempo_locked:
            return
        deck = self.engine.deck(deck_name)
        if deck.track is None:
            return
        analysis = deck.track.analysis
        base = float(deck.rate)
        if abs(base - 1.0) < config.STRETCH_DEADBAND:
            return  # already at its own tempo: nothing to glide
        # A glide still in flight is superseded, never interleaved with this one.
        stale = self.scheduler.cancel_matching(
            lambda cmd: getattr(cmd, "origin", "") == "master_glide"
        )
        if stale:
            gone = {(c.execute_at, c.rate) for c in stale}
            self._ride_steps = [s for s in self._ride_steps if s not in gone]
        bars = float(max(1, config.MASTER_GLIDE_BARS))
        # One step per bar, capped so a long glide does not flood the scheduler.
        steps = max(2, min(16, int(bars)))
        start_bar = math.floor(phrase.bar_at_frame(analysis, deck.position)) + 1
        queued = 0
        for i in range(1, steps + 1):
            # Linear in rate, so each step is the same size: a glide that
            # accelerates is a glide the room can hear.
            rate = base + (1.0 - base) * i / steps
            if i == steps:
                rate = 1.0  # land exactly, so the source swap triggers
            frame = phrase.frame_at_bar(analysis, start_bar + bars * i / steps)
            cmd = SetPitch(
                deck=deck_name,
                rate=round(rate, 6),
                execute_at=phrase.deck_frame_to_engine_frame(
                    deck, self.engine.frames_played, frame
                ),
                origin="master_glide",
                token=self._ride_token,
            )
            if self.supervisor.validate(cmd) is None:
                self.scheduler.submit(cmd)
                self._ride_steps.append((cmd.execute_at, cmd.rate))
                queued += 1
        if not queued:
            return
        self._ride_deck = deck_name
        self.session_log.write(
            "master_glide",
            deck=deck_name,
            track=analysis.title,
            from_bpm=round(analysis.bpm * base, 2),
            to_bpm=round(analysis.bpm, 2),
            trigger="transition finished",
            action=(
                f"{queued} step(s) over {bars:.0f} bars from {base:.4f}x to "
                f"1.0000x; beat clock follows to {analysis.bpm:.1f} BPM"
            ),
        )

    def _follow_ride(self) -> None:
        """Move the beat clock along with a ride that has landed a step.

        The clock counts the master deck. A ride pitches that deck up without
        telling it, so the clock would fall behind the pulse it is supposed to
        be counting -- and the supervisor, seeing the deck run ahead of the
        clock, would spend the ride correcting it back down. Following each
        landed step keeps the two on the same side.

        Control thread, from the autopilot tick: two float writes and a pair of
        dropped baselines, the same work a grid correction already does.
        """
        if not self._ride_steps:
            return
        now = self.engine.frames_played
        landed: tuple[int, float] | None = None
        while self._ride_steps and self._ride_steps[0][0] <= now:
            landed = self._ride_steps.pop(0)
        if landed is None or self.engine.master_deck != self._ride_deck:
            return
        before = self.engine.master_bpm
        after = self.engine.hand_master_to(self._ride_deck, rate=landed[1])
        if abs(after - before) < 1e-6:
            return
        for name in ("a", "b"):
            self.supervisor.forget_baseline(name)
        self.session_log.write(
            "tempo_ride_step",
            deck=self._ride_deck,
            from_bpm=round(before, 2),
            to_bpm=round(after, 2),
            trigger="ride step landed",
            action=f"beat clock now {after:.1f} BPM",
        )

    def _drop_unlanded_ride(self) -> None:
        """Forget a ride once the deck it was riding is no longer the mix.

        Its remaining steps would pitch a deck that has stopped, and -- worse --
        :meth:`_follow_ride` would read them as tempo the clock should follow.
        """
        if not self._ride_steps:
            return
        # By token, not by origin: a master glide's steps are ride steps too,
        # and matching "tempo_ride" alone once left a glide running after the
        # operator locked the clock.
        self.scheduler.cancel(self._ride_token)
        self._ride_token = CancelToken("tempo")
        self._ride_steps.clear()

    def _planned_master_bpm(self) -> float:
        """Where the beat clock will be once a ride in flight has landed."""
        master = self.engine.master_bpm
        if not self._ride_steps or self.engine.master_deck != self._ride_deck:
            return master
        deck = self.engine.deck(self._ride_deck)
        if deck.track is None:
            return master
        return deck.track.analysis.bpm * self._ride_steps[-1][1]

    def _drop_late_ride_steps(self, blend_starts_at: int) -> None:
        """Cancel ride steps that would still be moving during the blend.

        The ride is planned to be over well before this -- see
        :data:`RIDE_MARGIN_BARS` -- so this is a fence, not a mechanism. If it
        ever fires it is worth reading about, because the transition was
        matched to a tempo the deck was not going to reach in time, and the
        supervisor has to take up the difference.
        """
        if not self._ride_steps:
            return
        late = [step for step in self._ride_steps if step[0] >= blend_starts_at]
        if not late:
            return
        cutoff = min(step[0] for step in late)
        dropped = self.scheduler.cancel_matching(
            lambda cmd: getattr(cmd, "origin", "") == "tempo_ride"
            and getattr(cmd, "execute_at", 0) >= cutoff
        )
        self._ride_steps = [s for s in self._ride_steps if s[0] < blend_starts_at]
        self.session_log.write(
            "tempo_ride",
            deck=self._ride_deck,
            trigger="ride would still be moving when the blend starts",
            action=(
                f"{len(dropped)} step(s) dropped; the supervisor closes what "
                "is left"
            ),
        )

    def _beyond_stretch_range(self, incoming: TrackAnalysis) -> bool:
        """Is this track too far from the live tempo for any blend?

        Beyond the stretch range there is no rate that matches the decks, so
        the hand-over has to be a cut at the track's own tempo -- see
        :meth:`cue_next`'s last resort and :meth:`arm_transition`.
        """
        live = self.engine.deck(self.live_deck)
        if live.track is None or incoming.bpm <= 0:
            return False
        effective = live.track.analysis.bpm * max(live.rate, 1e-6)
        if effective <= 0:
            return False
        return abs(incoming.bpm - effective) / effective > config.MAX_STRETCH_RATIO

    def _rate_for(self, candidate: TrackAnalysis) -> float:
        """Resampling rate that puts ``candidate`` at the master tempo.

        Never further from 1.0 than :data:`config.MAX_STRETCH_RATIO`. Past that
        the stretcher refuses to make a key-locked copy, so the deck resamples
        the whole rate instead -- and a 120 BPM clock asked to play a 160 BPM
        track resamples it by 0.75, five semitones down. Matching is given up
        before pitch is: the track comes in at the nearest tempo the cap allows
        and the transition has to be one that does not need a long beatmatch.
        """
        master_bpm = self.engine.master_bpm
        if master_bpm <= 0 or candidate.bpm <= 0:
            return 1.0
        rate = master_bpm / candidate.bpm
        # Half or double time first: a 160 BPM track under a 120 clock counts
        # perfectly well at 80, and that lands inside the cap when 160 cannot.
        # Strictly inside the cap, not on it: the supervisor's check is
        # `abs(rate - 1.0) > MAX_RATE_DELTA` with no epsilon, and
        # `abs(1.08 - 1.0)` is 0.08000000000000007. Clamping to the limit
        # exactly produced a rate the supervisor then refused, which stalled
        # cueing altogether -- measured in the 20-track soak, where only 4 of
        # 20 tracks ever reached a deck.
        limit = config.MAX_STRETCH_RATIO * (1.0 - 1e-9)
        if abs(rate - 1.0) > limit:
            for folded in (rate * 2.0, rate / 2.0):
                if abs(folded - 1.0) <= limit:
                    return folded
        return float(min(1.0 + limit, max(1.0 - limit, rate)))

    def _load_or_refuse(self, analysis: TrackAnalysis, origin: str):
        """Decode a track, or say why not and remember it. Never raises.

        Control thread. Returns the loaded track, or None -- and a None is
        always accompanied by a log line naming the track and the reason, so a
        set that quietly stops choosing something has an explanation on disk.
        """
        try:
            return load_track(analysis)
        except Exception as exc:
            self._unplayable.add(analysis.track_id)
            reason = f"{type(exc).__name__}: {exc}"
            self.session_log.write(
                "track_unplayable",
                track=analysis.title,
                path=str(analysis.path),
                trigger=origin,
                action=f"excluded for this session: {reason}",
            )
            self.notify(
                f"[{origin}] {analysis.title} will not decode and has been "
                f"set aside: {reason}"
            )
            log.error("could not load %s: %s", analysis.path, exc)
            return None

    def _set_aside(self, track: TrackAnalysis) -> str | None:
        """Refuse a track whose analysis failed; say so once. Never raises."""
        reason = an.unusable_reason(track)
        if reason is not None and track.track_id not in self._unplayable:
            self._unplayable.add(track.track_id)
            self.session_log.write("track_unplayable", track=track.title,
                                   path=str(track.path), trigger="analysis check",
                                   action=f"excluded for this session: {reason}")
            self.notify(f"[autopilot] {track.title} set aside: {reason}")
        return reason

    def _selectable(self) -> set[str]:
        """Track ids autonomous selection must not offer: played, or broken."""
        for track in self.crate:
            self._set_aside(track)
        return self.played | self._unplayable

    def start_first_track(self, energy_direction: float = 0.0) -> bool:
        """Decode and start the opening track on deck A."""
        loaded = None
        # Walk down the ranking rather than dying on the first pick: an
        # unreadable opening track used to raise out of here and the set never
        # started at all.
        for _ in range(len(self.crate) or 1):
            first = select_next(
                self.crate, None, self._selectable(), energy_direction,
                set_phase=self._phase_arg(), history=self.history,
            )
            opener = self.set_plan.slots[0].track if self.set_plan and self.set_plan.slots else None
            if opener is not None and opener.track_id not in self._selectable():
                first = selector_mod.Candidate(
                    track=opener, score=0.0, bpm_delta_pct=0.0, key_relation="unknown",
                    energy_delta=0.0, penalties=("set plan opener",),
                )
            if first is None:
                break
            loaded = self._load_or_refuse(first.track, "autopilot")
            if loaded is not None:
                break
        if first is None or loaded is None:
            print("No playable tracks in the cache. Run "
                  "`python -m djai analyze <folder>` first.")
            return False
        start_frame = int(first.track.first_downbeat * SAMPLE_RATE)
        self.engine.submit(
            LoadTrack(
                deck="a",
                track=loaded,
                start_frame=start_frame,
                rate=1.0,
                master=True,
                origin="autopilot",
            )
        )
        self.engine.deck_a.gain.jump(0.0)
        self.engine.deck_a.set_gain(1.0)
        self.live_deck = "a"
        # Kept because the deck itself cannot be asked yet: a LoadTrack is
        # applied by the audio thread on its next callback, so reading
        # deck_a.track back here is a race that returns None.
        # Held only until the safety net has taken its own copy of the audio.
        # Releasing it there is not enough on its own -- a session that never
        # arms a fallback kept a whole decoded track pinned for the night --
        # so `_release_first_track` drops it once deck A is actually playing it.
        self._first_track = loaded
        self.played.add(first.track.track_id)
        self.history.append(first.track)
        self.set_state.history.append(first.track.track_id)
        self.set_state.save()
        self.session_log.write(
            "track_started",
            deck="a",
            track=first.track.title,
            bpm=first.track.bpm,
            camelot=first.track.camelot,
            trigger="session start",
            action="loaded and playing",
        )
        self.replan(f"opened on {first.track.title}")
        print(
            f"Now playing: {first.track.title}  "
            f"{first.track.bpm:.1f} BPM  {first.track.camelot} "
            f"({first.track.key_name})"
        )
        return True

    def cue_next(self, energy_direction: float = 0.0, origin: str = "autopilot") -> bool:
        """Pick, decode and cue the next track onto the idle deck.

        Returns True if a track was cued. Decoding happens on the calling
        thread, which is never the audio thread.
        """
        self._complete_handover()
        current = self.analysis_for(self.live_deck)
        idle = "b" if self.live_deck == "a" else "a"

        if self.engine.deck_in_transition(idle):
            self.notify(f"[autopilot] deck {idle} is mid-transition, not cueing")
            self._report_no_cue(f"deck {idle} is mid-transition", origin)
            return False
        idle_deck = self.engine.deck(idle)
        if idle_deck.transport is TransportState.PLAYING and idle_deck.gain.target > 0.0:
            # The engine has already handed over to this deck and the
            # bookkeeping in _autopilot_tick has not caught up: `live_deck` is
            # stale for a moment. Cueing here would load over the music the
            # room is hearing -- found by the Phase 2.3 fuzz as dead air.
            self._report_no_cue(f"deck {idle} is the one playing", origin)
            return False

        forced = self._forced_next
        if forced is not None:
            # The operator named this track. It is used once and then cleared,
            # so a forced pick never silently governs the rest of the night.
            self._forced_next = None
            loaded = self._load_or_refuse(forced, f"{origin}:forced")
            if loaded is None:
                self._report_no_cue(
                    f"{forced.title} will not decode", origin
                )
                return False
            cmd = LoadTrack(
                deck=idle,
                track=loaded,
                start_frame=int(forced.mix_in * SAMPLE_RATE),
                rate=self._rate_for(forced),
                play=False,
                origin=f"{origin}:forced",
            )
            rejection = self.supervisor.validate(cmd)
            if rejection is not None:
                self.notify(f"[override] {forced.title} was rejected by the supervisor")
                self._report_no_cue(
                    f"the forced track {forced.title} was rejected: {rejection.reason}",
                    origin,
                )
                return False
            self.engine.submit(cmd)
            self.engine.deck(idle).gain.jump(0.0)
            self._cued = loaded
            self.played.add(forced.track_id)
            self.history.append(forced)
            self._cue_decision = self.session_log.decision(
                "track_cued",
                self.engine.features.latest(),
                "manual override of the selector",
                deck=idle,
                track=forced.title,
                trigger=f"{origin} (forced by the operator)",
            )
            self._count_cue(forced.track_id)
            self._after_cue(forced)
            return True

        queued = self._next_queued()
        if queued is not None:
            return self._cue_candidate(queued, idle, energy_direction, origin, "")

        planned = self._planned_candidate(current)
        if planned is not None:
            return self._cue_candidate(planned, idle, energy_direction, origin, "; set plan")

        phase = self._phase_arg()
        # A journey, not a next track: the plan looks several tracks ahead and
        # only its first step is played. Re-planned at every cue, so it follows
        # what the room actually did rather than what was planned for it.
        # Planned from the tempo the mix is running at, not the playing
        # track's printed one: after a ride they are different numbers, and
        # planning from the label is what makes a cue arrive with a path the
        # arm then measures as unreachable.
        live_bpm = self._playing_bpm()
        travelling = (
            self.tempo_target is not None
            and live_bpm > 0
            and abs(live_bpm - self.tempo_target) >= 0.5
        )
        journey = plan_journey(
            self.crate, current, self._selectable(), energy_direction,
            set_phase=phase, history=self.history, cache_dir=self.cache_dir,
            weights=self.selector_weights, target_bpm=self.tempo_target,
            playing_bpm=live_bpm or None, travel=travelling,
        )
        self.journey = journey
        candidate = journey.first
        if candidate is not None and journey.steps[1:]:
            self.session_log.write(
                "journey",
                deck=idle,
                track=candidate.track.title,
                trigger=f"{origin} ({energy_direction:+.2f} energy)",
                action=journey.reason(),
                ahead=[c.track.title for c in journey.steps[1:]],
            )
        if candidate is None:
            candidate = select_next(
                self.crate, current, self._selectable(), energy_direction,
                set_phase=phase, history=self.history, cache_dir=self.cache_dir,
                weights=self.selector_weights,
            )
        if candidate is None:
            # Nothing new fits; allow repeats rather than falling silent.
            candidate = select_next(
                self.crate, current, set(), energy_direction,
                set_phase=phase, history=self.history, cache_dir=self.cache_dir,
                weights=self.selector_weights,
            )
            if candidate is None:
                # Nothing is within the stretch range, so no blend exists. The
                # room still has to hear something next: take the nearest tempo
                # and hand over with a cut, which needs no beat-matching. This
                # is the difference between a jarring change and the track
                # playing out into silence -- and silence is the one outcome
                # this program does not allow.
                candidate = select_nearest_tempo(self.crate, current, self._selectable())
                if candidate is None:
                    candidate = select_nearest_tempo(self.crate, current, set())
                    if candidate is not None:
                        self.played.clear()
                if candidate is None:
                    self.notify("[autopilot] no compatible track found")
                    if current is None:
                        why = "the crate offered no track at all"
                    else:
                        why = (
                            f"no track in the crate is within the "
                            f"+-{config.MAX_STRETCH_RATIO:.0%} stretch range of "
                            f"{current.title} ({current.bpm:.1f} BPM)"
                        )
                    self._report_no_cue(why, origin)
                    return False
                self.session_log.write(
                    "tempo_orphan",
                    deck=idle,
                    track=candidate.track.title,
                    trigger=(
                        f"nothing within +-{config.MAX_STRETCH_RATIO:.0%} of "
                        f"{current.title if current else '?'} "
                        f"({current.bpm:.1f} BPM)" if current else "no live track"
                    ),
                    action=(
                        f"cueing {candidate.track.title} "
                        f"({candidate.track.bpm:.1f} BPM, "
                        f"{candidate.bpm_delta_pct:+.1f}%) at its own tempo for a cut"
                    ),
                )
                self.notify(
                    f"[autopilot] no blend partner for this tempo - cutting to "
                    f"{candidate.track.title} ({candidate.track.bpm:.1f} BPM)"
                )
            self.played.clear()

        # A drop-aligned style asked for by name needs an incoming track whose
        # drop can actually be lined up with this one's. Found in the Part 4
        # soak: both drop styles fell back, one because the incoming track's
        # only drop left no runway. The selector's ranking still decides; this
        # walks down it to the best track the placement rule would accept, and
        # keeps the selector's first choice when none qualifies.
        drop_note = ""
        preset = (
            transition.preset_params(self.transition_style)
            if self.transition_style in transition.STYLES
            else None
        )
        live = self.engine.deck(self.live_deck)
        if preset is not None and preset.align_mode == "drop" and live.track is not None:
            bars = preset.length_bars + self.hold_extra_bars
            for option in rank_candidates(
                self.crate, current, self._selectable(), energy_direction,
                set_phase=phase, history=self.history, cache_dir=self.cache_dir,
                weights=self.selector_weights,
                playing_bpm=live_bpm or None, travel=travelling,
            ):
                if self._plan_drop_aligned(live, option.track, preset, bars) is None:
                    continue
                if option.track.track_id != candidate.track.track_id:
                    drop_note = (
                        f"; passed over {candidate.track.title}, whose drop "
                        "could not be lined up"
                    )
                candidate = option
                break
            else:
                drop_note = "; no track's drop could be lined up, best match kept"

        return self._cue_candidate(candidate, idle, energy_direction, origin, drop_note)

    def _planned_candidate(self, current):
        """The set plan's next slot, if there is a plan and it still stands.

        The slot must be unplayed, playable and reachable inside the cap from
        the tempo the mix is at now; otherwise the plan has gone stale and the
        journey chooses (the replan after that cue repairs the plan).
        """
        plan = self.set_plan
        if plan is None or not plan.slots:
            return None
        slot = next((x for x in plan.slots if not x.cued), None)
        if slot is None or slot.track.track_id in self._selectable():
            return None
        for c in rank_candidates(
            self.crate, current, self._selectable(), self.energy_direction,
            set_phase=self._phase_arg(), history=self.history, cache_dir=self.cache_dir,
            weights=self.selector_weights, playing_bpm=self._playing_bpm() or None,
            travel=True,
        ):
            if c.track.track_id == slot.track.track_id:
                return c
        return None

    def _next_queued(self):
        """The due operator cue as a Candidate, or None. Takes it off the queue.

        A cue whose file is gone is reported to the operator and the log,
        taken off the queue, and the next one is tried: the queue continues.
        """
        while True:
            entry = self._due_cue()
            if entry is None:
                return None
            track = self._track_by_id(entry["track_id"])
            if track is None or self._set_aside(track) or track.track_id in self._unplayable:
                self.cue_queue.remove(entry)
                self.set_state.save()
                why = f"cued track {entry['title']} is missing or unreadable; skipped"
                self.session_log.write("cue_missing", track=entry["title"],
                                       trigger="cue queue", action=why)
                self.notify(f"[cue] {why}")
                continue
            if not entry.get("bridge") and self._check_bridge(entry):
                return None      # the gap grew past the cap since it was queued
            self.cue_queue.remove(entry)
            self.set_state.save()
            if getattr(track, "quarantined", False):
                self._conservative[track.track_id] = "quarantined grid: echo out into it"
            if entry.get("bridge") == "tempo":
                self._conservative[track.track_id] = "tempo bridge: echo out, own tempo"
            current = self.analysis_for(self.live_deck)
            path = self._cap_path(track)
            return selector_mod.Candidate(
                track=track, score=0.0,
                bpm_delta_pct=(
                    (track.bpm - self._playing_bpm()) / self._playing_bpm() * 100.0
                    if self._playing_bpm() else 0.0
                ),
                key_relation=(selector_mod.key_relation(current.camelot, track.camelot)
                              if current else "unknown"),
                energy_delta=0.0, penalties=(f"cued by the operator (cue {entry['id']})",),
                tempo_path=path,
            )

    def _cue_candidate(self, candidate, idle: str, energy_direction: float,
                       origin: str, drop_note: str) -> bool:
        """Load, place and cue a chosen track: the tail every choice shares."""
        loaded = self._load_or_refuse(candidate.track, origin)
        if loaded is None:
            # Set aside and try again at once: the next tick would otherwise
            # rank the same unreadable file first all over again.
            return self.cue_next(energy_direction, origin=origin)
        loaded = self._at_mix_point(loaded)
        # How these two tempos are going to meet: straight, by riding the
        # playing deck toward it, or counted half or double. The 8% wall is
        # gone; what is left is a path or the absence of one.
        path = self.tempo_path_to(candidate.track)
        self.tempo_plan = path
        rate = path.rate_b if path.blendable else 1.0
        # Park the cued deck on its mix-in so `state` shows where it will
        # actually come in; arm_transition re-cues to the same point.
        start_frame = int(loaded.analysis.mix_in * SAMPLE_RATE)

        cmd = LoadTrack(
            deck=idle,
            track=loaded,
            start_frame=start_frame,
            rate=rate,
            play=False,
            origin=origin,
        )
        rejection = self.supervisor.validate(cmd)
        if rejection is not None:
            self._report_no_cue(
                f"{candidate.track.title} was rejected: {rejection.reason}", origin
            )
            return False

        self.engine.submit(cmd)
        self.engine.deck(idle).gain.jump(0.0)
        self.engine.deck(idle).metric_ratio = path.ratio if path.blendable else 1.0
        self._cued = loaded
        self.played.add(candidate.track.track_id)
        self.history.append(candidate.track)
        if path.technique not in ("direct", "none"):
            self.session_log.write(
                "tempo_path",
                deck=idle,
                track=candidate.track.title,
                trigger=f"{self.engine.master_bpm:.1f} -> {candidate.track.bpm:.1f} BPM",
                action=path.describe(),
                technique=path.technique,
                ratio=path.ratio,
                ride_percent=round(path.ride_percent, 4),
                bars=path.bars,
            )
        if path.ride_percent:
            self._schedule_tempo_ride(path)
        self._cue_decision = self.session_log.decision(
            "track_cued",
            self.engine.features.latest(),
            candidate.reason() + drop_note,
            deck=idle,
            track=candidate.track.title,
            trigger=f"{origin} ({energy_direction:+.2f} energy)",
        )
        # The pre-roll window: the transition is a phrase or more away, so this
        # is where a design is asked for. It runs on its own thread and nothing
        # waits for it -- if it is not back by the time the transition is
        # armed, the preset runs instead.
        self.start_transition_design(candidate.track)
        self._count_cue(candidate.track.track_id)
        self._after_cue(candidate.track)
        return True

    def _after_cue(self, track: TrackAnalysis) -> None:
        """Revise the plan within one track of whatever was just cued."""
        plan = self.set_plan
        if plan is None:
            return
        expected = plan.slots[0].track if plan.slots else None
        if expected is not None and expected.track_id != track.track_id:
            self.session_log.write(
                "plan_deviation", track=track.title, trigger="cue",
                action=f"planned {expected.title}, cued {track.title}",
            )
            self.replan(f"off-plan: {track.title} instead of {expected.title}")
        else:
            self.replan(f"cued {track.title}")

    def _at_mix_point(self, loaded: LoadedTrack) -> LoadedTrack:
        """Apply a one-shot ranked mix-in target (SPEC §3 `cue` action).

        A copy of the analysis with its mix-in moved, never the crate's own:
        the crate's entry is shared by every later selection. Placement reads
        the deck's analysis, so everything downstream agrees on the point.
        """
        rank, self.cue_mix_point = self.cue_mix_point, None
        regions = loaded.analysis.mix_in_regions
        if not rank or not regions:
            return loaded
        region = regions[min(rank, len(regions)) - 1]
        analysis = dataclasses.replace(
            loaded.analysis, mix_in=float(region["seconds"]),
            mix_in_bar=float(region["bar"]),
        )
        self.session_log.write(
            "mix_point_target", track=analysis.title,
            trigger=f"cue action: mix point {rank}",
            action=f"mix in at bar {region['bar']} (quality {region['quality']:.2f})",
        )
        return dataclasses.replace(loaded, analysis=analysis)

    def cued_deck(self) -> str:
        return "b" if self.live_deck == "a" else "a"

    def has_cued_track(self) -> bool:
        return self._cued is not None

    def reload_feedback(self) -> dict[str, Any]:
        """Work out the weights in use. Control thread; reads files.

        The stable model is the current stored version (djai.feedback); on the
        very first start with logs and no model, one is trained, which is what
        this method used to do every time. Tonight's layer is re-learned from
        this session's own log on every call -- at start, on every verdict or
        label, on a persona change -- so pressing a button changes the next
        pick rather than the next night, and never touches the stable model.
        """
        from djai import feedback as feedback_mod

        log_dir = self.session_log.path.parent
        if self._stable_model is None and not self._model_checked:
            # Once per session. The bootstrap reads every log but tonight's:
            # tonight belongs to the night layer until someone says `model
            # learn`, or it would be counted in both.
            self._model_checked = True
            mdir = feedback_mod.model_dir(log_dir)
            self._stable_model = feedback_mod.load(mdir)
            if self._stable_model is None and any(
                p != self.session_log.path and feedback_mod.read_log(p)
                for p in log_dir.glob("session_*.jsonl")
            ):
                self._stable_model = feedback_mod.train(
                    log_dir, note="first start", exclude=self.session_log.path)
        night = feedback_mod.night_layer(
            feedback_mod.read_log(self.session_log.path, self._night_from))
        eff = feedback_mod.effective(self._stable_model, night, self.persona)
        self.selector_weights = dict(eff["selector"])
        self.critic_weights = dict(eff["critic"])
        self.timing_bias_bars = int(eff["timing_bias_bars"])
        summary = dict(night.get("summary") or {})
        stable = self._stable_model or {}
        summary["model_version"] = stable.get("version")
        summary["stable"] = ((stable.get("global") or {}).get("summary") or {}).get("learning", "")
        summary["night"] = summary.get("learning", "")
        summary["learning"] = (
            f"model v{stable.get('version')}" if stable else "no stored model"
        ) + (f"; tonight: {summary['night']}" if summary["night"] else "")
        self.feedback_summary = summary
        return {"selector": self.selector_weights, "critic": self.critic_weights,
                "timing_bias_bars": self.timing_bias_bars, "summary": summary}

    def record_feedback(self, label: str, note: str = "") -> str:
        """The operator's word on what just happened. Any control thread.

        Written with everything a learner needs -- the transition's two tracks,
        its style, what the critic measured, how alike the pair was, the
        persona, and for timing labels where the blend sat -- then tonight's
        layer is re-learned at once.
        """
        from djai import feedback as feedback_mod

        if label not in feedback_mod.VERDICTS:
            raise ValueError(f"verdict must be one of {feedback_mod.VERDICTS}, got {label!r}")
        live = self.engine.deck(self.live_deck)
        other = self.engine.deck(self.cued_deck())
        outgoing = other.track.analysis.title if other.track is not None else None
        incoming = live.track.analysis.title if live.track is not None else None
        outcome = self.last_preview
        critic_rounds = getattr(outcome, "critic", []) if outcome is not None else []
        last = critic_rounds[-1] if critic_rounds else {}
        similarity = {}
        if self.journey is not None and getattr(self.journey, "first", None) is not None:
            similarity = dict(self.journey.first.similarity or {})
        context = {}
        if label in (feedback_mod.TOO_EARLY, feedback_mod.TOO_LATE):
            context = {"timing_bias_bars": self.timing_bias_bars,
                       "transition_active": self.engine.transition_active}
        if label in (feedback_mod.MORE_LIKE_THIS, feedback_mod.LESS_LIKE_THIS):
            context = {"track": incoming,
                       "track_id": live.track.analysis.track_id if live.track else None}
        feedback_mod.record(
            self.session_log, label, tracks=(outgoing, incoming),
            style=self.last_transition_choice[0] if self.last_transition_choice else "",
            critic_score=last.get("score"), worst_measure=last.get("worst") or "",
            similarity=similarity, note=note, persona=self.persona, context=context,
        )
        learned = self.reload_feedback()
        return f"Noted: {label.replace('_', ' ')}. {learned['summary'].get('learning', '')}"

    def model_text(self, words: str) -> str:
        """`model`, `model learn`, `model rollback [v]`, `model versions`."""
        from djai import feedback as feedback_mod

        log_dir = self.session_log.path.parent
        mdir = feedback_mod.model_dir(log_dir)
        parts = words.split()
        word = parts[0] if parts else ""
        if word == "learn":
            self._stable_model = feedback_mod.train(log_dir, note="operator: model learn")
            # Tonight so far is in the stable model now; the night layer starts
            # again from here rather than counting it twice.
            self._night_from = self.session_log.path.stat().st_size
            self.reload_feedback()
            self.session_log.write("model_trained", trigger="operator",
                                   action=f"v{self._stable_model['version']}",
                                   events=self._stable_model["events"])
            return (f"Trained model v{self._stable_model['version']} on "
                    f"{self._stable_model['events']} event(s).")
        if word == "rollback":
            try:
                version = feedback_mod.rollback(
                    mdir, int(parts[1].lstrip("v")) if len(parts) > 1 else None)
            except (ValueError, IndexError) as exc:
                return f"Cannot roll back: {exc}"
            self._stable_model = feedback_mod.load(mdir, version)
            self.reload_feedback()
            self.session_log.write("model_rollback", trigger="operator", action=f"v{version}")
            return f"Model v{version} is current."
        if word == "versions":
            have = feedback_mod.versions(mdir)
            now = feedback_mod.current_version(mdir)
            return ("Models: " + ", ".join(f"v{v}{'*' if v == now else ''}" for v in have)
                    if have else "No stored models.")
        return (f"{self.feedback_summary.get('learning', '')}. Selector x{self.selector_weights}, "
                f"critic x{self.critic_weights}, blend length {self.timing_bias_bars:+d} bars.")

    def _report_no_cue(self, reason: str, origin: str = "autopilot") -> None:
        """Say loudly that no next track is cued, and why. Any control thread.

        Every path that leaves the idle deck without a next track reports here,
        so the room going quiet is never silent in the log: a warning on the
        Python logger, a ``no_track_cued`` line in the session log, and a
        notice in the REPL and the UI. The same reason on the same live track
        repeats at most every :data:`NO_CUE_REPEAT_S`.
        """
        live = self.engine.deck(self.live_deck)
        title = live.track.analysis.title if live.track is not None else None
        key = (reason, title)
        now = time.monotonic()
        last = self._no_cue_reported.get(key)
        if last is not None and now - last < NO_CUE_REPEAT_S:
            return
        self._no_cue_reported[key] = now
        log.warning("no next track cued (%s, live deck %s): %s",
                    origin, self.live_deck, reason)
        self.session_log.write(
            "no_track_cued",
            deck=self.live_deck,
            track=title,
            trigger=origin,
            action=reason,
        )
        self.notify(f"[autopilot] WARNING: no next track cued - {reason}")

    # --- designed transitions -------------------------------------------------

    def _design_context(self, incoming: TrackAnalysis) -> dict:
        """What the designer is told. Facts about the music, and nothing else.

        Note what is absent: no frames, no seconds, no positions in either
        track, no bar numbers counted from anywhere but the transition itself.
        The model is not given the information it would need to compute timing,
        which is a stronger guarantee than asking it not to.
        """
        live = self.engine.deck(self.live_deck)
        a = live.track.analysis if live.track else None
        energy_now = 0.0
        if a is not None and a.beat_rms:
            beats = a.beats_np
            i = int(np.searchsorted(beats, live.position / SAMPLE_RATE))
            window = a.beat_rms[max(0, i - 16):max(1, i)]
            if window:
                energy_now = round(sum(window) / len(window), 4)
        return {
            "deck_a": {
                "bpm": round(float(a.bpm), 1) if a else None,
                "key": a.camelot if a else None,
                "energy": energy_now,
            },
            "deck_b": {
                "bpm": round(float(incoming.bpm), 1),
                "key": incoming.camelot,
                "energy": round(float(incoming.energy), 4),
            },
            # Name, label, and whether entering there leaves enough track to
            # play. `usable` is a yes/no, not a position: the model still gets
            # no timing, it just stops being asked to guess which cue works.
            #
            # Measured need: 5 of 6 rejections over 32 designs were the model
            # naming a cue that sat too near its track's mix-out, because
            # nothing in the context distinguished one cue from another.
            "hot_cues": [
                {
                    "name": f"hot_cue_{c['index']}",
                    "label": c.get("label", ""),
                    "usable": bool(
                        transition.entry_has_runway(
                            incoming, float(c["sample_position"]) / SAMPLE_RATE
                        )
                    ),
                }
                for c in (incoming.hot_cues or [])
            ],
            "energy_direction": round(float(self.energy_direction), 2),
        }

    def start_transition_design(self, incoming: TrackAnalysis) -> None:
        """Kick off a design during the pre-roll. Returns immediately.

        Runs on its own short-lived thread. The scheduler is never involved and
        never waits: if the answer has not arrived by the time the transition
        is armed, the rule-based preset runs instead and the late answer is
        discarded. A model is an improvement to reach for, not a dependency.
        """
        engine = self.intent_engine
        if engine is None or not getattr(engine, "available", False):
            return
        if self.transition_style != "auto":
            return          # the operator named a style; do not second-guess it

        slot: dict = {
            "track_id": incoming.track_id, "raw": None, "reason": "",
            "started_at": time.monotonic(),
        }
        context = self._design_context(incoming)

        def work() -> None:
            raw, reason = engine.design_transition(context)
            slot["raw"] = raw
            slot["reason"] = reason

        thread = threading.Thread(
            target=work, name="djai-transition-design", daemon=True
        )
        slot["thread"] = thread
        slot["context"] = context
        self._design = slot
        thread.start()

    def design_pending(self) -> bool:
        """Is a design still being written, with room to wait for it?

        The pre-roll window is around 105 seconds wide and a design takes a
        few, so the autopilot can afford to hold off arming. Without this it
        could not: cueing and arming are one 0.5 s tick apart, so the answer
        was never back in time and every transition quietly used a preset.

        The wait is bounded twice -- by a deadline, and by how much of the
        outgoing track is left. Nothing here blocks; it just declines to arm
        yet, and the next tick asks again.
        """
        slot = self._design
        if slot is None:
            return False
        thread = slot.get("thread")
        if thread is None or not thread.is_alive():
            return False
        if time.monotonic() - slot["started_at"] >= config.TRANSITION_DESIGN_WAIT_S:
            return False

        live = self.engine.deck(self.live_deck)
        if live.track is None:
            return False
        analysis = live.track.analysis
        remaining = (
            (analysis.mix_out * SAMPLE_RATE - live.position)
            / SAMPLE_RATE / max(live.rate, 1e-6)
        )
        # Only wait while there is still comfortable room before the blend has
        # to start. A design is never worth arriving late for.
        return remaining > self.autopilot_lead_seconds() * 0.5

    def take_design(self, incoming: TrackAnalysis) -> tuple[dict | None, str]:
        """The finished design for this track, without waiting for it.

        Returns ``(raw_params, reason)``. A design still in flight, or one for
        a different track, counts as absent -- both mean the preset runs.
        """
        slot = self._design
        if slot is None:
            return None, "no design was started"
        if slot["track_id"] != incoming.track_id:
            return None, "design was for a different track"
        thread = slot.get("thread")
        if thread is not None and thread.is_alive():
            return None, "design not ready in time"
        return slot["raw"], slot["reason"]

    def _params_for(
        self,
        choice: "transition.TransitionChoice",
        incoming: TrackAnalysis,
        safety_forced: bool,
        residual: float = 0.0,
    ) -> tuple["transition.TransitionParams", str, str]:
        """Decide the shape of this transition. ``(params, source, note)``.

        The order is the point. Safety first and unconditionally; then the
        operator, if they named a style; then the designed transition, if one
        arrived and survived validation; then the rule-based preset. Every step
        down that list is logged with its reason.
        """
        if safety_forced:
            return (
                transition.preset_params("cut"),
                "safety",
                f"safety override: {choice.rule}",
            )

        preset = transition.preset_params(choice.style)
        if self.transition_style != "auto":
            return preset, "operator", f"style requested: {self.transition_style}"

        raw, reason = self.take_design(incoming)
        if raw is None:
            return self._generated_or(
                preset, choice, incoming, residual, f"no design used ({reason})"
            )

        live = self.engine.deck(self.live_deck)
        outgoing = live.track.analysis if live.track else None
        params, why = self.supervisor.validate_transition_params(
            raw, incoming, outgoing
        )
        if params is None:
            self.session_log.write(
                "transition_design_rejected",
                trigger="llm transition design",
                action=f"rejected: {why}",
                params=raw,
                reason=why,
                fell_back_to=preset.name,
            )
            self.notify(f"[design] rejected ({why}) - using {preset.name}")
            return self._generated_or(
                preset, choice, incoming, residual, f"design rejected: {why}",
                source="rejected",
            )

        return params, "llm", "designed"

    def _mode_for(self, energy_delta: float) -> str:
        """The transition mode for this pair: the operator's, or by context."""
        if self.transition_mode in transition.MODES:
            return self.transition_mode
        persona = planner.PERSONAS.get(self.persona)
        if persona is not None and persona.mode in transition.MODES:
            return persona.mode
        rising = energy_delta > transition.ENERGY_RISE - 1.0
        if self.set_phase == "peak" or self.energy_direction > 0.3 or rising:
            return "showy"
        return "invisible"

    def _pair_context(
        self, choice: "transition.TransitionChoice", incoming: TrackAnalysis,
        residual: float,
    ) -> "transition.PairContext":
        """What the generator is told about this pair. Control thread."""
        live = self.engine.deck(self.live_deck)
        outgoing = live.track.analysis if live.track is not None else None
        if outgoing is None:
            return transition.PairContext(base_style=choice.style)
        out_q = outgoing.mix_out_regions[0]["quality"] if outgoing.mix_out_regions else None
        in_q = incoming.mix_in_regions[0]["quality"] if incoming.mix_in_regions else None
        quality = out_q * in_q if out_q is not None and in_q is not None else None
        position_s = float(live.position) / SAMPLE_RATE
        return transition.PairContext(
            base_style=choice.style,
            bpm_residual=float(residual),
            energy_delta=(
                float(incoming.energy) / float(outgoing.energy) - 1.0
                if outgoing.energy > 0 else 0.0
            ),
            keys_compatible=transition.camelot_compatible(
                outgoing.camelot, incoming.camelot
            ),
            mix_quality=quality,
            outro_falling=(
                transition._in_outro(outgoing, position_s)
                and transition._energy_falling(outgoing, position_s)
            ),
            drops=(
                transition.drop_cue(outgoing) is not None
                and transition.drop_cue(incoming) is not None
            ),
        )

    def _generated_or(
        self,
        preset: "transition.TransitionParams",
        choice: "transition.TransitionChoice",
        incoming: TrackAnalysis,
        residual: float,
        note: str,
        source: str = "generated",
    ) -> tuple["transition.TransitionParams", str, str]:
        """A transition generated for this pair, or the preset if none passes.

        Every candidate goes through the supervisor's own parameter check; one
        it refuses is never returned. Seeded from the set seed, the pair and
        the transition's index, so the same set generates the same shapes.
        """
        import zlib

        live = self.engine.deck(self.live_deck)
        outgoing = live.track.analysis if live.track is not None else None
        ctx = self._pair_context(choice, incoming, residual)
        mode = self._mode_for(ctx.energy_delta)
        seed = zlib.crc32(
            f"{self.set_seed}|{outgoing.track_id if outgoing else ''}|"
            f"{incoming.track_id}|{len(self.shape_history)}".encode()
        )

        def accept(params):
            ok, why = self.supervisor.validate_transition_params(
                dict(params.to_schema()), incoming, outgoing
            )
            return ok is not None, why

        generated = transition.generate_transition(
            ctx, mode, seed, self.shape_history, accept
        )
        if generated is None:
            return preset, "preset", f"{note}; no generated shape passed the supervisor"
        # The supervisor rebuilds params from the schema, dropping the name;
        # the generated object is what runs, having passed the same check.
        self.session_log.write(
            "transition_generated", trigger=note, action=generated.rule,
            shape=generated.shape, mode=mode, seed=seed, score=generated.score,
            params=generated.params.to_schema(),
        )
        return generated.params, source, f"{note}; generated {generated.rule}"

    def abort_armed_transition(self, trigger: str = "deck paused mid-transition") -> int:
        """Drop a transition that was scheduled but has not fired. Returns count.

        Aborting the crossfade in the engine is not enough on its own: the
        commands that were going to start it are still sitting in the
        scheduler, and they would fire into a mix that is no longer the one
        they were planned for. The incoming LoadTrack is dropped with them,
        because on its own it would start a track playing with nothing to
        blend it into.

        Only touches a transition this session armed, so an operator's queued
        drop onto a deck is left alone.
        """
        if not self._transition_armed:
            return 0
        dropped = self.scheduler.cancel_matching(
            lambda c: isinstance(c, (StartTransition, SwapStems))
            or (isinstance(c, LoadTrack) and c.play and not c.is_immediate)
        )
        self._transition_armed = False
        self._cued = None
        if dropped:
            self.session_log.write(
                "transition_aborted",
                trigger=trigger,
                action=f"dropped {len(dropped)} scheduled command(s)",
            )
        return len(dropped)

    def after_stop(self) -> str:
        """Control-thread bookkeeping for a STOP. Call after the command lands.

        The engine has already silenced both decks and put them in a defined
        state. This clears what the engine cannot see: commands still queued,
        which would otherwise fire into a stopped mix, and the autopilot, which
        must not answer a panic by starting a new track over the top of it.

        Deliberately not part of the panic command itself -- that goes straight
        onto the engine queue and shares no code with this layer, which is the
        whole point of the panic path.
        """
        dropped = self.scheduler.cancel_all()
        self._transition_armed = False
        self._cued = None
        self.hold_extra_bars = 0
        self.freeze(True)
        self.session_log.write(
            "stop_recovery",
            trigger="stop",
            action=(
                f"cleared {len(dropped)} queued command(s); decks hold their "
                f"tracks at bar 0 with automation held"
            ),
            dropped=len(dropped),
        )
        return f"Stopped. {len(dropped)} queued command(s) cleared."

    def set_transition_style(self, name: str) -> str:
        """Set the style the next blend uses. Validated, never trusted.

        The one thing the model is allowed to say about transitions is which
        style it wants, so this is the boundary where a name it invented gets
        stopped.
        """
        name = (name or "").strip().lower()
        reason = self.supervisor.validate_transition_style(name)
        if reason is not None:
            return f"Rejected: {reason}"
        self.transition_style = name
        self.session_log.write(
            "transition_style_set",
            trigger=f"style {name}",
            action=f"next transitions use {name}",
        )
        return (
            "Transitions will be chosen by the rules."
            if name == "auto"
            else f"Next transition: {name.replace('_', ' ')}."
        )

    def _phase_arg(self) -> str | None:
        """The set phase as the selector takes it: None when it is auto."""
        return None if self.set_phase == "auto" else self.set_phase

    def set_set_phase(self, name: str) -> str:
        """Set where the set is. Validated here; the model may only name one."""
        from djai.selector import SET_PHASES

        name = (name or "").strip().lower()
        if name not in SET_PHASES + ("auto",):
            return (
                f"Rejected: set phase {name!r} is not one of "
                f"{', '.join(SET_PHASES)}, or auto."
            )
        self.set_phase = name
        self.session_log.write(
            "set_phase",
            trigger=f"phase {name}",
            action=(
                "tracks chosen with no intensity target" if name == "auto"
                else f"next picks aim for a {name} intensity"
            ),
        )
        if name == "auto":
            return "Set phase off: tracks are chosen without an intensity target."
        return f"Set phase: {name}. The next pick aims for it."

    # --- hot cues -------------------------------------------------------------
    #
    # A hot cue is a stored marker inside a track that playback can jump to.
    # It has nothing to do with the cue OUTPUT, which is the headphone monitor
    # channel; the two are kept apart in the names for exactly that reason.

    def _persist_hot_cues(self, analysis: TrackAnalysis) -> None:
        """Write the track's sidecar back so cues survive a restart."""
        try:
            an.write_sidecar(analysis, self.cache_dir)
        except OSError as exc:
            self.notify(f"[hot cue] could not save to the sidecar: {exc}")

    def hot_cues(self, deck_name: str) -> list[dict]:
        deck = self.engine.deck(deck_name)
        if deck.track is None:
            return []
        return list(deck.track.analysis.hot_cues)

    def set_hot_cue(self, deck_name: str, index: int, label: str = "") -> str:
        """Store the deck's current playhead as hot cue ``index``."""
        deck = self.engine.deck(deck_name)
        if deck.track is None:
            return f"Deck {deck_name.upper()} is empty."
        if not 1 <= int(index) <= an.MAX_HOT_CUES:
            return f"Hot cue index must be 1..{an.MAX_HOT_CUES}."
        analysis = deck.track.analysis
        seconds = deck.position / SAMPLE_RATE
        cue = an.make_hot_cue(int(index), seconds, label or f"cue {index}")
        cues = [c for c in analysis.hot_cues if c.get("index") != int(index)]
        cues.append(cue)
        cues.sort(key=lambda c: c["index"])
        analysis.hot_cues = cues
        self._persist_hot_cues(analysis)
        self.session_log.write(
            "hot_cue_set", deck=deck_name, track=analysis.title,
            trigger=f"set hot cue {index}",
            action=f"stored at {seconds:.2f}s",
        )
        return f"Hot cue {index} set at {seconds:.1f}s on deck {deck_name.upper()}."

    def clear_hot_cue(self, deck_name: str, index: int) -> str:
        deck = self.engine.deck(deck_name)
        if deck.track is None:
            return f"Deck {deck_name.upper()} is empty."
        analysis = deck.track.analysis
        before = len(analysis.hot_cues)
        analysis.hot_cues = [
            c for c in analysis.hot_cues if c.get("index") != int(index)
        ]
        if len(analysis.hot_cues) == before:
            return f"Deck {deck_name.upper()} has no hot cue {index}."
        self._persist_hot_cues(analysis)
        self.session_log.write(
            "hot_cue_cleared", deck=deck_name, track=analysis.title,
            trigger=f"clear hot cue {index}", action="removed",
        )
        return f"Hot cue {index} cleared on deck {deck_name.upper()}."

    def jump_to_hot_cue(self, deck_name: str, index: int) -> str:
        """Jump the deck's playhead to a hot cue, on the grid.

        Two rules, both load-bearing. The landing point is snapped to the
        nearest downbeat, because a cue that fires half a bar out is worse than
        no cue. And on a deck that is playing, the jump itself is scheduled for
        that deck's next downbeat, so it leaves on a bar line and lands on a
        bar line -- which is what keeps its phase relationship to the master
        clock intact rather than displacing it by the jump distance.

        Refused outright mid-transition: the crossfade is driving both decks'
        gains against a fixed length, and moving one of them under it produces
        a blend of two different places in the same track.
        """
        deck = self.engine.deck(deck_name)
        if deck.track is None:
            return f"Deck {deck_name.upper()} is empty."
        analysis = deck.track.analysis
        cue = next(
            (c for c in analysis.hot_cues if c.get("index") == int(index)), None
        )
        if cue is None:
            return f"Deck {deck_name.upper()} has no hot cue {index}."

        if self.engine.deck_in_transition(deck_name) or self.engine.transition_active:
            self.session_log.write(
                "hot_cue_refused", deck=deck_name, track=analysis.title,
                trigger=f"jump to hot cue {index}",
                action="refused: a transition is in flight",
            )
            self.notify(
                f"[hot cue] jump refused - deck {deck_name.upper()} is "
                f"mid-transition"
            )
            return f"Refused: deck {deck_name.upper()} is mid-transition."

        target_s = an.cue_seconds(cue)
        if analysis.downbeats:
            target_s = min(analysis.downbeats, key=lambda d: abs(d - target_s))
        start_frame = int(round(target_s * SAMPLE_RATE))

        if deck.playing:
            at_deck = phrase.next_downbeat(deck)
            execute_at = phrase.deck_frame_to_engine_frame(
                deck, self.engine.frames_played, at_deck
            )
        else:
            execute_at = IMMEDIATE

        cmd = LoadTrack(
            deck=deck_name, track=deck.track, start_frame=start_frame,
            rate=deck.rate, play=deck.playing, execute_at=execute_at,
            origin=f"hot_cue:{index}",
        )
        if self.supervisor.validate(cmd) is not None:
            return f"Hot cue {index} was rejected by the supervisor."
        self.scheduler.submit(cmd)
        self.session_log.write(
            "hot_cue_jump", deck=deck_name, track=analysis.title,
            trigger=f"jump to hot cue {index}",
            action=(
                f"snapped to downbeat at {target_s:.2f}s"
                + ("" if not deck.playing else ", on this deck's next downbeat")
            ),
        )
        return (
            f"Deck {deck_name.upper()} -> hot cue {index} "
            f"({cue.get('label', '')}) at {target_s:.1f}s."
        )

    # --- beat grid correction -------------------------------------------------
    #
    # A person's correction of a track's tempo or bar line, persisted with
    # `grid_manually_corrected` so analysis never overwrites it. Tempo changes
    # are refused while the deck is in a blend: the other deck is matched to
    # this one's grid, and rewriting it mid-mix would break the sync. Every
    # refusal starts with "Refused" so the UI can tell one from a result.

    def _grid_deck(self, deck_name: str) -> tuple[Any, str | None]:
        if deck_name not in ("a", "b"):
            return None, f"Refused: unknown deck {deck_name!r}."
        deck = self.engine.deck(deck_name)
        if deck.track is None:
            return None, f"Refused: deck {deck_name.upper()} is empty."
        return deck, None

    def _blending(self, deck: Any) -> bool:
        other = self.engine.deck("b" if deck.name == "a" else "a")
        return bool(deck.playing and other.playing)

    def _apply_grid(self, deck: Any, bpm: float, first_downbeat: float, what: str) -> str:
        analysis = deck.track.analysis
        before = analysis.bpm
        an.regrid(analysis, bpm, first_downbeat)
        self._persist_hot_cues(analysis)  # writes the whole cache entry
        # A corrected master deck retunes the beat clock, and every phase
        # baseline captured against the old grid is dropped.
        self.engine.retune_master(deck.name)
        self.supervisor.forget_baseline("a")
        self.supervisor.forget_baseline("b")
        self.session_log.write(
            "grid_corrected",
            deck=deck.name,
            track=analysis.title,
            trigger=what,
            action=(
                f"{before:.3f} -> {analysis.bpm:.3f} BPM, "
                f"bar 1 at {analysis.first_downbeat:.3f}s"
            ),
        )
        return (
            f"Deck {deck.name.upper()} grid {what}: {analysis.bpm:.2f} BPM, "
            f"bar 1 at {analysis.first_downbeat:.3f}s. Saved."
        )

    def _refuse_mid_blend(self, deck: Any) -> str | None:
        if self._blending(deck):
            return (
                f"Refused: deck {deck.name.upper()} is in a blend. Change its "
                "tempo when it plays alone or is stopped; nudge works any time."
            )
        return None

    def _tempo_change(self, deck_name: str, factor: float, what: str) -> str:
        deck, refused = self._grid_deck(deck_name)
        if refused is None:
            refused = self._refuse_mid_blend(deck)
        if refused is not None:
            return refused
        analysis = deck.track.analysis
        bpm = analysis.bpm * factor
        if not 20.0 <= bpm <= 400.0:
            return f"Refused: {bpm:.1f} BPM is not a usable tempo."
        return self._apply_grid(deck, bpm, analysis.first_downbeat, what)

    def grid_halve(self, deck_name: str) -> str:
        return self._tempo_change(deck_name, 0.5, "halved")

    def grid_double(self, deck_name: str) -> str:
        return self._tempo_change(deck_name, 2.0, "doubled")

    def grid_nudge(self, deck_name: str, ms: float) -> str:
        """Shift every bar line by ``ms``. Allowed mid-blend: tempo is unchanged."""
        deck, refused = self._grid_deck(deck_name)
        if refused is not None:
            return refused
        ms = float(ms)
        if not abs(ms) <= 1000.0:  # also refuses NaN
            return "Refused: nudge by at most 1000 ms."
        analysis = deck.track.analysis
        return self._apply_grid(
            deck, analysis.bpm, analysis.first_downbeat + ms / 1000.0,
            f"nudged {ms:+.0f} ms",
        )

    def grid_set_downbeat(self, deck_name: str, seconds: float) -> str:
        """Put bar 1 at ``seconds``: the draggable handle in the grid view."""
        deck, refused = self._grid_deck(deck_name)
        if refused is not None:
            return refused
        seconds = float(seconds)
        analysis = deck.track.analysis
        if not 0.0 <= seconds <= analysis.duration_s:
            return "Refused: bar 1 has to be inside the track."
        return self._apply_grid(
            deck, analysis.bpm, seconds, f"bar 1 moved to {seconds:.3f}s"
        )

    def grid_tap(self, deck_name: str) -> str:
        """One tap. From the eighth steady tap on, each one sets the tempo."""
        deck, refused = self._grid_deck(deck_name)
        if refused is not None:
            return refused
        now = time.monotonic()
        taps = self._taps.setdefault(deck_name, [])
        if taps and now - taps[-1] > TAP_RESET_S:
            taps.clear()
        taps.append(now)
        del taps[:-32]
        bpm = an.bpm_from_taps(taps)
        if bpm is None:
            if len(taps) < an.MIN_TAPS:
                return f"Tap {len(taps)}/{an.MIN_TAPS}: keep tapping on the beat."
            return "Taps are uneven - keep tapping steadily."
        refused = self._refuse_mid_blend(deck)
        if refused is not None:
            return refused
        return self._apply_grid(
            deck, bpm, deck.track.analysis.first_downbeat,
            f"tapped ({len(taps)} taps)",
        )

    # --- transitions ---------------------------------------------------------

    @staticmethod
    def _drop_entry_frame(track_b, lead_bars: float) -> int | None:
        """Where deck B enters so its drop lands ``lead_bars`` into the window.

        The earliest drop that leaves room for the run-up before it and the
        usual runway after it, or None. The runway rule is the one both other
        routes into a deck obey; a drop-aligned entry is a third route, and a
        deck that enters near its own mix-out runs out into dead air.

        Shared by placement and by track selection, so a track picked for its
        drop is judged by exactly the rule that will later place it.
        """
        if track_b.bpm <= 0:
            return None
        bar_frames = 4 * 60.0 / track_b.bpm * SAMPLE_RATE
        for position in Session._drop_frames(track_b):
            entry = int(round(position - lead_bars * bar_frames))
            if entry < 0:
                continue
            if not transition.entry_has_runway(track_b, entry / SAMPLE_RATE):
                continue
            return entry
        return None

    @staticmethod
    def _drop_frames(track) -> list[float]:
        """Every drop in a track, in deck frames, earliest first.

        Labelled drop sections (Phase 3 structure) and hot cues labelled
        "drop", merged: a section start within a bar of a cue is the same
        drop, and the cue's position is kept.
        """
        cues = sorted(
            float(c["sample_position"])
            for c in (getattr(track, "hot_cues", None) or [])
            if c.get("label") == "drop"
        )
        bar_frames = 4 * 60.0 / track.bpm * SAMPLE_RATE if track.bpm > 0 else SAMPLE_RATE * 2
        found = list(cues)
        for section in getattr(track, "sections", None) or []:
            if section.get("label") != "drop":
                continue
            frame = float(phrase.frame_at_bar(track, float(section["start_bar"])))
            if frame >= 0 and all(abs(frame - c) > bar_frames for c in found):
                found.append(frame)
        return sorted(found)

    @staticmethod
    def _drop_lead_bars(params, bars: float) -> float:
        """Bars from the window's start to the moment the drops coincide.

        A double drop lines them up at the start, because the window IS the
        passage where they play together; everything else at the end.
        """
        return 0.0 if params.double_drop_bars > 0 else float(bars)

    def _plan_drop_aligned(
        self, live, track_b, params, bars: float
    ) -> "phrase.TransitionPlan | None":
        """Place a window so both tracks' drops land on the same frame.

        ``align_mode="drop"`` is the whole of what makes `drop_swap` and
        `double_drop` the things they are, and this is the only code that works
        out where such a window goes. The supervisor has already refused any
        design that asks for it without a drop hot cue on both tracks, so the
        cues can be relied on to exist; the model never sees either position.

        A double drop wants the two drops to coincide at the START of the
        window, because the window IS the passage where they play together.
        Everything else wants them at the END: the drop is the arrival that the
        run-up has been building towards.

        Every drop cue on each track is a candidate, earliest first. Deck A's
        must still be far enough ahead, pass the supervisor's early-start
        guard, and leave the window finished before A's mix-out. Deck B's must
        leave room for the run-up before it and the usual runway after it.

        Returns None when no pair qualifies, so the caller can fall back to
        phrase placement and say that it did.
        """
        analysis_a = live.track.analysis
        if analysis_a.bpm <= 0 or track_b.bpm <= 0:
            return None

        drops = self._drop_frames

        # Drop-aligned placement has its own, lower guard -- and only it does.
        from djai.supervisor import DROP_ALIGNED_MIN_START_FRACTION

        lead_bars = self._drop_lead_bars(params, bars)
        bar_frames_a = 4 * 60.0 / analysis_a.bpm * SAMPLE_RATE
        total_a = analysis_a.duration_s * SAMPLE_RATE
        mix_out_a = analysis_a.mix_out * SAMPLE_RATE

        # Deck A: the first drop whose window can still be had. Tracks often
        # carry more than one, and the autopilot arms late in a track, so the
        # first drop has usually gone by -- found in a soak, where a double drop
        # gave up on a passed drop at bar 38 with a usable one waiting at bar 85.
        start_frame = None
        drop_bar_a = 0
        for position in drops(analysis_a):
            # Snapped to the downbeat nearest the marker. A drop cue normally
            # sits on one already; rounding keeps it there when it is a few
            # milliseconds off, rather than shifting the window by a beat.
            candidate_bar = round(phrase.bar_at_frame(analysis_a, position))
            candidate = int(round(
                phrase.frame_at_bar(analysis_a, candidate_bar - lead_bars)
            ))
            # A bar of slack, so there is room for the command to be queued and
            # quantised before the frame it is meant to fire on.
            if candidate < live.position + bar_frames_a:
                continue
            # The supervisor refuses an automatic transition that starts too
            # early in the outgoing track. For drop-aligned placement that
            # guard is relaxed to its floor, and no further; a window below the
            # floor is passed over here instead.
            if total_a > 0 and candidate / total_a < DROP_ALIGNED_MIN_START_FRACTION:
                continue
            # And deck A has to still be playing when the window closes.
            if mix_out_a > 0 and candidate + bars * bar_frames_a > mix_out_a:
                continue
            start_frame, drop_bar_a = candidate, candidate_bar
            break
        if start_frame is None:
            return None

        entry_frame_b = self._drop_entry_frame(track_b, lead_bars)
        if entry_frame_b is None:
            return None
        return phrase.TransitionPlan(
            start_frame=start_frame,
            bars=bars,
            entry_frame_b=entry_frame_b,
            reason=(
                f"drops aligned: deck {self.live_deck.upper()} at bar "
                f"{drop_bar_a}, incoming from {entry_frame_b / SAMPLE_RATE:.1f}s"
            ),
        )

    def arm_transition(
        self, origin: str = "autopilot", at_next_phrase: bool = False,
        at_bars: int | None = None,
    ) -> int | None:
        """Schedule the bass swap so it *finishes* at the live deck's mix-out.

        Placement comes from :func:`djai.phrase.plan_transition`, which works
        backwards from mix-out. It is deliberately not
        :func:`~djai.phrase.next_phrase_boundary`: that answers "when is the
        next musically valid moment", and scheduling forwards from whenever a
        track happened to be selected is what left hand-offs anywhere between
        84% and 102% of the outgoing track.

        Returns the engine frame it will fire at, or None if it could not be
        armed.
        """
        self._complete_handover()
        live = self.engine.deck(self.live_deck)
        idle_name = self.cued_deck()
        incoming = self._cued
        if live.track is None or incoming is None:
            return None

        # Style first: it decides how long the blend is, and the length is an
        # input to placement, not something that can be retrofitted after it.
        # The tempo path is recomputed here rather than reused from the cue:
        # a ride scheduled then may have partly or wholly landed by now, and
        # the master clock is what says where the mix actually is.
        path = self.tempo_path_to(incoming.analysis)
        # The tempo the clock will be counting when the blend starts, which
        # after a ride still in flight is not the one it counts now. Arming
        # against today's number and blending at tomorrow's is how the two
        # decks walk apart while they are both audible.
        master_bpm = self._planned_master_bpm() or (
            live.track.analysis.bpm * max(live.rate, 1e-6)
        )
        ratio = path.ratio if path.blendable else 1.0
        counted = incoming.analysis.bpm * ratio
        matched_rate = master_bpm / counted if counted > 0 else 1.0
        limit = config.MAX_STRETCH_RATIO
        rate_b = min(max(matched_rate, 1.0 - limit), 1.0 + limit)
        # What is left after the deck has stretched as far as it may: the ride
        # in flight closes the rest, and this is the gap a style rule should
        # judge rather than the raw difference between two printed tempos.
        residual = (
            abs(counted * rate_b - master_bpm) / master_bpm if master_bpm > 0 else 1.0
        )
        choice = transition.choose_transition(
            live, incoming.analysis,
            {"style": self.transition_style,
             "bpm_delta": residual if path.blendable else None},
        )

        # Safety beats the designer, unconditionally. A weak grid or a tempo
        # gap too wide to blend forces a cut whatever anyone asked for.
        safety_forced = choice.style == "cut" and (
            "grid confidence" in choice.rule or "BPM delta" in choice.rule
        )
        conservative = self._conservative.get(incoming.analysis.track_id)
        if conservative is not None:
            # An operator's cue the grid or the tempo cannot hold a long blend
            # on: an echo out into its clean mix-in, never a cut (Phase 3.2).
            safety_forced = False
            params = transition.preset_params("echo_out")
            source, design_note = "cue", conservative
            choice = choice._replace(style="echo_out", rule=f"operator cue: {conservative}")
        elif at_bars and not safety_forced:
            # "Play now": a short showy gesture, generated like any other.
            ctx = self._pair_context(choice, incoming.analysis, residual)
            outgoing = live.track.analysis
            short = transition.generate_transition(
                ctx, "showy", len(self.shape_history), self.shape_history,
                lambda prm: (
                    prm.length_bars <= at_bars and self.supervisor.validate_transition_params(
                        prm.to_schema(), incoming.analysis, outgoing)[0] is not None, ""),
            )
            params = short.params if short else transition.preset_params("echo_out")
            source, design_note = "play_now", (short.rule if short else "echo_out preset")
        else:
            params, source, design_note = self._params_for(
                choice, incoming.analysis, safety_forced,
                residual if path.blendable else 0.0,
            )
        if params.name == "cut":
            choice = choice._replace(style="cut")

        #: Set when the window was placed by lining the two tracks' drops up,
        #: which also settles where deck B enters: the cue-based entry rules
        #: below must not then move it.
        drop_aligned = False

        if choice.style == "cut":
            # A hard switch belongs on a bar line, not at the end of a fade
            # that is not happening. One block, on the next downbeat.
            plan = phrase.TransitionPlan(
                start_frame=phrase.next_downbeat(live),
                bars=0.0,
                entry_frame_b=int(round(incoming.analysis.mix_in * SAMPLE_RATE)),
            )
        else:
            # The design's own length, which is an INPUT to placement. Where
            # the transition sits is still worked out backwards from mix-out by
            # plan_transition; nothing here decides that.
            requested_bars = params.length_bars + self.hold_extra_bars
            if self.timing_bias_bars:
                # Learned from "too early" / "too late": a shorter blend, planned
                # back from mix-out, starts later. Never shorter than the floor
                # a blend needs, nor than the design itself asked for.
                floor = min(phrase.MIN_TRANSITION_BARS, params.length_bars)
                requested_bars = max(floor, requested_bars + self.timing_bias_bars)
            if at_bars:
                # The next line of `at_bars` bars at least a bar away: close
                # enough to be "now", far enough to schedule on a bar line.
                analysis_a = live.track.analysis
                bar = phrase.bar_at_frame(analysis_a, live.position)
                target = (math.floor(bar / at_bars) + 1) * at_bars
                if target - bar < 1.0:
                    target += at_bars
                plan = phrase.TransitionPlan(
                    start_frame=int(round(phrase.frame_at_bar(analysis_a, target))),
                    bars=params.length_bars,
                    entry_frame_b=int(round(incoming.analysis.mix_in * SAMPLE_RATE)),
                    reason=f"play now: next {at_bars}-bar line",
                )
            elif at_next_phrase:
                # "Go now": the user asked, so this is an alignment question,
                # not a placement one. Next musically valid moment, requested
                # length, mix-out ignored. This is the only caller of
                # next_phrase_boundary, and it stays that way.
                plan = phrase.TransitionPlan(
                    start_frame=phrase.next_phrase_boundary(live),
                    bars=requested_bars,
                    entry_frame_b=int(
                        round(incoming.analysis.mix_in * SAMPLE_RATE)
                    ),
                )
            elif params.align_mode == "drop":
                plan = self._plan_drop_aligned(live, incoming.analysis, params,
                                               requested_bars)
                if plan is None:
                    # Say so rather than running a "drop swap" that is not one.
                    align_note = "drop alignment unavailable, placed on a phrase"
                    design_note = f"{design_note}; {align_note}"
                    self.notify(f"[autopilot] {align_note}")
                    plan = phrase.plan_transition(
                        live, incoming.analysis, requested_bars
                    )
                else:
                    drop_aligned = True
            else:
                plan = None
                if (
                    self.transition_style == "auto"
                    and live.track.analysis.sections_labelled("drop")
                    and incoming.analysis.sections_labelled("drop")
                ):
                    # Both tracks have labelled drops: the default is drop to
                    # drop. Falls through to outro-over-intro / mix-out
                    # placement when no pair of drops can be lined up.
                    plan = self._plan_drop_aligned(
                        live, incoming.analysis, params, requested_bars
                    )
                    if plan is not None:
                        drop_aligned = True
                if plan is None:
                    plan = phrase.plan_transition(
                        live, incoming.analysis, requested_bars
                    )
        if plan is None:
            return None

        execute_at = phrase.deck_frame_to_engine_frame(
            live, self.engine.frames_played, plan.start_frame
        )
        bpm = self.engine.master_bpm or live.track.analysis.bpm
        if choice.style == "cut":
            bars = 0.0
            total_frames = self.engine.blocksize
        else:
            bars = plan.bars if plan.bars > 0 else transition.TRANSITION_BARS
            total_frames = transition.transition_frames(
                bpm, SAMPLE_RATE, transition_bars=bars
            )

        # Entry method: where deck B starts from. Independent of the curve.
        # A designed entry_point wins over the rule-based one, having already
        # been checked against the cues this track actually has.
        entry_frame_b = plan.entry_frame_b
        entry_note = "its mix_in"
        cue_index = None if drop_aligned else params.hot_cue_index()
        if drop_aligned:
            entry_note = (
                f"{entry_frame_b / SAMPLE_RATE:.1f}s, placed so its drop lands "
                f"on deck {self.live_deck.upper()}'s"
            )
        elif choice.entry == "cue_jump_in" and cue_index is None:
            cue_index = choice.hot_cue_index
        if cue_index is not None:
            cue = next(
                (c for c in incoming.analysis.hot_cues
                 if c.get("index") == cue_index),
                None,
            )
            if cue is not None:
                entry_frame_b = int(cue["sample_position"])
                entry_note = (
                    f"hot cue {cue['index']} ({cue.get('label', '')}) at "
                    f"{entry_frame_b / SAMPLE_RATE:.1f}s"
                )

        # Two vocals over each other is refused by the supervisor, whatever
        # asked for the transition. A vocal-aware design first tries to move
        # out of the way -- deck A starting earlier, or deck B entering later --
        # and only if nothing works does the mix fall back to a cut.
        if choice.style != "cut" and bars > 0:
            env = self._envelope_for(params, total_frames, bpm)
            clash = self._clash(live, incoming, plan.start_frame, entry_frame_b,
                                bars, env, params.name, log=True)
            if clash is not None:
                # Stems first: dropping deck A's vocals for the blend removes
                # the clash without moving the placement or falling back to a
                # cut, which is what the brief asks for.
                stem_fix = self._stem_clash_fix(live, plan, bars)
                if stem_fix is not None:
                    entry_note = f"{entry_note} ({stem_fix})"
                    design_note = f"{design_note}; {stem_fix}"
                    self.session_log.write(
                        "vocal_clash_avoided", trigger=params.name, action=stem_fix,
                        reason=clash,
                    )
                    clash = None
            if clash is not None:
                moved = (
                    self._avoid_vocal_clash(live, incoming, plan.start_frame,
                                            entry_frame_b, bars, env, params.name,
                                            drop_aligned)
                    if params.vocal_aware else None
                )
                if moved is not None:
                    start_frame, entry_frame_b, move_note = moved
                    plan = phrase.TransitionPlan(
                        start_frame=start_frame, bars=plan.bars,
                        entry_frame_b=entry_frame_b,
                        reason=f"{plan.reason + '; ' if plan.reason else ''}{move_note}",
                    )
                    execute_at = phrase.deck_frame_to_engine_frame(
                        live, self.engine.frames_played, plan.start_frame
                    )
                    entry_note = f"{entry_note} ({move_note})"
                    self.session_log.write(
                        "vocal_clash_avoided", trigger=params.name, action=move_note
                    )
                else:
                    # A cut where the blend would have finished: the hand-off
                    # stays where it was planned, just without the overlap.
                    analysis_a = live.track.analysis
                    end_bar = phrase.bar_at_frame(analysis_a, plan.start_frame) + bars
                    cut_frame = int(round(phrase.frame_at_bar(analysis_a, math.floor(end_bar))))
                    params = transition.preset_params("cut")
                    choice = choice._replace(style="cut")
                    design_note = f"{design_note}; vocal clash, cut instead"
                    plan = phrase.TransitionPlan(
                        start_frame=cut_frame, bars=0.0,
                        entry_frame_b=int(round(incoming.analysis.mix_in * SAMPLE_RATE)),
                        reason="vocal clash could not be avoided; hard cut at the planned hand-off",
                    )
                    execute_at = phrase.deck_frame_to_engine_frame(
                        live, self.engine.frames_played, plan.start_frame
                    )
                    bars = 0.0
                    total_frames = self.engine.blocksize
                    entry_frame_b = plan.entry_frame_b
                    entry_note = "its mix_in (cut after a vocal clash)"
                    drop_aligned = False
                    self.session_log.write(
                        "vocal_clash_fallback", trigger="vocal clash",
                        action="hard cut at the planned hand-off", reason=clash,
                    )

        # Build the envelope now, on this thread. The audio thread never does.
        # Presets keep their own builders so they stay bit-identical to what
        # they have always produced; only designed shapes go the other way.
        if params.name in transition.STYLES:
            armed_style = self.engine.arm_transition_plan(
                params.name, total_frames, bpm
            )
        else:
            armed_style = self.engine.arm_transition_params(
                params, total_frames, bpm
            )
        self.last_transition_choice = (
            armed_style, f"{choice.rule} [{source}: {design_note}]"
        )
        self.design_counts[source] += 1
        self.style_counts[armed_style] += 1
        self.shape_history.append(transition.shape_signature(params))

        # Beyond the stretch range nothing can be matched, so the incoming deck
        # plays at its own tempo and takes the beat clock with it: the outgoing
        # deck is one block from being silent, and leaving the clock behind
        # would have the supervisor resyncing the new track against a tempo
        # nothing is playing.
        unmatched = not path.blendable
        load = LoadTrack(
            deck=idle_name,
            track=incoming,
            # Its mix-in, never its first sample: entering on an intro that has
            # no beat is the other half of the placement bug.
            start_frame=entry_frame_b,
            rate=1.0 if unmatched else rate_b,
            play=True,
            execute_at=execute_at,
            origin=origin,
            master=unmatched,
        )
        swap = StartTransition(
            from_deck=self.live_deck,
            to_deck=idle_name,
            total_frames=total_frames,
            execute_at=execute_at,
            origin=origin,
            drop_aligned=drop_aligned,
        )
        if self.supervisor.validate(load) is not None:
            return None
        # The mid-track guard lives on StartTransition, so it has to be checked
        # too -- validating only the load would leave the regression uncovered.
        if self.supervisor.validate(swap) is not None:
            return None
        self._drop_late_ride_steps(swap.execute_at)
        #: The tempo the set is at going into this transition, so the hand-over
        #: at the far end can say what the journey actually moved.
        self._clock_at_arm = master_bpm
        # Recorded BEFORE the commands are queued: the pre-roll hook can fire as
        # soon as the swap is in the scheduler, and it identifies the transition
        # it is allowed to touch by these exact command objects.
        self._armed_preview = {
            "load": load,
            "swap": swap,
            "params": params,
            "source": source,
            "bars": bars,
            "bpm": bpm,
            "start_frame": plan.start_frame,
            "drop_aligned": drop_aligned,
            "previewed": False,
        }
        if self._transition_armed:
            # Re-armed -- `go` over a blend the autopilot already queued. The
            # earlier pair is withdrawn, or both fire and the second lands in
            # the first's blend (found by the Phase 2.3 fuzz).
            self.scheduler.cancel_matching(
                lambda c: isinstance(c, StartTransition)
                or (isinstance(c, LoadTrack) and c.play and not c.is_immediate)
            )
        self.scheduler.submit(load)
        self.scheduler.submit(swap)

        # The blend owns both decks from here. A plan or an energy move queued
        # before it was armed must not land in it: a fader step on a deck in a
        # running blend ends the blend part-way (manual level outranks
        # automation), and the Phase 2.3 fuzz traced a silent set to exactly
        # that. Withdrawn, with any gesture they opened closed.
        for token in (self._action_token, self._energy_token):
            if token is not None:
                for cmd in glue.closing_commands(self.scheduler.cancel(token)):
                    self.scheduler.submit(cmd)

        self._transition_armed = True
        # Stem moves are planned only now: they are swaps around a transition
        # that is already queued, never a reason for one to exist.
        try:
            self._start_stem_moves(live, incoming, params, plan, bars)
        except Exception as exc:  # noqa: BLE001 - the transition stands
            self._pending_stem_moves = []
            log.warning("stem move planning failed: %s", exc)
        if plan.reason:
            self.notify(f"[autopilot] {plan.reason}")
        self.session_log.write(
            "transition_armed",
            track_pair=[live.track.analysis.title, incoming.analysis.title],
            position_bars=round(
                phrase.bar_at_frame(live.track.analysis, live.position), 2
            ),
            trigger=origin,
            action=(
                f"{armed_style} over {bars:.0f} bar(s) at engine frame "
                f"{execute_at}, ending at bar "
                f"{phrase.bar_at_frame(live.track.analysis, plan.start_frame) + bars:.1f} "
                f"(mix_out bar {live.track.analysis.mix_out_bar:.1f}); "
                f"incoming enters at {entry_note}"
            ),
            plan_reason=plan.reason,
            is_cut=plan.is_cut or armed_style == "cut",
            # Every transition says which style ran and which rule chose it.
            transition_style=armed_style,
            transition_entry=choice.entry,
            style_rule=choice.rule,
            # Everything needed to debug a transition that sounded wrong:
            # what was asked for, what the model returned, whether it survived
            # validation, and what actually ran.
            requested_energy=round(float(self.energy_direction), 2),
            design_source=source,
            design_note=design_note,
            design_raw=(self._design or {}).get("raw"),
            params_used=params.to_schema(),
        )
        self.notify(
            f"[autopilot] {armed_style} into {incoming.analysis.title} "
            f"- {choice.rule}"
        )
        return execute_at

    # --- stem moves -----------------------------------------------------------

    def stems_ready(self, analysis: TrackAnalysis) -> bool:
        """Are separated stems on disk for this track, and are they wanted?"""
        if not self.stems_enabled:
            return False
        return stems_mod.cached(analysis.track_id, self.cache_dir) is not None

    def _stem_clash_fix(self, live, plan, bars: float) -> str | None:
        """Plan deck A's vocals out of the blend. Returns a note, or None.

        Only decides; :meth:`_start_stem_moves` does the decoding, once the
        transition is actually queued.
        """
        if live.track is None or bars <= 0:
            return None
        if not self.stems_ready(live.track.analysis):
            return None
        self._pending_stem_moves.append({
            "deck": self.live_deck,
            "mix": "instrumental",
            "deck_frame": float(plan.start_frame),
            "reason": "two vocals: deck A plays instrumental through the blend",
        })
        return "vocal clash avoided by a stem move (deck A instrumental)"

    def _start_stem_moves(self, live, incoming, params, plan, bars: float) -> None:
        """Decide and schedule this transition's stem moves. Control thread.

        Each move is a buffer swap on a bar line, so the audio callback still
        reads one buffer per deck. The decode and the sum happen on a worker
        thread: if a mix is not ready in time the move is skipped and the
        transition runs on EQ alone, exactly as it did before stems existed.
        """
        moves = list(self._pending_stem_moves)
        self._pending_stem_moves = []
        idle_name = self.cued_deck()
        a = live.track.analysis if live.track is not None else None
        b = incoming.analysis
        if a is None or bars <= 0 or params.name == "cut":
            self._pending_stem_moves = []
            return
        have_a = self.stems_ready(a)
        have_b = self.stems_ready(b)
        start_bar = phrase.bar_at_frame(a, plan.start_frame)
        entry_bar_b = phrase.bar_at_frame(b, plan.entry_frame_b)
        a_moved = {m["deck"] for m in moves}

        # Acapella over the next instrumental: deck A is singing, deck B enters
        # without vocals of its own. Only when nothing else has claimed deck A.
        if (
            have_a and have_b and not a_moved and bars >= 8
            and a.has_vocals_between(start_bar, start_bar + bars)
            and not b.has_vocals_between(entry_bar_b, entry_bar_b + bars)
        ):
            moves.append({
                "deck": self.live_deck, "mix": "acapella",
                "deck_frame": float(plan.start_frame),
                "reason": "acapella over the incoming instrumental",
            })
            moves.append({
                "deck": idle_name, "mix": "instrumental",
                "deck_frame": float(plan.entry_frame_b),
                "reason": "incoming plays instrumental under the acapella",
            })
            a_moved.add(self.live_deck)

        # A clean bass hand-over, and then the groove: the low end leaves deck A
        # at the swap bar the style already uses, and its drums four bars later,
        # rather than both being EQ'd away.
        swap_bar = params.low_swap_bar if params.low_swap_bar is not None else bars / 2
        if have_a and self.live_deck not in a_moved and bars >= 8:
            moves.append({
                "deck": self.live_deck, "mix": "no_bass",
                "deck_frame": float(phrase.frame_at_bar(a, start_bar + swap_bar)),
                "reason": f"clean bass hand-over at bar {swap_bar:.0f} of the blend",
            })
            if bars >= 12 and swap_bar + 4 < bars:
                moves.append({
                    "deck": self.live_deck, "mix": "vocals_and_other",
                    "deck_frame": float(phrase.frame_at_bar(a, start_bar + swap_bar + 4)),
                    "reason": "drum swap: the incoming groove carries from here",
                })

        if not moves:
            self.stem_moves = []
            return
        # Whatever was swapped on the incoming deck goes back to its own master
        # at the hand-over: the stems do not sum back to it exactly.
        for move in list(moves):
            if move["deck"] == idle_name:
                moves.append({
                    "deck": idle_name, "mix": "original",
                    "deck_frame": float(
                        plan.entry_frame_b + bars * 4 * b.beat_period * SAMPLE_RATE
                    ),
                    "reason": "back to the master once the blend is over",
                })
                break

        self.stem_moves = moves
        originals = {
            self.live_deck: live.track,
            idle_name: incoming,
        }
        thread = threading.Thread(
            target=self._run_stem_moves, args=(moves, originals),
            name="djai-stem-moves", daemon=True,
        )
        thread.start()

    #: A stem swap this close to its bar line is not worth landing late.
    STEM_MOVE_MARGIN_S: float = 0.5

    def _run_stem_moves(self, moves: list[dict], originals: dict) -> None:
        """Build each mix and queue its swap. WORKER THREAD; decodes FLAC."""
        for move in moves:
            deck_name = move["deck"]
            original = originals.get(deck_name)
            if original is None:
                continue
            try:
                if move["mix"] == "original":
                    mix_track = self._stem_origin.get(deck_name, original)
                else:
                    mix_track = stems_mod.load_mix(original, self.cache_dir, move["mix"])
            except Exception as exc:  # noqa: BLE001 - EQ alone is the fallback
                self.notify(f"[stems] {move['mix']} failed: {type(exc).__name__}: {exc}")
                log.warning("stem mix %s failed: %s", move["mix"], exc)
                continue
            if mix_track is None:
                continue
            deck = self.engine.deck(deck_name)
            if deck.track is None or deck.track.analysis is not original.analysis:
                continue  # that deck has moved on; this mix is for old music
            self._stem_origin.setdefault(deck_name, original)
            execute_at = phrase.deck_frame_to_engine_frame(
                deck, self.engine.frames_played, move["deck_frame"]
            )
            if (execute_at - self.engine.frames_played) / SAMPLE_RATE < self.STEM_MOVE_MARGIN_S:
                self.session_log.write(
                    "stem_move_skipped", deck=deck_name, trigger=move["mix"],
                    action="the mix was not ready in time; EQ alone",
                )
                continue
            self.scheduler.submit(SwapStems(
                deck=deck_name, mix=move["mix"], track=mix_track,
                execute_at=execute_at, origin="stems",
            ))
            self.session_log.write(
                "stem_move", deck=deck_name, track=original.analysis.title,
                trigger=move["mix"], action=move["reason"],
            )

    def _envelope_for(self, params, total_frames: int, bpm: float):
        """The envelope the engine will run for these params. Control thread."""
        if params.name in transition.STYLES:
            return transition.build_envelope(
                params.name, int(total_frames), self.engine.blocksize, bpm, SAMPLE_RATE
            )
        return transition.build_envelope_from_params(
            params, int(total_frames), self.engine.blocksize, bpm, SAMPLE_RATE
        )

    # --- transition preview (PRE-ROLL WORKER THREAD) -------------------------

    #: Where a transition's shape came from, for the ones the preview leaves
    #: alone: a style the operator named is theirs, and a safety cut is not a
    #: transition to improve.
    PREVIEW_SKIP_SOURCES = ("operator", "safety")

    def _on_pre_roll(self, cmd: Any) -> None:
        """Preview the armed transition, then commit what the preview decided.

        PRE-ROLL WORKER THREAD, handed a command by the scheduler some seconds
        before it is due. Never the tick, never the audio thread. Whatever
        happens in here -- a slow render, a dead model, an exception -- the
        commands already in the scheduler fire on time; the most this can do
        is swap them for better ones while there is still time to.
        """
        if not isinstance(cmd, StartTransition):
            return
        armed = self._armed_preview
        if armed is None or armed["swap"] is not cmd or armed["previewed"]:
            return  # a manual or superseded transition, or already previewed
        armed["previewed"] = True
        params = armed["params"]
        if (
            params.name == "cut"
            or armed["bars"] <= 0
            or armed["source"] in self.PREVIEW_SKIP_SOURCES
        ):
            return
        live = self.engine.deck(cmd.from_deck)
        load = armed["load"]
        if live.track is None or load.track is None:
            return

        self._preview_event("started", params.name)
        # A drop-aligned entry was placed to line the drops up; the preview
        # renders it where it was placed, and nothing it decides may move it.
        preview_params = (
            dataclasses.replace(params, entry_point="mix_in")
            if armed["drop_aligned"] else params
        )
        state = preview_mod.DeckAState(
            analysis=live.track.analysis,
            loaded=live.track,
            transition_at_frame=float(armed["start_frame"]),
        )
        outcome = revise_mod.run_preview(
            state, load.track.analysis, load.track, float(load.start_frame),
            preview_params,
            intent_engine=self.intent_engine,
            supervisor=self.supervisor,
            session_log=self.session_log,
            blocksize=self.engine.blocksize,
            weights=self.critic_weights or None,
        )
        try:
            note = self._commit_preview(cmd, armed, outcome, live)
        except Exception as exc:  # noqa: BLE001 - the armed transition stands
            note = f"commit failed ({type(exc).__name__}: {exc}); armed transition stands"
        outcome.events.append(note)
        self.last_preview = outcome
        self.session_log.write(
            "preview_commit", trigger=f"preview {outcome.status}", action=note,
            elapsed_ms=round(outcome.elapsed_ms, 1),
        )
        self.notify(f"[preview] {outcome.status} in {outcome.elapsed_ms:.0f} ms - {note}")
        self._preview_event("finished", outcome)

    def _commit_preview(self, cmd, armed: dict, outcome, live) -> str:
        """Swap the queued transition for the previewed one, if it is safe to.

        Returns a one-line account of what happened, for the log and the UI.
        """
        margin_s = (cmd.execute_at - self.engine.frames_played) / SAMPLE_RATE
        if margin_s < config.PREVIEW_COMMIT_MARGIN_S:
            return (f"{margin_s:.1f} s to the hand-off is too close to change "
                    f"anything; armed transition stands")
        if outcome.force_cut:
            return self._preview_force_cut(cmd, armed, live)
        if not outcome.changed:
            return "armed transition stands"

        original = armed["params"]
        new = outcome.params
        load = armed["load"]
        notes: list[str] = []

        # A revised preset is no longer the preset, and the arm path builds a
        # named preset from its own builder -- which would quietly throw the
        # revision away. So it is renamed and armed through the params path.
        # A generated shape (SPEC §4) already arms by params; it is renamed
        # too, so the log says the preview changed it.
        if new.name in transition.STYLES or new.name.startswith("gen_"):
            new = dataclasses.replace(new, name=f"{new.name}+preview")
        if armed["drop_aligned"]:
            new = dataclasses.replace(new, entry_point=original.entry_point)

        # Length. Placement was worked back from mix-out for the ARMED length,
        # so a revision may make the blend shorter -- it then ends early, which
        # is safe -- but never longer, which would run past mix-out.
        # Scaled, not subtracted: placement may already have shortened the
        # design to fit before mix-out (measured in a soak: a 24-bar design
        # armed as 9 bars), and taking the revision's bar count off the armed
        # length then went negative.
        armed_bars = float(armed["bars"])
        ratio = float(new.length_bars) / max(float(original.length_bars), 1e-9)
        bars = armed_bars * ratio
        if bars <= 0.0:
            new = dataclasses.replace(new, length_bars=original.length_bars)
            bars = armed_bars
            notes.append("kept the armed length")
        if bars > armed_bars + 1e-9:
            new = dataclasses.replace(new, length_bars=original.length_bars)
            bars = float(armed["bars"])
            notes.append("kept the armed length (a longer blend would pass mix-out)")
        total_frames = transition.transition_frames(
            armed["bpm"], SAMPLE_RATE, transition_bars=bars
        )

        # Entry. Only a real cue, with runway, that does not start a vocal
        # clash the supervisor would refuse.
        entry_frame = load.start_frame
        if new.entry_point != original.entry_point and not armed["drop_aligned"]:
            candidate = preview_mod.resolve_entry_frame(
                load.track.analysis, new, load.start_frame
            )
            env = self._envelope_for(new, total_frames, armed["bpm"])
            clash = self._clash(live, load.track, armed["start_frame"], candidate,
                                bars, env, new.name, log=False)
            if (
                candidate != load.start_frame
                and transition.entry_has_runway(load.track.analysis, candidate / SAMPLE_RATE)
                and clash is None
            ):
                entry_frame = int(round(candidate))
            else:
                new = dataclasses.replace(new, entry_point=original.entry_point)
                notes.append("kept the armed entry (no usable later cue)")

        new_load = dataclasses.replace(load, start_frame=int(entry_frame))
        new_swap = dataclasses.replace(cmd, total_frames=int(total_frames))
        why = self._replace_armed(cmd, load, new_load, new_swap, armed,
                                  lambda: self.engine.arm_transition_params(
                                      new, total_frames, armed["bpm"]))
        if why:
            return why
        armed.update(params=new, bars=bars)
        changed = ", ".join(
            f"{r.field} {r.before!r}->{r.after!r}" for r in outcome.revisions
        )
        self.last_transition_choice = (new.name, self.last_transition_choice[1])
        return "revised: " + changed + ("; " + "; ".join(notes) if notes else "")

    def _preview_force_cut(self, cmd, armed: dict, live) -> str:
        """The grids do not agree: a cut where the blend would have finished.

        Same shape as the vocal-clash fallback in :meth:`arm_transition` -- the
        hand-off stays where it was planned, just without the overlap.
        """
        analysis_a = live.track.analysis
        end_bar = phrase.bar_at_frame(analysis_a, armed["start_frame"]) + armed["bars"]
        cut_frame = int(round(phrase.frame_at_bar(analysis_a, math.floor(end_bar))))
        execute_at = phrase.deck_frame_to_engine_frame(
            live, self.engine.frames_played, cut_frame
        )
        if (execute_at - self.engine.frames_played) / SAMPLE_RATE < config.PREVIEW_COMMIT_MARGIN_S:
            return "grid clash, but too close to the hand-off to cut; armed transition stands"
        load = armed["load"]
        new_load = dataclasses.replace(
            load, execute_at=execute_at,
            start_frame=int(round(load.track.analysis.mix_in * SAMPLE_RATE)),
        )
        new_swap = dataclasses.replace(
            cmd, execute_at=execute_at, total_frames=self.engine.blocksize,
            drop_aligned=False,
        )
        why = self._replace_armed(cmd, load, new_load, new_swap, armed,
                                  lambda: self.engine.arm_transition_plan(
                                      "cut", self.engine.blocksize, armed["bpm"]))
        if why:
            return why
        cut = transition.preset_params("cut")
        armed.update(params=cut, bars=0.0, drop_aligned=False)
        self.last_transition_choice = ("cut", "preview: grids disagree, forced cut")
        self.session_log.write(
            "preview_forced_cut", trigger="phase coherence below threshold",
            action="hard cut at the planned hand-off",
        )
        return "grids disagree: forced a cut at the planned hand-off"

    def _replace_armed(self, cmd, load, new_load, new_swap, armed: dict, arm) -> str | None:
        """Atomically swap the queued pair for a new pair. None on success.

        Validated first, like every command arm_transition queues. Then both
        originals are taken out together; if either has already gone -- fired,
        cancelled by a panic, superseded by a re-arm -- whatever was taken is put
        straight back and nothing changes. The envelope is armed before the new
        StartTransition is queued, so the command can never run an old shape.
        """
        for c in (new_load, new_swap):
            rejection = self.supervisor.validate(c)
            if rejection is not None:
                return f"supervisor refused the revision ({rejection}); armed transition stands"
        with self._preview_lock:
            if self._armed_preview is not armed:
                return "a new transition was armed meanwhile; left alone"
            taken = self.scheduler.cancel_matching(lambda c: c is cmd or c is load)
            if len(taken) != 2:
                for c in taken:
                    self.scheduler.submit(c)
                return "the transition changed underneath the preview; armed transition stands"
            arm()
            armed.update(load=new_load, swap=new_swap)
            self.scheduler.submit(new_load)
            self.scheduler.submit(new_swap)
        return None

    def _clash(self, live, incoming, start_frame, entry_frame_b, bars, env,
               style: str, log: bool) -> str | None:
        """The supervisor's vocal-clash verdict for one placement."""
        a, b = live.track.analysis, incoming.analysis
        return self.supervisor.check_vocal_clash(
            a, b,
            phrase.bar_at_frame(a, start_frame),
            phrase.bar_at_frame(b, entry_frame_b),
            bars, env, style=style, log=log,
        )

    #: Moves tried to get two vocals apart, in bars: deck A earlier, deck B
    #: later, then both. Every one keeps the grid and the transition's length.
    VOCAL_MOVES: tuple[tuple[int, int], ...] = (
        (-8, 0), (0, 8), (-16, 0), (0, 16), (-8, 8), (-16, 16),
    )

    def _avoid_vocal_clash(self, live, incoming, start_frame, entry_frame_b, bars,
                           env, style: str, drop_aligned: bool):
        """A nearby placement with no vocal clash, or None.

        Returns ``(start_frame, entry_frame_b, note)``. Every candidate obeys the
        rules the original did: deck A's start is ahead of the playhead and past
        the supervisor's early-start guard, and deck B keeps its runway.
        """
        from djai.supervisor import (
            DROP_ALIGNED_MIN_START_FRACTION,
            MIN_TRANSITION_START_FRACTION,
        )

        a, b = live.track.analysis, incoming.analysis
        guard = DROP_ALIGNED_MIN_START_FRACTION if drop_aligned else MIN_TRANSITION_START_FRACTION
        total_a = a.duration_s * SAMPLE_RATE
        start_bar = phrase.bar_at_frame(a, start_frame)
        entry_bar = phrase.bar_at_frame(b, entry_frame_b)
        now_bar = phrase.bar_at_frame(a, live.position)
        for move_a, move_b in self.VOCAL_MOVES:
            new_start_bar = start_bar + move_a
            if new_start_bar <= now_bar + 1:
                continue
            new_start = phrase.frame_at_bar(a, new_start_bar)
            if total_a > 0 and new_start / total_a < guard:
                continue
            new_entry = phrase.frame_at_bar(b, entry_bar + move_b)
            if new_entry < 0 or not transition.entry_has_runway(b, new_entry / SAMPLE_RATE):
                continue
            if self._clash(live, incoming, new_start, new_entry, bars, env, style,
                           log=False) is None:
                parts = []
                if move_a:
                    parts.append(f"deck A starts {-move_a} bars earlier")
                if move_b:
                    parts.append(f"deck B enters {move_b} bars later")
                return (int(round(new_start)), int(round(new_entry)),
                        "vocal clash avoided: " + " and ".join(parts))
        return None

    def bars_until(self, execute_at: int) -> float:
        live = self.engine.deck(self.live_deck)
        return phrase.bars_until(live, self.engine.frames_played, execute_at)

    # --- autopilot -----------------------------------------------------------

    def autopilot_lead_seconds(self) -> float:
        """How much of the live track must remain before arming a transition."""
        bpm = self.engine.master_bpm
        if bpm <= 0:
            bpm = 128.0
        phrase_s = transition.transition_frames(bpm, SAMPLE_RATE) / SAMPLE_RATE
        extra_bars = self.hold_extra_bars
        if extra_bars:
            phrase_s *= 1.0 + extra_bars / transition.TRANSITION_BARS
        return phrase_s * AUTOPILOT_LEAD_PHRASES + AUTOPILOT_MARGIN_SECONDS

    @staticmethod
    def _seconds_to_mix_out(deck) -> float:
        """Real seconds until ``deck`` reaches its track's mix-out."""
        mix_out_frame = deck.track.analysis.mix_out * SAMPLE_RATE
        return (mix_out_frame - deck.position) / SAMPLE_RATE / max(deck.rate, 1e-6)

    def _autopilot_loop(self) -> None:
        """Keep music playing: cue ahead, then arm the transition."""
        while not self._stop.wait(AUTOPILOT_TICK):
            try:
                self._autopilot_tick()
            except Exception as exc:
                self.notify(f"[autopilot] error: {type(exc).__name__}: {exc}")
                log.exception("autopilot tick raised")
                try:
                    self._report_no_cue(
                        f"the autopilot tick raised {type(exc).__name__}: {exc}"
                    )
                except Exception:
                    pass  # reporting must never be what kills the loop

    def _release_first_track(self) -> None:
        """Drop the opening track once it is no longer the only copy.

        `_first_track` exists so the safety net can be armed from whatever
        opened the set, before anything else is decoded. Once deck A holds that
        audio -- or the fallback has copied it -- this reference is redundant,
        and keeping it pinned a whole decoded track (84.7 MB, measured) for the
        length of the session.
        """
        first = self._first_track
        if first is None:
            return
        if self.watchdog is not None or self.engine.deck_a.track is first:
            self._first_track = None

    def _release_stretches(self) -> None:
        """Free stretched copies the finished transition left unused.

        Only the tracks the decks hold and the one that is cued are kept.
        """
        keep = {
            deck.track.analysis.track_id
            for deck in (self.engine.deck_a, self.engine.deck_b)
            if deck.track is not None
        }
        if self._cued is not None:
            keep.add(self._cued.analysis.track_id)
        released = self.engine.release_stretches(keep)
        if released:
            self.session_log.write(
                "stretch_released",
                trigger="transition complete",
                action=f"freed {released} stretched copy(ies) no deck uses",
            )

    def _recover_from_silence(self) -> None:
        """Nothing is playing: hard-start the next track on the idle deck."""
        if not self.has_cued_track():
            self.cue_next(origin="recovery")
            return

        incoming = self._cued
        if incoming is None:
            return
        idle = self.cued_deck()
        self.engine.submit(
            LoadTrack(
                deck=idle,
                track=incoming,
                start_frame=int(incoming.analysis.mix_in * SAMPLE_RATE),
                rate=1.0,
                play=True,
                master=True,
                execute_at=IMMEDIATE,
                origin="recovery",
            )
        )
        self.engine.deck(idle).gain.jump(0.0)
        self.engine.deck(idle).set_gain(1.0)
        self.live_deck = idle
        self._cued = None
        self._transition_armed = False
        self.session_log.write(
            "recovered_from_silence",
            deck=idle,
            track=incoming.title,
            trigger="live deck ended with no transition in flight",
            action="started the cued track immediately at the master tempo",
        )
        self.notify(f"[autopilot] deck ran out - starting {incoming.title} now")

    #: Real seconds of audio left at which a deck about to run dry is covered:
    #: a tick to cue, a tick to arm, the next bar line and a one-bar blend.
    RUN_DRY_S: float = 10.0

    def _run_dry(self, live) -> bool:
        """The live deck's audio is about to stop with no blend to cover it.

        A truncated or damaged file ends long before its analysed mix-out, or a
        blend never armed. Waiting for ``ended`` cost a tick to cue and a tick
        to start: 518 ms of dead air, measured. Instead the cued track comes in
        on the next bar line through the play-now path -- a real transition, so
        the hand-over bookkeeping is the ordinary one. Returns True when it
        acted, so nothing else runs this tick.
        """
        if (self.frozen or self.engine.transition_active or live.loop_active or live.ended
                or live.transport is not TransportState.PLAYING or live.track is None):
            return False
        left_s = live.remaining_frames / SAMPLE_RATE / max(live.rate, 1e-6)
        if left_s > self.RUN_DRY_S:
            return False
        end_at = self.engine.frames_played + left_s * SAMPLE_RATE
        bar = 4.0 * SAMPLE_RATE * 60.0 / max(self.engine.master_bpm or 120.0, 1.0)
        if self._transition_armed:
            starts = [c.execute_at for c in self.scheduler.pending()
                      if isinstance(c, StartTransition)]
            if starts and min(starts) + bar < end_at:
                return False          # the armed blend starts in time
            cued = self._cued
            self.abort_armed_transition(trigger="live deck runs dry before the blend")
            self._cued = cued
        if not self.has_cued_track():
            self.cue_next(origin="recovery")   # lands on the deck next block
            return True
        at = self.arm_transition(origin="recovery", at_bars=1)
        if at is None:
            return False   # silence recovery covers the end
        self.session_log.write(
            "run_dry", track=self._cued.title if self._cued else None,
            trigger=f"live deck's audio ends in {left_s:.1f} s, before its mix-out",
            action=f"blend on the next bar line, {self.bars_until(at):.2f} bars out",
        )
        self.notify(f"[autopilot] {live.track.analysis.title} is running out - "
                    "bringing the next track in on the bar")
        return True

    def _complete_handover(self) -> bool:
        """Bring the bookkeeping up to date with a hand-over the engine made.

        Returns True if one was completed. Called from the autopilot tick and
        at the top of everything that reads ``live_deck`` to decide where the
        next track goes -- a cue or an arm between the engine finishing a blend
        and the next tick otherwise acts on the deck that just stopped, and
        loads over the one the room is hearing (Phase 2.3 fuzz, three routes).
        """
        live = self.engine.deck(self.live_deck)
        if live.track is None:
            return False
        # The transition finished: the incoming deck is now live. Also when no
        # blend was armed but the other deck is the one the room hears -- a
        # cancel that raced the blend's own ending, or a hand-over by hand.
        # The bookkeeping follows the audio, never the other way round.
        if not self.engine.transition_active and (
            self._transition_armed
            or self.engine.deck(self.cued_deck()).gain.target > 0.0
        ):
            other = self.cued_deck()
            incoming = self.engine.deck(other)
            if (
                incoming.transport is TransportState.PLAYING
                and live.transport is not TransportState.PLAYING
            ):
                self.live_deck = other
                self._transition_armed = False
                self.hold_extra_bars = 0
                self._cued = None
                incoming_deck = self.engine.deck(other)
                ratio = float(getattr(incoming_deck, "metric_ratio", 1.0))
                metric = abs(ratio - 1.0) > 1e-6
                # The beat clock follows the deck that is now the mix. Whatever
                # tempo the set arrived at through this hand-over -- a ride, a
                # half-time count, or simply the track's own speed -- is the
                # tempo the next one is planned from. Leaving the clock on the
                # track that just ended pins the whole night to what it opened
                # at and quietly undoes every journey: the next track is
                # stretched back and eventually cut for being too far away.
                # The engine hands the clock over in the callback as the
                # crossfade ends, so by now it has usually already moved: the
                # tempo the set was *at* is the one recorded when this
                # transition was armed, and that is what this compares against.
                before = self._clock_at_arm or self.engine.master_bpm
                if metric:
                    for name in ("a", "b"):
                        self.engine.deck(name).metric_ratio = 1.0
                for name in ("a", "b"):
                    self.supervisor.forget_baseline(name)
                after = self.engine.hand_master_to(other)
                if self._cue_decision is not None:
                    # The window over the cue closes here: the blend it was
                    # taken for has finished, so there is now an outcome.
                    self.session_log.outcome(
                        self._cue_decision,
                        self.engine.features.latest(),
                        trigger="transition finished",
                        action=f"handed over to deck {other}",
                    )
                    self._cue_decision = None
                # Everything this transition was holding goes now. Each of
                # these pins a whole decoded track -- 70-120 MB -- and none of
                # them was ever cleared, only overwritten by the next
                # transition. Measured over a 40-minute set: 31 LoadedTracks
                # alive holding 2.4 GB while two tracks had been played, which
                # is what drove the machine into paging and put 40 blocks over
                # the callback budget in the 4-hour soak.
                #
                # `_armed_preview` is identity-compared under `_preview_lock`
                # (see _replace_armed), and None correctly reads there as "a
                # new transition was armed meanwhile", so dropping it is safe
                # as well as necessary.
                with self._preview_lock:
                    self._armed_preview = None
                self._stem_origin.clear()
                self.stem_moves = []
                self._pending_stem_moves = []
                # The hand-over alone cannot move the clock -- it adopts the
                # tempo this deck was already matched to. The glide is what
                # actually takes the set to the incoming track's own BPM.
                self._schedule_master_glide(other)
                if self.room is not None:
                    self.room.handover()
                    self._room_deferred = False
                if metric or abs(after - before) > 0.05:
                    self.session_log.write(
                        "metric_handover" if metric else "tempo_handover",
                        deck=other,
                        track=(
                            incoming_deck.track.analysis.title
                            if incoming_deck.track is not None else None
                        ),
                        from_bpm=round(before, 2),
                        to_bpm=round(after, 2),
                        trigger=(
                            f"x{ratio:g} counted mix finished" if metric
                            else "mix finished at a new tempo"
                        ),
                        action=f"beat clock now {after:.1f} BPM",
                    )
                self._drop_unlanded_ride()
                self._release_stretches()
                new = self.engine.deck(self.live_deck).track
                if new is not None:
                    self.notify(f"[autopilot] now playing {new.analysis.title}")
                    self.session_log.write(
                        "transition_complete",
                        deck=self.live_deck,
                        track=new.analysis.title,
                        trigger="transition finished",
                        action="deck is now live",
                    )
                return True
        return False

    def _autopilot_tick(self) -> None:
        if self.engine.stop_requested:
            return

        # Before anything looks at the tempo: the clock follows a ride in
        # flight, so what the cue and the arm measure is where the mix is.
        self._follow_ride()
        try:
            self._room_tick()
        except Exception as exc:  # noqa: BLE001 - the loop is never worth the set
            log.warning("closed loop tick raised: %s", exc)

        # Arm the safety net as soon as there is decoded audio to fall back to.
        # On an idle start nothing is decoded at launch, so a deck reaching
        # PLAYING is the first moment a fallback can exist at all.
        if self.autoarm_fallback and self.watchdog is None and self.fallback_enabled:
            for name in ("a", "b"):
                deck = self.engine.deck(name)
                if deck.transport is TransportState.PLAYING and deck.track:
                    if self.arm_fallback(deck.track):
                        self.notify(f"[fallback] armed: {deck.track.title}")
                    break

        # The fallback player owns the output: the engine no longer drains
        # commands or advances a deck, so nothing below can cue anything.
        watchdog = self.watchdog
        if watchdog is not None and watchdog.tripped:
            self._report_no_cue(
                f"the fallback player has the output ({watchdog.reason}); the "
                "engine is not advancing, so nothing can be cued until it is "
                "released"
            )

        self._release_first_track()

        live = self.engine.deck(self.live_deck)

        # Follow the deck the room actually hears. On an idle start the operator
        # picks which deck opens the set; later they may hand the mix over by
        # hand, pausing the live deck and starting the other one. Either way the
        # autopilot has to count down on the deck that is PLAYING. Watching the
        # paused one left it waiting on a deck that never reaches mix-out while
        # the playing deck ran out uncued -- the 2026-09-16 "never cues the next
        # song" regression.
        # Nothing is started here, so a pause is still never read as a hand-off
        # cue. Not while a transition is armed or running: both decks are
        # PLAYING then, and the hand-off below is what moves `live_deck`.
        if (
            (live.transport is not TransportState.PLAYING or live.ended)
            and not self._transition_armed
            and not self.engine.transition_active
        ):
            other_name = self.cued_deck()
            other = self.engine.deck(other_name)
            if other.transport is TransportState.PLAYING and not other.ended:
                handed_over = live.track is not None
                self.live_deck = other_name
                live = other
                # Whatever was cued was cued onto the deck that is now live.
                self._cued = None
                if handed_over:
                    self.notify(
                        f"[autopilot] following deck {other_name.upper()}: "
                        f"{live.track.analysis.title} is the one playing"
                    )

        if live.track is None:
            return

        if self._complete_handover():
            return

        if self._run_dry(live):
            return

        if self.engine.transition_active or self._transition_armed:
            # Armed, yet nothing queued and nothing running: the hand-off this
            # tick waits for will never come, and silence recovery below is
            # never reached. Reported, not repaired -- see STUCK_ARM_S.
            if (
                self._transition_armed
                and not self.engine.transition_active
                and not any(
                    isinstance(c, StartTransition) for c in self.scheduler.pending()
                )
            ):
                now = time.monotonic()
                if self._stuck_arm_since is None:
                    self._stuck_arm_since = now
                elif now - self._stuck_arm_since >= STUCK_ARM_S:
                    self._report_no_cue(
                        "a transition is armed but nothing is queued or running; "
                        "the autopilot is waiting on a hand-off that will not come"
                    )
            else:
                self._stuck_arm_since = None
            return
        self._stuck_arm_since = None

        # "Play now" was cued last tick; its LoadTrack has landed, so arm it on
        # the next 4-bar line with a glue swell on the way out. The operator's
        # own request, so it runs even while automation is held.
        if self._play_now is not None and self.has_cued_track():
            landed = self.engine.deck(self.cued_deck()).track
            if landed is not None and landed.analysis.track_id == self._play_now:
                self._arm_play_now(live)
            return

        # Held automation gates the scheduler itself, not just the banner in
        # the UI. It comes first so that nothing below it -- cueing, arming, or
        # silence recovery -- can route around it. Recovery used to run even
        # when frozen, on the reasoning that freezing meant "stop choosing" and
        # not "let the room go quiet"; holding now means the automation makes
        # no move at all, and `cut` and `stop` remain the deliberate ways to
        # silence a room.
        if self.frozen:
            if not self.has_cued_track() and (
                live.ended
                or (
                    live.transport is TransportState.PLAYING
                    and self._seconds_to_mix_out(live) <= self.autopilot_lead_seconds()
                )
            ):
                self._report_no_cue(
                    "automation is held, and the live deck is inside its cue "
                    "window; nothing will be cued until `resume`"
                )
            return

        # Paused and ended are different events and get different handlers.
        # Collapsing them into "the deck is not advancing" is what let a pause
        # be read as a hand-off cue and start the other deck.
        transport = live.transport
        if transport is TransportState.PAUSED:
            return

        # The live deck ran out with no transition in flight -- the mix is
        # silent. Waiting for the next 32-bar boundary of a stopped deck would
        # extend that silence indefinitely, so start the next track now.
        if live.ended:
            self._recover_from_silence()
            return

        # Cued-and-waiting, or anything else that is not advancing: not a
        # transition candidate.
        if transport is not TransportState.PLAYING:
            return

        # Count down to mix-out, not to the end of the file. Placement is
        # anchored to mix-out, so waking on the file's end would arm late on any
        # track with a long outro and force plan_transition to shorten or cut a
        # blend that had plenty of room.
        if self._seconds_to_mix_out(live) > self.autopilot_lead_seconds():
            return

        if not self.has_cued_track():
            # Cue, then stop. The LoadTrack is applied by the audio thread, so
            # the track is not on the deck yet and arming now would always
            # fail; the next tick sees it and arms.
            self.cue_next()
            return

        # A design is still being written and there is room to wait for it.
        # Nothing blocks: the next tick asks again.
        if self.design_pending():
            return

        if self._arm_held(live):
            return

        execute_at = self.arm_transition()
        if execute_at is not None:
            bars = self.bars_until(execute_at)
            title = self._cued.title if self._cued else "?"
            self.notify(f"[autopilot] bass swap into {title} in {bars:.0f} bars")

    # --- state blobs ---------------------------------------------------------

    def model_state(self) -> dict[str, Any]:
        """The minimal state the local model needs to pick an action.

        Deliberately five fields. llama3.1:8b degrades when given a large
        object to read past -- deck gains, EQ positions, queued command
        descriptions and the played-track titles were all noise it could not
        use, since it may not name tracks and cannot set gains. What is left is
        only what distinguishes one action from another.
        """
        live = self.engine.deck_state(self.live_deck)
        return {
            "bpm": round(self.engine.master_bpm, 1),
            "key": live.camelot,
            "bars_in": round(live.position_bars, 1),
            "transition": self.engine.transition_active,
            "played": len(self.played),
        }

    def state_blob(self) -> dict[str, Any]:
        """Full state, for the human-facing `state` command. Not sent to the model."""
        decks = {}
        for name in ("a", "b"):
            st = self.engine.deck_state(name)
            decks[name] = {
                "title": st.title,
                "playing": st.playing,
                "live": name == self.live_deck,
                "bpm_playing": st.bpm,
                "bpm_native": st.native_bpm,
                "key": st.key,
                "camelot": st.camelot,
                "position_bars": st.position_bars,
                "remaining_seconds": st.remaining_seconds,
                "gain": st.gain,
                "eq_low_mid_high": list(st.eq),
            }
        pending = self.scheduler.pending()
        return {
            "decks": decks,
            "master_bpm": round(self.engine.master_bpm, 2),
            "transition_active": self.engine.transition_active,
            "transition_progress_bars": round(
                self.engine.transition_progress_bars, 1
            ),
            "queued_commands": [
                {
                    "command": c.describe(),
                    "bars_until": round(self.bars_until(c.execute_at), 1),
                }
                for c in pending[:8]
            ],
            "played_this_session": [
                t.title for t in self.crate if t.track_id in self.played
            ],
            "crate_size": len(self.crate),
            "underruns": self.engine.underruns,
            "supervisor_interventions": self.supervisor.interventions,
            "frozen": self.frozen,
            "forced_next": self._forced_next.title if self._forced_next else None,
            "cue_deck": self.engine.cue_deck,
            "cue_mode": self.engine.cue_mode,
            "recording": (
                {
                    "path": str(self.recorder.path),
                    "seconds": round(self.recorder.seconds_written, 1),
                    "dropped_frames": self.recorder.dropped_frames,
                    "error": self.recorder.error,
                }
                if self.recorder is not None
                else None
            ),
            "fallback": (
                {
                    "armed": self.watchdog is not None,
                    "active": bool(self.watchdog and self.watchdog.tripped),
                    "reason": self.watchdog.reason if self.watchdog else None,
                    "track": self.watchdog.player.title if self.watchdog else None,
                }
            ),
        }

    def format_state(self) -> str:
        blob = self.state_blob()
        lines = [f"master {blob['master_bpm']:.1f} BPM"]
        for name, d in blob["decks"].items():
            if d["title"] is None:
                lines.append(f"  deck {name}: empty")
                continue
            flag = "LIVE" if d["live"] else "cued" if not d["playing"] else "playing"
            lines.append(
                f"  deck {name} [{flag}]: {d['title']} | "
                f"{d['bpm_playing']} BPM ({d['bpm_native']} native) | "
                f"{d['camelot']} | bar {d['position_bars']:.0f} | "
                f"gain {d['gain']:.2f} | eq {d['eq_low_mid_high']} | "
                f"{d['remaining_seconds']:.0f}s left"
            )
        if blob["transition_active"]:
            lines.append(
                f"  transition running, bar {blob['transition_progress_bars']:.0f}"
                f" of {transition.TRANSITION_BARS:.0f}"
            )
        for q in blob["queued_commands"]:
            lines.append(f"  queued: {q['command']} in {q['bars_until']:.0f} bars")

        flags = []
        if blob["frozen"]:
            flags.append("AUTOMATION FROZEN")
        if blob["forced_next"]:
            flags.append(f"forced next: {blob['forced_next']}")
        if blob["cue_deck"]:
            flags.append(f"cue: deck {blob['cue_deck']} ({blob['cue_mode']})")
        fb = blob["fallback"]
        if fb["active"]:
            flags.append(f"FALLBACK ACTIVE ({fb['reason']})")
        if flags:
            lines.append("  " + " | ".join(flags))

        rec = blob["recording"]
        if rec is not None:
            note = f"  recording {rec['seconds']:.0f}s -> {rec['path']}"
            if rec["dropped_frames"]:
                note += f" [{rec['dropped_frames']} frames dropped]"
            if rec["error"]:
                note += f" [STOPPED: {rec['error']}]"
            lines.append(note)

        lines.append(
            f"  underruns {blob['underruns']} | "
            f"supervisor interventions {blob['supervisor_interventions']}"
        )
        return "\n".join(lines)


# --- panic path --------------------------------------------------------------


def handle_panic(session: Session, word: str) -> bool:
    """Dispatch a panic command immediately. Returns True if the REPL should exit.

    Deliberately the first thing the REPL checks and the shortest path in the
    program: text -> command -> engine queue, no LLM, no scheduler, no
    validation. The next audio callback picks it up.
    """
    word = word.strip().lower()
    if word == "cut":
        session.engine.submit(Cut(deck="master", execute_at=IMMEDIATE, origin="panic"))
        session.session_log.write(
            "panic", trigger="cut", action="master gain to 0 immediately"
        )
        print("cut.")
        return False
    if word in ("killbass", "kill bass"):
        session.engine.submit(
            KillBass(deck="master", killed=True, execute_at=IMMEDIATE, origin="panic")
        )
        session.session_log.write(
            "panic", trigger="killbass", action="low band to 0 on both decks"
        )
        print("bass killed.")
        return False
    if word in ("stop", "quit", "exit"):
        session.engine.submit(Stop(execute_at=IMMEDIATE, origin="panic"))
        session.session_log.write("panic", trigger="stop", action="both decks stopped")
        # Audio first, bookkeeping second: the command is already on the engine
        # queue by the time anything below runs.
        session.after_stop()
        print("stopping.")
        return True
    return False


# --- REPL --------------------------------------------------------------------


def cue_intent(session: Session, intent: Intent) -> str:
    """A chat song request: the model's search text, looked up in code."""
    query = intent.params.get("query")
    if not isinstance(query, str) or not query.strip():
        return "I did not catch a song name. Say `cue <song>`."
    mode = intent.params.get("mode")
    after = intent.params.get("after")
    after = after if isinstance(after, int) and not isinstance(after, bool) else 0
    return session.cue_song(query, mode if isinstance(mode, str) else "next",
                            max(0, min(after, 50)))


def apply_intent(session: Session, intent: Intent) -> None:
    """Turn a parsed intent into scheduled commands, printing what happens."""
    action = intent.action

    if action == "describe_state":
        print(session.format_state())
        return

    if action == "skip_queued":
        dropped = session.scheduler.cancel_all()
        session._transition_armed = False
        session.session_log.write(
            "commands_skipped",
            trigger="user skip_queued",
            action=f"dropped {len(dropped)} queued command(s)",
        )
        print(f"Dropped {len(dropped)} queued command(s).")
        return

    if action == "hold_blend":
        session.hold_extra_bars = intent.hold_bars
        print(f"Holding the blend {intent.hold_bars} bars longer.")
        return

    if action == "set_transition_style":
        print(session.set_transition_style(str(intent.params.get("style", ""))))
        return

    if action == "set_phase":
        print(session.set_set_phase(str(intent.params.get("phase", ""))))
        return

    if action == "cue_track":
        print(cue_intent(session, intent))
        return

    if action in ("next_track", "set_energy"):
        direction = intent.energy_direction
        # Remembered so the transition designer knows which way the operator
        # wants the set to go, not just which track to play next.
        session.energy_direction = direction
        # The room hears it now, not a track from now.
        now = session.energy_correction(direction)
        if now:
            print(now)
        if action == "set_energy" and session.has_cued_track():
            # Re-cue: the queued choice was made under the old direction.
            session.scheduler.cancel_matching(lambda c: isinstance(c, LoadTrack))
        if not session.cue_next(direction, origin="user"):
            print("Could not find a compatible track for that.")
            return
        cued = session._cued
        if cued is not None:
            print(f"Cued: {cued.title} ({cued.analysis.bpm:.1f} BPM)")
        if action == "next_track" and not session._transition_armed:
            execute_at = session.arm_transition(origin="user", at_next_phrase=True)
            if execute_at is not None:
                bars = session.bars_until(execute_at)
                print(f"Bass swap fires in {bars:.0f} bars.")
        return

    # "none" and anything unhandled: nothing is scheduled.


#: Manual override words. Like the panic words, these are matched before the
#: model ever sees the line: they are the controls an operator reaches for when
#: the automation is doing the wrong thing, so routing them through the thing
#: that is doing the wrong thing would be a poor design.
OVERRIDE_WORDS = {
    "freeze", "hold", "resume", "unfreeze", "go", "now", "force", "cue", "transitions",
    "suggest", "queue", "bridge", "plan", "persona",
    # Back out of the blend in flight (Phase 2.1).
    "cancel",
    # The action language (Phase 2.3): `act <request>` or `act {json plan}`.
    "act",
    # `hotcue` is a stored marker in a track; `cue` above is the headphone
    # output. Two different things, two different words, deliberately.
    "hotcue", "style",
    # Where to take the tempo: `tempo 128`, or `tempo off` to stop steering it.
    "tempo",
    # Phase 4: `feedback <label>` and the stable model (`model learn|rollback`).
    "feedback", "model",
}


def _handle_tempo(session: Session, rest: str) -> str:
    """`tempo 128` aims the set at 128 BPM; `tempo off` stops steering it.

    The journey rides toward the target across tracks rather than jumping: a
    set at 100 reaches 128 over several mixes, each one blendable, which is
    the whole point of the planner.
    """
    text = (rest or "").strip().lower()
    if not text or text in ("?", "what"):
        if session.tempo_target:
            return f"Heading for {session.tempo_target:.0f} BPM."
        return "No tempo target; say `tempo 128` to set one."
    if text in ("off", "none", "stop", "clear"):
        session.tempo_target = None
        session.session_log.write(
            "tempo_target", trigger="operator", action="no tempo target",
        )
        return "Tempo target cleared."
    try:
        target = float(text.split()[0])
    except (TypeError, ValueError):
        return f"Say `tempo 128`, not `tempo {text}`."
    if not 60.0 <= target <= 200.0:
        return f"{target:.0f} BPM is outside 60-200."
    session.tempo_target = target
    now = session.engine.master_bpm
    session.session_log.write(
        "tempo_target",
        trigger=f"operator: {target:.1f} BPM",
        action=f"riding from {now:.1f} BPM toward {target:.1f} across the set",
    )
    return f"Heading for {target:.0f} BPM, a track at a time (now {now:.0f})."


def handle_override(session: Session, text: str) -> str | None:
    """Run a manual-override command. Returns the reply, or None if not one.

    Shared by the REPL and the web UI so both surfaces behave identically and
    neither depends on the LLM being alive.
    """
    parts = text.strip().split(maxsplit=1)
    if not parts:
        return None
    word = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    # "That worked", "too early", "more like this": the operator's word on
    # what just happened, matched on the whole line before the model sees it.
    from djai import feedback as feedback_mod

    label = feedback_mod.parse_phrase(text)
    if label is not None:
        return session.record_feedback(label)
    if word == "feedback":
        label = rest.lower().replace(" ", "_").replace("'", "")
        if label not in feedback_mod.VERDICTS:
            return "Say `feedback` with one of: " + ", ".join(feedback_mod.VERDICTS) + "."
        return session.record_feedback(label)
    if word == "model":
        return session.model_text(rest.lower())

    if word in ("freeze", "hold") and not rest:
        return session.freeze(True)
    if word in ("resume", "unfreeze") and not rest:
        return session.freeze(False)
    if word == "cancel" and not rest:
        return session.cancel_transition()
    if word == "act":
        if not rest:
            return "Say `act <request>` or `act {json plan}`."
        return actions_mod.act(session, rest, session.intent_engine).summary()
    if word in ("explain", "why") and not rest:
        return session.explain()
    if word == "mode":
        return session.set_mode(rest.lower())
    if word in ("mc", "mic"):
        if rest.lower() not in ("", "on", "off"):
            return "Say `mc on` or `mc off`."
        return session.set_mc(rest.lower() == "on" if rest else not session.mc)
    if word in ("go", "now") and not rest:
        return session.force_transition()
    if word == "force":
        if not rest:
            return "Say `force <track>` - a title, an id, or part of one."
        return session.force_next(rest)
    if word == "cue":
        target = rest.lower()
        if target in ("off", "none", ""):
            return session.set_cue(None)
        if target in ("a", "b"):
            return session.set_cue(target)
        query, mode, after = _parse_cue_request(rest)
        return session.cue_song(query, mode, after)
    if word == "suggest":
        n = int(rest) if rest.isdigit() else 5
        rows = session.suggest(n)
        if not rows:
            return "Nothing to suggest: every eligible track has played."
        return "\n".join(r.line() for r in rows)
    if word == "queue":
        return _handle_queue(session, rest)
    if word == "bridge":
        return session.resolve_bridge(rest.lower())
    if word == "plan":
        if rest:
            try:
                session.set_minutes = max(10.0, min(480.0, float(rest)))
            except ValueError:
                return "Say `plan` or `plan <minutes>`."
        plan = session.replan("planned" if session.set_plan is None else "operator asked")
        return plan.summary() if plan else "No plan."
    if word == "persona":
        if not rest:
            return f"Persona: {session.persona}. Choose from {', '.join(planner.PERSONAS)}."
        return session.set_persona(rest.lower())
    if word == "hotcue":
        return _handle_hotcue(session, rest)
    if word == "tempo":
        return _handle_tempo(session, rest)
    if word == "style":
        return _handle_style(session, rest)
    if word == "transitions":
        mode = rest.lower()
        if mode not in transition.MODES + ("auto",):
            return (f"Transitions are {session.transition_mode}. "
                    "Say `transitions invisible`, `showy` or `auto`.")
        session.transition_mode = mode
        return f"Transitions: {mode}."
    if word == "grid":
        return _handle_grid(session, rest)
    if word in ("loop", "roll", "jump", "pitch", "sync", "quantize"):
        reply = _handle_performance(session, word, rest)
        if reply is not None:
            return reply
    if word == "phase":
        if not rest:
            return (
                f"Set phase is {session.set_phase}. "
                "Say `phase <warmup|build|peak|cooldown|auto>`."
            )
        return session.set_set_phase(rest)
    if word == "keylock":
        return _handle_keylock(session, rest)
    if word == "filter":
        return _handle_filter(session, rest)
    return None


_CUE_MODE_RE = re.compile(
    r"\s+(?:(?P<now>now|play now)|(?P<next>next|play next)|"
    r"after\s+(?P<n>\d+)(?:\s+tracks?)?)\s*$", re.IGNORECASE)


def _parse_cue_request(text: str) -> tuple[str, str, int]:
    """``<query> [next|now|after N]`` -> (query, mode, after)."""
    m = _CUE_MODE_RE.search(" " + text)
    if not m:
        return text.strip(), "next", 0
    query = (" " + text)[: m.start()].strip()
    if m.group("now"):
        return query, "now", 0
    if m.group("n"):
        return query, "after", int(m.group("n"))
    return query, "next", 0


def _handle_queue(session: Session, rest: str) -> str:
    parts = rest.split()
    if not parts:
        return session.queue_text()
    try:
        if parts[0] == "remove" and len(parts) == 2:
            return session.queue_remove(int(parts[1]))
        if parts[0] == "move" and len(parts) == 3:
            return session.queue_move(int(parts[1]), int(parts[2]))
    except ValueError:
        pass
    return "Say `queue`, `queue remove <n>` or `queue move <from> <to>`."


def _bars_text(bars: float) -> str:
    fractions = {0.125: "1/8 bar", 0.25: "1/4 bar", 0.5: "1/2 bar", 1.0: "1 bar"}
    return fractions.get(bars, f"{bars:g} bars")


def _parse_bars(word: str) -> float | None:
    """``1/8``, ``0.5``, ``2`` -> bars."""
    try:
        if "/" in word:
            num, den = word.split("/", 1)
            return float(num) / float(den)
        return float(word)
    except (ValueError, ZeroDivisionError):
        return None


def _handle_performance(session: Session, word: str, rest: str) -> str | None:
    """Loops, rolls, jumps, pitch, sync and quantize from the REPL.

    `loop [a|b] in|out|exit|halve|double|<beats>`, `roll [a|b] <1/8..4>|off`,
    `jump [a|b] <+-1|4|8|16>`, `pitch [a|b] <+-pct>|reset`, `sync [a|b]`,
    `quantize on|off`. Anything that does not parse returns None, so a
    sentence such as "loop roll it out" still reaches the model.
    """
    words = rest.lower().split()
    if word == "quantize":
        if len(words) == 1 and words[0] in ("on", "off"):
            return session.set_quantize(words[0] == "on")
        if not words:
            return f"Quantize is {'on' if session.quantize else 'off'}. Say `quantize on` or `quantize off`."
        return None
    deck = session.live_deck
    if words and words[0] in ("a", "b"):
        deck = words.pop(0)
    if word == "sync":
        return session.sync(deck) if not words else None
    if len(words) != 1:
        return None
    arg = words[0]
    if word == "loop":
        if arg == "in":
            return session.loop_in(deck)
        if arg == "out":
            return session.loop_out(deck)
        if arg in ("exit", "off"):
            return session.loop_exit(deck)
        if arg == "halve":
            return session.loop_resize(deck, 0.5)
        if arg == "double":
            return session.loop_resize(deck, 2.0)
        beats = _parse_bars(arg)
        return None if beats is None else session.auto_loop(deck, beats)
    if word == "roll":
        if arg == "off":
            return session.roll_off(deck)
        bars = _parse_bars(arg)
        return None if bars is None else session.roll(deck, bars)
    if word == "jump":
        try:
            bars = int(arg)
        except ValueError:
            return None
        return session.beat_jump(deck, bars)
    if word == "pitch":
        if arg in ("reset", "0"):
            return session.set_pitch(deck, 0.0)
        try:
            return session.set_pitch(deck, float(arg.rstrip("%")))
        except ValueError:
            return None
    return None


def _handle_keylock(session: Session, rest: str) -> str:
    """`keylock [a|b] on|off`, or no argument to report both decks."""
    words = rest.lower().split()
    deck = session.live_deck
    if words and words[0] in ("a", "b"):
        deck = words.pop(0)
    if not words:
        states = ", ".join(
            f"{name.upper()} {'on' if session.engine.deck(name).key_lock else 'off'}"
            for name in ("a", "b")
        )
        return f"Key lock: {states}. Say `keylock [a|b] on` or `keylock [a|b] off`."
    if len(words) == 1 and words[0] in ("on", "off"):
        return session.set_key_lock(deck, words[0] == "on")
    return "Say `keylock [a|b] on` or `keylock [a|b] off`."


def _handle_filter(session: Session, rest: str) -> str | None:
    """`filter [a|b] <-1..1> [res <0..1>]`, or `filter [a|b] off`.

    Anything that does not parse returns None, so a sentence such as "filter
    this one out" still reaches the model instead of a usage message.
    """
    words = rest.lower().split()
    deck = session.live_deck
    if words and words[0] in ("a", "b"):
        deck = words.pop(0)
    if words == ["off"]:
        return session.set_filter(deck, 0.0)
    if not words:
        return None
    try:
        position = float(words[0])
    except ValueError:
        return None
    resonance = None
    if len(words) == 3 and words[1] in ("res", "resonance"):
        try:
            resonance = float(words[2])
        except ValueError:
            return None
    elif len(words) != 1:
        return None
    return session.set_filter(deck, position, resonance)


def _handle_hotcue(session: Session, rest: str) -> str:
    """`hotcue [set|clear] [a|b] <n>` -- jump by default."""
    words = rest.lower().split()
    if not words:
        return (
            "Say `hotcue <n>` to jump, `hotcue set <n>` to store the playhead, "
            "or `hotcue clear <n>`. Add `a` or `b` to pick a deck."
        )
    action = "jump"
    if words[0] in ("set", "clear", "jump"):
        action = words.pop(0)
    deck = session.live_deck
    if words and words[0] in ("a", "b"):
        deck = words.pop(0)
    if not words:
        return f"Say which hot cue: `hotcue {action} <1-{an.MAX_HOT_CUES}>`."
    try:
        index = int(words[0])
    except ValueError:
        return f"`{words[0]}` is not a hot cue number."
    if action == "set":
        return session.set_hot_cue(deck, index, " ".join(words[1:]))
    if action == "clear":
        return session.clear_hot_cue(deck, index)
    return session.jump_to_hot_cue(deck, index)


def _handle_grid(session: Session, rest: str) -> str:
    """`grid [a|b] halve|double|nudge <ms>|tap|downbeat <seconds>`."""
    words = rest.lower().split()
    deck = session.live_deck
    if words and words[0] in ("a", "b"):
        deck = words.pop(0)
    usage = (
        "Say `grid [a|b] halve`, `grid [a|b] double`, `grid [a|b] nudge <ms>`, "
        f"`grid [a|b] tap` ({an.MIN_TAPS}+ times, on the beat) or "
        "`grid [a|b] downbeat <seconds>`."
    )
    if not words:
        return usage
    op, args = words[0], words[1:]
    if op == "halve":
        return session.grid_halve(deck)
    if op == "double":
        return session.grid_double(deck)
    if op == "tap":
        return session.grid_tap(deck)
    if op in ("nudge", "downbeat"):
        try:
            value = float(args[0])
        except (IndexError, ValueError):
            return usage
        if op == "nudge":
            return session.grid_nudge(deck, value)
        return session.grid_set_downbeat(deck, value)
    return usage


def _handle_style(session: Session, rest: str) -> str:
    """`style <name|auto>` -- what the next transition should be."""
    name = rest.strip().lower()
    if not name:
        return (
            f"Transition style is {session.transition_style}. "
            f"Say `style <{'|'.join(transition.STYLE_CHOICES)}>`."
        )
    return session.set_transition_style(name)


def repl(session: Session, intent_engine: IntentEngine) -> None:
    print(BANNER)
    while True:
        for line in session.drain_notices():
            print(line)

        try:
            text = input("djai> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            handle_panic(session, "stop")
            return

        for line in session.drain_notices():
            print(line)

        if not text:
            continue

        lowered = text.lower()
        if lowered in PANIC_WORDS:
            if handle_panic(session, lowered):
                return
            continue

        if lowered in ("state", "status"):
            print(session.format_state())
            continue
        if lowered in ("help", "?"):
            print(BANNER)
            continue

        # Manual override, before the model: these must work when it is down.
        reply = handle_override(session, text)
        if reply is not None:
            print(reply)
            continue

        intent = intent_engine.interpret(text, session.model_state())
        session.session_log.write(
            "intent",
            trigger=text,
            action=f"{intent.action} {intent.params}",
            ok=intent.ok,
            error=intent.error,
            fallback=intent.fallback,
            raw=intent.raw,
            latency_s=round(intent.latency_s, 2),
        )

        if intent.fallback:
            # Not an error path: the local model failed, the keyword chain
            # answered, and the session carries on.
            print(f"[local model unavailable: {intent.fallback} - used keywords]")

        if intent.reply:
            print(intent.reply)
        apply_intent(session, intent)


# --- subcommands -------------------------------------------------------------


def cmd_stems(args: argparse.Namespace) -> int:
    """Precompute stems for the crate. Offline, slow, and never during a set."""
    from djai import stems as stems_mod

    if not stems_mod.available():
        print(
            "Stem separation needs PyTorch and torchaudio: pip install .[beats]",
            file=sys.stderr,
        )
        return 1
    crate = load_crate(Path(args.cache))
    if not crate:
        print("No tracks in the cache. Run `python -m djai analyze <folder>` first.")
        return 1
    if args.only:
        wanted = [t for t in crate if args.only.lower() in t.title.lower()]
        if not wanted:
            print(f"No cached track matches {args.only!r}", file=sys.stderr)
            return 1
        crate = wanted

    started = time.time()
    done = skipped = failed = 0
    written = 0
    for i, track in enumerate(crate, 1):
        if not args.force and stems_mod.cached(track.track_id, Path(args.cache)):
            skipped += 1
            print(f"[{i}/{len(crate)}] cached   {track.title}")
            continue
        t0 = time.time()
        try:
            manifest = stems_mod.precompute(track, Path(args.cache), force=args.force)
        except Exception as exc:  # one bad file must not stop the crate
            failed += 1
            log.error("stem separation failed for %s: %s", track.path, exc)
            print(f"[{i}/{len(crate)}] FAILED   {track.title}: {exc}", file=sys.stderr)
            continue
        size = sum(s["bytes"] for s in manifest["stems"].values())
        written += size
        done += 1
        print(
            f"[{i}/{len(crate)}] stems    {track.title}  "
            f"{time.time() - t0:.1f}s  {size / 2**20:.0f} MB  gain {manifest['gain']:.3f}"
        )
    print(
        f"\n{done} separated, {skipped} already cached, {failed} failed; "
        f"{written / 2**30:.2f} GB written in {(time.time() - started) / 60:.1f} min"
    )
    return 1 if failed else 0


def cmd_analyze(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    if not folder.exists():
        print(f"No such folder: {folder}", file=sys.stderr)
        return 1
    an.BEAT_TRACKER = args.tracker
    started = time.time()
    results, analyzed, skipped = analyze_folder(
        folder, Path(args.cache), force=args.force
    )
    print(
        f"\n{len(results)} track(s) in the crate: "
        f"{analyzed} analyzed, {skipped} already cached "
        f"({time.time() - started:.1f}s)"
    )
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    """Measure agreement with a person's structure annotations (SPEC §2).

    One JSON file per track in the folder: ``{"track": <title or id>,
    "sections": [{"label", "start_bar", "end_bar"}, ...], "vocal_bars":
    [[start, end], ...]}``, bars counted from the first downbeat, ends
    exclusive. Agreement is always measured first, against what analysis
    produced; ``--apply`` then records the annotations as authoritative
    corrections, which re-analysis never overwrites.
    """
    folder = Path(args.folder)
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    if not files:
        print(f"No annotation files (*.json) in {folder}", file=sys.stderr)
        return 1
    cache = Path(args.cache)
    crate = an.load_crate(cache)
    reports: list[dict] = []
    for f in files:
        try:
            ann = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"{f.name}: unreadable ({exc})")
            continue
        needle = str(ann.get("track") or f.stem)
        ta = _resolve_track(crate, needle)
        if ta is None:
            print(f"{f.name}: no cached track matches {needle!r}")
            continue
        reason = understanding.validate_structure(ann.get("sections"), ann.get("vocal_bars"))
        if reason is not None:
            print(f"{f.name}: {reason}")
            continue
        report = understanding.structure_agreement(ta, ann)
        report["already_corrected"] = bool(ta.structure_manually_corrected)
        reports.append(report)

        def pct(key: str) -> str:
            v = report.get(key)
            return "  n/a" if v is None else f"{v:5.0%}"

        print(
            f"{ta.title[:40]:40}  labels {pct('label_accuracy')}  "
            f"bounds found {pct('boundary_recall')} real {pct('boundary_precision')}  "
            f"vocals P {pct('vocal_precision')} R {pct('vocal_recall')}"
            + ("  (already corrected: compares with itself)" if report["already_corrected"] else "")
        )
        for d in report["label_disagreements"]:
            print(f"    bars {d['start_bar']}-{d['end_bar']}: ours {d['ours']}, yours {d['theirs']}")
        if report["missed_boundaries"] or report["false_boundaries"]:
            print(
                f"    missed boundaries {report['missed_boundaries']}, "
                f"false {report['false_boundaries']}"
            )
        if args.apply:
            understanding.correct_structure(
                ta, ann.get("sections"), ann.get("vocal_bars") if "vocal_bars" in ann else None
            )
            an.write_sidecar(ta, cache)

    fresh = [r for r in reports if not r["already_corrected"]]
    bars = sum(r["bars_compared"] for r in fresh)
    agree = sum(r["bars_compared"] * (r["label_accuracy"] or 0.0) for r in fresh)
    few = " (SPEC §2 asks for 10 or more)" if len(fresh) < 10 else ""
    if bars:
        print(f"\n{len(fresh)} track(s) compared{few}; "
              f"bar-label accuracy {agree / bars:.1%} over {bars} bars")
    else:
        print(f"\n{len(fresh)} track(s) compared{few}; no overlapping bars to score")
    if args.apply:
        print(f"Recorded {len(reports)} annotation(s) as manual structure corrections.")
    if args.report:
        Path(args.report).write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"Report written to {args.report}")
    return 0


def _resolve_track(crate: list[TrackAnalysis], needle: str) -> TrackAnalysis | None:
    """Find a cached track by id, exact title, path, or unique substring."""
    for t in crate:
        if needle in (t.track_id, t.title) or Path(t.path) == Path(needle):
            return t
    matches = [t for t in crate if needle.lower() in t.title.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(sorted(t.title for t in matches)[:6])
        print(f"{needle!r} matches {len(matches)} tracks: {names} ...", file=sys.stderr)
    return None


def run_shutdown_steps(*steps) -> list[str]:
    """Run every teardown step, whatever happens to the ones before it.

    Each step stands on its own. A Ctrl+C during the first one used to skip
    the rest, and the step it skipped was the one that closes the session --
    and with it the recording. Losing a set's recording to an impatient
    keypress at the end of the night is not a trade worth making.

    ``KeyboardInterrupt`` is caught deliberately, not by accident: every step
    passed here is bounded, so there is nothing left worth interrupting.

    Returns the names of the steps that did not close cleanly.
    """
    failed: list[str] = []
    for what, close in steps:
        if close is None:
            continue
        try:
            close()
        except BaseException as exc:  # noqa: BLE001 - see the docstring
            failed.append(what)
            print(
                f"  [!] {type(exc).__name__} while closing {what}; "
                f"carrying on with the rest of the shutdown",
                file=sys.stderr,
            )
    return failed


def cmd_render(args: argparse.Namespace) -> int:
    from djai.render import render_transition

    crate = load_crate(Path(args.cache))
    if not crate:
        print(f"No analysed tracks in {args.cache}.", file=sys.stderr)
        return 1

    track_a = _resolve_track(crate, args.track_a)
    track_b = _resolve_track(crate, args.track_b)
    if track_a is None or track_b is None:
        missing = args.track_a if track_a is None else args.track_b
        print(f"No cached track matching {missing!r}.", file=sys.stderr)
        return 1

    styles = (
        list(transition.STYLES) if args.style == "all" else [args.style]
    )
    print(f"A: {track_a.title}  {track_a.bpm:.2f} BPM  {track_a.camelot}")
    print(f"B: {track_b.title}  {track_b.bpm:.2f} BPM  {track_b.camelot}")

    import os
    import shutil

    stem = f"{track_a.title}__{track_b.title}"
    temp_dir: Path | None = None
    if args.out:
        base = Path(args.out)  # an explicit destination is always kept
    elif args.keep_renders:
        base = Path("renders") / stem
    else:
        # Renders are for inspection, and 25 left behind had grown to 254 MB.
        # By default they go to a scratch folder inside renders/ that is
        # removed when this command exits; --keep-renders keeps them.
        temp_dir = Path("renders") / (
            f"{RENDER_TEMP_PREFIX}{os.getpid()}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        base = temp_dir / stem
    try:
        for style in styles:
            out = base if len(styles) == 1 else Path(f"{base}__{style}")
            print(f"\n[{style}]")
            started = time.time()
            result = render_transition(
                track_a, track_b, out, blocksize=args.blocksize,
                transition_bars=args.bars, style=style,
                key_lock=False if args.no_key_lock else None,
            )
            elapsed = time.time() - started
            print(
                f"Rendered {result.duration_s:.1f}s of audio in {elapsed:.1f}s "
                f"({result.duration_s / max(elapsed, 1e-9):.0f}x realtime), "
                f"deck B rate {result.rate_b:.5f}"
            )
            print(f"  {result.wav_path}")
            print(f"  {result.envelope_path}")
            print(f"  {result.grid_path}")
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\nRemoved {temp_dir}. Pass --keep-renders to keep the files.")
    return 0


def cmd_reference(args: argparse.Namespace) -> int:
    from djai.reference import analyze_sets, write_profile

    sets_dir = Path(args.sets)
    markers = Path(args.markers)
    if not sets_dir.is_dir():
        print(f"No such folder: {sets_dir}", file=sys.stderr)
        return 1
    if not markers.exists():
        print(f"No such markers file or folder: {markers}", file=sys.stderr)
        return 1
    if markers.is_dir() and not sorted(markers.glob("*.txt")):
        print(f"No .txt markers files in {markers}", file=sys.stderr)
        return 1

    started = time.time()
    measurements, profile = analyze_sets(sets_dir, markers)
    if not measurements:
        print(f"No markers parsed from {markers}.", file=sys.stderr)
        return 1

    out = write_profile(profile, Path(args.out))
    m = profile["measures"]
    print(
        f"\n{profile['n_transitions']} transition(s) measured across "
        f"{profile['n_sets']} set(s), {profile['n_rejected']} rejected "
        f"({time.time() - started:.1f}s)"
    )
    for key in ("length_bars", "bpm_delta_pct", "level_change_db", "low_dip_db"):
        s = m.get(key, {})
        if s.get("n"):
            print(
                f"  {key:<16} median {s['median']:>8.2f}   "
                f"IQR [{s['iqr'][0]:.2f}, {s['iqr'][1]:.2f}]   n={s['n']}"
            )
    print(f"  grid alignment   {profile['grid_alignment']}")

    # Per set as well as pooled. Different DJs in different genres blend at
    # different lengths, and a pooled median across them can land in the gap
    # between the two and describe neither.
    if len(profile.get("by_set", {})) > 1:
        print("\n  per set:")
        for name, s in profile["by_set"].items():
            lb = s["measures"].get("length_bars", {})
            if not lb.get("n"):
                continue
            print(
                f"    {name[:44]:<46} n={s['n']:<3} {s['median_bpm']:>6.1f} BPM  "
                f"length {lb['median']:>6.2f} bars "
                f"[{lb['iqr'][0]:.1f}-{lb['iqr'][1]:.1f}]"
            )

    print(f"\nWrote {out}")
    if profile["n_transitions"] < 30:
        print(
            f"\n[!] Only {profile['n_transitions']} usable transitions. "
            "At least 30 are needed before these medians should move any "
            "config default.",
            file=sys.stderr,
        )
    return 0


#: Scratch folders `render` creates inside renders/ start with this, so `clean`
#: can recognise one a crashed run left behind.
RENDER_TEMP_PREFIX = "_tmp_"


def cmd_clean(args: argparse.Namespace) -> int:
    """List what can be deleted, with sizes, and delete it only on a typed yes."""
    from djai import housekeeping as hk
    from djai.supervisor import LOG_ROTATE_KEEP

    items = hk.find_reclaimable(
        renders_dir=Path(args.renders),
        record_dir=Path(args.record_dir),
        log_dir=Path(args.logs),
        cache_dir=Path(args.cache),
        keep_recordings=config.RECORD_RETENTION,
        keep_logs=LOG_ROTATE_KEEP,
    )
    if not items:
        print("Nothing to clean.")
        return 0

    total = sum(item.size for item in items)
    print(f"clean would delete {len(items)} item(s), {total / 2**20:,.1f} MB in all:\n")
    for item in items:
        print(f"  {item.size / 2**20:9.1f} MB  {item.path}  ({item.reason})")
    print(
        "\nNot touched: your music, reference sets, current cache entries, "
        "current session logs and the newest recordings."
    )
    try:
        answer = input("Type 'yes' to delete these: ")
    except EOFError:
        answer = ""
    if answer.strip().lower() != "yes":
        print("Nothing deleted.")
        return 0

    deleted, freed, failed = hk.delete(items)
    print(f"Deleted {deleted} item(s), freed {freed / 2**20:,.1f} MB.")
    for path, error in failed:
        print(f"  [!] could not delete {path}: {error}", file=sys.stderr)
    return 1 if failed else 0


def cmd_import(args: argparse.Namespace) -> int:
    """Import grids, cues and playlists from a Rekordbox XML export or Serato."""
    from collections import Counter as _Counter

    from djai import library_import as li

    rules: list[tuple[str, str]] = []
    for rule in args.relocate:
        if "=" not in rule:
            print(f"--relocate wants OLD=NEW, got {rule!r}", file=sys.stderr)
            return 2
        old, new = rule.split("=", 1)
        rules.append((old, new))
    source = Path(args.path)
    if not source.exists():
        print(f"No such file or folder: {source}", file=sys.stderr)
        return 1
    if args.source == "rekordbox":
        tracks = li.parse_rekordbox_xml(source)
    else:
        tracks = li.parse_serato(source, Path(args.root) if args.root else None)
    print(f"{len(tracks)} track(s) in {source}")
    summaries = li.apply_import(
        tracks, Path(args.cache), analyze_missing=not args.no_analyze,
        relocate_rules=rules,
    )
    for summary in summaries:
        print(summary.line())
    counts = _Counter(s.status for s in summaries)
    print(
        f"\n{len(summaries)} file(s): "
        + ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    )
    return 1 if counts.get("failed") else 0


def cmd_devices(_: argparse.Namespace) -> int:
    for index, name, api, channels in list_output_devices():
        print(f"{index:3d}  {name[:52]:52s} [{api}]  {channels}ch")
    return 0


def _parse_cue_channels(raw: str | None) -> tuple[int, int] | None:
    """``"3,4"`` -> ``(3, 4)``. 1-based, as every audio interface labels them."""
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace(":", ",").split(",")]
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"--cue-channels wants two channel numbers, got {raw!r}")
    lo, hi = int(parts[0]), int(parts[1])
    if hi != lo + 1 or lo < 1:
        raise ValueError(
            f"--cue-channels wants an adjacent 1-based pair like 3,4; got {raw!r}"
        )
    return (lo, hi)


def cmd_preflight(args: argparse.Namespace) -> int:
    recording = (args.record or config.RECORD_ENABLED) and not args.no_record
    record_dir = Path(args.record_dir) if recording else None

    def progress(i: int, total: int, title: str) -> None:
        if i == 1 or i % 10 == 0 or i == total:
            print(f"      decoding {i}/{total}: {title[:48]}", flush=True)

    print(f"Pre-flight: cache {args.cache}, logs {args.logs}")
    report = run_preflight(
        cache_dir=Path(args.cache),
        log_dir=Path(args.logs),
        blocksize=args.blocksize,
        require_ollama=not args.no_llm,
        record_dir=record_dir,
        decode_limit=args.decode_limit,
        progress=progress if args.verbose else None,
    )

    print()
    for check in report.checks:
        print(check)
        for warning in check.warnings:
            print(f"       [!] {warning}")

    print()
    if report.ok:
        print("PREFLIGHT PASSED - ready to play.")
        return 0
    print(f"PREFLIGHT FAILED - {len(report.failures)} problem(s):")
    for check in report.failures:
        print(f"  {check.name}: {check.detail}")
    return 1


def cmd_play(args: argparse.Namespace) -> int:
    crate = load_crate(Path(args.cache))
    if not crate:
        print(
            f"No analysed tracks in {args.cache}. "
            "Run `python -m djai analyze <folder>` first.",
            file=sys.stderr,
        )
        return 1
    print(f"Crate: {len(crate)} track(s) from {args.cache}")

    try:
        cue_channels = _parse_cue_channels(args.cue_channels)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if cue_channels is not None and args.cue_device is not None:
        print(
            "--cue-device and --cue-channels are alternatives: the first opens a "
            "second stream, the second routes within one device.",
            file=sys.stderr,
        )
        return 1

    intent_engine = IntentEngine()
    if args.no_llm:
        # The documented switch for this: with `available` false, every
        # interpret() returns from the keyword chain without touching HTTP.
        intent_engine.available = False
        print("Model disabled (--no-llm): commands are matched by keyword only.")
    else:
        # Warm the model on its own thread, and open the audio stream without
        # waiting for it. A cold 8B load off disk measured 62 s on this machine
        # -- past the 60 s warmup budget -- and warming before the stream opened
        # meant a minute in which the room could not be played at all. Nothing
        # downstream waits on this: the autopilot's cueing and selection are
        # rule-based, and a transition design that is not back in time runs as
        # the preset instead.
        print(
            f"Loading model in the background... "
            f"({intent_engine.model} at {intent_engine.base_url})"
        )

        def warm() -> None:
            ok, detail = intent_engine.warmup()
            if ok:
                print(f"[model] {detail}")
            elif intent_engine.available:
                print(f"[model] [!] {detail}")
            else:
                print(
                    f"[model] [!] {detail}\n"
                    "[model] Continuing in fallback-only mode: typed commands are "
                    "matched by keyword instead.\n"
                    f"[model] Start Ollama and re-run to enable the model "
                    f"(expected at {intent_engine.base_url})."
                )

        threading.Thread(target=warm, name="djai-model-warmup", daemon=True).start()

    record_path = None
    # Off unless asked for. `--no-record` is still accepted and still wins, so
    # a script written when recording was the default keeps working.
    if (args.record or config.RECORD_ENABLED) and not args.no_record:
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        record_path = Path(args.record_dir) / f"set_{stamp}.flac"

    session = Session(
        crate,
        device=args.device,
        blocksize=args.blocksize,
        log_dir=Path(args.logs),
        cue_device=args.cue_device,
        cue_channels=cue_channels,
        record_path=record_path,
        fallback_enabled=not args.no_fallback,
        cache_dir=Path(args.cache),
    )
    # Used only to design transitions during the pre-roll, on its own thread.
    session.intent_engine = intent_engine
    # Crash-safe set state (Phase 3.2/3.3): a killed set resumes here.
    print(session.attach_set_state(Path(args.logs) / "set_state.json"))
    # SPEC §6: the set is planned and critic-scored, and both are logged,
    # before anything plays.
    plan = session.replan("planned")
    if plan is not None:
        print(plan.summary())
    try:
        session.start()
    except Exception as exc:
        print(f"Could not start audio: {exc}", file=sys.stderr)
        print("Try `python -m djai devices` and pass --device <index>.", file=sys.stderr)
        return 1

    print(f"Audio device {session.engine.device}, blocksize {args.blocksize}")
    if session.engine.cue_mode == "none":
        if args.cue_device is not None or cue_channels is not None:
            print("  [!] cue output unavailable - running MASTER ONLY")
        else:
            print("  Cue output: none configured - master only")
    else:
        print(f"  Cue output: {session.engine.cue_mode} (say `cue a` / `cue b`)")
    if record_path is not None:
        print(f"Recording: {record_path}")
    print(f"Session log: {session.session_log.path}")

    ui = None
    if args.ui:
        from djai.ui_server import UIServer

        ui = UIServer(session, intent_engine, host=args.ui_host, port=args.ui_port)
        try:
            ui.start()
            print(f"UI: {ui.url}   (the terminal below stays live)")
        except Exception as exc:
            print(f"Could not start the UI: {exc}", file=sys.stderr)
            ui = None

    try:
        if args.autostart:
            if not session.start_first_track():
                return 1
            if session.arm_fallback():
                print(f"Fallback armed: {session.watchdog.player.title}")
        else:
            # Idle by default. Nothing is selected, decoded or scheduled until
            # an operator loads a deck: no selector call, no transition plan and
            # no model call happens while both decks are EMPTY.
            print(
                "Idle: both decks empty. Load a deck from the UI, or type "
                "`play <track>` here. Use --autostart to open with a track."
            )
            # The safety net needs decoded audio, and nothing is decoded yet, so
            # it arms itself once the first track is actually playing.
            session.autoarm_fallback = True
            print("Fallback: arms once the first track starts.")
        repl(session, intent_engine)
    finally:
        run_shutdown_steps(
            ("the web UI", ui.stop if ui is not None else None),
            ("the session", session.shutdown),
            ("the model client", intent_engine.close),
        )
        print(
            f"Session over. {session.engine.frames_played / SAMPLE_RATE:.0f}s played, "
            f"{session.engine.underruns} underrun(s), "
            f"{session.supervisor.interventions} supervisor intervention(s)."
        )
        if session.recorder is not None:
            rec = session.recorder
            note = f"Recording: {rec.path} ({rec.seconds_written:.0f}s)"
            if rec.dropped_frames:
                note += f", {rec.dropped_frames} frames dropped by the writer"
            if rec.error:
                note += f", STOPPED EARLY: {rec.error}"
            print(note)
        if session.watchdog is not None and session.watchdog.tripped:
            print(f"[!] The fallback player took over: {session.watchdog.reason}")
        print(f"Log: {session.session_log.path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="djai", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="analyse a folder of tracks (offline)")
    p_analyze.add_argument("folder", help="folder of audio files, searched recursively")
    p_analyze.add_argument("--cache", default=str(DEFAULT_CACHE_DIR))
    p_analyze.add_argument(
        "--force", action="store_true", help="re-analyse even if cached"
    )
    p_analyze.add_argument(
        "--tracker", choices=an.BEAT_TRACKERS, default=an.BEAT_TRACKER,
        help="beat tracker: Beat This! when installed (auto), required "
             "(beat_this), or the original librosa detector",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    p_annotate = sub.add_parser(
        "annotate",
        help="compare analysed structure with your annotations; --apply keeps them",
    )
    p_annotate.add_argument("folder", help="folder of per-track annotation .json files")
    p_annotate.add_argument("--cache", default=str(DEFAULT_CACHE_DIR))
    p_annotate.add_argument(
        "--apply", action="store_true",
        help="after reporting, record the annotations as manual corrections",
    )
    p_annotate.add_argument("--report", help="also write the full report as JSON here")
    p_annotate.set_defaults(func=cmd_annotate)

    p_play = sub.add_parser("play", help="start the engine and the REPL")
    p_play.add_argument("--cache", default=str(DEFAULT_CACHE_DIR))
    p_play.add_argument("--logs", default=str(DEFAULT_LOG_DIR))
    p_play.add_argument("--device", type=int, default=None, help="output device index")
    p_play.add_argument("--blocksize", type=int, default=DEFAULT_BLOCKSIZE)
    p_play.add_argument(
        "--ui", action="store_true", help="also serve the web UI (REPL stays live)"
    )
    p_play.add_argument("--ui-host", default="127.0.0.1")
    p_play.add_argument("--ui-port", type=int, default=8765)
    p_play.add_argument(
        "--cue-device", type=int, default=None,
        help="second output device for pre-listen (see `djai devices`)",
    )
    p_play.add_argument(
        "--cue-channels", default=None, metavar="LO,HI",
        help="pre-listen on this 1-based channel pair of the master device, "
             "e.g. 3,4 (alternative to --cue-device)",
    )
    p_play.add_argument(
        "--no-llm", action="store_true",
        help="never call Ollama; keyword matching only",
    )
    p_play.add_argument(
        "--record-dir", default=config.RECORD_DIR,
        help="where to write the session recording",
    )
    p_play.add_argument(
        "--record", action="store_true",
        help="record the master output to FLAC (off by default)",
    )
    p_play.add_argument(
        "--no-record", action="store_true",
        help="never record, even if RECORD_ENABLED is set",
    )
    p_play.add_argument(
        "--no-fallback", action="store_true",
        help="do not arm the fallback player (not recommended in a club)",
    )
    p_play.add_argument(
        "--autostart", action="store_true",
        help="pick a track and start playing at launch (default: start idle)",
    )
    p_play.set_defaults(func=cmd_play)

    p_render = sub.add_parser(
        "render", help="render one A->B transition offline for inspection"
    )
    p_render.add_argument("track_a", help="cached track: id, title, or substring")
    p_render.add_argument("track_b", help="cached track: id, title, or substring")
    p_render.add_argument("--out", default=None, help="output path stem")
    p_render.add_argument("--cache", default=str(DEFAULT_CACHE_DIR))
    p_render.add_argument(
        "--style", default="bass_swap",
        choices=list(transition.STYLES) + ["all"],
        help="transition style to render, or `all` for one render per style",
    )
    p_render.add_argument("--blocksize", type=int, default=None)
    p_render.add_argument(
        "--bars", type=float, default=None, help="transition length in bars"
    )
    p_render.add_argument(
        "--keep-renders", action="store_true",
        help="keep the rendered files in renders/ (default: removed on exit)",
    )
    p_render.add_argument(
        "--no-key-lock", action="store_true",
        help="resample deck B to tempo instead of stretching it at original pitch",
    )
    p_render.set_defaults(func=cmd_render)

    p_ref = sub.add_parser(
        "reference", help="measure transitions in recorded sets you supply"
    )
    p_ref.add_argument("sets", help="folder of recorded DJ sets")
    p_ref.add_argument(
        "markers",
        help="markers .txt file, or a folder of them (one per set)",
    )
    p_ref.add_argument("--out", default="reference_profile.json")
    p_ref.set_defaults(func=cmd_reference)

    p_pre = sub.add_parser(
        "preflight", help="check everything needed to play, before the doors open"
    )
    p_pre.add_argument("--cache", default=str(DEFAULT_CACHE_DIR))
    p_pre.add_argument("--logs", default=str(DEFAULT_LOG_DIR))
    p_pre.add_argument("--blocksize", type=int, default=DEFAULT_BLOCKSIZE)
    p_pre.add_argument(
        "--no-llm", action="store_true",
        help="do not require Ollama (the set will run rule-based)",
    )
    p_pre.add_argument("--record-dir", default=config.RECORD_DIR)
    p_pre.add_argument(
        "--record", action="store_true",
        help="also check space for a recording (recording is off by default)",
    )
    p_pre.add_argument(
        "--no-record", action="store_true", help="do not check recording space"
    )
    p_pre.add_argument(
        "--decode-limit", type=int, default=None, metavar="N",
        help="decode only the first N tracks (0 = all; default from config)",
    )
    p_pre.add_argument(
        "-v", "--verbose", action="store_true", help="print decode progress"
    )
    p_pre.set_defaults(func=cmd_preflight)

    p_import = sub.add_parser(
        "import",
        help="import beat grids, hot cues and playlists from Rekordbox XML or Serato",
    )
    p_import.add_argument("source", choices=["rekordbox", "serato"])
    p_import.add_argument(
        "path", help="a Rekordbox XML export, or a _Serato_ folder or .crate file"
    )
    p_import.add_argument("--cache", default=str(DEFAULT_CACHE_DIR))
    p_import.add_argument(
        "--root", default=None,
        help="Serato: where its relative track paths start (default: the _Serato_ folder's drive)",
    )
    p_import.add_argument(
        "--relocate", action="append", default=[], metavar="OLD=NEW",
        help="rewrite a path prefix, for a library that has moved (repeatable)",
    )
    p_import.add_argument(
        "--no-analyze", action="store_true",
        help="only update tracks already in the cache; do not analyse new ones",
    )
    p_import.set_defaults(func=cmd_import)

    p_stems = sub.add_parser(
        "stems", help="precompute drum/bass/vocal/other stems (offline, GPU)"
    )
    p_stems.add_argument("--cache", default=str(DEFAULT_CACHE_DIR))
    p_stems.add_argument(
        "--force", action="store_true", help="re-separate even if cached"
    )
    p_stems.add_argument(
        "--only", default="", metavar="TITLE",
        help="only tracks whose title contains this",
    )
    p_stems.set_defaults(func=cmd_stems)

    p_clean = sub.add_parser(
        "clean", help="report reclaimable disk space; delete it on confirmation"
    )
    p_clean.add_argument("--cache", default=str(DEFAULT_CACHE_DIR))
    p_clean.add_argument("--logs", default=str(DEFAULT_LOG_DIR))
    p_clean.add_argument("--record-dir", default=config.RECORD_DIR)
    p_clean.add_argument("--renders", default="renders")
    p_clean.set_defaults(func=cmd_clean)

    p_dev = sub.add_parser("devices", help="list usable audio output devices")
    p_dev.set_defaults(func=cmd_devices)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
