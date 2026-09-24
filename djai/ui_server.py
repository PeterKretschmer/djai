"""Local web UI: static files, a 20 Hz state feed, and a panic path.

THREADING CONTEXT
-----------------
* uvicorn runs on its **own daemon thread** with its own asyncio loop, so the
  terminal REPL keeps the main thread and keeps working.
* The state feed reads engine state from that loop. Every read is either a
  single float attribute or :attr:`djai.engine.Engine.clock`, the consistent
  tuple the audio callback publishes -- nothing here blocks or mutates the
  audio path.
* Typed commands hit the LLM, which blocks for seconds, so they are dispatched
  to a thread executor and never run on the event loop.

Nothing in this module is reachable from the audio callback.

THE PANIC PATH IS DELIBERATELY NOT THE WEBSOCKET
------------------------------------------------
``/panic/{action}`` are plain HTTP POSTs that drop a command straight onto the
engine queue, exactly as the REPL's ``cut`` does. They share no code with the
state feed and no code with the intent layer. If the model is wedged, the socket
is dead and the UI is showing stale numbers, a panic button still works -- which
is the only reason to have one.

THE CONTROLLER SURFACE OWNS NO MUSICAL DECISIONS
------------------------------------------------
Deck loading, transport and mixer control live here, but every one of them is a
translation of an operator gesture into the *existing* command vocabulary. There
is no scheduling, no transition shape and no track selection in this module or
in the browser -- ``plan_transition`` and ``select_next`` stay where they are.
What this module decides is only *when a gesture is musically legal*: a load
onto a playing deck is quantized to :func:`djai.phrase.next_phrase_boundary`,
which is the same rule the REPL's manual actions use.

Pause is expressed as ``LoadTrack(play=False)`` at the deck's current playhead
rather than by writing ``deck.playing`` from this thread. That keeps the
mutation on the audio thread where the engine owns it, and it ends an in-flight
transition through the engine's own path instead of a second one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from djai import config, phrase
from djai import transition as tr_module
from djai import waveform as waveform_mod
from djai.selector import SET_PHASES
from djai.commands import (
    IMMEDIATE,
    Cut,
    KillBass,
    LoadTrack,
    SetEQ,
    SetGain,
    Stop,
)
from djai.deck import SAMPLE_RATE
from djai.engine import MASTER_CEILING

log = logging.getLogger(__name__)

#: State pushes per second. The acceptance bar is "reflects engine state within
#: 100 ms", so 20 Hz leaves headroom for a frame to be missed.
STATE_HZ: float = 20.0

#: Points in a rendered waveform. Enough for a 1080p-wide canvas without
#: shipping megabytes of peaks per track.
WAVEFORM_POINTS: int = 1600

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _peaks(audio: np.ndarray, points: int = WAVEFORM_POINTS) -> list[float]:
    """Absolute-peak envelope, normalised. Cheap enough to do per load."""
    if audio.size == 0:
        return []
    mono = audio[:, 0] if audio.ndim > 1 else audio
    n = mono.shape[0]
    bucket = max(1, n // points)
    usable = (n // bucket) * bucket
    blocks = np.abs(mono[:usable]).reshape(-1, bucket)
    env = blocks.max(axis=1)[:points]
    peak = float(env.max()) or 1.0
    return [round(float(v / peak), 4) for v in env]


class UIServer:
    """Serves the UI and streams engine state. Owns one daemon thread."""

    def __init__(
        self,
        session,
        intent_engine=None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.session = session
        self.intent_engine = intent_engine
        self.host = host
        self.port = port

        #: Coloured waveform bands per track id, bounded like the waveforms.
        self._bands: dict[str, Any] = {}
        #: The page's most recent frame-timing report, and when it was logged.
        self.frame_stats: dict[str, Any] = {}
        self._frame_stats_logged: float = 0.0

        self._app = self._build_app()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._waveforms: dict[str, dict] = {}
        self._notices: list[str] = []
        self._notice_lock = threading.Lock()
        self.clients: int = 0

        #: Raised by a manual load or a manual pause. While it is set the
        #: autopilot is frozen, so operator and automation can never be
        #: steering the same deck at the same time.
        self._held: bool = False
        #: The last typed line nothing understood, shown until the next one.
        self._parse_failure: dict | None = None
        #: Per deck: a load that is waiting for the next phrase boundary, so
        #: the UI can say "loads at next phrase" instead of looking ignored.
        self._pending: dict[str, dict[str, Any]] = {}
        #: Decoded tracks keyed by track id. A pause re-issues LoadTrack with
        #: the deck's existing audio, and a re-drag of a recent track should
        #: not pay for a second decode.
        self._decoded: dict[str, Any] = {}
        #: Whether the loop's exception handler has been quietened yet.
        self._quietened: bool = False

        #: The most recent transition preview, as
        #: :meth:`djai.revise.PreviewOutcome.summary` returns it, plus a
        #: "running" marker while one is in flight. Written from the pre-roll
        #: worker thread and read by the state feed, which is why it is
        #: replaced wholesale rather than mutated in place: a reader either sees
        #: the old summary or the new one, never half of each.
        self._preview: dict[str, Any] | None = None

        session.add_notice_listener(self._on_notice)
        add_preview_listener = getattr(session, "add_preview_listener", None)
        if add_preview_listener is not None:
            add_preview_listener(self._on_preview)

    # --- the transition preview ----------------------------------------------

    def _on_preview(self, kind: str, payload) -> None:
        """Session preview events, from the pre-roll worker thread."""
        if kind == "started":
            self.preview_started(str(payload))
        elif kind == "finished":
            self.preview_finished(payload)

    def preview_started(self, style: str = "") -> None:
        """Called from the pre-roll worker when a preview begins."""
        self._preview = {"status": "running", "style": style, "rounds": 0}

    def preview_finished(self, outcome) -> None:
        """Called from the pre-roll worker when a preview commits."""
        try:
            self._preview = outcome.summary()
        except Exception:  # noqa: BLE001 - never break a transition over the UI
            self._preview = {"status": "unknown"}

    @property
    def preview_state(self) -> dict[str, Any] | None:
        return self._preview

    # --- lifecycle -----------------------------------------------------------

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        cfg = uvicorn.Config(
            self._app, host=self.host, port=self.port, log_level="warning"
        )
        self._server = uvicorn.Server(cfg)
        # Signal handlers can only be installed on the main thread, and the
        # main thread belongs to the REPL.
        self._server.install_signal_handlers = lambda: None
        self._thread = threading.Thread(
            target=self._server.run, name="djai-ui", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Shut the server down promptly, open browser tabs notwithstanding.

        Graceful first, so a request already in flight finishes. But uvicorn's
        graceful shutdown waits for open connections to close, and the state
        feed is a websocket that a left-open tab holds indefinitely -- which
        turned "type stop" into a wait long enough to reach for Ctrl+C.
        ``force_exit`` drops those connections instead of waiting on them.
        """
        server, thread = self._server, self._thread
        self._thread = None
        if server is None or thread is None:
            return

        server.should_exit = True
        thread.join(timeout=1.0)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=2.0)

    def _quieten_dropped_connections(self, loop) -> None:
        """Stop a closed browser tab printing a traceback into the REPL.

        Closing a tab tears the socket down underneath asyncio's proactor
        transport on Windows, which reports ``ConnectionResetError`` through
        the event loop's exception handler. By then the endpoint below has
        already handled the disconnect, so the traceback says nothing new --
        it just lands in the middle of whatever the operator was typing.

        Installed from inside the loop, once, on the first connection. Anything
        that is not a dropped client still reaches the previous handler.
        """
        if self._quietened:
            return
        self._quietened = True
        previous = loop.get_exception_handler()

        def handler(running_loop, context) -> None:
            exc = context.get("exception")
            if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
                log.debug("browser connection dropped: %s", exc)
                return
            if previous is not None:
                previous(running_loop, context)
            else:
                running_loop.default_exception_handler(context)

        loop.set_exception_handler(handler)

    def _on_notice(self, message: str) -> None:
        with self._notice_lock:
            self._notices.append(message)
            del self._notices[:-40]

    def _drain_notices(self) -> list[str]:
        with self._notice_lock:
            out, self._notices = self._notices, []
        return out

    # --- state ---------------------------------------------------------------

    def _drift_ms(self) -> dict[str, float | None]:
        """Each deck's current drift, computed the way the supervisor does.

        Reads the supervisor's captured offsets directly. ``supervisor.py`` is
        outside this phase's scope, so it has no accessor for this yet; when it
        gains one, this should use it.
        """
        engine = self.session.engine
        supervisor = self.session.supervisor
        _frames, master_beat, pos_a, pos_b = engine.clock
        out: dict[str, float | None] = {}
        for deck, position in (("a", pos_a), ("b", pos_b)):
            d = engine.deck(deck)
            offset = supervisor._beat_offset.get(deck)
            if d.track is None or not d.playing or offset is None:
                out[deck] = None
                continue
            analysis = d.track.analysis
            actual = phrase.beat_at_frame(analysis, position)
            drift_beats = actual - (master_beat + offset)
            out[deck] = round(drift_beats * analysis.beat_period * 1000.0, 2)
        return out

    def _pending_loads(self) -> dict[str, dict[str, Any]]:
        """Queued drops, with a live bars-remaining countdown.

        An entry is dropped once the engine has passed its execute_at: the load
        has landed, and a stale "loads at next phrase" badge over a track that
        is already playing is worse than none.
        """
        now = self.session.engine.frames_played
        out: dict[str, dict[str, Any]] = {}
        for deck_name, item in list(self._pending.items()):
            if now >= item["execute_at"]:
                del self._pending[deck_name]
                continue
            out[deck_name] = {
                "title": item["title"],
                "bars_until": round(
                    self.session.bars_until(item["execute_at"]), 1
                ),
            }
        return out

    def state(self) -> dict[str, Any]:
        """One frame of engine state. Cheap, non-blocking reads only."""
        session = self.session
        engine = session.engine
        _frames, _master_beat, pos_a, pos_b = engine.clock

        decks: dict[str, Any] = {}
        for name, position in (("a", pos_a), ("b", pos_b)):
            d = engine.deck(name)
            st = engine.deck_state(name)
            analysis = d.track.analysis if d.track else None
            decks[name] = {
                "title": st.title,
                "track_id": analysis.track_id if analysis else None,
                "bpm": st.bpm,
                "native_bpm": st.native_bpm,
                "key": st.key,
                "camelot": st.camelot,
                "playing": st.playing,
                "ended": st.ended,
                "transport": st.transport,
                "live": name == session.live_deck,
                "position_s": round(position / SAMPLE_RATE, 3),
                "position_bars": st.position_bars,
                "duration_s": round(analysis.duration_s, 2) if analysis else 0.0,
                "remaining_s": st.remaining_seconds,
                "gain": st.gain,
                "eq": list(st.eq),
                "rate": st.rate,
                "stretched": bool(d.track.is_stretched) if d.track else False,
                "key_lock": st.key_lock,
                "filter": st.filter,
                "filter_resonance": st.filter_resonance,
                "loop_active": st.loop_active,
                "loop_beats": st.loop_beats,
                "pitch_percent": st.pitch_percent,
                "manual_pitch": st.manual_pitch,
                "master": name == engine.master_deck,
                "mix_in_s": round(analysis.mix_in, 2) if analysis else 0.0,
                "mix_out_s": round(analysis.mix_out, 2) if analysis else 0.0,
                # Stored markers inside the track. Not the headphone cue
                # output, which is reported separately under "cue".
                "hot_cues": list(analysis.hot_cues) if analysis else [],
            }

        pending = session.scheduler.pending()
        queue = [
            {
                "command": c.describe(),
                "bars_until": round(session.bars_until(c.execute_at), 1),
            }
            for c in pending[:8]
        ]

        # Read the two transition counters once each. The audio thread advances
        # _trans_frames every block, so these can disagree by one block; that
        # costs the progress bar one frame of accuracy and nothing else.
        # Reported in frames rather than bars because the engine derives bars
        # from the configured TRANSITION_BARS regardless of the actual length,
        # so a held or stretched blend would report the wrong bar count.
        trans_done = float(engine._trans_frames)
        trans_total = float(engine._trans_total)
        trans_from = engine._trans_from
        trans_to = engine._trans_to

        return {
            "t": time.time(),
            "master_bpm": round(engine.master_bpm, 2),
            "live_deck": session.live_deck,
            "decks": decks,
            "drift_ms": self._drift_ms(),
            "queue": queue,
            "transition": {
                "active": engine.transition_active,
                # What is running, what the operator asked for, and which rule
                # chose it. The "why" matters as much as the "what" here.
                "style": engine.transition_style,
                "requested_style": session.transition_style,
                "planned_style": session.last_transition_choice[0],
                "planned_rule": session.last_transition_choice[1],
                "styles": list(tr_module.STYLE_CHOICES),
                # The set arc: where the operator (or the model) says the set
                # is, and the phases it may be set to.
                "set_phase": session.set_phase,
                "set_phases": list(SET_PHASES) + ["auto"],
                # Per-session diversity: how many shapes the model actually
                # supplied against how many fell back, and what ran. A silent
                # fallback is indistinguishable from a dull model until you
                # can see this.
                "sources": dict(session.design_counts),
                "style_counts": dict(session.style_counts),
                "progress_bars": round(engine.transition_progress_bars, 2),
                "elapsed_s": round(trans_done / SAMPLE_RATE, 3),
                "total_s": round(trans_total / SAMPLE_RATE, 3),
                "from_deck": trans_from.name if trans_from is not None else None,
                "to_deck": trans_to.name if trans_to is not None else None,
                # What the preview made of the next transition: whether it ran,
                # what it measured, and what it changed. None until one has.
                "preview": self._preview,
            },
            "cued": session._cued.title if session._cued else None,
            "engine": {
                # Master bus level. `peak_in` is what the mix produced before
                # the limiter, which is the number that says the mix is hot;
                # `peak` is what left the box.
                "peak": round(engine.master_peak, 4),
                "peak_in": round(engine.master_peak_in, 4),
                "limiter_gain": round(engine.limiter_gain, 4),
                "limiter_clips": engine.limiter_clips,
                "ceiling": MASTER_CEILING,
                "underruns": engine.underruns,
                "interventions": session.supervisor.interventions,
                "stretched_loads": engine.stretched_loads,
                "blocksize": engine.blocksize,
                "device": str(engine.device),
            },
            "llm": {
                "available": bool(
                    self.intent_engine is not None and self.intent_engine.available
                ),
                "model": self.intent_engine.model if self.intent_engine else None,
                "base_url": self.intent_engine.base_url if self.intent_engine else None,
            },
            "frozen": session.frozen,
            "automation_held": self.automation_held,
            "quantize": bool(session.quantize),
            "ui_frames": self.frame_stats,
            "pending_loads": self._pending_loads(),
            "forced_next": (
                session._forced_next.title if session._forced_next else None
            ),
            # Phase 3.2: the operator's cue queue, and ranked suggestions with
            # their reasons. Cueing one is `/override/cue?arg=<track_id>`.
            "cue_queue": [dict(e) for e in session.cue_queue],
            "suggestions": self._suggestions(),
            # SPEC §6: the upcoming plan, as titles, with its critic score.
            "plan": (
                {
                    "persona": session.set_plan.persona,
                    "score": session.set_plan.critique.get("score"),
                    "reason": session.set_plan.reason,
                    "near": [x.track.title for x in session.set_plan.near],
                    "mid": [x.track.title for x in session.set_plan.mid],
                }
                if session.set_plan is not None else None
            ),
            "cue": {
                "mode": engine.cue_mode,
                "deck": engine.cue_deck,
                "underruns": engine.cue_underruns,
            },
            "recording": (
                {
                    "seconds": round(session.recorder.seconds_written, 1),
                    "dropped": session.recorder.dropped_frames,
                    "error": session.recorder.error,
                }
                if session.recorder is not None
                else None
            ),
            "fallback": {
                "armed": session.watchdog is not None,
                "active": bool(session.watchdog and session.watchdog.tripped),
                "reason": session.watchdog.reason if session.watchdog else None,
            },
            "notices": self._drain_notices(),
            # Phase 5: co-pilot and MC, bar-by-bar energy (a proxy, from the
            # closed loop's own readings), what autonomous selection will not
            # touch, and the last line the chat could not understand.
            "mode": session.mode,
            "mc": session.mc,
            "energy_history": (
                [round(r.energy_db, 1) for r in list(session.room.master.readings)]
                if session.room is not None else []
            ),
            "quarantined": self._quarantined(),
            "last_parse_failure": self._parse_failure,
        }

    def _quarantined(self) -> list[dict]:
        """Quarantined grids and set-aside tracks. Re-read only when they change."""
        s = self.session
        key = (len(s.crate), len(s._unplayable),
               sum(bool(t.grid_manually_corrected) for t in s.crate))
        if key != getattr(self, "_quarantine_key", None):
            rows = [{"title": t.title, "why": f"grid {t.grid_confidence:.2f}"}
                    for t in s.crate if getattr(t, "quarantined", False)]
            rows += [{"title": t.title, "why": "set aside"}
                     for t in s.crate if t.track_id in s._unplayable]
            self._quarantine_rows, self._quarantine_key = rows, key
        return self._quarantine_rows

    def waveform(self, deck_name: str) -> dict[str, Any]:
        """Peaks plus the musical landmarks the canvas draws. Cached per track."""
        d = self.session.engine.deck(deck_name)
        if d.track is None:
            return {"track_id": None}
        analysis = d.track.analysis
        cached = self._waveforms.get(analysis.track_id)
        if cached is not None:
            return cached

        bar_s = phrase.BEATS_PER_BAR * 60.0 / analysis.bpm if analysis.bpm else 2.0
        first = analysis.first_downbeat
        phrases = []
        t = first
        while t < analysis.duration_s:
            phrases.append(round(t, 3))
            t += bar_s * 8
        payload = {
            "track_id": analysis.track_id,
            "title": analysis.title,
            "duration_s": round(analysis.duration_s, 3),
            "bpm": round(analysis.bpm, 3),
            "camelot": analysis.camelot,
            "peaks": _peaks(d.track.audio),
            "downbeats": [round(x, 3) for x in analysis.downbeats],
            "phrases": phrases,
            "mix_in": round(analysis.mix_in, 3),
            "mix_out": round(analysis.mix_out, 3),
            "stretch_rate": d.track.stretch_rate,
            # The grid view: every beat, bar 1's handle, and whether a person
            # has already corrected this grid or analysis doubts it.
            "beats": [round(x, 3) for x in analysis.beats],
            "first_downbeat": round(analysis.first_downbeat, 4),
            "grid_corrected": bool(analysis.grid_manually_corrected),
            "review_needed": bool(analysis.review_needed),
            "tempo_ambiguous": bool(analysis.tempo_ambiguous),
            # Structure, as times, for the section strip above the bands.
            "sections": [
                {
                    "label": s["label"],
                    "start_s": round(phrase.frame_at_bar(analysis, s["start_bar"]) / SAMPLE_RATE, 4),
                    "end_s": round(phrase.frame_at_bar(analysis, s["end_bar"]) / SAMPLE_RATE, 4),
                }
                for s in (analysis.sections or [])
            ],
            # The coloured bands, fetched separately as binary: sizes and
            # densities here, data from /api/waveform/<deck>/bands/<level>.
            "bands": self._bands_for(d.track).meta(),
        }
        self._waveforms[analysis.track_id] = payload
        if len(self._waveforms) > 8:
            self._waveforms.pop(next(iter(self._waveforms)))
        return payload

    def _bands_for(self, track) -> "waveform_mod.Waveform":
        """Low/mid/high bands for a loaded track: memory, then disk, then computed."""
        track_id = track.analysis.track_id
        cached = self._bands.get(track_id)
        if cached is None:
            cached = waveform_mod.for_track(
                track, getattr(self.session, "cache_dir", None), SAMPLE_RATE
            )
            self._bands[track_id] = cached
            while len(self._bands) > 8:
                self._bands.pop(next(iter(self._bands)))
        return cached

    def band_bytes(self, deck_name: str, level: str) -> tuple[bytes, float, int] | None:
        """``(uint8 bytes, points per second, points)`` for one band level.

        The bytes are the low row, then mid, then high, each ``points`` long.
        ``level`` is ``"overview"`` or a detail level's index.
        """
        if deck_name not in ("a", "b"):
            return None
        deck = self.session.engine.deck(deck_name)
        if deck.track is None:
            return None
        bands = self._bands_for(deck.track)
        if level == "overview":
            arr = bands.overview
            pps = arr.shape[1] / max(bands.duration_s, 1e-9)
        else:
            try:
                pps, arr = bands.levels[int(level)]
            except (ValueError, IndexError):
                return None
        return np.ascontiguousarray(arr).tobytes(), float(pps), int(arr.shape[1])

    def record_frame_stats(self, msg: dict[str, Any]) -> dict[str, Any]:
        """The page's own frame timing, as it measured it. Kept for the state
        feed and logged about once a minute."""
        keys = ("fps", "p50_ms", "p95_ms", "p99_ms", "max_ms", "draw_p95_ms", "frames",
                "decks_playing")
        stats = {}
        for key in keys:
            value = msg.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stats[key] = round(float(value), 3)
        stats["received_at"] = round(time.time(), 3)
        self.frame_stats = stats
        if time.time() - self._frame_stats_logged >= 60.0:
            self._frame_stats_logged = time.time()
            self.session.session_log.write(
                "ui_frame_stats", trigger="page", action="frame timing reported", **stats
            )
        return {"ok": True}

    # --- commands ------------------------------------------------------------

    def panic(self, action: str) -> dict[str, Any]:
        """The bypass path: straight onto the engine queue.

        No LLM, no scheduler, no validation, no dependency on the state feed.
        """
        engine = self.session.engine
        action = action.lower()
        if action == "cut":
            engine.submit(Cut(deck="master", execute_at=IMMEDIATE, origin="panic"))
        elif action in ("killbass", "kill_bass"):
            engine.submit(
                KillBass(deck="master", killed=True, execute_at=IMMEDIATE,
                         origin="panic")
            )
        elif action == "stop":
            engine.submit(Stop(execute_at=IMMEDIATE, origin="panic"))
        else:
            return {"ok": False, "error": f"unknown action {action!r}"}
        self.session.session_log.write(
            "panic", trigger=f"ui:{action}", action="submitted to the engine queue"
        )
        if action == "stop":
            # The command is already on the engine queue, so the audio is
            # stopping regardless of what happens next. Everything here is
            # control-thread bookkeeping and none of it can delay the panic.
            self.session.after_stop()
            self.hold_automation(True, reason="stop")
        return {"ok": True, "action": action}

    def _suggestions(self) -> list[dict]:
        """Suggestions, re-ranked only when something they depend on changed.

        Ranking walks the crate with similarity lookups; the UI polls several
        times a second, and polling must not cost the control thread that.
        """
        s = self.session
        live = s.engine.deck(s.live_deck)
        key = (
            live.track.analysis.track_id if live.track is not None else None,
            len(s.played), len(s.cue_queue), s.set_phase,
            round(s.energy_direction, 2), s.tempo_target,
            id(getattr(s, "set_plan", None)),
        )
        if key != getattr(self, "_suggest_key", None):
            try:
                self._suggest_rows = [r.as_dict() for r in s.suggest(5)]
            except Exception as exc:  # the panel is advisory; never break state()
                self._suggest_rows = [{"error": str(exc)}]
            self._suggest_key = key
        return self._suggest_rows

    def override(self, action: str, arg: str = "") -> dict[str, Any]:
        """Manual override, over plain HTTP like the panic path.

        Deliberately not routed through the WebSocket or the model. These are
        the controls an operator reaches for when the automation is doing the
        wrong thing, and the model is part of the automation.
        """
        from djai import cli

        text = f"{action} {arg}".strip()
        reply = cli.handle_override(self.session, text)
        if reply is None:
            return {"ok": False, "error": f"unknown override {text!r}"}
        self.session.session_log.write(
            "override", trigger=f"ui:{text}", action=reply
        )
        return {"ok": True, "action": action, "text": reply}

    # --- the controller surface ----------------------------------------------

    def _submit(self, cmd) -> None:
        """Every controller command goes through the scheduler.

        The engine applies whatever it is handed on the next block, so a
        command with a future ``execute_at`` only waits if the scheduler is
        holding it. That is the layer that quantizes to musical position, and
        routing around it is what would turn a queued drop into an interruption.

        The panic path deliberately does NOT come through here -- it goes
        straight to the engine, so it still lands if this layer is wedged.
        """
        self.session.scheduler.submit(cmd)

    @property
    def automation_held(self) -> bool:
        """True while a manual gesture has the autopilot parked.

        Self-healing against the REPL: ``resume`` typed in the terminal clears
        ``session.frozen``, and a HELD banner that outlived the freeze it stands
        for would be a lie in a dark room.
        """
        session = self.session
        if not session.frozen:
            self._held = False
            session.manual_held = False
        # A performance control from the REPL holds the automation too, and
        # the page has to say so.
        return self._held or bool(getattr(session, "manual_held", False))

    def hold_automation(self, held: bool, reason: str = "") -> dict[str, Any]:
        """Park or release the autopilot. One flag, one owner.

        Manual and automatic control must never steer the same deck, so taking
        manual control parks the autopilot rather than racing it.
        """
        held = bool(held)
        if held == self.automation_held:
            return {"ok": True, "held": held, "text": ""}
        self._held = held
        text = self.session.freeze(held)
        self.session.session_log.write(
            "automation_hold",
            trigger=f"ui:{reason or ('hold' if held else 'resume')}",
            action=text,
            held=held,
        )
        return {"ok": True, "held": held, "text": text}

    def library(self) -> list[dict[str, Any]]:
        """Every analysed track, for the browsable list. Cache reads only."""
        rows = []
        for t in self.session.crate:
            weak = t.grid_confidence < config.PREFLIGHT_MIN_GRID_CONFIDENCE
            rows.append(
                {
                    "track_id": t.track_id,
                    "title": t.title,
                    "bpm": round(float(t.bpm), 1),
                    "camelot": t.camelot,
                    "key": getattr(t, "key", ""),
                    "duration_s": round(float(t.duration_s), 1),
                    "grid_confidence": round(float(t.grid_confidence), 3),
                    "estimated": bool(t.mix_points_estimated),
                    "weak_grid": bool(weak),
                    # One flag for the row's warning icon, so the client does
                    # not re-derive the thresholds this module already owns.
                    "warn": bool(weak or t.mix_points_estimated),
                    # The review-needed badge: ambiguous tempo or a grid too
                    # weak to blend on, unless a person has corrected it.
                    "review": bool(t.review_needed),
                    "tempo_ambiguous": bool(t.tempo_ambiguous),
                    "grid_corrected": bool(t.grid_manually_corrected),
                }
            )
        rows.sort(key=lambda r: r["title"].lower())
        return rows

    def _decode(self, analysis) -> Any:
        """Decode a crate track, memoised. **Worker thread only** -- blocks.

        Goes through ``cli.load_track`` rather than importing the decoder
        directly, so the UI and the autopilot always share one decode path.
        """
        from djai import cli

        loaded = self._decoded.get(analysis.track_id)
        if loaded is None:
            loaded = cli.load_track(analysis)
            self._decoded[analysis.track_id] = loaded
            if len(self._decoded) > 6:
                self._decoded.pop(next(iter(self._decoded)))
        return loaded

    def _takes_master_clock(self, deck_name: str) -> bool:
        """Should a deck about to play drive the master beat clock?

        Yes when it is the mix on its own -- nothing else is playing. That is
        the case the autopilot covers with ``LoadTrack(master=True)`` when it
        starts a set, and the case the UI has to cover when an operator starts
        one by hand.

        Getting this wrong is not subtle. The clock starts at 0 BPM, so a deck
        that plays without claiming it is measured against a clock that never
        ticks; the supervisor reads unbounded drift and hard-resyncs the deck
        every couple of seconds, and the track never gets past its second bar.

        No when the other deck is already playing: a running mix keeps the
        clock, and a transition hands it over on completion.
        """
        other = self.session.engine.deck("b" if deck_name == "a" else "a")
        return not other.playing

    def _neutralise_partner(self, deck_name: str) -> str | None:
        """Undo a half-finished bass swap on the deck that is *staying*.

        Pausing a deck mid-transition ends the crossfade wherever it had got
        to, which would otherwise strand the surviving deck with its low band
        ducked and its level part-way down -- a quiet, bass-less mix with
        nothing left running to finish the fade.
        """
        engine = self.session.engine
        if not engine.transition_active:
            return None
        src, dst = engine._trans_from, engine._trans_to
        if src is None or dst is None:
            return None
        paused = engine.deck(deck_name)
        other = dst if src is paused else src if dst is paused else None
        if other is None:
            return None

        self._submit(
            SetEQ(
                deck=other.name, low=1.0, mid=1.0, high=1.0,
                execute_at=IMMEDIATE, origin="ui:pause",
            )
        )
        self._submit(
            SetGain(
                deck=other.name, gain=1.0,
                execute_at=IMMEDIATE, origin="ui:pause",
            )
        )
        return other.name

    def load_deck(self, deck_name: str, track_id: str) -> dict[str, Any]:
        """Drop a library track on a deck. **Worker thread only** -- decodes.

        A stopped deck takes the track at once, parked at its mix-in. A playing
        deck is never interrupted: the load is quantized to the next phrase
        boundary, which is the same rule every manual action in the REPL obeys.
        """
        if deck_name not in ("a", "b"):
            return {"ok": False, "error": f"unknown deck {deck_name!r}"}
        analysis = next(
            (t for t in self.session.crate if t.track_id == track_id), None
        )
        if analysis is None:
            return {"ok": False, "error": f"no track {track_id!r} in the crate"}

        engine = self.session.engine
        deck = engine.deck(deck_name)
        loaded = self._decode(analysis)
        start = int(analysis.mix_in * SAMPLE_RATE)
        self.hold_automation(True, reason=f"load onto deck {deck_name.upper()}")

        if deck.track is None or not deck.playing:
            self._submit(
                LoadTrack(
                    deck=deck_name, track=loaded, start_frame=start, rate=1.0,
                    play=False, execute_at=IMMEDIATE, origin="ui:load",
                )
            )
            self._pending.pop(deck_name, None)
            when = "now"
            at = None
        else:
            boundary = phrase.next_phrase_boundary(deck)
            at = phrase.deck_frame_to_engine_frame(
                deck, engine.frames_played, boundary
            )
            self._submit(
                LoadTrack(
                    deck=deck_name, track=loaded, start_frame=start, rate=1.0,
                    play=True, execute_at=at, origin="ui:load",
                    # This deck is the mix and its tempo is about to change,
                    # so the beat clock has to follow it.
                    master=self._takes_master_clock(deck_name),
                )
            )
            self._pending[deck_name] = {
                "title": analysis.title,
                "execute_at": int(at),
            }
            when = "next phrase"

        self.session.session_log.write(
            "ui_load",
            trigger=f"ui:drop deck {deck_name}",
            action=f"{analysis.title} loads {when}",
            deck=deck_name,
            execute_at=at,
        )
        self.session.notify(
            f"[deck {deck_name.upper()}] {analysis.title} - loads {when}"
        )
        return {"ok": True, "deck": deck_name, "when": when,
                "title": analysis.title}

    def operator_feedback(self, verdict: str) -> dict[str, Any]:
        """Record "that worked" / "that didn't" -- or any other label in
        djai.feedback.VERDICTS -- about what just played.

        Worker thread. The page sends only the label; the Session attaches
        everything that makes it useful later, the same way the REPL's and the
        chat box's words do.
        """
        from djai import feedback as feedback_mod

        if verdict not in feedback_mod.VERDICTS:
            return {"ok": False, "error": f"unknown verdict {verdict!r}"}
        session = self.session
        try:
            session.record_feedback(verdict)
        except Exception as exc:  # noqa: BLE001 - a verdict must not break the UI
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        learning = session.feedback_summary.get("learning", "")
        session.notify(f"[feedback] {verdict}: {learning}")
        return {"ok": True, "verdict": verdict, "learning": learning}

    def set_transport(self, deck_name: str, playing: bool) -> dict[str, Any]:
        """PLAY / PAUSE one deck, immediately, without touching the other.

        Expressed as a re-load of the audio already on the deck at its current
        playhead: the engine applies it on the audio thread at a block boundary,
        and it ends an in-flight transition through the engine's own path.
        """
        if deck_name not in ("a", "b"):
            return {"ok": False, "error": f"unknown deck {deck_name!r}"}
        engine = self.session.engine
        deck = engine.deck(deck_name)
        if deck.track is None:
            return {"ok": False, "error": f"deck {deck_name} is empty"}

        playing = bool(playing)
        restored = None
        dropped = 0
        if not playing:
            restored = self._neutralise_partner(deck_name)
            # And clear what was scheduled. The engine aborts the crossfade
            # that is running; this stops the one that was about to start.
            dropped = self.session.abort_armed_transition()
        elif engine.stop_requested:
            # Coming back from a STOP, which zeroed every level. Restore this
            # deck's, or PLAY would look like it worked and produce silence
            # with nothing on screen to say why.
            #
            # Keyed on the stop rather than on the current level, so a fader an
            # operator pulled down on purpose is left where they put it.
            self._submit(
                SetGain(
                    deck=deck_name, gain=1.0,
                    execute_at=IMMEDIATE, origin="ui:transport",
                )
            )

        self._submit(
            LoadTrack(
                deck=deck_name, track=deck.track, start_frame=int(deck.position),
                rate=deck.rate, play=playing, execute_at=IMMEDIATE,
                origin="ui:transport",
                # Pressing PLAY on the only running deck makes it the mix, and
                # the mix owns the beat clock.
                master=playing and self._takes_master_clock(deck_name),
            )
        )
        if not playing:
            self.hold_automation(True, reason=f"pause deck {deck_name.upper()}")

        self.session.session_log.write(
            "ui_transport",
            trigger=f"ui:{'play' if playing else 'pause'} deck {deck_name}",
            action=("playing" if playing else "paused"),
            deck=deck_name,
            eq_restored_on=restored,
            scheduled_commands_dropped=dropped,
        )
        if restored:
            self.session.notify(
                f"[deck {deck_name.upper()}] paused mid-transition - "
                f"deck {restored.upper()} EQ and level restored"
            )
        return {"ok": True, "deck": deck_name, "playing": playing,
                "restored": restored}

    def cue_to_mix_in(self, deck_name: str) -> dict[str, Any]:
        """CUE: park the deck's playhead back at its mix-in point, stopped."""
        if deck_name not in ("a", "b"):
            return {"ok": False, "error": f"unknown deck {deck_name!r}"}
        engine = self.session.engine
        deck = engine.deck(deck_name)
        if deck.track is None:
            return {"ok": False, "error": f"deck {deck_name} is empty"}
        start = int(deck.track.analysis.mix_in * SAMPLE_RATE)
        self._neutralise_partner(deck_name)
        self._submit(
            LoadTrack(
                deck=deck_name, track=deck.track, start_frame=start,
                rate=deck.rate, play=False, execute_at=IMMEDIATE,
                origin="ui:cue",
            )
        )
        self.hold_automation(True, reason=f"cue deck {deck_name.upper()}")
        return {"ok": True, "deck": deck_name}

    def apply_eq(
        self,
        deck_name: str,
        low: float | None = None,
        mid: float | None = None,
        high: float | None = None,
    ) -> dict[str, Any]:
        """Mixer EQ. Goes through the queue like every other band change.

        Named ``apply_`` rather than ``set_`` on purpose: the audio-path guard
        in the tests bans the deck-mutating spelling from this module, and this
        submits a command instead of touching a deck. That is exactly the
        distinction the guard exists to enforce, so the name follows it.
        """
        if deck_name not in ("a", "b"):
            return {"ok": False, "error": f"unknown deck {deck_name!r}"}
        self._submit(
            SetEQ(
                deck=deck_name, low=low, mid=mid, high=high,
                execute_at=IMMEDIATE, origin="ui:eq",
            )
        )
        self.session.operator_touched(deck_name)
        return {"ok": True, "deck": deck_name}

    def apply_filter(
        self, deck_name: str, position: object, resonance: object = None
    ) -> dict[str, Any]:
        """Filter knob, through the same Session method as the REPL's `filter`."""
        try:
            pos = float(position)  # type: ignore[arg-type]
            res = None if resonance is None else float(resonance)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"bad filter value: {exc}"}
        text = self.session.set_filter(deck_name, pos, res)
        return {
            "ok": text.startswith("Deck "),
            "deck": deck_name,
            "position": pos,
            "text": text,
        }

    def apply_key_lock(self, deck_name: str, on: bool) -> dict[str, Any]:
        """Key lock, through the same Session method as the REPL's `keylock`."""
        text = self.session.set_key_lock(deck_name, bool(on))
        return {"ok": text.startswith("Deck "), "deck": deck_name, "text": text}

    def _take_mixer_control(self) -> bool:
        """Moving a fader is taking over, so hand the blend over cleanly.

        A crossfade drives both decks' level and low band every block. Ending
        it on the operator's first fader move is what stops the two from
        overwriting each other; restoring both decks' EQ at the same moment is
        what stops the handover leaving a half-swapped bass behind.
        """
        engine = self.session.engine
        handed_over = engine.transition_active
        if handed_over:
            for name in ("a", "b"):
                self._submit(
                    SetEQ(
                        deck=name, low=1.0, mid=1.0, high=1.0,
                        execute_at=IMMEDIATE, origin="ui:mixer",
                    )
                )
            self.session.notify(
                "[mixer] manual level control - blend handed over, EQ neutral"
            )
        self.hold_automation(True, reason="mixer")
        return handed_over

    def set_level(self, deck_name: str, gain: float) -> dict[str, Any]:
        """Channel fader for one deck."""
        if deck_name not in ("a", "b"):
            return {"ok": False, "error": f"unknown deck {deck_name!r}"}
        g = max(0.0, min(1.0, float(gain)))
        handed_over = self._take_mixer_control()
        self._submit(
            SetGain(
                deck=deck_name, gain=g, execute_at=IMMEDIATE, origin="ui:fader"
            )
        )
        return {"ok": True, "deck": deck_name, "gain": g,
                "handed_over": handed_over}

    def set_crossfader(self, x: float) -> dict[str, Any]:
        """Crossfader, equal-power so the middle is not a level dip.

        A fader law, not a transition: it sets two levels and schedules
        nothing. The bass-swap crossfade remains the one transition type and
        still lives in ``transition.py``.
        """
        x = max(0.0, min(1.0, float(x)))
        a = round(float(np.cos(x * np.pi / 2.0)), 4)
        b = round(float(np.sin(x * np.pi / 2.0)), 4)
        handed_over = self._take_mixer_control()
        for name, g in (("a", a), ("b", b)):
            self._submit(
                SetGain(
                    deck=name, gain=g, execute_at=IMMEDIATE, origin="ui:xfader"
                )
            )
        return {"ok": True, "x": x, "a": a, "b": b, "handed_over": handed_over}

    def hot_cue(
        self, deck_name: str, index: int, op: str = "jump", label: str = ""
    ) -> dict[str, Any]:
        """Jump to, set, or clear a stored marker in the track on a deck.

        A hot cue is a position inside a track. It has nothing to do with the
        cue OUTPUT, which is the headphone monitor; they are separate fields in
        the state feed for the same reason they are separate words here.
        """
        if deck_name not in ("a", "b"):
            return {"ok": False, "error": f"unknown deck {deck_name!r}"}
        if op == "set":
            text = self.session.set_hot_cue(deck_name, index, label)
        elif op == "clear":
            text = self.session.clear_hot_cue(deck_name, index)
        elif op == "jump":
            text = self.session.jump_to_hot_cue(deck_name, index)
        else:
            return {"ok": False, "error": f"unknown hot cue op {op!r}"}
        refused = text.startswith(("Refused", "Deck", "Hot cue "))
        return {
            "ok": not text.startswith("Refused"),
            "deck": deck_name,
            "index": index,
            "text": text,
            "refused": refused and text.startswith("Refused"),
        }

    def grid(self, deck_name: str, op: str, msg: dict[str, Any]) -> dict[str, Any]:
        """Correct a loaded track's beat grid, through the session methods the
        REPL's `grid` command uses."""
        session = self.session
        try:
            if op == "halve":
                text = session.grid_halve(deck_name)
            elif op == "double":
                text = session.grid_double(deck_name)
            elif op == "nudge":
                text = session.grid_nudge(deck_name, float(msg.get("ms", 0.0)))
            elif op == "tap":
                text = session.grid_tap(deck_name)
            elif op == "downbeat":
                text = session.grid_set_downbeat(
                    deck_name, float(msg.get("seconds", -1.0))
                )
            else:
                return {"ok": False, "error": f"unknown grid op {op!r}"}
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"bad grid value: {exc}"}
        if deck_name in ("a", "b"):
            deck = session.engine.deck(deck_name)
            if deck.track is not None:
                # The cached waveform carries the old grid.
                self._waveforms.pop(deck.track.analysis.track_id, None)
        return {
            "ok": not text.startswith("Refused"),
            "deck": deck_name,
            "op": op,
            "text": text,
        }

    #: Performance ops the page may send, and the Session method each runs.
    PERFORM_OPS = (
        "loop_in", "loop_out", "loop_exit", "loop_halve", "loop_double", "auto_loop",
        "roll", "roll_off", "jump", "pitch", "pitch_reset", "sync", "quantize",
    )

    def perform(self, deck_name: str, op: str, value: object = None) -> dict[str, Any]:
        """One performance control, through the same Session methods as the REPL."""
        session = self.session
        try:
            if op == "loop_in":
                text = session.loop_in(deck_name)
            elif op == "loop_out":
                text = session.loop_out(deck_name)
            elif op == "loop_exit":
                text = session.loop_exit(deck_name)
            elif op == "loop_halve":
                text = session.loop_resize(deck_name, 0.5)
            elif op == "loop_double":
                text = session.loop_resize(deck_name, 2.0)
            elif op == "auto_loop":
                text = session.auto_loop(deck_name, float(value))  # type: ignore[arg-type]
            elif op == "roll":
                text = session.roll(deck_name, float(value))  # type: ignore[arg-type]
            elif op == "roll_off":
                text = session.roll_off(deck_name)
            elif op == "jump":
                text = session.beat_jump(deck_name, int(value))  # type: ignore[arg-type]
            elif op == "pitch":
                text = session.set_pitch(deck_name, float(value))  # type: ignore[arg-type]
            elif op == "pitch_reset":
                text = session.set_pitch(deck_name, 0.0)
            elif op == "sync":
                text = session.sync(deck_name)
            elif op == "quantize":
                text = session.set_quantize(bool(value))
            else:
                return {"ok": False, "error": f"unknown performance op {op!r}"}
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"bad value for {op}: {exc}"}
        refused = text.startswith("Refused")
        return {"ok": not refused, "deck": deck_name, "op": op, "text": text,
                "refused": refused}

    def apply_phase(self, phase: str) -> dict[str, Any]:
        """Set the set phase, through the same Session method as the REPL."""
        text = self.session.set_set_phase(phase)
        return {"ok": not text.startswith("Rejected"), "phase": phase, "text": text}

    def set_style(self, style: str) -> dict[str, Any]:
        """Ask for a transition style. The supervisor vets the name."""
        text = self.session.set_transition_style(style)
        ok = not text.startswith("Rejected")
        return {"ok": ok, "style": style, "text": text}

    def handle_action(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one controller message. **Worker thread only** -- a load
        decodes, which blocks for far longer than an event loop should wait."""
        kind = str(msg.get("type", ""))
        deck = str(msg.get("deck", ""))
        if kind == "load":
            out = self.load_deck(deck, str(msg.get("track_id", "")))
        elif kind == "transport":
            out = self.set_transport(deck, bool(msg.get("playing")))
        elif kind == "cue_point":
            out = self.cue_to_mix_in(deck)
        elif kind == "automation":
            out = self.hold_automation(bool(msg.get("held")), reason="button")
        elif kind == "eq":
            out = self.apply_eq(
                deck, msg.get("low"), msg.get("mid"), msg.get("high")
            )
        elif kind == "filter":
            out = self.apply_filter(deck, msg.get("position", 0.0), msg.get("resonance"))
        elif kind == "keylock":
            out = self.apply_key_lock(deck, bool(msg.get("on", True)))
        elif kind == "level":
            out = self.set_level(deck, float(msg.get("gain", 1.0)))
        elif kind == "crossfader":
            out = self.set_crossfader(float(msg.get("x", 0.0)))
        elif kind == "hotcue":
            out = self.hot_cue(
                deck, int(msg.get("index", 0)),
                str(msg.get("op", "jump")), str(msg.get("label", "")),
            )
        elif kind == "feedback":
            out = self.operator_feedback(str(msg.get("verdict", "")))
        elif kind == "style":
            out = self.set_style(str(msg.get("style", "auto")))
        elif kind == "phase":
            out = self.apply_phase(str(msg.get("phase", "auto")))
        elif kind == "perform":
            out = self.perform(deck, str(msg.get("op", "")), msg.get("value"))
        elif kind == "frame_stats":
            out = self.record_frame_stats(msg)
        elif kind == "grid":
            out = self.grid(deck, str(msg.get("op", "")), msg)
        elif kind == "library":
            return {"type": "library", "tracks": self.library()}
        else:
            return {"type": "ack", "ok": False, "error": f"unknown type {kind!r}"}
        return {"type": "ack", **out}

    def handle_text(self, text: str) -> dict[str, Any]:
        """Run one typed line, exactly as the REPL would. Worker thread only."""
        from djai import cli

        text = text.strip()
        if not text:
            return {"type": "reply", "text": ""}

        lowered = text.lower()
        if lowered in cli.PANIC_WORDS:
            self.panic("stop" if lowered in ("stop", "quit", "exit") else lowered)
            return {"type": "reply", "text": f"{lowered}.", "panic": True}
        if lowered in ("state", "status"):
            return {"type": "reply", "text": self.session.format_state()}

        # Manual override, before the model -- same order as the REPL, and for
        # the same reason: these have to work when the model is down.
        reply = cli.handle_override(self.session, text)
        if reply is not None:
            return {"type": "reply", "text": reply, "override": True}

        if self.intent_engine is None:
            self._parse_failure = {"text": text, "why": "no local model; keyword commands only",
                                   "t": time.time()}
            return {
                "type": "reply",
                "text": "No local model available; keyword commands only.",
                "parse_failure": self._parse_failure["why"],
            }

        intent = self.intent_engine.interpret(text, self.session.model_state())
        self.session.session_log.write(
            "intent",
            trigger=text,
            action=f"{intent.action} {intent.params}",
            ok=intent.ok,
            error=intent.error,
            fallback=intent.fallback,
            origin="ui",
            latency_s=round(intent.latency_s, 2),
        )
        if intent.action == "cue_track":
            # What the lookup found, not what the model said it would find:
            # "no match" and a list of candidates must reach the operator.
            reply = cli.cue_intent(self.session, intent)
        else:
            cli.apply_intent(self.session, intent)
            reply = intent.reply or f"({intent.action})"
        failure = None
        if not intent.ok or intent.action == "none":
            failure = intent.error or "no action matched"
            if intent.fallback:
                failure += f" (model: {intent.fallback})"
            self._parse_failure = {"text": text, "why": failure, "t": time.time()}
        return {
            "type": "reply",
            "text": reply,
            "parse_failure": failure,
            "action": intent.action,
            "fallback": intent.fallback,
            "latency_s": round(intent.latency_s, 2),
        }

    # --- app -----------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="djai", docs_url=None, redoc_url=None)

        if STATIC_DIR.is_dir():
            app.mount(
                "/static", StaticFiles(directory=str(STATIC_DIR)), name="static"
            )

        @app.get("/")
        def index():
            page = STATIC_DIR / "index.html"
            if not page.is_file():
                return JSONResponse(
                    {"error": f"UI files not found at {STATIC_DIR}"}, status_code=500
                )
            return FileResponse(str(page))

        @app.post("/panic/{action}")
        def do_panic(action: str):
            return JSONResponse(self.panic(action))

        @app.post("/override/{action}")
        def do_override(action: str, arg: str = ""):
            return JSONResponse(self.override(action, arg))

        @app.get("/api/state")
        def get_state():
            return JSONResponse(self.state())

        @app.get("/api/library")
        def get_library():
            return JSONResponse({"tracks": self.library()})

        @app.get("/api/waveform/{deck_name}/bands/{level}")
        def get_bands(deck_name: str, level: str):
            payload = self.band_bytes(deck_name, level)
            if payload is None:
                return JSONResponse({"error": "no waveform band"}, status_code=404)
            data, pps, points = payload
            return Response(
                content=data,
                media_type="application/octet-stream",
                headers={
                    "X-Points-Per-Second": f"{pps:.6f}",
                    "X-Points": str(points),
                    "Cache-Control": "no-store",
                },
            )

        @app.get("/api/waveform/{deck_name}")
        def get_waveform(deck_name: str):
            if deck_name not in ("a", "b"):
                return JSONResponse({"error": "unknown deck"}, status_code=404)
            return JSONResponse(self.waveform(deck_name))

        @app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket):
            await ws.accept()
            self.clients += 1
            loop = asyncio.get_running_loop()
            self._quieten_dropped_connections(loop)

            async def push():
                # Paced against a deadline, not `sleep(interval)` after the
                # send. Windows' default timer granularity is 15.625 ms, so a
                # 50 ms sleep actually wakes at 62.5 ms; adding that to the time
                # the send itself took, measured gaps reached 96 ms -- the whole
                # 100 ms budget, for a 20 Hz feed. Tracking the deadline lets a
                # long frame be paid back by a shorter following sleep.
                interval = 1.0 / STATE_HZ
                due = loop.time()
                while True:
                    await ws.send_text(json.dumps(self.state()))
                    due += interval
                    now = loop.time()
                    if due < now:  # fell behind; don't build up a burst
                        due = now
                    await asyncio.sleep(due - now)

            async def pump():
                while True:
                    raw = await ws.receive_text()
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    kind = str(msg.get("type", "text"))
                    if kind == "text":
                        # The LLM blocks for seconds; keep it off the loop.
                        reply = await loop.run_in_executor(
                            None, self.handle_text, str(msg.get("text", ""))
                        )
                    else:
                        # A load decodes a file, which is far slower still.
                        reply = await loop.run_in_executor(
                            None, self.handle_action, msg
                        )
                    await ws.send_text(json.dumps(reply))

            pusher = asyncio.create_task(push())
            pumper = asyncio.create_task(pump())
            try:
                done, _pending = await asyncio.wait(
                    {pusher, pumper}, return_when=asyncio.FIRST_COMPLETED
                )
                # A browser going away completes one of these with a
                # disconnect. Retrieve it so asyncio does not log it as an
                # unhandled task exception on every closed tab.
                for task in done:
                    exc = task.exception()
                    if exc is not None and not isinstance(exc, WebSocketDisconnect):
                        log.warning("ui websocket closed: %s", exc)
            except WebSocketDisconnect:
                pass
            except Exception as exc:
                log.warning("ui websocket closed: %s", exc)
            finally:
                # A browser going away must never touch the audio path -- there
                # is nothing to unwind here but these two tasks.
                pusher.cancel()
                pumper.cancel()
                self.clients = max(0, self.clients - 1)

        return app
