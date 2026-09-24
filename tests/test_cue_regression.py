"""Regression: the next track is cued before the live deck reaches mix-out.

THREADING CONTEXT: main thread (pytest). Real Sessions on synthetic audio, never
``start()``ed: no device, no background threads. The live deck is *played* --
the audio callback is driven block by block -- and the autopilot is ticked once
per :data:`djai.cli.AUTOPILOT_TICK` of audio, the pace its thread keeps.

The 2026-09-16 regression: an operator paused the live deck and started the
other one by hand. The autopilot kept counting down on the paused deck, which
never reaches mix-out, so the deck the room heard ran out with nothing cued and
nothing in the log to say so.
"""

from __future__ import annotations

import threading

from djai import cli
from djai.deck import SAMPLE_RATE, TransportState
from djai.ui_server import UIServer
from tests.test_engine import drive
from tests.test_hardening import idle_session  # noqa: F401 - a fixture
from tests.test_integration import log_events
from tests.test_integration import session as _session_fixture

session = _session_fixture

BLOCK = 512
BLOCKS_PER_TICK = int(round(cli.AUTOPILOT_TICK * SAMPLE_RATE / BLOCK))


def play_to_outro(s: cli.Session, deck_name: str) -> float | None:
    """Play ``deck_name`` toward its mix-out, ticking the autopilot on time.

    Starts a little before the autopilot's lead window so the whole approach is
    played rather than jumped. Returns the deck's seconds-to-mix-out at the tick
    the next track was on the idle deck, or None if it got to mix-out without.
    """
    deck = s.engine.deck(deck_name)
    analysis = deck.track.analysis
    start_s = analysis.mix_out - s.autopilot_lead_seconds() - 32 * analysis.beat_period
    deck.position = max(float(deck.position), start_s * SAMPLE_RATE)
    drive(s.engine, 2, BLOCK)
    mix_out_frame = analysis.mix_out * SAMPLE_RATE
    while deck.position < mix_out_frame:
        s._autopilot_tick()
        drive(s.engine, BLOCKS_PER_TICK, BLOCK)
        idle = s.engine.deck("b" if deck_name == "a" else "a")
        cued = s._cued
        if (
            cued is not None
            and idle.track is not None
            and idle.track.analysis.track_id == cued.analysis.track_id
        ):
            return (mix_out_frame - deck.position) / SAMPLE_RATE
    return None


def events(s: cli.Session, name: str) -> list[dict]:
    return [e for e in log_events(s) if e["event"] == name]


# --- the regression -------------------------------------------------------------


def test_the_autopilot_cues_and_loads_the_next_track_before_mix_out(session):
    left = play_to_outro(session, session.live_deck)
    assert left is not None, "the live deck reached mix-out with nothing cued"
    assert left > 0.0
    assert events(session, "track_cued")
    assert not events(session, "no_track_cued")


def test_a_hand_over_by_hand_is_followed_and_the_next_track_is_cued(idle_session):
    """The operator pauses the live deck and starts the other one."""
    s = idle_session
    ui = UIServer(s, intent_engine=None)
    ui.load_deck("a", "mid")
    ui.load_deck("b", "opener")
    drive(s.engine, 2, BLOCK)
    ui.hold_automation(False)
    ui.set_transport("a", True)
    drive(s.engine, 4, BLOCK)
    ui.set_transport("a", False)        # pause the live deck ...
    ui.set_transport("b", True)         # ... and start the other by hand
    ui.hold_automation(False)
    drive(s.engine, 4, BLOCK)
    assert s.engine.deck("a").transport is TransportState.PAUSED
    assert s.engine.deck("b").transport is TransportState.PLAYING

    left = play_to_outro(s, "b")

    assert s.live_deck == "b"
    assert left is not None, "deck B reached mix-out with nothing cued"
    assert left > 0.0
    cued = events(s, "track_cued")
    assert cued and cued[-1]["deck"] == "a"


def test_a_live_deck_that_ran_out_is_not_restarted_over_the_playing_one(idle_session):
    """Following the playing deck, not silence recovery, when the operator has
    already started the other deck."""
    s = idle_session
    ui = UIServer(s, intent_engine=None)
    ui.load_deck("a", "mid")
    ui.load_deck("b", "opener")
    drive(s.engine, 2, BLOCK)
    ui.set_transport("a", True)
    ui.set_transport("b", True)
    ui.hold_automation(False)
    drive(s.engine, 4, BLOCK)
    a = s.engine.deck("a")
    a.position = a.track.audio.shape[0] - BLOCK
    drive(s.engine, 4, BLOCK)
    assert a.ended

    s._autopilot_tick()
    drive(s.engine, 4, BLOCK)

    assert s.live_deck == "b"
    assert s.engine.deck("b").track.analysis.track_id == "opener"
    assert s.engine.deck("b").transport is TransportState.PLAYING
    assert not events(s, "recovered_from_silence")


# --- loud logging on every path that leaves nothing cued ------------------------


def test_no_compatible_track_is_logged_loudly_and_not_flooded(session, monkeypatch, caplog):
    # Every way in: since Phase 2 an empty ranking falls back to the nearest
    # tempo and a cut, and since Phase 4 the cue starts from a journey. This
    # path is what is left when selection offers nothing at all.
    from djai.selector import Journey

    monkeypatch.setattr(cli, "plan_journey", lambda *a, **k: Journey(steps=(), score=0.0))
    monkeypatch.setattr(cli, "select_next", lambda *a, **k: None)
    monkeypatch.setattr(cli, "select_nearest_tempo", lambda *a, **k: None)
    notices: list[str] = []
    session.add_notice_listener(notices.append)

    with caplog.at_level("WARNING", logger="djai.cli"):
        assert session.cue_next() is False
        assert session.cue_next() is False

    logged = events(session, "no_track_cued")
    assert len(logged) == 1, "a repeat inside NO_CUE_REPEAT_S must not flood the log"
    assert "stretch range" in logged[0]["action"]
    assert any("no next track cued" in r.getMessage() for r in caplog.records)
    assert any("WARNING: no next track cued" in n for n in notices)


def test_a_supervisor_rejection_of_the_cue_is_logged(session, monkeypatch):
    from djai.supervisor import Rejection

    monkeypatch.setattr(
        session.supervisor, "validate",
        lambda cmd: Rejection(command=cmd, reason="test refusal"),
    )
    assert session.cue_next() is False
    logged = events(session, "no_track_cued")
    assert logged and "test refusal" in logged[0]["action"]


def test_held_automation_inside_the_cue_window_is_logged(session):
    session.freeze(True)
    left = play_to_outro(session, session.live_deck)
    assert left is None, "nothing may be cued while automation is held"
    logged = events(session, "no_track_cued")
    assert logged and "automation is held" in logged[0]["action"]


def test_an_engaged_fallback_is_logged(session):
    class Tripped:
        tripped = True
        reason = "audio callback stalled for 547 ms"

        def stop(self):
            pass

    session.watchdog = Tripped()
    session._autopilot_tick()
    logged = events(session, "no_track_cued")
    assert logged and "fallback player has the output" in logged[0]["action"]


def test_an_armed_transition_with_nothing_queued_is_logged(session, monkeypatch):
    monkeypatch.setattr(cli, "STUCK_ARM_S", 0.0)
    session._transition_armed = True
    assert not session.scheduler.pending()
    session._autopilot_tick()
    session._autopilot_tick()
    logged = events(session, "no_track_cued")
    assert logged and "nothing is queued or running" in logged[0]["action"]


def test_an_exception_in_the_tick_is_logged(session, monkeypatch):
    monkeypatch.setattr(cli, "AUTOPILOT_TICK", 0.001)

    def boom():
        session._stop.set()
        raise RuntimeError("selector blew up")

    monkeypatch.setattr(session, "_autopilot_tick", boom)
    worker = threading.Thread(target=session._autopilot_loop)
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    logged = events(session, "no_track_cued")
    assert logged and "RuntimeError: selector blew up" in logged[0]["action"]
