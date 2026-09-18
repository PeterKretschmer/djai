"""STOP: a panic with a defined post-state, and a way back from it.

THREADING CONTEXT: main thread (pytest). The callback is driven directly.

What STOP used to do: zero both gains, clear `playing`, end any transition,
set `stop_requested`. It left `paused` untouched, so the transport state a
deck reported afterwards depended on what it had been doing before; it left
the playheads where they were and the EQ wherever the last transition had
put it; it left queued commands in the scheduler; and nothing ever cleared
`stop_requested`, so the autopilot never ran again. A later PLAY reloaded the
deck against a gain of zero and produced silence with nothing to explain it.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from djai.commands import IMMEDIATE, SetEQ, StartTransition, Stop
from djai.deck import SAMPLE_RATE, TransportState
from djai.ui_server import UIServer
from tests.test_engine import drive
from tests.test_integration import session as _session_fixture

session = _session_fixture

BLOCK = 512


@pytest.fixture
def ui(session):
    return UIServer(session, intent_engine=None), session


def level(buf) -> float:
    return float(np.max(np.abs(buf)))


def stop_via_ui(server, session):
    server.panic("stop")
    drive(session.engine, 4, BLOCK)


# --- the post-state -----------------------------------------------------------


def test_stop_leaves_both_decks_loaded_and_stopped(ui):
    server, session = ui
    engine = session.engine
    server.load_deck("b", "mid")
    drive(engine, 2, BLOCK)
    server.set_transport("b", True)
    drive(engine, 20, BLOCK)

    stop_via_ui(server, session)

    for name in ("a", "b"):
        deck = engine.deck(name)
        assert deck.track is not None, "STOP must not unload a track"
        assert deck.transport is TransportState.LOADED_STOPPED, (
            f"deck {name} is {deck.transport}"
        )


def test_stop_rewinds_the_playheads_to_zero(ui):
    server, session = ui
    engine = session.engine
    drive(engine, 60, BLOCK)
    assert engine.deck("a").position > 0

    stop_via_ui(server, session)
    for name in ("a", "b"):
        deck = engine.deck(name)
        if deck.track is not None:
            assert deck.position == 0.0, f"deck {name} at {deck.position}"


def test_stop_zeroes_the_gains_and_restores_neutral_eq(ui):
    server, session = ui
    engine = session.engine
    # Leave the EQ somewhere a transition might have put it.
    engine.submit(SetEQ(deck="a", low=0.0, mid=0.3, execute_at=IMMEDIATE,
                        origin="test"))
    drive(engine, 4, BLOCK)
    assert engine.deck_state("a").eq[0] < 0.5

    stop_via_ui(server, session)
    drive(engine, 30, BLOCK)

    for name in ("a", "b"):
        st = engine.deck_state(name)
        assert st.gain == pytest.approx(0.0, abs=0.02), f"deck {name} still open"
        assert all(v == pytest.approx(1.0, abs=0.05) for v in st.eq), (
            f"deck {name} kept a shaped EQ: {st.eq}"
        )


def test_stop_makes_the_room_silent(ui):
    server, session = ui
    engine = session.engine
    assert level(drive(engine, 8, BLOCK)) > 0.05

    stop_via_ui(server, session)
    drive(engine, 20, BLOCK)
    assert level(drive(engine, 20, BLOCK)) < 0.01


def test_stop_clears_the_scheduler_queue(ui):
    server, session = ui
    engine = session.engine
    server.load_deck("a", "banger")          # queued at the next phrase
    assert len(session.scheduler) > 0

    stop_via_ui(server, session)
    assert len(session.scheduler) == 0, "queued commands must not fire after a stop"


def test_stop_holds_the_automation(ui):
    server, session = ui
    stop_via_ui(server, session)
    assert session.frozen
    assert server.automation_held, "the UI must show that it is held"


def test_stop_ends_an_in_flight_transition(ui):
    server, session = ui
    engine = session.engine
    engine.deck("b").attach(engine.deck("a").track, 0)
    engine.arm_transition_plan("bass_swap", SAMPLE_RATE * 20, 125.0)
    engine.submit(StartTransition(from_deck="a", to_deck="b",
                                  total_frames=SAMPLE_RATE * 20,
                                  execute_at=IMMEDIATE, origin="test"))
    drive(engine, 10, BLOCK)
    assert engine.transition_active

    stop_via_ui(server, session)
    assert not engine.transition_active


def test_stop_does_not_tear_down_the_stream(ui):
    """It is a panic, not a shutdown. The device stays open."""
    server, session = ui
    engine = session.engine
    before = engine._stream
    stop_via_ui(server, session)
    assert engine._stream is before
    # And the callback keeps being served, silently.
    buf = drive(engine, 10, BLOCK)
    assert buf.shape[0] == 10 * BLOCK, "the callback stopped being called"
    assert level(buf) < 0.01


# --- getting back --------------------------------------------------------------


def test_play_after_stop_resumes_audio_with_no_restart(ui):
    """The acceptance bar. No new process, no reload, just PLAY."""
    server, session = ui
    engine = session.engine
    stop_via_ui(server, session)
    assert level(drive(engine, 20, BLOCK)) < 0.01

    out = server.set_transport("a", True)
    assert out["ok"]
    drive(engine, 40, BLOCK)

    assert engine.deck("a").transport is TransportState.PLAYING
    assert level(drive(engine, 20, BLOCK)) > 0.05, "PLAY after STOP was silent"


def test_play_after_stop_works_on_either_deck(ui):
    server, session = ui
    engine = session.engine
    server.load_deck("b", "mid")
    drive(engine, 2, BLOCK)
    stop_via_ui(server, session)

    assert server.set_transport("b", True)["ok"]
    drive(engine, 40, BLOCK)
    assert engine.deck("b").transport is TransportState.PLAYING
    assert level(drive(engine, 20, BLOCK)) > 0.05


def test_playing_again_clears_the_stopped_flag(ui):
    """Otherwise the autopilot stays dead for the life of the process."""
    server, session = ui
    engine = session.engine
    stop_via_ui(server, session)
    assert engine.stop_requested

    server.set_transport("a", True)
    drive(engine, 4, BLOCK)
    assert not engine.stop_requested


def test_the_autopilot_runs_again_once_the_hold_is_released(ui):
    server, session = ui
    engine = session.engine
    stop_via_ui(server, session)

    server.set_transport("a", True)
    drive(engine, 8, BLOCK)
    server.hold_automation(False)

    # It should now be willing to cue again rather than returning at the top.
    session._autopilot_tick()
    assert not engine.stop_requested
    assert not session.frozen


def test_a_stop_from_the_repl_reaches_the_same_state(session):
    """The two surfaces must not disagree about what STOP means."""
    from djai import cli

    engine = session.engine
    drive(engine, 40, BLOCK)
    assert cli.handle_panic(session, "stop") is True
    drive(engine, 4, BLOCK)

    assert engine.deck("a").transport is TransportState.LOADED_STOPPED
    assert engine.deck("a").position == 0.0
    assert session.frozen
    assert len(session.scheduler) == 0


# --- and it is still a panic ---------------------------------------------------


def test_stop_still_acts_in_under_100ms(ui):
    server, session = ui
    engine = session.engine
    drive(engine, 8, BLOCK)

    started = time.perf_counter()
    server.panic("stop")
    request_ms = (time.perf_counter() - started) * 1000

    blocks = 0
    while blocks < 40:
        buf = drive(engine, 1, BLOCK)
        blocks += 1
        if level(buf) < 0.01:
            break
    audio_ms = blocks * BLOCK / SAMPLE_RATE * 1000

    assert request_ms < 100.0, f"the request itself took {request_ms:.1f} ms"
    assert audio_ms < 100.0, f"took {audio_ms:.1f} ms of audio to go quiet"


def test_stop_does_not_go_through_the_scheduler(ui):
    """The panic path stays direct, whatever the bookkeeping does afterwards."""
    server, session = ui
    before = session.engine.queue.qsize()
    server.panic("stop")
    assert session.engine.queue.qsize() == before + 1, "must land on the engine queue"

