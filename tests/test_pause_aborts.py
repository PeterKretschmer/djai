"""Pausing a deck mid-transition aborts the transition.

THREADING CONTEXT: main thread (pytest). The callback is driven directly, so
"within one audio block" means exactly that.

The trace, for the record. Envelope advancement lives in
`Engine._advance_transition`, called from `Engine.callback` under
`if self._trans_active:`. It advances by `self._trans_frames += frames` -- a
block count -- and the only gate was that boolean. Neither the paused deck's
state nor the other deck's was consulted, so a crossfade over a paused deck
kept writing gains for a deck that was not playing and left the deck that was
playing stranded at whatever the curve had reached.
"""

from __future__ import annotations

import csv
import io

import numpy as np
import pytest

from djai.commands import IMMEDIATE, StartTransition
from djai.deck import SAMPLE_RATE, TransportState
from djai.ui_server import UIServer
from tests.test_engine import drive
from tests.test_integration import log_events
from tests.test_integration import session as _session_fixture

session = _session_fixture

BLOCK = 512


@pytest.fixture
def mid_transition(session):
    """A running bass swap between two playing decks, part-way through."""
    server = UIServer(session, intent_engine=None)
    engine = session.engine
    server.load_deck("b", "mid")
    drive(engine, 2, BLOCK)
    server.set_transport("b", True)
    drive(engine, 2, BLOCK)

    engine.arm_transition_plan("bass_swap", SAMPLE_RATE * 20, 125.0)
    engine.submit(
        StartTransition(from_deck="a", to_deck="b", total_frames=SAMPLE_RATE * 20,
                        execute_at=IMMEDIATE, origin="test")
    )
    drive(engine, 20, BLOCK)
    assert engine.transition_active
    assert engine.deck_state("b").eq[0] < 1.0, "the swap should be under way"
    return server, session, engine


# --- the abort ------------------------------------------------------------------


def test_pausing_halts_the_envelope_within_one_block(mid_transition):
    server, _session, engine = mid_transition
    before = engine._trans_frames

    server.set_transport("a", False)
    drive(engine, 1, BLOCK)          # exactly one block

    assert not engine.transition_active, "the transition must be over"
    frozen = engine._trans_frames
    drive(engine, 40, BLOCK)
    assert engine._trans_frames == frozen, "the envelope kept advancing"
    assert frozen <= before + BLOCK, "it advanced more than one block"


def test_advancement_is_gated_on_deck_state_not_the_flag(session):
    """The gate the trace was about: stop a deck behind the engine's back."""
    engine = session.engine
    engine.deck("b").attach(engine.deck("a").track, 0)
    engine.arm_transition_plan("bass_swap", SAMPLE_RATE * 20, 125.0)
    engine.submit(
        StartTransition(from_deck="a", to_deck="b", total_frames=SAMPLE_RATE * 20,
                        execute_at=IMMEDIATE, origin="test")
    )
    drive(engine, 10, BLOCK)
    assert engine.transition_active

    # Not through any command: just stop the deck. The flag still says active.
    engine.deck("a").playing = False
    drive(engine, 1, BLOCK)
    assert not engine.transition_active, (
        "advancement was gated on the flag, not on whether the decks are playing"
    )


def test_the_other_decks_eq_and_gain_return_to_neutral(mid_transition):
    """Recorded per block, the way the envelope CSV records a render."""
    server, _session, engine = mid_transition

    rows = []

    def capture():
        st = engine.deck_state("b")
        rows.append({
            "block": len(rows),
            "b_gain": round(st.gain, 6),
            "b_low": round(st.eq[0], 6),
            "b_mid": round(st.eq[1], 6),
            "b_high": round(st.eq[2], 6),
            "active": int(engine.transition_active),
        })

    for _ in range(4):
        drive(engine, 1, BLOCK)
        capture()
    server.set_transport("a", False)
    for _ in range(60):
        drive(engine, 1, BLOCK)
        capture()

    # Written out as a CSV so the record is inspectable, not just asserted.
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    assert buf.getvalue().count("\n") == len(rows) + 1

    during = rows[0]
    after = rows[-1]
    assert during["b_low"] < 1.0, "the swap had taken the low band down"
    assert after["b_gain"] == pytest.approx(1.0, abs=0.02), buf.getvalue()[-400:]
    assert after["b_low"] == pytest.approx(1.0, abs=0.02)
    assert after["b_mid"] == pytest.approx(1.0, abs=0.02)
    assert after["b_high"] == pytest.approx(1.0, abs=0.02)
    assert after["active"] == 0


def test_pausing_one_deck_never_alters_the_others_transport(mid_transition):
    server, _session, engine = mid_transition
    before = engine.deck("b").transport
    assert before is TransportState.PLAYING

    server.set_transport("a", False)
    drive(engine, 40, BLOCK)

    assert engine.deck("b").transport is before, "deck B's transport changed"
    assert engine.deck("b").playing


def test_the_paused_deck_keeps_its_playhead(mid_transition):
    server, _session, engine = mid_transition
    where = engine.deck("a").position

    server.set_transport("a", False)
    drive(engine, 40, BLOCK)

    assert engine.deck("a").transport is TransportState.PAUSED
    assert abs(engine.deck("a").position - where) < BLOCK * 3, (
        "the playhead moved after the pause"
    )


def test_the_pause_holds_the_automation(mid_transition):
    server, session, engine = mid_transition
    server.set_transport("a", False)
    assert server.automation_held and session.frozen


def test_a_scheduled_transition_is_cleared_too(session):
    """The engine aborts what is running; this clears what was about to start."""
    server = UIServer(session, intent_engine=None)
    engine = session.engine
    assert session.cue_next(origin="test")
    drive(engine, 4, BLOCK)
    assert session.arm_transition(origin="test") is not None
    assert len(session.scheduler) > 0
    assert session._transition_armed

    server.set_transport(session.live_deck, False)
    drive(engine, 4, BLOCK)

    assert len(session.scheduler) == 0, "a scheduled transition survived the pause"
    assert not session._transition_armed
    events = [e for e in log_events(session) if e["event"] == "transition_aborted"]
    assert events


def test_an_operators_own_queued_drop_is_left_alone(session):
    """Only a transition this session armed is cleared."""
    server = UIServer(session, intent_engine=None)
    engine = session.engine
    server.load_deck("a", "banger")          # queued at the next phrase
    queued = len(session.scheduler)
    assert queued > 0
    assert not session._transition_armed

    server.set_transport("a", False)
    assert len(session.scheduler) == queued, "the operator's drop was dropped"


# --- getting going again ----------------------------------------------------------


def test_play_after_pause_resumes_with_no_residual_transition(mid_transition):
    server, _session, engine = mid_transition
    server.set_transport("a", False)
    drive(engine, 20, BLOCK)
    where = engine.deck("a").position

    assert server.set_transport("a", True)["ok"]
    drive(engine, 20, BLOCK)

    assert engine.deck("a").transport is TransportState.PLAYING
    assert engine.deck("a").position > where, "it did not resume from where it was"
    assert not engine.transition_active, "the aborted transition came back"
    assert engine._trans_env is None


def test_the_resumed_deck_is_audible(mid_transition):
    server, _session, engine = mid_transition
    server.set_transport("a", False)
    drive(engine, 20, BLOCK)
    server.set_transport("a", True)
    drive(engine, 20, BLOCK)
    out = drive(engine, 20, BLOCK)
    assert float(np.max(np.abs(out))) > 0.05
