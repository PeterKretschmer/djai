"""Hot cues: stored markers inside a track that playback can jump to.

THREADING CONTEXT: main thread (pytest). The audio callback is driven directly.

Note the naming, which is load-bearing throughout: a `hot_cue` is a position
inside a track; the `cue_output` is the headphone monitor. They are different
features and share no code, and these tests assert that separation holds at the
edges where the two words meet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from djai import analysis as an
from djai.commands import IMMEDIATE, StartTransition
from djai.deck import SAMPLE_RATE
from djai.ui_server import UIServer
from tests.test_engine import drive
from tests.test_integration import log_events, make_analysis
from tests.test_integration import session as _session_fixture

session = _session_fixture

BLOCK = 512


def run(session, blocks):
    """Drive the engine AND the scheduler, the way the two threads would.

    A jump on a playing deck is scheduled for that deck's next downbeat, so a
    test that only drives the callback never sees it land.
    """
    for _ in range(blocks):
        session.scheduler.tick(session.engine.frames_played)
        drive(session.engine, 1, BLOCK)


@pytest.fixture
def ui(session):
    return UIServer(session, intent_engine=None), session


# --- the schema ---------------------------------------------------------------


def test_the_schema_version_was_bumped_for_hot_cues():
    assert an.ANALYSIS_VERSION >= 4


def test_a_cue_round_trips_through_seconds_and_samples():
    cue = an.make_hot_cue(3, 12.5, "drop")
    assert cue == {"index": 3, "sample_position": int(12.5 * an.HOT_CUE_SR),
                   "label": "drop"}
    assert an.cue_seconds(cue) == pytest.approx(12.5)


def test_hot_cue_positions_are_in_deck_frames():
    """A cue that has to be converted before use is one that gets converted
    wrongly. The stored rate is the deck's."""
    assert an.HOT_CUE_SR == SAMPLE_RATE


# --- auto-population ----------------------------------------------------------


def test_cue_one_and_two_are_always_the_mix_points():
    ta = make_analysis("t", 128.0, "8A", 0.5, Path("x.wav"))
    cues = an.auto_hot_cues(ta)
    assert cues[0]["index"] == 1 and cues[0]["label"] == "mix in"
    assert cues[1]["index"] == 2 and cues[1]["label"] == "mix out"
    assert an.cue_seconds(cues[0]) == pytest.approx(ta.mix_in, abs=0.01)
    assert an.cue_seconds(cues[1]) == pytest.approx(ta.mix_out, abs=0.01)


def test_a_flat_track_gets_no_structural_cues():
    """Constant energy has no boundaries, so only the mix points are stored."""
    ta = make_analysis("flat", 128.0, "8A", 0.5,
                       Path("x.wav"))
    assert len(an.auto_hot_cues(ta)) == 2


def test_a_step_in_energy_on_a_downbeat_becomes_a_cue():
    ta = make_analysis("stepped", 120.0, "8A", 0.5,
                       Path("x.wav"))
    # A clean doubling of energy, on a downbeat, well inside the track.
    rms = list(ta.beat_rms)
    step = (len(rms) // 2) // 4 * 4          # land it on a downbeat
    for i in range(step, len(rms)):
        rms[i] = rms[i] * 3.0
    ta.beat_rms = rms
    ta.hot_cues = []

    cues = an.auto_hot_cues(ta)
    assert len(cues) > 2, "the boundary should have been detected"
    detected = [c for c in cues if c["label"] in ("drop", "break")]
    assert detected, "a rise in energy is a drop"
    at = an.cue_seconds(detected[0])
    assert min(abs(at - d) for d in ta.downbeats) < 0.01, "must land on a downbeat"


def test_no_more_than_the_documented_ceiling_of_cues():
    ta = make_analysis("busy", 120.0, "8A", 0.5,
                       Path("x.wav"))
    rms = list(ta.beat_rms)
    for i in range(0, len(rms), 32):          # a boundary every 8 bars
        for j in range(i, min(i + 32, len(rms))):
            rms[j] = 0.2 if (i // 32) % 2 else 1.0
    ta.beat_rms = rms
    ta.hot_cues = []
    cues = an.auto_hot_cues(ta)
    assert len(cues) <= 6, "two mix points plus at most four detected"
    assert len(cues) <= an.MAX_HOT_CUES


# --- persistence --------------------------------------------------------------


def test_cues_persist_across_a_restart(session, tmp_path):
    session.cache_dir = tmp_path
    deck = session.live_deck
    analysis = session.engine.deck(deck).track.analysis

    session.set_hot_cue(deck, 5, "my marker")
    sidecar = tmp_path / f"{analysis.track_id}.json"
    assert sidecar.exists(), "setting a cue must write the sidecar"
    assert sidecar.with_suffix(".npz").exists(), "and its array file"

    # Reload the way a later run would.
    reloaded = an.load_cached(analysis.track_id, tmp_path)
    assert reloaded is not None, "the saved entry must load back"
    stored = next(c for c in reloaded.hot_cues if c["index"] == 5)
    assert stored["label"] == "my marker"
    assert stored["sample_position"] > 0


def test_clearing_a_cue_persists_too(session, tmp_path):
    session.cache_dir = tmp_path
    deck = session.live_deck
    session.set_hot_cue(deck, 4, "temp")
    session.clear_hot_cue(deck, 4)
    analysis = session.engine.deck(deck).track.analysis
    sidecar = tmp_path / f"{analysis.track_id}.json"
    stored = json.loads(sidecar.read_text(encoding="utf-8"))
    assert all(c["index"] != 4 for c in stored["hot_cues"])


def test_setting_a_cue_twice_replaces_rather_than_duplicates(session, tmp_path):
    session.cache_dir = tmp_path
    deck = session.live_deck
    session.set_hot_cue(deck, 3, "first")
    drive(session.engine, 40, BLOCK)
    session.set_hot_cue(deck, 3, "second")
    cues = [c for c in session.hot_cues(deck) if c["index"] == 3]
    assert len(cues) == 1 and cues[0]["label"] == "second"


# --- jumping ------------------------------------------------------------------


def test_a_jump_snaps_to_the_nearest_downbeat(session, tmp_path):
    session.cache_dir = tmp_path
    engine = session.engine
    deck_name = session.live_deck
    deck = engine.deck(deck_name)
    analysis = deck.track.analysis

    # A cue deliberately off the grid, between two downbeats.
    off_grid = analysis.downbeats[8] + 0.31
    analysis.hot_cues = [an.make_hot_cue(6, off_grid, "off grid")]

    session.jump_to_hot_cue(deck_name, 6)

    # Catch the playhead on the block the jump actually lands, rather than
    # wherever it has played on to afterwards.
    seq = deck.load_seq
    landed = None
    for _ in range(600):
        session.scheduler.tick(engine.frames_played)
        drive(engine, 1, BLOCK)
        if deck.load_seq != seq:
            landed = deck.position / SAMPLE_RATE
            break

    assert landed is not None, "the jump never landed"
    nearest = min(analysis.downbeats, key=lambda d: abs(d - off_grid))
    assert abs(landed - nearest) < 0.05, (
        f"landed at {landed:.3f}s, nearest downbeat {nearest:.3f}s "
        f"(cue was at {off_grid:.3f}s)"
    )


def test_a_jump_is_refused_during_a_transition_and_logged(session):
    engine = session.engine
    deck_name = session.live_deck
    analysis = engine.deck(deck_name).track.analysis
    analysis.hot_cues = [an.make_hot_cue(2, analysis.mix_out, "mix out")]

    engine.deck("b").attach(engine.deck("a").track, 0)
    engine.arm_transition_plan("bass_swap", SAMPLE_RATE * 20, 125.0)
    engine.submit(
        StartTransition(
            from_deck="a", to_deck="b", total_frames=SAMPLE_RATE * 20,
            execute_at=IMMEDIATE, origin="test",
        )
    )
    drive(engine, 4, BLOCK)
    assert engine.transition_active

    before = engine.deck(deck_name).position
    reply = session.jump_to_hot_cue(deck_name, 2)
    drive(engine, 6, BLOCK)

    assert "Refused" in reply
    assert engine.deck(deck_name).position > before, "the deck kept playing"
    events = [e for e in log_events(session) if e["event"] == "hot_cue_refused"]
    assert events, "a refused jump must be logged"
    assert "transition" in events[-1]["action"]


def test_a_jump_on_a_playing_deck_keeps_it_playing(session, tmp_path):
    session.cache_dir = tmp_path
    engine = session.engine
    deck_name = session.live_deck
    deck = engine.deck(deck_name)
    deck.track.analysis.hot_cues = [
        an.make_hot_cue(7, deck.track.analysis.downbeats[12], "later")
    ]
    seq = deck.load_seq
    session.jump_to_hot_cue(deck_name, 7)
    run(session, 600)
    assert deck.load_seq != seq, "the jump should have landed"
    assert deck.playing, "a cue jump must not stop the deck"


def test_jumping_to_a_cue_that_does_not_exist_says_so(session):
    assert "no hot cue" in session.jump_to_hot_cue(session.live_deck, 8).lower()


def test_an_empty_deck_refuses_every_hot_cue_operation(session):
    idle = session.cued_deck()
    for reply in (
        session.jump_to_hot_cue(idle, 1),
        session.set_hot_cue(idle, 1),
        session.clear_hot_cue(idle, 1),
    ):
        assert "empty" in reply.lower()


# --- the UI surface -----------------------------------------------------------


def test_the_state_feed_carries_hot_cues_separately_from_the_cue_output(ui):
    server, session = ui
    state = server.state()
    deck = state["decks"][session.live_deck]
    assert "hot_cues" in deck, "markers live on the deck"
    assert "cue" in state and "mode" in state["cue"], "the monitor is separate"
    # The two must never be conflated: the monitor has no notion of an index.
    assert "index" not in state["cue"]
    json.dumps(state)


def test_the_ui_can_set_jump_and_clear_a_hot_cue(ui, tmp_path):
    server, session = ui
    session.cache_dir = tmp_path
    deck = session.live_deck

    out = server.hot_cue(deck, 4, "set", "ui marker")
    assert out["ok"]
    assert any(c["index"] == 4 for c in session.hot_cues(deck))

    assert server.hot_cue(deck, 4, "jump")["ok"]
    assert server.hot_cue(deck, 4, "clear")["ok"]
    assert not any(c["index"] == 4 for c in session.hot_cues(deck))


def test_the_ui_refuses_an_unknown_hot_cue_operation(ui):
    server, session = ui
    out = server.hot_cue(session.live_deck, 1, "scratch")
    assert out["ok"] is False and "unknown hot cue op" in out["error"]


def test_the_page_has_hot_cue_buttons_and_a_style_picker(ui):
    from fastapi.testclient import TestClient

    server, _session = ui
    body = TestClient(server._app).get("/").text
    for needed in ("hotcues-a", "hotcues-b", "style-pick", "trans-style",
                   "trans-rule"):
        assert needed in body, f"missing {needed} in index.html"


# --- the style request --------------------------------------------------------


def test_the_supervisor_rejects_an_invented_transition_style(session):
    reply = session.set_transition_style("beat_juggle")
    assert reply.startswith("Rejected")
    assert session.transition_style == "auto", "a bad name must not take effect"
    events = [
        e for e in log_events(session) if e["event"] == "transition_style_rejected"
    ]
    assert events


def test_every_known_style_is_accepted(session):
    from djai import transition as tr

    for style in tr.STYLE_CHOICES:
        assert not session.set_transition_style(style).startswith("Rejected")
        assert session.transition_style == style

