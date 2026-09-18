"""Phase 4: performance controls, measured.

THREADING CONTEXT: main thread (pytest). The engine callback is driven
directly, and the scheduler is ticked deterministically rather than by its
thread, so every "on the next beat" claim is checked against real frames.

The controls are loops and rolls, beat jump, the pitch fader, sync and
quantize -- each from both the REPL and the UI, both of which run the same
Session methods.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from djai import cli, phrase
from djai.commands import IMMEDIATE, Cut, ExitLoop, LoadTrack, SetLoop, StartTransition
from djai.deck import SAMPLE_RATE, Deck, LoadedTrack
from tests.test_engine import drive, loud_track
from tests.test_integration import session as _session_fixture

session = _session_fixture

BLOCK = 512


# --- helpers -----------------------------------------------------------------


def _run(session, blocks: int = 4, block: int = BLOCK) -> None:
    """Release anything due and run the callback, as the live path would."""
    for _ in range(blocks):
        session.scheduler.tick(session.engine.frames_played)
        drive(session.engine, 1, block)


def _live(session):
    deck = session.engine.deck(session.live_deck)
    deck.playing = True
    return deck


def _events(session, name: str) -> list[dict]:
    session.session_log._file.flush()
    return [
        entry
        for line in session.session_log.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for entry in [json.loads(line)]
        if entry["event"] == name
    ]


def _beat(deck, frame=None) -> float:
    return phrase.beat_at_frame(deck.track.analysis, deck.position if frame is None else frame)


# --- loops -------------------------------------------------------------------


def test_loop_exit_preserves_phase(session):
    """A slip loop hands the playhead back where straight playback would be."""
    engine = session.engine
    deck = _live(session)
    rate = deck.rate
    start_position = deck.position
    fpb = deck.track.analysis.beat_period * SAMPLE_RATE

    # One beat, driven well past its length, so the loop genuinely laps before
    # it is left.
    engine.submit(SetLoop(deck=deck.name, start_frame=deck.position,
                          length_frames=fpb, execute_at=IMMEDIATE))
    drive(engine, 1, BLOCK)
    assert deck.loop_active
    blocks = int(3.5 * fpb / (BLOCK * rate))
    drive(engine, blocks, BLOCK)
    assert deck.position < start_position + blocks * BLOCK * rate, "the loop must hold it back"

    engine.submit(ExitLoop(deck=deck.name, execute_at=IMMEDIATE))
    drive(engine, 2, BLOCK)
    assert not deck.loop_active
    straight = start_position + (blocks + 3) * BLOCK * rate
    assert deck.position == pytest.approx(straight, abs=1.0), (
        "the playhead should resume where it would have been"
    )


def test_loop_in_out_halve_double_and_exit_from_the_repl(session):
    deck = _live(session)
    fpb = deck.track.analysis.beat_period * SAMPLE_RATE

    assert "loop in at" in cli.handle_override(session, f"loop {deck.name} in")
    drive(session.engine, int(4 * fpb / BLOCK) + 1, BLOCK)
    assert "looping 4 beats" in cli.handle_override(session, f"loop {deck.name} out")
    _run(session)
    assert deck.loop_active
    assert deck.loop_region[1] == pytest.approx(4 * fpb, rel=1e-6)

    assert "halved" in cli.handle_override(session, f"loop {deck.name} halve")
    _run(session)
    assert deck.loop_region[1] == pytest.approx(2 * fpb, rel=1e-6)

    assert "doubled" in cli.handle_override(session, f"loop {deck.name} double")
    _run(session)
    assert deck.loop_region[1] == pytest.approx(4 * fpb, rel=1e-6)

    assert "in phase" in cli.handle_override(session, f"loop {deck.name} exit")
    _run(session, blocks=int(fpb / BLOCK) + 4)
    assert not deck.loop_active


def test_an_autoloop_starts_on_a_beat_line(session):
    deck = _live(session)
    analysis = deck.track.analysis
    reply = cli.handle_override(session, f"loop {deck.name} 8")
    assert "loops 8 beats" in reply
    _run(session, blocks=int(analysis.beat_period * SAMPLE_RATE / BLOCK) + 4)
    assert deck.loop_active
    start_beat = phrase.beat_at_frame(analysis, deck.loop_region[0])
    assert start_beat == pytest.approx(round(start_beat), abs=1e-6)


def test_a_roll_loops_from_the_beat_just_played_and_releases_in_phase(session):
    deck = _live(session)
    analysis = deck.track.analysis
    fpb = analysis.beat_period * SAMPLE_RATE

    assert "rolling 1/4 bar" in cli.handle_override(session, f"roll {deck.name} 1/4")
    _run(session, blocks=2)
    assert deck.loop_active
    assert deck.loop_region[1] == pytest.approx(fpb, rel=1e-6)
    start_beat = phrase.beat_at_frame(analysis, deck.loop_region[0])
    assert start_beat == pytest.approx(round(start_beat), abs=1e-6)

    before = deck.position
    slip = deck._slip
    assert "in phase" in cli.handle_override(session, f"roll {deck.name} off")
    _run(session, blocks=2)
    assert not deck.loop_active
    assert deck.position >= before
    assert deck.position == pytest.approx(slip + 2 * BLOCK * deck.rate, abs=2.0)


@pytest.mark.parametrize("bars", [0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
def test_every_roll_size_is_accepted(session, bars):
    deck = _live(session)
    text = cli.handle_override(session, f"roll {deck.name} {bars:g}")
    assert not text.startswith("Refused"), text
    _run(session, blocks=2)
    assert deck.loop_active
    cli.handle_override(session, f"roll {deck.name} off")
    _run(session, blocks=2)


def test_an_unparseable_loop_line_still_reaches_the_model(session):
    assert cli.handle_override(session, "loop roll it out into the next one") is None
    assert cli.handle_override(session, "jump around a bit") is None
    assert cli.handle_override(session, "pitch it up somehow") is None


# --- beat jump ----------------------------------------------------------------


def test_a_quantized_beat_jump_lands_exactly_on_the_grid(session, capsys):
    """The jump is an exact grid distance, applied at a beat line."""
    engine = session.engine
    deck = _live(session)
    analysis = deck.track.analysis
    fpb = analysis.beat_period * SAMPLE_RATE
    session.set_quantize(True)

    reply = cli.handle_override(session, f"jump {deck.name} 4")
    assert "jumps +4 bars on the next beat" in reply
    expected = 16 * fpb

    seq = deck.load_seq
    before = after = None
    for _ in range(400):
        position = deck.position
        session.scheduler.tick(engine.frames_played)
        drive(engine, 1, BLOCK)
        if deck.load_seq != seq:
            before, after = position, deck.position
            break
    assert before is not None, "the jump never fired"

    distance = after - before - BLOCK * deck.rate
    beat_before = phrase.beat_at_frame(analysis, before)
    past_line_ms = (beat_before - np.floor(beat_before)) * analysis.beat_period * 1000
    with capsys.disabled():
        print(f"\n    beat jump +4 bars: moved {distance:.2f} frames (grid {expected:.2f}, "
              f"error {distance - expected:+.3f}); fired {past_line_ms:.2f} ms past the "
              f"beat line (one {BLOCK}-frame block is {BLOCK / SAMPLE_RATE * 1000:.1f} ms)")
    # Exactly the requested distance, on top of the block that was played.
    assert distance == pytest.approx(expected, abs=1.0)
    # And it fired at a beat line: the playhead was within one block past one.
    assert 0 <= beat_before - np.floor(beat_before) < BLOCK * deck.rate / fpb + 1e-6
    # The grid distance is a whole number of bars, so the phase is unchanged.
    assert (phrase.beat_at_frame(analysis, after - BLOCK * deck.rate)
            - beat_before) == pytest.approx(16.0, abs=1e-3)


def test_an_unquantized_beat_jump_fires_at_once(session):
    engine = session.engine
    deck = _live(session)
    session.set_quantize(False)
    try:
        seq = deck.load_seq
        # Forward: the fixture's deck sits near the start of its track, where a
        # backward jump is (rightly) refused.
        assert "jumps +1 bars now" in cli.handle_override(session, f"jump {deck.name} 1")
        _run(session, blocks=2)
        assert deck.load_seq != seq
    finally:
        session.set_quantize(True)


def test_a_jump_off_the_end_of_the_track_is_refused(session):
    deck = _live(session)
    assert cli.handle_override(session, f"jump {deck.name} -16").startswith("Refused")
    assert "1, 4, 8 or 16" in cli.handle_override(session, f"jump {deck.name} 3")


# --- pitch and sync -------------------------------------------------------------


def test_the_pitch_fader_moves_the_deck_and_resets(session):
    deck = _live(session)
    native = deck.track.analysis.bpm

    reply = cli.handle_override(session, f"pitch {deck.name} 4")
    assert f"{native * 1.04:.1f} BPM" in reply
    _run(session)
    assert deck.rate == pytest.approx(1.04)
    assert deck.manual_pitch

    assert cli.handle_override(session, f"pitch {deck.name} reset")
    _run(session)
    assert deck.rate == pytest.approx(1.0)
    assert cli.handle_override(session, f"pitch {deck.name} 12").startswith("Refused")


def test_the_master_pitch_fader_moves_the_beat_clock(session):
    engine = session.engine
    deck = engine.deck(engine.master_deck)
    deck.playing = True
    before = engine.master_bpm
    cli.handle_override(session, f"pitch {deck.name} 2")
    _run(session)
    assert engine.master_bpm == pytest.approx(before * 1.02, rel=1e-6)
    cli.handle_override(session, f"pitch {deck.name} reset")
    _run(session)


def test_the_supervisor_leaves_a_manually_pitched_deck_alone(session):
    engine = session.engine
    other = "b" if engine.master_deck == "a" else "a"
    deck = engine.deck(other)
    deck.attach(engine.deck(engine.master_deck).track, 0)
    deck.playing = True
    deck.manual_pitch = True
    assert session.supervisor._expected_beat(deck, deck.position, engine.master_beat) is None
    deck.manual_pitch = False


def test_sync_matches_tempo_and_phase(session, capsys):
    engine = session.engine
    master_name = engine.master_deck
    master = engine.deck(master_name)
    master.playing = True
    other = "b" if master_name == "a" else "a"
    deck = engine.deck(other)

    # The same track on the other deck, deliberately off tempo and off phase.
    track = master.track
    engine.submit(LoadTrack(deck=other, track=track, start_frame=int(master.position + 977),
                            rate=1.03, play=True, execute_at=IMMEDIATE))
    drive(engine, 2, BLOCK)
    assert deck.playing

    reply = cli.handle_override(session, f"sync {other}")
    assert "synced to" in reply, reply
    _run(session, blocks=2)

    assert deck.rate == pytest.approx(engine.master_bpm / deck.track.analysis.bpm, rel=1e-6)
    assert not deck.manual_pitch
    master_frac = _beat(master) % 1.0
    own_frac = _beat(deck) % 1.0
    diff = abs(master_frac - own_frac)
    residual = min(diff, 1.0 - diff)
    with capsys.disabled():
        print(f"\n    sync: rate 1.0300 -> {deck.rate:.4f}; phase error after "
              f"{residual:.4f} beats ({residual * deck.track.analysis.beat_period * 1000:.2f} ms)")
    assert residual < 0.02, f"phase still {diff:.3f} beats out"


def test_syncing_the_master_to_itself_is_refused(session):
    master = session.engine.master_deck
    assert "is the master" in cli.handle_override(session, f"sync {master}")


# --- quantize, holds and refusals ------------------------------------------------


def test_quantize_toggles_from_the_repl(session):
    assert cli.handle_override(session, "quantize off") == "Quantize off."
    assert session.quantize is False
    assert "Quantize is off" in cli.handle_override(session, "quantize")
    assert cli.handle_override(session, "quantize on") == "Quantize on."
    assert session.quantize is True


def test_every_manual_control_holds_the_automation(session):
    session.freeze(False)
    deck = _live(session)
    assert not session.manual_held
    cli.handle_override(session, f"roll {deck.name} 1")
    assert session.frozen and session.manual_held
    _run(session, blocks=2)

    from djai.ui_server import UIServer

    ui = UIServer(session, intent_engine=None)
    assert ui.state()["automation_held"] is True
    assert ui.hold_automation(False)["ok"]
    assert not session.frozen and not session.manual_held
    assert ui.state()["automation_held"] is False
    cli.handle_override(session, f"roll {deck.name} off")
    _run(session, blocks=2)


def test_a_manual_control_during_a_transition_is_refused_and_logged(session):
    engine = session.engine
    engine.deck("b").attach(engine.deck("a").track, 0)
    engine.deck("b").playing = True
    engine.arm_transition_plan("bass_swap", SAMPLE_RATE * 30, 125.0)
    engine.submit(StartTransition(from_deck="a", to_deck="b",
                                  total_frames=SAMPLE_RATE * 30, execute_at=IMMEDIATE))
    drive(engine, 2, BLOCK)
    assert engine.transition_active

    try:
        for line in (f"roll a 1", "jump a 4", "pitch a 2", "loop a in", "sync b"):
            assert cli.handle_override(session, line).startswith("Refused"), line
        refusals = _events(session, "manual_refused")
        assert len(refusals) >= 5
        assert all("mid-transition" in e["action"] for e in refusals)
        # Quantize is not a deck gesture, so it still works.
        assert cli.handle_override(session, "quantize on") == "Quantize on."
    finally:
        engine._abort_transition()
        drive(engine, 2, BLOCK)


def test_panic_still_cuts_in_under_100ms_with_a_loop_and_pitch_engaged(session, capsys):
    engine = session.engine
    deck = _live(session)
    fpb = deck.track.analysis.beat_period * SAMPLE_RATE
    block = engine.blocksize

    cli.handle_override(session, f"pitch {deck.name} 3")
    cli.handle_override(session, f"roll {deck.name} 1")
    _run(session, blocks=4, block=block)
    deck.gain.jump(1.0)
    assert deck.loop_active

    engine.submit(Cut(deck="master", execute_at=IMMEDIATE, origin="panic"))
    blocks_to_silence = 0
    for _ in range(8):
        out = drive(engine, 1, block)
        blocks_to_silence += 1
        if float(np.max(np.abs(out))) < 1e-4:
            break
    latency_ms = blocks_to_silence * block / SAMPLE_RATE * 1000
    with capsys.disabled():
        print(f"\n    panic cut with a roll and +3% pitch engaged: silent after "
              f"{blocks_to_silence} block(s) of {block} frames = {latency_ms:.1f} ms")
    assert latency_ms < 100.0, f"cut took {latency_ms:.1f} ms"


# --- the UI runs the same methods -------------------------------------------------


def test_every_control_works_from_the_ui(session):
    from djai.ui_server import UIServer

    engine = session.engine
    ui = UIServer(session, intent_engine=None)
    deck = _live(session)
    d = deck.name

    assert ui.handle_action({"type": "perform", "deck": d, "op": "auto_loop", "value": 4})["ok"]
    _run(session, blocks=int(deck.track.analysis.beat_period * SAMPLE_RATE / BLOCK) + 4)
    assert deck.loop_active
    state = ui.state()["decks"][d]
    assert state["loop_active"] is True
    assert state["loop_beats"] == pytest.approx(4.0, abs=0.01)

    assert ui.handle_action({"type": "perform", "deck": d, "op": "loop_halve"})["ok"]
    _run(session)
    assert deck.loop_region[1] == pytest.approx(
        2 * deck.track.analysis.beat_period * SAMPLE_RATE, rel=1e-6
    )
    assert ui.handle_action({"type": "perform", "deck": d, "op": "loop_exit"})["ok"]
    _run(session, blocks=int(deck.track.analysis.beat_period * SAMPLE_RATE / BLOCK) + 4)
    assert not deck.loop_active

    assert ui.handle_action({"type": "perform", "deck": d, "op": "roll", "value": 0.5})["ok"]
    _run(session, blocks=2)
    assert deck.loop_active
    assert ui.handle_action({"type": "perform", "deck": d, "op": "roll_off"})["ok"]
    _run(session, blocks=2)

    assert ui.handle_action({"type": "perform", "deck": d, "op": "jump", "value": 1})["ok"]
    _run(session, blocks=200)
    assert ui.handle_action({"type": "perform", "deck": d, "op": "pitch", "value": -2})["ok"]
    _run(session)
    assert engine.deck(d).rate == pytest.approx(0.98)
    assert ui.state()["decks"][d]["pitch_percent"] == pytest.approx(-2.0, abs=0.01)
    assert ui.handle_action({"type": "perform", "deck": d, "op": "pitch_reset"})["ok"]
    _run(session)

    assert ui.handle_action({"type": "perform", "deck": d, "op": "quantize", "value": False})["ok"]
    assert ui.state()["quantize"] is False
    assert ui.handle_action({"type": "perform", "deck": d, "op": "quantize", "value": True})["ok"]

    bad = ui.handle_action({"type": "perform", "deck": d, "op": "nonsense"})
    assert not bad["ok"]


def test_the_page_has_every_performance_control():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")
    for deck in ("a", "b"):
        assert f'id="perf-{deck}"' in html
    assert 'id="btn-quantize"' in html
    assert 'type: "perform"' in js
    for op in ("loop_in", "loop_out", "loop_exit", "loop_halve", "loop_double",
               "auto_loop", "roll", "roll_off", "jump", "pitch", "pitch_reset",
               "sync", "quantize"):
        assert f'"{op}"' in js, op
    # Rolls are momentary: pressed they roll, released they let go.
    assert "pointerdown" in js and "pointerup" in js
