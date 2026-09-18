"""Phase 1: a transition keeps the beatmatch it inherits.

THREADING CONTEXT: main thread (pytest). The callback is driven directly.

The supervisor audit's root cause, measured before any fix. A 30-minute run
logged 488 interventions, every one "behind", nearly all within a minute of a
transition, a hard resync roughly every bar. `_advance_transition` assigned the
envelope's RATE_A column -- documented as a multiplier, 1.0 for normal -- as an
absolute playback rate. So every transition forced the outgoing deck from its
beatmatched rate back to 1.0: a deck at 1.030 dropped to 1.000, a 2.91% error
the 0.5% nudge ceiling could never catch, and the supervisor resynced it every
bar for the rest of the blend.
"""

from __future__ import annotations

import pytest

from djai import transition as tr
from djai.commands import IMMEDIATE, LoadTrack, SetRate, StartTransition
from djai.deck import SAMPLE_RATE
from djai.engine import Engine
from tests.test_engine import drive, loud_track

BLOCK = 512
MASTER_BPM = 127.72
TRACK_BPM = 124.0
#: The outgoing deck's beatmatched rate: a 124 BPM track under a 127.72 master.
MATCHED = MASTER_BPM / TRACK_BPM


def _rig(style: str) -> tuple[Engine, int]:
    """Deck B is the master at 1.0; deck A is matched up to it and goes out."""
    engine = Engine(blocksize=BLOCK)
    master = loud_track(330.0, MASTER_BPM, 120.0)
    outgoing = loud_track(220.0, TRACK_BPM, 120.0)
    engine.submit(LoadTrack(deck="b", track=master, start_frame=0, rate=1.0,
                            master=True, play=True))
    engine.submit(LoadTrack(deck="a", track=outgoing, start_frame=0, rate=MATCHED,
                            play=True))
    drive(engine, 4, BLOCK)
    total = tr.transition_frames(MASTER_BPM, SAMPLE_RATE,
                                 transition_bars=tr.style_bars(style))
    engine.arm_transition_plan(style, total, MASTER_BPM)
    engine.submit(StartTransition(from_deck="a", to_deck="b", total_frames=total,
                                  execute_at=IMMEDIATE))
    return engine, total


@pytest.mark.parametrize("style", [s for s in tr.STYLES if s not in ("cut", "backspin")])
def test_a_transition_keeps_the_outgoing_decks_beatmatched_rate(style):
    engine, total = _rig(style)
    drive(engine, max(1, int(total * 0.5) // BLOCK), BLOCK)
    assert engine.transition_active, "the check has to happen mid-blend"
    assert engine.deck_a.rate == pytest.approx(MATCHED, rel=1e-9), (
        f"{style} moved the outgoing deck off its matched rate"
    )


def test_a_supervisor_nudge_during_a_transition_is_left_alone():
    """The envelope may only move the rate when its own value changes."""
    engine, _ = _rig("bass_swap")
    drive(engine, 8, BLOCK)
    nudged = MATCHED * 1.004
    engine.submit(SetRate(deck="a", rate=nudged, execute_at=IMMEDIATE, origin="supervisor"))
    drive(engine, 40, BLOCK)
    assert engine.transition_active
    assert engine.deck_a.rate == pytest.approx(nudged, rel=1e-9)


def test_a_backspin_reverses_relative_to_the_matched_rate():
    engine, total = _rig("backspin")
    drive(engine, int(total * 0.99) // BLOCK, BLOCK)
    assert engine.transition_active
    rate = engine.deck_a.rate
    assert rate < 0.0, "the record never went backwards"
    assert tr.BACKSPIN_END_RATE * MATCHED - 0.05 <= rate, (
        f"reversed at {rate:.3f}, beyond the spin's own floor scaled to the match"
    )


def test_aborting_a_backspin_restores_the_matched_rate():
    engine, total = _rig("backspin")
    drive(engine, int(total * 0.95) // BLOCK, BLOCK)
    assert engine.deck_a.rate < 0.0
    engine.deck_b.playing = False            # a pause: the transition aborts
    drive(engine, 1, BLOCK)
    assert not engine.transition_active
    assert engine.deck_a.rate == pytest.approx(MATCHED, rel=1e-9)


def test_the_supervisor_does_not_correct_a_deck_playing_backwards(tmp_path):
    from djai.scheduler import Scheduler
    from djai.supervisor import SessionLog, Supervisor

    engine, total = _rig("backspin")
    supervisor = Supervisor(engine, Scheduler(engine), [], session_log=SessionLog(tmp_path))
    try:
        supervisor.check_drift()             # capture baselines
        drive(engine, int(total * 0.95) // BLOCK, BLOCK)
        assert engine.deck_a.rate < 0.0
        before = supervisor.interventions
        for _ in range(10):
            supervisor.check_drift()
            drive(engine, 1, BLOCK)
        assert supervisor.interventions == before, "a spinning deck is not drift"
    finally:
        supervisor.log.close()
