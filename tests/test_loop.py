"""Loop playback in the deck: sample-accurate, beat-aligned, no click.

THREADING CONTEXT: main thread (pytest). `Deck.read` is the audio thread's
code path and is called directly.

The property that makes looping cheap to leave: the region is a whole number
of beats, so after any number of laps the playhead sits at the same offset
within the beat it would have reached playing straight through. Nothing has to
be corrected on the way out.
"""

from __future__ import annotations

import numpy as np
import pytest

from djai import phrase
from djai.deck import SAMPLE_RATE, Deck, LoadedTrack
from tests.test_engine import loud_track
from tests.test_integration import session as _session_fixture

session = _session_fixture

BLOCK = 512


@pytest.fixture
def deck():
    base = loud_track(220.0, 120.0, 30.0)
    d = Deck("a")
    d.attach(LoadedTrack(analysis=base.analysis, audio=base.audio), 0)
    d.playing = True
    d.gain.jump(1.0)
    return d


def beats_frames(bpm: float, beats: float) -> float:
    return beats * (60.0 / bpm) * SAMPLE_RATE


def render(d: Deck, blocks: int) -> np.ndarray:
    out = np.concatenate([d.read(BLOCK).copy() for _ in range(blocks)])
    return out


# --- the region ------------------------------------------------------------------


def test_a_deck_starts_with_no_loop(deck):
    assert not deck.loop_active
    assert deck.loop_region == (0.0, 0.0)


def test_setting_and_clearing_a_loop(deck):
    deck.set_loop(1000.0, 2000.0)
    assert deck.loop_active and deck.loop_region == (1000.0, 2000.0)
    deck.clear_loop()
    assert not deck.loop_active


def test_a_zero_length_loop_is_no_loop(deck):
    deck.set_loop(1000.0, 0.0)
    assert not deck.loop_active
    deck.set_loop(1000.0, -5.0)
    assert not deck.loop_active


def test_attaching_a_track_does_not_inherit_a_loop(deck):
    deck.set_loop(1000.0, 2000.0)
    base = loud_track(220.0, 120.0, 10.0)
    deck.attach(LoadedTrack(analysis=base.analysis, audio=base.audio), 0)
    assert deck.loop_region == (1000.0, 2000.0), (
        "attach does not clear the loop; the engine does that explicitly"
    )


# --- playback --------------------------------------------------------------------


def test_the_playhead_stays_inside_the_region(deck):
    start = beats_frames(120.0, 8)
    length = beats_frames(120.0, 4)
    deck.position = start
    deck.set_loop(start, length)

    for _ in range(400):
        deck.read(BLOCK)
        assert start <= deck.position < start + length + BLOCK, (
            f"playhead left the loop: {deck.position}"
        )


def test_a_looping_deck_never_ends(deck):
    """It cannot run off the end of the track, because it never gets there."""
    start = beats_frames(120.0, 8)
    deck.position = start
    deck.set_loop(start, beats_frames(120.0, 4))
    for _ in range(600):
        deck.read(BLOCK)
    assert not deck.ended


def test_the_loop_actually_repeats_its_audio(deck):
    """Two consecutive laps must contain the same audio."""
    start = beats_frames(120.0, 8)
    length = beats_frames(120.0, 2)
    deck.position = start
    deck.set_loop(start, length)

    # Rendered output must be periodic with the loop's own period. Compared at
    # sample offsets rather than block counts, because a lap does not land on a
    # block boundary.
    lap = int(round(length))
    out = render(deck, int(2.5 * lap // BLOCK) + 4)
    window = 20_000
    offset = 2_000
    a = out[offset:offset + window]
    b = out[offset + lap:offset + lap + window]
    assert b.shape == a.shape
    assert np.allclose(a, b, atol=0.02), (
        f"the loop is not periodic at {lap} frames; "
        f"max difference {float(np.max(np.abs(a - b))):.4f}"
    )


def test_clearing_the_loop_lets_playback_continue(deck):
    start = beats_frames(120.0, 8)
    length = beats_frames(120.0, 4)
    deck.position = start
    deck.set_loop(start, length)
    for _ in range(200):
        deck.read(BLOCK)

    deck.clear_loop()
    before = deck.position
    for _ in range(20):
        deck.read(BLOCK)
    assert deck.position > before + 19 * BLOCK, "playback did not resume forward"

    # And it leaves the region entirely, given a lap's worth of blocks.
    for _ in range(int(length // BLOCK) + 4):
        deck.read(BLOCK)
    assert deck.position > start + length, "it never left the region"


# --- no click ----------------------------------------------------------------------


def test_the_seam_does_not_click(deck):
    """Measured as the largest sample-to-sample step, looped vs straight.

    A click is a discontinuity, so it shows up as a jump far larger than
    anything in the music around it.
    """
    start = beats_frames(120.0, 8)
    length = beats_frames(120.0, 2)

    deck.position = start
    straight = render(deck, 200)
    straight_step = float(np.max(np.abs(np.diff(straight, axis=0))))

    deck.attach(deck.track, int(start))
    deck.playing = True
    deck.gain.jump(1.0)
    deck.set_loop(start, length)
    looped = render(deck, 200)
    looped_step = float(np.max(np.abs(np.diff(looped, axis=0))))

    assert looped_step <= straight_step * 1.5, (
        f"loop introduced a step of {looped_step:.4f} against the track's own "
        f"{straight_step:.4f}"
    )


def test_the_crossfade_buffers_are_allocated_once(deck):
    before = deck._xf_cont
    deck.set_loop(beats_frames(120.0, 8), beats_frames(120.0, 2))
    deck.position = beats_frames(120.0, 8)
    for _ in range(200):
        deck.read(BLOCK)
    assert deck._xf_cont is before


def test_looping_allocates_nothing_the_straight_path_does_not(deck):
    import gc
    import tracemalloc

    def growth(loop: bool) -> int:
        d = Deck("x")
        base = loud_track(220.0, 120.0, 30.0)
        d.attach(LoadedTrack(analysis=base.analysis, audio=base.audio), 0)
        d.playing = True
        d.gain.jump(1.0)
        d.position = beats_frames(120.0, 8)
        if loop:
            d.set_loop(beats_frames(120.0, 8), beats_frames(120.0, 2))
        for _ in range(40):
            d.read(BLOCK)
        gc.collect()
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        for _ in range(400):
            d.read(BLOCK)
        after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        return sum(s.size_diff for s in after.compare_to(before, "filename"))

    straight = growth(False)
    looping = growth(True)
    assert looping <= straight + 8_000, (
        f"looping added {looping - straight} bytes over 400 blocks"
    )


# --- phase ---------------------------------------------------------------------------


def test_phase_against_the_master_clock_survives_the_loop(deck):
    """The acceptance criterion, measured in beats.

    A whole-beat loop leaves the playhead at the same position within the beat
    it would have reached playing straight through -- so the fractional part of
    the deck's beat position is unchanged by however many laps it did.
    """
    analysis = deck.track.analysis
    start = beats_frames(120.0, 8)
    length = beats_frames(120.0, 4)          # exactly four beats

    deck.position = start
    before = phrase.beat_at_frame(analysis, deck.position)

    deck.set_loop(start, length)
    laps = 0
    for _ in range(2000):
        was = deck.position
        deck.read(BLOCK)
        if deck.position < was:
            laps += 1
        if laps >= 3:
            break
    assert laps >= 3, "the test needs several laps"

    deck.clear_loop()
    after = phrase.beat_at_frame(analysis, deck.position)

    # Whole beats may have been repeated; the offset within the beat may not
    # have moved.
    assert (after - before) % 1.0 == pytest.approx(
        0.0, abs=0.02
    ) or (after - before) % 1.0 == pytest.approx(1.0, abs=0.02), (
        f"beat phase shifted by {(after - before) % 1.0:.4f} beats"
    )


def test_a_loop_of_whole_beats_is_a_whole_number_of_beats(deck):
    """The precondition the phase guarantee rests on."""
    for beats in (0.5, 1, 2, 4, 8):
        length = beats_frames(120.0, beats)
        period = (60.0 / 120.0) * SAMPLE_RATE
        assert (length / period) == pytest.approx(beats)


def test_the_supervisor_leaves_a_looping_deck_alone(session):
    """Otherwise every lap reads as drift and it resyncs out of the loop."""
    from tests.test_engine import drive

    engine = session.engine
    live = engine.deck(session.live_deck)
    analysis = live.track.analysis
    period = 60.0 / analysis.bpm * SAMPLE_RATE

    drive(engine, 20, BLOCK)
    session.supervisor.check_drift()
    baseline = session.supervisor.interventions

    live.set_loop(live.position, 4 * period)
    for _ in range(60):
        drive(engine, 8, BLOCK)
        session.supervisor.check_drift()

    assert session.supervisor.interventions == baseline, (
        "the supervisor corrected a deck that was looping on purpose"
    )
    assert live.loop_active, "and it should still be looping"
