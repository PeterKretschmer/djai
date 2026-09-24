"""The transition library: four styles, one precomputed envelope mechanism.

THREADING CONTEXT: main thread (pytest). The audio callback is driven directly,
so every "audio thread" assertion here is made against the real callback.

The property that matters most is a negative one: whatever the style, the
callback does one array lookup and some assignments. No curve maths, no
allocation, and no branch on style ever reaches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from djai import transition as tr
from djai.commands import IMMEDIATE, Cut, StartTransition
from djai.deck import SAMPLE_RATE
from tests.test_engine import drive
from tests.test_integration import session as _session_fixture

session = _session_fixture

BLOCK = 512


# --- the envelope ------------------------------------------------------------


def test_every_style_builds_an_envelope_of_the_right_shape():
    for style in tr.STYLES:
        env = tr.build_envelope(style, SAMPLE_RATE * 8, BLOCK)
        assert env.dtype == np.float32
        assert env.shape[1] == tr.N_COLS
        assert env.shape[0] >= 2
        assert np.isfinite(env).all()


def test_an_unknown_style_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError):
        tr.build_envelope("scratch_and_spin", SAMPLE_RATE, BLOCK)


def test_the_precomputed_bass_swap_matches_the_original_curve():
    """The shape is unchanged; only where it is evaluated moved.

    bass_swap is specified as untouched, so the array has to agree with
    `gains_at` at every block rather than merely look similar.
    """
    total = SAMPLE_RATE * 12
    env = tr.build_envelope("bass_swap", total, BLOCK)
    rows = total // BLOCK
    for i in range(rows):
        t = (i * BLOCK) / total
        expected = tr.gains_at(t * tr.TRANSITION_BARS, tr.TRANSITION_BARS)
        assert env[i, tr.FROM_GAIN] == pytest.approx(expected.from_gain, abs=1e-6)
        assert env[i, tr.TO_GAIN] == pytest.approx(expected.to_gain, abs=1e-6)
        assert env[i, tr.FROM_LOW] == pytest.approx(expected.from_low, abs=1e-6)
        assert env[i, tr.TO_LOW] == pytest.approx(expected.to_low, abs=1e-6)


def test_every_style_ends_handed_over():
    """However a style ends, it ends with B carrying the mix on its own."""
    for style in tr.STYLES:
        env = tr.build_envelope(style, SAMPLE_RATE * 6, BLOCK)
        last = env[-1]
        assert last[tr.FROM_GAIN] == pytest.approx(0.0)
        assert last[tr.TO_GAIN] == pytest.approx(1.0)
        assert last[tr.TO_LOW] == pytest.approx(1.0)
        assert last[tr.TO_MID] == pytest.approx(1.0)
        assert last[tr.TO_HIGH] == pytest.approx(1.0)
        assert last[tr.ECHO_SEND] == pytest.approx(0.0)


def test_bass_swap_automation_moves_every_block():
    env = tr.build_envelope("bass_swap", SAMPLE_RATE * 12, BLOCK)
    a = env[:-1, tr.FROM_GAIN]
    moved = int(np.sum(np.abs(np.diff(a)) > 1e-6))
    assert moved >= len(a) - 2, "gains must change per block, not step once"


def test_exactly_one_deck_owns_the_low_end_through_a_bass_swap():
    env = tr.build_envelope("bass_swap", SAMPLE_RATE * 12, BLOCK)
    for row in env[:-1]:
        owners = (row[tr.FROM_LOW] > 0.5) + (row[tr.TO_LOW] > 0.5)
        assert owners == 1, "never both decks, never neither"


def test_filter_sweep_turns_deck_a_filter_toward_high_pass():
    """Phase 2: a real resonant high-pass, no longer the EQ bands in turn."""
    env = tr.build_envelope("filter_sweep", SAMPLE_RATE * 16, BLOCK)
    knob = env[:-1, tr.FROM_FILTER]
    assert knob[0] == pytest.approx(0.0, abs=0.01), "starts open"
    assert knob.max() == pytest.approx(tr.FILTER_SWEEP_DEPTH, abs=0.01)
    assert np.all(np.diff(knob) >= -1e-7), "a high-pass sweeping upward only"
    assert int(np.sum(np.abs(np.diff(knob)) > 1e-6)) > 10, "must move per block"
    assert np.all(env[:-1, tr.FILTER_RES] == pytest.approx(tr.FILTER_SWEEP_RESONANCE))
    # The EQ is left alone: the filter does the work now.
    for band in (tr.FROM_LOW, tr.FROM_MID, tr.FROM_HIGH):
        assert np.all(env[:, band] == 1.0)
    assert np.all(env[:, tr.TO_FILTER] == 0.0), "only the outgoing deck filters"
    assert env[-1, tr.FROM_FILTER] == 0.0, "hands back an open filter"


def test_echo_out_sends_and_then_stops_sending():
    env = tr.build_envelope("echo_out", SAMPLE_RATE * 4, BLOCK)
    send = env[:, tr.ECHO_SEND]
    assert send.max() > 0.5, "the send has to actually open"
    assert send[0] == pytest.approx(0.0, abs=0.05)
    assert send[-1] == pytest.approx(0.0), "the tail must close out"
    dry = env[:-1, tr.FROM_GAIN]
    assert dry[0] > dry[len(dry) // 2] > dry[-1], "dry signal falls throughout"


def test_only_the_echo_styles_ever_open_the_send():
    # Phase 3: the reverb tail is fed by the echo, and filter_echo is a filter
    # sweep with one. Nothing else may touch the delay line.
    echo_styles = {"echo_out", "reverb_out", "filter_echo"}
    for style in tr.STYLES:
        env = tr.build_envelope(style, SAMPLE_RATE * 4, BLOCK)
        peak = float(env[:, tr.ECHO_SEND].max())
        if style in echo_styles:
            assert peak > 0.5
        else:
            assert peak == 0.0, f"{style} must not touch the delay line"


def test_the_echo_delay_is_beat_synced():
    at_128 = tr.echo_delay_frames(128.0, SAMPLE_RATE)
    at_64 = tr.echo_delay_frames(64.0, SAMPLE_RATE)
    assert at_64 == pytest.approx(at_128 * 2, rel=0.01)
    expected = tr.ECHO_DELAY_BEATS * 60.0 / 128.0 * SAMPLE_RATE
    assert at_128 == pytest.approx(expected, rel=0.01)


# --- the engine running them --------------------------------------------------


def _start(session, style, total_frames):
    engine = session.engine
    if engine.transition_active:
        # A fresh blend. The engine never restarts one part-way (Phase 2.3:
        # a second start landing in a running blend was a click), so a test
        # that wants a new one ends the old one first.
        engine._end_transition()
    engine.arm_transition_plan(style, total_frames, 125.0)
    engine.submit(
        StartTransition(
            from_deck="a", to_deck="b", total_frames=total_frames,
            execute_at=IMMEDIATE, origin="test",
        )
    )


def test_a_cut_hands_over_within_a_single_block(session):
    """Deck A to zero, deck B to full, one block. That is the whole style."""
    engine = session.engine
    engine.deck("b").attach(engine.deck("a").track, 0)
    _start(session, "cut", BLOCK)
    drive(engine, 1, BLOCK)

    assert engine.deck_state("a").gain == pytest.approx(0.0, abs=0.02)
    assert engine.deck_state("b").gain == pytest.approx(1.0, abs=0.02)
    assert not engine.transition_active, "a cut is over as soon as it happens"


def test_a_filter_sweep_drives_the_outgoing_deck_filter_knob(session):
    engine = session.engine
    engine.deck("b").attach(engine.deck("a").track, 0)
    total = SAMPLE_RATE * 8
    _start(session, "filter_sweep", total)

    drive(engine, 1, BLOCK)
    assert engine.deck_state("a").filter == pytest.approx(0.0, abs=0.05)

    drive(engine, int(total / BLOCK * 0.9), BLOCK)
    state = engine.deck_state("a")
    assert state.filter > 0.7, "knob well toward high-pass"
    assert state.filter_resonance == pytest.approx(tr.FILTER_SWEEP_RESONANCE)
    assert engine.deck_state("b").filter == pytest.approx(0.0)


def test_a_transition_hands_back_a_neutral_eq_and_an_open_filter(session):
    """A sweep must not leave the deck high-passed for its next track."""
    engine = session.engine
    engine.deck("b").attach(engine.deck("a").track, 0)
    total = SAMPLE_RATE * 4
    _start(session, "filter_sweep", total)
    drive(engine, total // BLOCK + 8, BLOCK)

    for name in ("a", "b"):
        state = engine.deck_state(name)
        assert all(v == pytest.approx(1.0, abs=0.05) for v in state.eq), (
            f"deck {name} kept a filtered EQ: {state.eq}"
        )
        assert state.filter == 0.0, f"deck {name} kept its filter at {state.filter}"
        assert engine.deck(name).filter_pos.cur == 0.0 or not engine.deck(name).playing


def test_echo_out_puts_a_tail_into_the_mix(session):
    """The delay has to be audible after the dry signal has gone."""
    engine = session.engine
    engine.deck("b").attach(None, 0)          # B silent: only the tail can sound
    total = SAMPLE_RATE * 2
    _start(session, "echo_out", total)

    # Run to where the dry signal is essentially out but the send is still open.
    drive(engine, int(total / BLOCK * 0.7), BLOCK)
    buf = drive(engine, 8, BLOCK)
    assert float(np.max(np.abs(buf))) > 1e-4, "no echo tail reached the mix"


def test_the_echo_ring_starts_clean_once_the_previous_tail_has_died(session):
    engine = session.engine
    engine.deck("b").attach(engine.deck("a").track, 0)
    _start(session, "echo_out", SAMPLE_RATE)
    drive(engine, 20, BLOCK)
    assert float(np.max(np.abs(engine._echo_ring))) > 0.0

    engine._echo_ring_left = 0          # the last tail has rung out
    _start(session, "echo_out", SAMPLE_RATE)
    # The plan is consumed on the next drained block, which clears the ring.
    drive(engine, 1, BLOCK)
    assert engine._echo_write <= BLOCK


def test_a_still_ringing_tail_carries_into_the_next_transition(session):
    """Cutting a tail short to start the next transition clean would be the
    same fault as cutting it at the hand-over."""
    engine = session.engine
    engine.deck("b").attach(engine.deck("a").track, 0)
    _start(session, "echo_out", SAMPLE_RATE)
    drive(engine, 20, BLOCK)
    assert engine._echo_ring_left > 0
    write_before = engine._echo_write

    _start(session, "echo_out", SAMPLE_RATE)
    drive(engine, 1, BLOCK)
    assert engine._echo_write == (write_before + BLOCK) % engine._echo_len, (
        "the ring must not have been reset"
    )


def _to_the_hand_over(session, style: str = "echo_out", seconds: float = 2.0) -> None:
    """Run a transition into a silent deck B until it has handed over."""
    engine = session.engine
    engine.deck("b").attach(None, 0)          # B silent: only the tail can sound
    total = int(SAMPLE_RATE * seconds)
    _start(session, style, total)
    drive(engine, total // BLOCK + 2, BLOCK)
    assert not engine.transition_active


def test_echo_repeats_ring_out_after_the_hand_over(session):
    """Deck A has stopped and B is silent: anything left is the echo, and it
    must decay away by itself rather than stopping dead."""
    engine = session.engine
    _to_the_hand_over(session)
    after = drive(engine, 8, BLOCK)
    assert float(np.max(np.abs(after))) > 1e-3, "the tail was cut at the hand-over"

    # It decays: louder soon after the hand-over than near the end of its run.
    early = float(np.sqrt(np.mean(after ** 2)))
    drive(engine, max(0, engine._echo_ring_left // BLOCK - 8), BLOCK)
    late = float(np.sqrt(np.mean(drive(engine, 8, BLOCK) ** 2)))
    assert late < early * 0.2, f"no decay: early rms {early:.5f}, late {late:.5f}"

    drive(engine, 8, BLOCK)
    assert engine._echo_ring_left <= 0
    assert float(np.max(np.abs(drive(engine, 8, BLOCK)))) < 1e-4, "and then stops"


def test_a_panic_cut_silences_a_ringing_echo_at_once(session):
    engine = session.engine
    _to_the_hand_over(session)
    assert engine._echo_ring_left > 0
    engine.submit(Cut(deck="master", execute_at=IMMEDIATE, origin="panic"))
    drive(engine, 1, BLOCK)
    assert engine._echo_ring_left == 0
    assert float(np.max(np.abs(drive(engine, 4, BLOCK)))) < 1e-4


def _heap_growth(engine, blocks=400):
    """Bytes the heap grows across ``blocks`` callbacks."""
    import gc
    import tracemalloc

    buf = np.zeros((BLOCK, 2), dtype=np.float32)
    for _ in range(20):                     # settle caches first
        engine.callback(buf, BLOCK, None, None)
    gc.collect()
    # Deep tracebacks, so an allocation made on another thread can be told
    # apart by its thread bootstrap frame. tracemalloc records every thread,
    # and a full single-process run once caught a stretch worker left running
    # by an earlier test reading an 82 MB Rubber Band output inside this
    # window. The callback here runs on the main thread, which never passes
    # through threading.py's bootstrap -- so excluding those traces measures
    # exactly the thread the test is about.
    tracemalloc.start(40)
    other_threads = [tracemalloc.Filter(False, "*threading.py", all_frames=True)]
    before = tracemalloc.take_snapshot().filter_traces(other_threads)
    for _ in range(blocks):
        engine.callback(buf, BLOCK, None, None)
    after = tracemalloc.take_snapshot().filter_traces(other_threads)
    tracemalloc.stop()
    return sum(s.size_diff for s in after.compare_to(before, "filename"))


def _heap_slope(engine, blocks=400) -> float:
    """Bytes of heap growth per callback, with fixed costs cancelled out.

    Growth over ``blocks`` callbacks and over twice as many, differenced.
    Anything that costs the same however long it runs -- tracemalloc's own
    bookkeeping, a cache an earlier test left half-warm -- appears in both and
    cancels. What remains grows with every block, which is the only kind of
    growth an allocation in the audio path can produce.
    """
    shorter = _heap_growth(engine, blocks)
    longer = _heap_growth(engine, 2 * blocks)
    return (longer - shorter) / blocks


def test_the_echo_delay_line_allocates_nothing_the_mixer_does_not(session):
    """The acceptance bar for `echo_out`: the delay costs no allocation.

    Measured against the same callback running a bass swap, which is the only
    meaningful comparison: the interpreter allocates a little for any work at
    all, and what matters is that the delay line adds none of it. A per-block
    buffer in the send would show up as growth proportional to the block count.

    Compared as growth per block rather than as a total. The total moved by a
    few hundred bytes with whatever the tests before it had left behind --
    enough to cross a fixed 8 KB margin in a full single-process run while
    passing alone. A per-block rate is immune to that and still sees a
    block-sized buffer, which costs thousands of bytes every callback.
    """
    engine = session.engine
    engine.deck("b").attach(engine.deck("a").track, 0)

    _start(session, "bass_swap", SAMPLE_RATE * 60)
    drive(engine, 4, BLOCK)
    baseline = _heap_slope(engine)

    _start(session, "echo_out", SAMPLE_RATE * 60)
    drive(engine, 8, BLOCK)
    assert engine._echo_send > 0.0, "the send should be open for this measurement"
    with_echo = _heap_slope(engine)
    assert engine._echo_send > 0.0, "and still open at the end of it"

    # A stereo float32 buffer the size of one block is thousands of bytes a
    # callback; 64 bytes a block is far below that and far above the jitter of
    # a rate taken over 400 differenced blocks.
    assert with_echo <= baseline + 64.0, (
        f"echo grows the heap {with_echo - baseline:.1f} bytes per block more "
        f"than a bass swap (baseline {baseline:.1f}, with echo {with_echo:.1f})"
    )


def test_the_echo_ring_is_allocated_once_and_never_resized(session):
    """Whatever the tempo, the ring is the one that was built at startup."""
    engine = session.engine
    ring = engine._echo_ring
    for bpm in (60.0, 128.0, 175.0):
        engine.arm_transition_plan("echo_out", SAMPLE_RATE * 4, bpm)
        assert engine._echo_ring is ring, "the ring must never be rebuilt"
        assert 0 < engine._armed_plan[3] < engine._echo_len


# --- choosing ----------------------------------------------------------------


class _FakeDeck:
    def __init__(self, track, position=0.0, rate=1.0):
        self.track = track
        self.position = position
        self.rate = rate


class _Loaded:
    def __init__(self, analysis):
        self.analysis = analysis


def _deck_for(analysis, position_s=0.0):
    return _FakeDeck(_Loaded(analysis), position=position_s * SAMPLE_RATE)


def test_a_weak_grid_forces_a_cut_and_says_so(session):
    a, b = session.crate[0], session.crate[1]
    object.__setattr__(b, "grid_confidence", 0.05)
    choice = tr.choose_transition(_deck_for(a), b)
    assert choice.style == "cut"
    assert "grid confidence" in choice.rule


def test_a_big_tempo_gap_forces_a_cut(session):
    a, b = session.crate[0], session.crate[1]
    object.__setattr__(b, "bpm", a.bpm * 1.5)
    choice = tr.choose_transition(_deck_for(a), b)
    assert choice.style == "cut"
    assert "BPM delta" in choice.rule


def test_clashing_keys_with_close_tempo_choose_a_filter_sweep(session):
    a, b = session.crate[0], session.crate[1]
    object.__setattr__(b, "bpm", a.bpm)
    object.__setattr__(a, "camelot", "1A")
    object.__setattr__(b, "camelot", "7B")     # as far around the wheel as it gets
    object.__setattr__(b, "energy", a.energy)
    choice = tr.choose_transition(_deck_for(a), b)
    assert choice.style == "filter_sweep"
    assert "keys clash" in choice.rule


def test_rising_energy_on_compatible_keys_stays_a_bass_swap(session):
    a, b = session.crate[0], session.crate[1]
    object.__setattr__(b, "bpm", a.bpm)
    object.__setattr__(a, "camelot", "8A")
    object.__setattr__(b, "camelot", "8A")
    object.__setattr__(b, "energy", a.energy * 2)
    choice = tr.choose_transition(_deck_for(a), b)
    assert choice.style == "bass_swap"
    assert "energy rising" in choice.rule


def test_an_explicit_request_wins_over_the_rules(session):
    a, b = session.crate[0], session.crate[1]
    object.__setattr__(b, "grid_confidence", 0.01)   # would otherwise force a cut
    choice = tr.choose_transition(_deck_for(a), b, {"style": "echo_out"})
    assert choice.style == "echo_out"
    assert "requested explicitly" in choice.rule


def test_a_drop_hot_cue_on_deck_b_chooses_the_cue_jump_entry(session):
    from djai import analysis as an

    a, b = session.crate[0], session.crate[1]
    bar_s = 4 * 60.0 / b.bpm
    b.hot_cues = [
        an.make_hot_cue(1, b.mix_in, "mix in"),
        an.make_hot_cue(3, 20 * bar_s, "drop"),
    ]
    choice = tr.choose_transition(_deck_for(a), b)
    assert choice.entry == "cue_jump_in"
    assert choice.hot_cue_index == 3
    assert "cue_jump_in" in choice.rule


def test_a_late_drop_cue_is_not_used_as_an_entry(session):
    """Regression: a drop near the end starves the deck of runway.

    Seen in a soak. Deck B entered on a drop three quarters through its track,
    ran out almost as soon as the blend handed over, and the recovery path's
    gap was audible as dead air. An entry has to leave real track behind it.
    """
    from djai import analysis as an

    a, b = session.crate[0], session.crate[1]
    bar_s = 4 * 60.0 / b.bpm
    too_late = b.mix_out - (tr.MIN_ENTRY_RUNWAY_BARS - 8) * bar_s
    b.hot_cues = [
        an.make_hot_cue(1, b.mix_in, "mix in"),
        an.make_hot_cue(4, too_late, "drop"),
    ]
    choice = tr.choose_transition(_deck_for(a), b)
    assert choice.entry == "mix_in", "a late drop must not be used as an entry"
    assert choice.hot_cue_index is None


def test_a_drop_cue_too_early_to_skip_anything_is_not_used(session):
    from djai import analysis as an

    a, b = session.crate[0], session.crate[1]
    bar_s = 4 * 60.0 / b.bpm
    b.hot_cues = [an.make_hot_cue(4, 2 * bar_s, "drop")]
    assert tr.choose_transition(_deck_for(a), b).entry == "mix_in"


def test_the_earliest_qualifying_drop_wins(session):
    """Skip the intro, keep the track."""
    from djai import analysis as an

    a, b = session.crate[0], session.crate[1]
    bar_s = 4 * 60.0 / b.bpm
    early = 16 * bar_s
    later = b.mix_out - (tr.MIN_ENTRY_RUNWAY_BARS + 4) * bar_s
    b.hot_cues = [
        an.make_hot_cue(5, later, "drop"),
        an.make_hot_cue(3, early, "drop"),
    ]
    choice = tr.choose_transition(_deck_for(a), b)
    assert choice.hot_cue_index == 3


def test_no_drop_cue_means_the_incoming_deck_enters_at_its_mix_in(session):
    a, b = session.crate[0], session.crate[1]
    b.hot_cues = []
    choice = tr.choose_transition(_deck_for(a), b)
    assert choice.entry == "mix_in"
    assert choice.hot_cue_index is None


def test_camelot_compatibility_is_the_wheel_not_string_equality():
    assert tr.camelot_compatible("8A", "8A")
    assert tr.camelot_compatible("8A", "8B")      # relative major/minor
    assert tr.camelot_compatible("8A", "9A")      # one step, same mode
    assert tr.camelot_compatible("12A", "1A")     # the wheel wraps
    assert not tr.camelot_compatible("8A", "2A")
    assert not tr.camelot_compatible("8A", None)


def test_every_style_the_chooser_can_return_is_one_the_engine_implements(session):
    a, b = session.crate[0], session.crate[1]
    for style in tr.STYLE_CHOICES:
        choice = tr.choose_transition(_deck_for(a), b, {"style": style})
        assert choice.style in tr.STYLES
        tr.build_envelope(choice.style, SAMPLE_RATE, BLOCK)
