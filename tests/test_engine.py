"""Engine, scheduler and transition tests.

THREADING CONTEXT: main thread (pytest). The audio callback is driven directly
here rather than by PortAudio, which makes every timing assertion deterministic
and lets the whole mix be inspected sample by sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from djai import phrase, transition
from djai.commands import IMMEDIATE, Cut, KillBass, LoadTrack, SetEQ, StartTransition
from djai.deck import CHANNELS, SAMPLE_RATE, LoadedTrack
from djai.engine import Engine
from djai.scheduler import Scheduler
from tests.test_deck import make_track


def drive(engine: Engine, n_blocks: int, block: int = 512) -> np.ndarray:
    """Run the audio callback n_blocks times, returning the whole output."""
    out = np.zeros((n_blocks * block, CHANNELS), dtype=np.float32)
    buf = np.zeros((block, CHANNELS), dtype=np.float32)
    for i in range(n_blocks):
        engine.callback(buf, block, None, None)
        out[i * block : (i + 1) * block] = buf
    return out


def loud_track(freq: float, bpm: float = 120.0, seconds: float = 30.0) -> LoadedTrack:
    """A track that is a single steady tone, so each deck is identifiable."""
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    sig = 0.5 * np.sin(2 * np.pi * freq * t)
    audio = np.stack([sig, sig], axis=1).astype(np.float32)
    audio = np.vstack([audio, np.zeros((2, CHANNELS), dtype=np.float32)])
    base = make_track(n_frames=n, bpm=bpm)
    return LoadedTrack(analysis=base.analysis, audio=audio)


def test_engine_mixes_both_decks():
    engine = Engine(blocksize=512)
    engine.submit(LoadTrack(deck="a", track=loud_track(200.0), master=True))
    engine.submit(LoadTrack(deck="b", track=loud_track(3000.0)))
    engine.deck_a.gain.jump(0.5)
    engine.deck_b.gain.jump(0.5)

    out = drive(engine, 200)[8192:]
    spec = np.abs(np.fft.rfft(out[:, 0] * np.hanning(out.shape[0])))
    freqs = np.fft.rfftfreq(out.shape[0], 1 / SAMPLE_RATE)

    def level(f):
        band = (freqs > f * 0.9) & (freqs < f * 1.1)
        return float(np.max(spec[band]))

    assert level(200.0) > 1e3, "deck A missing from the mix"
    assert level(3000.0) > 1e3, "deck B missing from the mix"


def test_master_clock_advances_at_the_master_tempo():
    engine = Engine(blocksize=512)
    engine.submit(LoadTrack(deck="a", track=loud_track(200.0, bpm=124.0), master=True))
    blocks = 200
    drive(engine, blocks)
    seconds = blocks * 512 / SAMPLE_RATE
    assert engine.master_beat == pytest.approx(seconds * 124.0 / 60.0, rel=1e-6)
    assert engine.master_bpm == pytest.approx(124.0, rel=1e-9)


def test_callback_never_raises_even_with_a_broken_deck():
    """An exception escaping the callback would tear down the stream."""
    engine = Engine(blocksize=512)
    engine.submit(LoadTrack(deck="a", track=loud_track(200.0), master=True))
    drive(engine, 2)
    engine.deck_a.track = "not a track"  # type: ignore[assignment]
    buf = np.zeros((512, CHANNELS), dtype=np.float32)
    engine.callback(buf, 512, None, None)  # must not raise
    assert not np.any(buf), "a failed callback must emit silence"
    assert engine.underruns > 0, "the failure should be counted"


def test_cut_takes_effect_within_one_block():
    engine = Engine(blocksize=512)
    engine.submit(LoadTrack(deck="a", track=loud_track(200.0), master=True))
    engine.deck_a.gain.jump(1.0)
    drive(engine, 20)

    engine.submit(Cut(deck="master", execute_at=IMMEDIATE))
    after = drive(engine, 2)
    # The ramp completes inside the first block after the command.
    assert np.max(np.abs(after[512:])) < 1e-6, "cut did not take effect in one block"


def test_cut_latency_is_under_100ms():
    block = 512
    engine = Engine(blocksize=block)
    engine.submit(LoadTrack(deck="a", track=loud_track(200.0), master=True))
    engine.deck_a.gain.jump(1.0)
    drive(engine, 20)
    engine.submit(Cut(deck="master", execute_at=IMMEDIATE))

    buf = np.zeros((block, CHANNELS), dtype=np.float32)
    blocks_to_silence = 0
    for _ in range(20):
        engine.callback(buf, block, None, None)
        blocks_to_silence += 1
        if np.max(np.abs(buf)) < 1e-6:
            break
    latency_ms = blocks_to_silence * block / SAMPLE_RATE * 1000
    assert latency_ms < 100.0, f"cut took {latency_ms:.1f} ms"


def test_killbass_removes_low_end_from_both_decks():
    engine = Engine(blocksize=512)
    engine.submit(LoadTrack(deck="a", track=loud_track(60.0), master=True))
    engine.submit(LoadTrack(deck="b", track=loud_track(60.0)))
    engine.deck_a.gain.jump(0.5)
    engine.deck_b.gain.jump(0.5)
    drive(engine, 100)

    engine.submit(KillBass(deck="master", killed=True, execute_at=IMMEDIATE))
    after = drive(engine, 200)[SAMPLE_RATE:]
    assert np.max(np.abs(after)) < 0.02, "60 Hz survived a master bass kill"


def test_set_eq_only_touches_the_named_bands():
    engine = Engine(blocksize=512)
    engine.submit(LoadTrack(deck="a", track=loud_track(200.0), master=True))
    engine.submit(SetEQ(deck="a", low=0.25, execute_at=IMMEDIATE))
    drive(engine, 2)
    assert engine.deck_a.eq_low.target == pytest.approx(0.25)
    assert engine.deck_a.eq_mid.target == pytest.approx(1.0)
    assert engine.deck_a.eq_high.target == pytest.approx(1.0)


# --- transition --------------------------------------------------------------


def test_transition_endpoints():
    length = transition.TRANSITION_BARS
    start = transition.gains_at(0.0)
    assert start.from_gain == pytest.approx(1.0)
    assert start.from_low == pytest.approx(1.0)
    assert start.to_gain == pytest.approx(0.0)
    assert start.to_low == pytest.approx(0.0), "incoming must enter with no bass"

    end = transition.gains_at(length)
    assert end.from_gain == pytest.approx(0.0)
    assert end.to_gain == pytest.approx(1.0)
    assert end.to_low == pytest.approx(1.0)
    assert end.from_low == pytest.approx(0.0)


def test_crossfade_is_equal_power_everywhere():
    """The measured fault was both decks sitting at unity gain for 12 bars,
    which summed +5 to +9 dB over one deck and clipped 26-45% of blocks.
    Constant power is the contract that prevents it."""
    length = transition.TRANSITION_BARS
    for i in range(401):
        bar = length * i / 400.0
        g = transition.gains_at(bar)
        power = g.from_gain**2 + g.to_gain**2
        assert power == pytest.approx(1.0, abs=1e-9), (
            f"power {power:.4f} at bar {bar:.2f}, not constant"
        )
        assert not (g.from_gain > 0.999 and g.to_gain > 0.999), (
            f"both decks at unity at bar {bar:.2f}"
        )


def test_gains_are_monotonic_across_the_transition():
    length = transition.TRANSITION_BARS
    bars = [length * i / 200.0 for i in range(201)]
    from_gains = [transition.gains_at(b).from_gain for b in bars]
    to_gains = [transition.gains_at(b).to_gain for b in bars]
    assert all(a >= b - 1e-12 for a, b in zip(from_gains, from_gains[1:]))
    assert all(a <= b + 1e-12 for a, b in zip(to_gains, to_gains[1:]))
    mid = transition.gains_at(length / 2.0)
    assert mid.from_gain == pytest.approx(2**-0.5, abs=1e-9)
    assert mid.to_gain == pytest.approx(2**-0.5, abs=1e-9)


def test_exactly_one_deck_owns_the_low_band_at_every_point():
    """Never both, never neither -- so the low end is never doubled or absent."""
    length = transition.TRANSITION_BARS
    for i in range(401):
        g = transition.gains_at(length * i / 400.0)
        lows = sorted((g.from_low, g.to_low))
        assert lows[0] == pytest.approx(0.0), "both decks have some low band"
        assert lows[1] == pytest.approx(1.0), "neither deck has a full low band"
        assert g.from_low + g.to_low == pytest.approx(1.0)


def test_bass_hands_over_at_the_midpoint():
    length = transition.TRANSITION_BARS
    just_before = transition.gains_at(length * transition.BASS_SWAP_AT - 1e-6)
    just_after = transition.gains_at(length * transition.BASS_SWAP_AT + 1e-6)
    assert just_before.from_low == pytest.approx(1.0)
    assert just_before.to_low == pytest.approx(0.0)
    assert just_after.from_low == pytest.approx(0.0)
    assert just_after.to_low == pytest.approx(1.0)


def test_transition_length_is_config_driven():
    from djai import config

    assert transition.TRANSITION_BARS == config.TRANSITION_BARS
    # 24 bars, measured from reference sets rather than chosen -- see
    # docs/tuning.md. Asserted as a plausible musical phrase length rather than
    # a literal, so retuning does not break the test that proves it is tunable.
    assert config.TRANSITION_BARS in (8.0, 16.0, 24.0, 32.0, 40.0)
    # An explicit length overrides the default without touching module state.
    g = transition.gains_at(4.0, transition_bars=8.0)
    assert g.from_gain == pytest.approx(2**-0.5, abs=1e-9)
    frames = transition.transition_frames(120.0, SAMPLE_RATE, transition_bars=8.0)
    assert frames == pytest.approx(8 * 4 * 0.5 * SAMPLE_RATE, rel=1e-6)


def test_transition_is_clamped_outside_its_range():
    before = transition.gains_at(-5.0)
    after = transition.gains_at(transition.TRANSITION_BARS * 3)
    assert before.to_gain == pytest.approx(0.0)
    assert before.from_gain == pytest.approx(1.0)
    assert after.from_gain == pytest.approx(0.0)
    assert after.to_gain == pytest.approx(1.0)


def test_transition_runs_to_completion_and_hands_over():
    bpm = 120.0
    engine = Engine(blocksize=512)
    engine.submit(LoadTrack(deck="a", track=loud_track(200.0, bpm, 120.0), master=True))
    engine.submit(
        LoadTrack(deck="b", track=loud_track(3000.0, bpm, 120.0), play=False)
    )
    engine.deck_a.gain.jump(1.0)
    drive(engine, 10)

    total = transition.transition_frames(bpm, SAMPLE_RATE)
    engine.submit(
        StartTransition(from_deck="a", to_deck="b", total_frames=total)
    )
    drive(engine, 4)
    assert engine.transition_active
    assert engine.deck_b.playing, "incoming deck must start playing"
    assert engine.deck_b.eq_low.target == pytest.approx(0.0, abs=1e-3)

    # Run past the end of the transition.
    drive(engine, total // 512 + 8)
    assert not engine.transition_active
    assert not engine.deck_a.playing, "outgoing deck should be stopped"
    assert engine.deck_b.gain.target == pytest.approx(1.0)
    assert engine.deck_b.eq_low.target == pytest.approx(1.0)
    assert engine.master_deck == "b", "master clock should follow the incoming deck"


def test_transition_output_is_continuous_and_never_silent():
    """No dropout, no clipping runaway, no discontinuity through the swap."""
    bpm = 120.0
    engine = Engine(blocksize=512)
    engine.submit(LoadTrack(deck="a", track=loud_track(200.0, bpm, 120.0), master=True))
    engine.submit(
        LoadTrack(deck="b", track=loud_track(200.0, bpm, 120.0), play=False)
    )
    engine.deck_a.gain.jump(1.0)
    drive(engine, 10)

    total = transition.transition_frames(bpm, SAMPLE_RATE)
    engine.submit(StartTransition(from_deck="a", to_deck="b", total_frames=total))
    out = drive(engine, total // 512)

    assert np.all(np.isfinite(out))
    # Envelope per 0.1 s window: never silent, never a jump.
    win = SAMPLE_RATE // 10
    env = np.array(
        [np.max(np.abs(out[i : i + win, 0])) for i in range(0, out.shape[0] - win, win)]
    )
    assert env.min() > 0.05, f"mix dropped to {env.min():.3f} during the transition"
    assert np.max(np.abs(np.diff(env))) < 0.35, "envelope discontinuity in the mix"


# --- scheduler ---------------------------------------------------------------


def test_scheduler_holds_until_the_playhead_arrives():
    engine = Engine(blocksize=512)
    sched = Scheduler(engine)
    cmd = Cut(deck="master", execute_at=10_000)
    sched.submit(cmd)

    assert sched.tick(9_999) == [], "fired early"
    assert len(sched) == 1
    assert sched.tick(10_000) == [cmd], "did not fire on time"
    assert len(sched) == 0


def test_scheduler_releases_in_musical_order():
    engine = Engine(blocksize=512)
    sched = Scheduler(engine)
    late = Cut(deck="a", execute_at=3000)
    early = Cut(deck="b", execute_at=1000)
    sched.submit(late)
    sched.submit(early)
    assert sched.tick(5000) == [early, late]


def test_scheduler_passes_immediate_commands_straight_through():
    engine = Engine(blocksize=512)
    sched = Scheduler(engine)
    sched.submit(Cut(deck="master", execute_at=IMMEDIATE))
    assert len(sched) == 0, "immediate commands must not be held"
    assert engine.queue.qsize() == 1


def test_scheduler_cancel_all_drops_everything():
    engine = Engine(blocksize=512)
    sched = Scheduler(engine)
    sched.submit(Cut(deck="a", execute_at=1000))
    sched.submit(Cut(deck="b", execute_at=2000))
    assert len(sched.cancel_all()) == 2
    assert sched.tick(10_000) == []


def test_full_queue_is_reported_not_swallowed():
    engine = Engine(blocksize=512, queue_size=2)
    dropped = []
    sched = Scheduler(engine, on_dropped=dropped.append)
    for _ in range(5):
        sched.submit(Cut(deck="master", execute_at=IMMEDIATE))
    assert len(dropped) == 3, "overflow must be surfaced, not silently lost"


# --- phrase ------------------------------------------------------------------


def test_phrase_boundaries_land_on_32_bar_multiples():
    track = loud_track(200.0, bpm=120.0, seconds=300.0)
    engine = Engine(blocksize=512)
    engine.submit(LoadTrack(deck="a", track=track, master=True))
    drive(engine, 1)

    deck = engine.deck_a
    boundary = phrase.next_phrase_boundary(deck)
    bar = phrase.bar_at_frame(track.analysis, boundary)
    assert bar == pytest.approx(32.0, abs=1e-3)
    assert bar % phrase.BARS_PER_PHRASE == pytest.approx(0.0, abs=1e-3)


def test_next_phrase_boundary_is_strictly_in_the_future():
    track = loud_track(200.0, bpm=120.0, seconds=300.0)
    engine = Engine(blocksize=512)
    engine.submit(LoadTrack(deck="a", track=track, master=True))
    drive(engine, 1)
    deck = engine.deck_a

    exact = phrase.frame_at_bar(track.analysis, 32.0)
    nxt = phrase.next_phrase_boundary(deck, after_frame=exact)
    assert nxt > exact, "sitting on a boundary must return the next one"
    assert phrase.bar_at_frame(track.analysis, nxt) == pytest.approx(64.0, abs=1e-3)


def test_a_late_release_is_compensated_at_the_cue_point():
    """A command released late must start where it would have been on time.

    The scheduler polls at 2 ms and commands apply at the top of a callback, so
    a cue can land a block after its musical moment. Without compensation that
    is a systematic phase error at the start of every transition.
    """
    engine = Engine(blocksize=512)
    track = loud_track(200.0, bpm=120.0, seconds=60.0)
    engine.submit(LoadTrack(deck="a", track=track, master=True))
    drive(engine, 10)  # frames_played = 5120

    late_by = 700  # the boundary was 700 frames ago
    engine.submit(
        LoadTrack(
            deck="b",
            track=track,
            start_frame=1000,
            rate=1.0,
            execute_at=engine.frames_played - late_by,
        )
    )
    engine.callback(np.zeros((512, CHANNELS), np.float32), 512, None, None)
    # Started at the cue point advanced by exactly the lateness, then one block.
    assert engine.deck_b.position == pytest.approx(1000 + late_by + 512)


def test_an_immediate_command_is_not_compensated():
    engine = Engine(blocksize=512)
    track = loud_track(200.0, bpm=120.0, seconds=60.0)
    engine.submit(LoadTrack(deck="a", track=track, master=True))
    drive(engine, 10)
    engine.submit(LoadTrack(deck="b", track=track, start_frame=1000, play=False))
    drive(engine, 1)
    assert engine.deck_b.position == pytest.approx(1000)


def test_cue_point_at_the_first_downbeat_is_bar_zero():
    """Regression: cueing exactly on the first downbeat reported bar -1.

    beat_at_frame returns a tiny negative float there, and Python's % maps that
    to ~3.9999 instead of 0, shifting bar numbering and every phrase boundary.
    """
    track = loud_track(200.0, bpm=124.0, seconds=300.0)
    a = track.analysis
    cue = a.first_downbeat * SAMPLE_RATE
    assert phrase.bar_at_frame(a, cue) == pytest.approx(0.0, abs=1e-6)
    assert phrase.downbeat_offset_beats(a) == pytest.approx(0.0)
    assert phrase.bar_at_frame(a, phrase.frame_at_bar(a, 32.0)) == pytest.approx(32.0)


def test_bar_and_beat_conversions_round_trip():
    track = loud_track(200.0, bpm=127.3, seconds=120.0)
    a = track.analysis
    for bar in (0.0, 1.0, 7.5, 32.0, 129.25):
        frame = phrase.frame_at_bar(a, bar)
        assert phrase.bar_at_frame(a, frame) == pytest.approx(bar, abs=1e-6)
    for beat in (0.0, 3.0, 128.0):
        frame = phrase.frame_at_beat(a, beat)
        assert phrase.beat_at_frame(a, frame) == pytest.approx(beat, abs=1e-6)
