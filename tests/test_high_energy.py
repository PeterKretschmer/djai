"""Loops and the five high-energy transitions, against the Part 4 criteria.

THREADING CONTEXT: main thread (pytest). No audio device: the engine callback
is driven directly, which is the same code the audio thread runs.

What is defended here, in order: the envelope says what each style does, in
beats; the control thread turns beats into frames; the audio thread only
assigns; a loop leaves the deck exactly where straight playback would be; the
rendered audio does not click and does not clip; and a drop-aligned design
cannot run on a track that has no drop to align.
"""

from __future__ import annotations

import csv
import gzip
import inspect

import numpy as np
import pytest
import soundfile as sf

from djai import analysis, phrase
from djai import transition as tr
from djai.commands import IMMEDIATE, LoadTrack, StartTransition
from djai.deck import CHANNELS, SAMPLE_RATE, Deck, LoadedTrack
from djai.engine import Engine
from djai.render import ENVELOPE_COLUMNS, render_transition
from tests import synth
from tests.test_designed_transitions import GOOD, FakeModel, _arm
from tests.test_engine import drive, loud_track
from tests.test_integration import log_events
from tests.test_integration import session as _session_fixture

session = _session_fixture

BLOCK = 512
BPM = 120.0
HIGH_ENERGY = ("loop_roll_out", "drop_swap", "beat_repeat_in", "backspin", "double_drop")


def frames_per_beat(bpm: float = BPM) -> float:
    return 60.0 / bpm * SAMPLE_RATE


# --- the envelope: beats in, frames out ------------------------------------------


def _env(style: str, bpm: float = BPM):
    total = tr.transition_frames(bpm, SAMPLE_RATE, transition_bars=tr.style_bars(style))
    return tr.build_envelope(style, total, BLOCK, bpm, SAMPLE_RATE)


def _bars_of(env, bpm: float = BPM) -> np.ndarray:
    """The bar each row is READ for, given the engine advances before lookup.

    This is the clock loop regions are resolved against.
    """
    rows = np.arange(env.shape[0])
    return np.maximum(0, rows - 1) * BLOCK / (4 * frames_per_beat(bpm))


def _shape_bars(env, bpm: float = BPM) -> np.ndarray:
    """The bar each row's gains and rate were BUILT for: row i is t = i blocks.

    One block ahead of the read clock, as every style has always been. Shape
    assertions use this so a boundary row is not misjudged by that block.
    """
    return np.arange(env.shape[0]) * BLOCK / (4 * frames_per_beat(bpm))


def test_every_preset_reports_the_length_it_is_armed_with():
    """The renderer arms from style_bars, the live path from the preset.

    `cut` is the one exception and is armed as a single block by both.
    """
    for style in tr.STYLES:
        if style == "cut":
            continue
        assert tr.preset_params(style).length_bars == tr.style_bars(style), style


def test_loop_roll_out_halves_four_two_one_half_every_two_bars():
    env = _env("loop_roll_out")
    bars = _bars_of(env)
    fpb = frames_per_beat()
    for step, loop_bars in enumerate(tr.LOOP_ROLL_BARS):
        lo = step * tr.LOOP_ROLL_STEP_BARS
        rows = (bars >= lo + 0.1) & (bars < lo + tr.LOOP_ROLL_STEP_BARS - 0.1)
        assert rows.any()
        assert np.allclose(env[rows, tr.LOOP_BEATS], loop_bars * 4)
        assert np.allclose(env[rows, tr.LOOP_LEN], loop_bars * 4 * fpb, rtol=1e-6)

    # Halving moves the loop's end, never its start.
    engaged = env[:, tr.LOOP_LEN] > 0
    assert len(set(np.round(env[engaged, tr.LOOP_START], 3).tolist())) == 1


def test_beat_repeat_in_stutters_only_the_final_bar():
    env = _env("beat_repeat_in")
    bars = _bars_of(env)
    total = tr.style_bars("beat_repeat_in")
    stutter = env[:, tr.LOOP_BEATS] > 0
    assert np.allclose(env[stutter, tr.LOOP_BEATS], 4.0 / 16)
    assert np.all(bars[stutter] >= total - tr.BEAT_REPEAT_BARS - 0.05)
    assert stutter[:-1][bars[:-1] > total - 0.9].all(), "the whole last bar stutters"
    # A stutter starts on a beat line.
    starts = env[stutter, tr.LOOP_START] / frames_per_beat()
    assert np.allclose(starts, np.round(starts), atol=1e-3)


def test_backspin_goes_through_zero_into_reverse_over_the_last_bar():
    env = _env("backspin")
    bars = _shape_bars(env)
    total = tr.style_bars("backspin")
    rate = env[:, tr.RATE_A]
    assert np.allclose(rate[bars < total - tr.BACKSPIN_BARS], 1.0)
    assert float(rate[:-1].min()) == pytest.approx(tr.BACKSPIN_END_RATE, abs=0.05)
    assert np.all(np.diff(rate[:-1]) <= 1e-6), "the spin only slows"
    # And the gain decays with it rather than stopping dead.
    assert np.all(np.diff(env[:-1, tr.FROM_GAIN]) <= 1e-6)


def test_double_drop_holds_both_decks_under_the_headroom():
    env = _env("double_drop")
    head = tr.DOUBLE_DROP_HEADROOM
    assert head < 1.0
    assert np.all(env[:-1, tr.FROM_GAIN] <= head + 1e-6)
    shared = _shape_bars(env) < tr.DOUBLE_DROP_BARS * 0.75
    assert np.allclose(env[shared, tr.FROM_GAIN], head)
    assert np.allclose(env[shared, tr.TO_GAIN], head)
    # Summed power of the shared passage is under a single deck at unity.
    assert np.all(env[shared, tr.FROM_GAIN] ** 2 + env[shared, tr.TO_GAIN] ** 2 < 1.0)
    assert np.all(env[:, tr.FROM_LOW] == 0.0), "never two basslines"


def test_every_style_hands_back_a_straight_deck():
    for style in tr.STYLES:
        env = _env(style)
        assert env[-1, tr.LOOP_LEN] == 0.0, style
        assert env[-1, tr.RATE_A] == 1.0, style


def test_no_tempo_means_no_loop_rather_than_a_guessed_one():
    total = tr.transition_frames(BPM, SAMPLE_RATE, transition_bars=8)
    env = tr.build_envelope("loop_roll_out", total, BLOCK, 0.0, SAMPLE_RATE)
    assert not np.any(env[:, tr.LOOP_LEN])


# --- designed transitions: the six fields do what they say ------------------------


def _designed(**fields) -> np.ndarray:
    p = tr.TransitionParams(**{"length_bars": 16, "curve": "equal_power", **fields})
    total = tr.transition_frames(BPM, SAMPLE_RATE, transition_bars=p.length_bars)
    return tr.build_envelope_from_params(p, total, BLOCK, BPM, SAMPLE_RATE)


def test_a_design_with_the_high_energy_set_off_leaves_deck_a_straight():
    env = _designed()
    assert not np.any(env[:, tr.LOOP_LEN])
    assert np.all(env[:, tr.RATE_A] == 1.0)


def test_a_designed_halving_roll_runs_over_the_end_of_the_window():
    env = _designed(length_bars=16, loop_out_bars=4, loop_halving=True)
    bars = _bars_of(env)
    beats = env[:-1, tr.LOOP_BEATS]
    assert set(np.unique(beats).tolist()) == {0.0, 16.0, 8.0, 4.0, 2.0}
    assert np.all(beats[bars[:-1] < 16 - 8 - 0.05] == 0.0), "roll spans the last 8 bars"


def test_a_designed_loop_without_halving_holds_one_length():
    env = _designed(length_bars=16, loop_out_bars=2)
    assert set(np.unique(env[:-1, tr.LOOP_BEATS]).tolist()) == {0.0, 8.0}


@pytest.mark.parametrize("division", [8, 16])
def test_a_designed_stutter_uses_its_division(division):
    env = _designed(beat_repeat_division=division)
    assert set(np.unique(env[:-1, tr.LOOP_BEATS]).tolist()) == {0.0, 4.0 / division}


def test_a_designed_backspin_is_sized_in_bars_of_the_window():
    env = _designed(length_bars=16, backspin_bars=2)
    bars = _bars_of(env)
    rate = env[:-1, tr.RATE_A]
    assert np.all(rate[bars[:-1] < 14 - 0.05] == 1.0)
    assert float(rate.min()) < 0.0


def test_a_designed_double_drop_applies_the_headroom():
    env = _designed(length_bars=16, double_drop_bars=8)
    shared = _bars_of(env) < 8 - 0.05
    assert np.allclose(env[shared, tr.FROM_GAIN], tr.DOUBLE_DROP_HEADROOM)
    assert np.allclose(env[shared, tr.TO_GAIN], tr.DOUBLE_DROP_HEADROOM)


# --- the engine: assigns, never computes -------------------------------------------


def _engine_mid_transition(style: str, bpm: float = BPM):
    """Deck A on a downbeat, deck B silent under it, the style armed and started."""
    eng = Engine(blocksize=BLOCK)
    a = loud_track(220.0, bpm, 120.0)
    b = loud_track(330.0, bpm, 120.0)
    start = int(round(phrase.frame_at_bar(a.analysis, 8)))
    eng.submit(LoadTrack(deck="a", track=a, start_frame=start, master=True, play=True))
    eng.submit(LoadTrack(deck="b", track=b, start_frame=0, play=True))
    eng.deck_a.gain.jump(1.0)
    eng.deck_b.gain.jump(0.0)
    total = tr.transition_frames(bpm, SAMPLE_RATE, transition_bars=tr.style_bars(style))
    eng.arm_transition_plan(style, total, bpm)
    eng.submit(StartTransition(from_deck="a", to_deck="b", total_frames=total,
                               execute_at=IMMEDIATE))
    return eng, start, total


def _blocks_to_bar(bar: float, bpm: float = BPM) -> int:
    return int(bar * 4 * frames_per_beat(bpm) // BLOCK)


def test_the_callback_contains_no_loop_arithmetic():
    """The stop condition, as a regression guard."""
    src = inspect.getsource(Engine._advance_transition)
    # `bpm` is deliberately not banned: the function hands the master clock to
    # the incoming deck when a transition completes, which predates loops and
    # is not loop arithmetic. What is banned is turning beats into frames.
    for banned in ("floor", "frames_per_beat", "60.0", "LOOP_BEATS"):
        assert banned not in src, f"{banned!r} is back in the audio path"
    assert "LOOP_LEN" in src and "LOOP_START" in src, "reads the resolved region"
    assert not hasattr(Engine, "_apply_loop")


def test_the_engine_puts_the_stutter_on_a_beat_line():
    eng, start, _ = _engine_mid_transition("beat_repeat_in")
    drive(eng, _blocks_to_bar(3.5), BLOCK)
    deck = eng.deck_a
    assert deck.loop_active
    loop_start, loop_len = deck.loop_region
    assert loop_len == pytest.approx(frames_per_beat() / 4, abs=0.1)
    beat = (loop_start - start) / frames_per_beat()
    assert beat == pytest.approx(round(beat), abs=1e-3)
    assert loop_start <= deck.position < loop_start + loop_len + BLOCK


def test_the_engine_rolls_the_loop_down():
    eng, start, _ = _engine_mid_transition("loop_roll_out")
    seen = []
    done = 0
    for step in range(len(tr.LOOP_ROLL_BARS)):
        target = _blocks_to_bar(step * tr.LOOP_ROLL_STEP_BARS + 1.0)
        drive(eng, target - done, BLOCK)
        done = target
        seen.append(eng.deck_a.loop_region)
    lengths = [round(length / frames_per_beat(), 3) for _, length in seen]
    assert lengths == [16.0, 8.0, 4.0, 2.0]
    assert len({round(s, 3) for s, _ in seen}) == 1, "the loop-in point moved"


def test_the_engine_spins_deck_a_backwards():
    eng, _, _ = _engine_mid_transition("backspin")
    drive(eng, _blocks_to_bar(3.9), BLOCK)
    assert eng.deck_a.rate < 0.0


@pytest.mark.parametrize("style", ["beat_repeat_in", "loop_roll_out"])
def test_phase_against_the_master_clock_survives_leaving_a_loop(style):
    """Aborted mid-loop, deck A must be where straight playback would have it.

    The quarter-beat stutter is the demanding case: any exit that simply
    continued from inside the loop would land off the beat.
    """
    eng, start, _ = _engine_mid_transition(style)
    n = _blocks_to_bar(3.3 if style == "beat_repeat_in" else 5.3)
    drive(eng, n, BLOCK)
    assert eng.deck_a.loop_active
    eng.deck_b.playing = False            # a pause: Part 2 aborts the transition
    drive(eng, 1, BLOCK)
    assert not eng.deck_a.loop_active
    assert eng.deck_a.rate == 1.0
    straight = start + (n + 1) * BLOCK
    assert eng.deck_a.position == pytest.approx(straight, abs=1.0)

    analysis_a = eng.deck_a.track.analysis
    phase = phrase.beat_at_frame(analysis_a, eng.deck_a.position) % 1.0
    expected = phrase.beat_at_frame(analysis_a, straight) % 1.0
    assert phase == pytest.approx(expected, abs=1e-4)


def test_leaving_a_sub_beat_loop_does_not_click():
    """The worst case: a pure tone whose phase is inverted across the jump.

    A quarter beat at 120 BPM is 27.5 cycles of 220 Hz, so the slip position
    is exactly half a cycle away from where the loop was. No crossfade can make
    that seamless; what it can do is spread the swing across its length. The
    bound is the tone's own slope plus the full swing shared out over the
    crossfade, and the counterfactual -- the same exit with the declick
    skipped -- is what a click looks like.
    """
    from djai.deck import _LOOP_XFADE

    base = loud_track(220.0, BPM, 30.0)
    peak = float(np.max(np.abs(base.audio)))

    def fresh():
        d = Deck("a")
        d.attach(LoadedTrack(analysis=base.analysis, audio=base.audio), 0)
        d.playing = True
        d.gain.jump(1.0)
        d.position = 8 * frames_per_beat()
        return d

    straight = fresh()
    ref = np.concatenate([straight.read(BLOCK).copy() for _ in range(60)])
    ref_step = float(np.max(np.abs(np.diff(ref, axis=0))))

    lap = frames_per_beat() * 0.25
    # Exit after 12 blocks: one completed lap, so the slip position is an odd
    # number of laps -- 27.5 cycles each -- from the looped one, which is the
    # phase flip. The last seam (5512 frames) falls in block 10, so blocks 11
    # onwards contain the exit and nothing else.
    exit_blocks = 12
    assert int(exit_blocks * BLOCK // lap) % 2 == 1

    def run(declick: bool) -> tuple[float, float]:
        d = fresh()
        d.set_loop(d.position, lap)
        out = [d.read(BLOCK).copy() for _ in range(exit_blocks)]
        d.clear_loop()
        if not declick:
            d._exit_pending = False
        out += [d.read(BLOCK).copy() for _ in range(20)]
        audio = np.concatenate(out)
        everything = float(np.max(np.abs(np.diff(audio, axis=0))))
        around_exit = audio[(exit_blocks - 1) * BLOCK:]
        return everything, float(np.max(np.abs(np.diff(around_exit, axis=0))))

    bound = ref_step + 2.0 * peak / _LOOP_XFADE
    smoothed_all, smoothed_exit = run(True)
    _, raw_exit = run(False)
    assert smoothed_all <= bound * 1.05, (
        f"seams or exit stepped {smoothed_all:.4f} over the bound {bound:.4f}"
    )
    assert raw_exit > smoothed_exit * 5, (
        f"the exit is not a discontinuity to begin with ({raw_exit:.4f}), "
        "so this test is not testing the declick"
    )


def test_high_energy_transitions_allocate_nothing_bass_swap_does_not():
    """Net allocation across callbacks, the project's standard measure.

    Measured over the part of each transition where its gesture runs, against
    the same number of bass-swap blocks.
    """
    import gc
    import tracemalloc

    def growth(style: str, from_bar: float, blocks: int) -> int:
        eng, _, _ = _engine_mid_transition(style)
        drive(eng, _blocks_to_bar(from_bar), BLOCK)
        buf = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
        gc.collect()
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        for _ in range(blocks):
            eng.callback(buf, BLOCK, None, None)
        after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        return sum(s.size_diff for s in after.compare_to(before, "filename"))

    baseline = growth("bass_swap", 3.0, 150)
    windows = {
        "loop_roll_out": (1.5, 600),     # through three halvings
        "beat_repeat_in": (2.9, 150),
        "backspin": (2.9, 150),
        "drop_swap": (2.9, 150),
        "double_drop": (8.5, 150),
    }
    for style, (from_bar, blocks) in windows.items():
        grew = growth(style, from_bar, blocks)
        assert grew <= baseline + 8_000, f"{style} grew {grew - baseline} bytes"


# --- rendered audio -----------------------------------------------------------------


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    folder = tmp_path_factory.mktemp("hightracks")
    synth.render_track(folder / "a.wav", bpm=126.0, root="A", minor=True, bars=48, seed=1)
    synth.render_track(folder / "b.wav", bpm=128.0, root="E", minor=True, bars=48, seed=2)
    return analysis.analyze_file(folder / "a.wav"), analysis.analyze_file(folder / "b.wav")


@pytest.fixture(scope="module")
def renders(pair, tmp_path_factory):
    a, b = pair
    out = tmp_path_factory.mktemp("highrender")
    return {
        style: render_transition(a, b, out / style, style=style)
        for style in HIGH_ENERGY + ("bass_swap",)
    }


@pytest.fixture(scope="module")
def hot_pair(tmp_path_factory):
    """The same pair, normalised to 0.99: two drops as hot as a master gets."""
    folder = tmp_path_factory.mktemp("hottracks")
    for name, bpm, root, seed in (("a", 126.0, "A", 1), ("b", 128.0, "E", 2)):
        path = folder / f"{name}.wav"
        synth.render_track(path, bpm=bpm, root=root, minor=True, bars=48, seed=seed)
        audio, sr = sf.read(str(path))
        sf.write(str(path), audio / float(np.max(np.abs(audio))) * 0.99, sr,
                 subtype="PCM_16")
    return analysis.analyze_file(folder / "a.wav"), analysis.analyze_file(folder / "b.wav")


LANES = ("a_gain", "a_low", "a_mid", "a_high", "b_gain", "b_low", "b_mid", "b_high")


@pytest.mark.parametrize("style", HIGH_ENERGY)
def test_each_style_renders_with_a_valid_envelope_csv(renders, style):
    r = renders[style]
    audio, sr = sf.read(str(r.wav_path))
    assert sr == SAMPLE_RATE and audio.shape[1] == 2
    assert np.all(np.isfinite(audio))

    with gzip.open(r.envelope_path, "rt", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == ENVELOPE_COLUMNS
    during = [row for row in rows if row["in_transition"] == "1"]
    assert len(during) == pytest.approx(r.transition_frames / r.blocksize, abs=2)
    idx = [int(row["block"]) for row in during]
    assert idx == list(range(idx[0], idx[-1] + 1)), "contiguous"


@pytest.mark.parametrize("style", HIGH_ENERGY)
def test_each_style_automates_continuously(renders, style):
    """No lane steps, and the automation arrives where it is going.

    A step is allowed on the transition's first block only: a double drop
    slams deck A's bass out on the downbeat, which is the move.
    """
    r = renders[style]
    during = [e for e in r.envelope if e["in_transition"] == 1]
    for lane in LANES:
        values = np.array([e[lane] for e in during])
        worst = float(np.max(np.abs(np.diff(values[1:])))) if len(values) > 2 else 0.0
        assert worst <= 0.1, f"{style} {lane} stepped by {worst:.3f}"
    moving = sum(
        1 for lane in ("a_gain", "b_gain")
        if np.ptp([e[lane] for e in during]) > 0.3
    )
    assert moving == 2, f"{style}: both decks' levels must actually move"
    end = during[-1]
    assert end["a_gain"] <= 0.1 and end["b_gain"] >= 0.9


@pytest.mark.parametrize("style", ["loop_roll_out", "beat_repeat_in"])
def test_loop_boundaries_do_not_click_in_the_rendered_wav(renders, style):
    """The largest sample-to-sample step, against a plain bass swap's."""
    def worst_step(r):
        s = r.transition_start_frame
        window = r.mix[s:s + r.transition_frames]
        return float(np.max(np.abs(np.diff(window, axis=0))))

    looped = worst_step(renders[style])
    plain = worst_step(renders["bass_swap"])
    assert looped <= plain * 1.5, f"{style} stepped {looped:.4f} against {plain:.4f}"


@pytest.mark.parametrize("style", HIGH_ENERGY)
def test_no_style_clips(renders, style):
    mix = renders[style].mix
    assert float(np.max(np.abs(mix))) <= 0.99
    assert int(np.sum(np.abs(mix) >= 1.0)) == 0


def test_double_drop_stays_under_0_99_on_hot_tracks(hot_pair, tmp_path, monkeypatch):
    a, b = hot_pair
    r = render_transition(a, b, tmp_path / "dd", style="double_drop")
    peak = float(np.max(np.abs(r.mix)))
    assert peak < 0.99, f"double_drop peaked at {peak:.4f}"
    assert int(np.sum(np.abs(r.mix) >= 1.0)) == 0

    during = [e for e in r.envelope if e["in_transition"] == 1][2:]
    assert all(e["a_gain"] <= tr.DOUBLE_DROP_HEADROOM + 1e-3 for e in during)

    # And the headroom is doing the work, not just the limiter behind it.
    monkeypatch.setattr(tr, "DOUBLE_DROP_HEADROOM", 1.0)
    unguarded = render_transition(a, b, tmp_path / "dd_unity", style="double_drop")
    assert float(np.max(np.abs(unguarded.mix))) > peak


# --- drops: validation and placement ----------------------------------------------


def _drop_cue(track, bar: float, index: int = 3) -> dict:
    return analysis.make_hot_cue(index, phrase.frame_at_bar(track, bar) / SAMPLE_RATE, "drop")


def _without_drops(track) -> None:
    track.hot_cues = [c for c in track.hot_cues if c.get("label") != "drop"]


@pytest.mark.parametrize("which, raw", [
    ("drop_swap", dict(GOOD, align_mode="drop")),
    ("double_drop", dict(GOOD, double_drop_bars=12, align_mode="drop")),
])
def test_a_drop_design_needs_a_drop_cue_on_both_tracks(session, which, raw):
    a, b = session.crate[0], session.crate[1]
    for missing, label in ((b, "incoming"), (a, "outgoing")):
        a.hot_cues = [_drop_cue(a, 16)]
        b.hot_cues = [_drop_cue(b, 10)]
        _without_drops(missing)
        params, why = session.supervisor.validate_transition_params(raw, b, a)
        assert params is None, f"{which} ran without a drop on the {label} track"
        assert which in why and label in why, why

    a.hot_cues = [_drop_cue(a, 16)]
    b.hot_cues = [_drop_cue(b, 10)]
    params, why = session.supervisor.validate_transition_params(raw, b, a)
    assert params is not None, why
    assert params.align_mode == "drop"


def test_a_drop_swap_without_a_drop_cue_is_rejected_and_logged(session):
    for track in session.crate:
        _without_drops(track)
    session.intent_engine = FakeModel(dict(GOOD, align_mode="drop"))
    _arm(session)
    style, rule = session.last_transition_choice
    assert style != "llm" and "rejected" in rule

    events = [e for e in log_events(session) if e["event"] == "transition_design_rejected"]
    assert events, "the rejection must be logged"
    assert "drop_swap" in events[-1]["reason"]
    assert "drop hot cue" in events[-1]["reason"]
    assert events[-1]["params"]["align_mode"] == "drop"


def _probe(track, position: float = 0.0) -> Deck:
    d = Deck("probe")
    d.track = LoadedTrack(analysis=track, audio=np.zeros((2, CHANNELS), np.float32))
    d.position = position
    return d


def _late_bar(track, fraction: float = 0.7) -> int:
    """A whole bar this far through the track: past the supervisor's start guard."""
    return int(track.duration_s / (4 * 60.0 / track.bpm) * fraction)


def test_a_drop_swap_ends_its_window_on_both_drops(session):
    a, b = session.crate[0], session.crate[1]
    drop_a = _late_bar(a)
    a.hot_cues = [_drop_cue(a, drop_a)]
    b.hot_cues = [_drop_cue(b, 10)]
    params = tr.preset_params("drop_swap")
    plan = session._plan_drop_aligned(_probe(a), b, params, params.length_bars)
    assert plan is not None
    assert plan.start_frame == pytest.approx(phrase.frame_at_bar(a, drop_a - 4), abs=1)
    b_drop = b.hot_cues[0]["sample_position"]
    bar_b = 4 * 60.0 / b.bpm * SAMPLE_RATE
    assert plan.entry_frame_b == pytest.approx(b_drop - 4 * bar_b, abs=1)


def test_a_double_drop_starts_its_window_on_both_drops(session):
    a, b = session.crate[0], session.crate[1]
    drop_a = _late_bar(a)
    a.hot_cues = [_drop_cue(a, drop_a)]
    b.hot_cues = [_drop_cue(b, 10)]
    params = tr.preset_params("double_drop")
    plan = session._plan_drop_aligned(_probe(a), b, params, params.length_bars)
    assert plan is not None
    assert plan.start_frame == pytest.approx(phrase.frame_at_bar(a, drop_a), abs=1)
    assert plan.entry_frame_b == pytest.approx(b.hot_cues[0]["sample_position"], abs=1)


def test_a_drop_the_start_guard_would_refuse_is_declined_not_armed(session):
    """The supervisor's early-start guard stays as it is; placement defers to it."""
    from djai.supervisor import MIN_TRANSITION_START_FRACTION

    a, b = session.crate[0], session.crate[1]
    early = max(2, _late_bar(a, MIN_TRANSITION_START_FRACTION * 0.5))
    a.hot_cues = [_drop_cue(a, early)]
    b.hot_cues = [_drop_cue(b, 10)]
    params = tr.preset_params("double_drop")
    assert session._plan_drop_aligned(_probe(a), b, params, params.length_bars) is None


def test_a_drop_already_behind_the_playhead_cannot_be_aligned(session):
    a, b = session.crate[0], session.crate[1]
    a.hot_cues = [_drop_cue(a, 8)]
    b.hot_cues = [_drop_cue(b, 10)]
    late = _probe(a, phrase.frame_at_bar(a, 10))
    params = tr.preset_params("double_drop")
    assert session._plan_drop_aligned(late, b, params, params.length_bars) is None


def test_the_next_drop_ahead_is_used_when_an_earlier_one_has_passed(session):
    """Regression from the Part 4 soak: a passed first drop hid a usable second."""
    a, b = session.crate[0], session.crate[1]
    passed, ahead = 4, _late_bar(a)
    a.hot_cues = [_drop_cue(a, passed, 3), _drop_cue(a, ahead, 5)]
    b.hot_cues = [_drop_cue(b, 10)]
    params = tr.preset_params("double_drop")
    probe = _probe(a, phrase.frame_at_bar(a, passed + 2))
    plan = session._plan_drop_aligned(probe, b, params, params.length_bars)
    assert plan is not None
    assert plan.start_frame == pytest.approx(phrase.frame_at_bar(a, ahead), abs=1)


def test_an_incoming_drop_without_runway_is_not_used(session):
    """The runway rule every other route into a deck obeys applies here too."""
    a, b = session.crate[0], session.crate[1]
    a.hot_cues = [_drop_cue(a, _late_bar(a))]
    bar_b = 4 * 60.0 / b.bpm
    too_late = b.mix_out - (tr.MIN_ENTRY_RUNWAY_BARS - 8) * bar_b
    b.hot_cues = [analysis.make_hot_cue(4, too_late, "drop")]
    params = tr.preset_params("double_drop")
    assert session._plan_drop_aligned(_probe(a), b, params, params.length_bars) is None

    b.hot_cues.append(_drop_cue(b, 10, 3))
    plan = session._plan_drop_aligned(_probe(a), b, params, params.length_bars)
    assert plan is not None
    assert plan.entry_frame_b == pytest.approx(b.hot_cues[-1]["sample_position"], abs=1)


def test_a_window_that_would_outlast_deck_a_is_not_used(session):
    a, b = session.crate[0], session.crate[1]
    a.hot_cues = [analysis.make_hot_cue(3, a.mix_out - 2 * 4 * 60.0 / a.bpm, "drop")]
    b.hot_cues = [_drop_cue(b, 10)]
    params = tr.preset_params("double_drop")    # 12 bars from the drop
    assert session._plan_drop_aligned(_probe(a), b, params, params.length_bars) is None


# --- the operator can reach every style ------------------------------------------


def test_the_web_style_picker_is_built_from_the_server_list():
    """One list of styles, on the server. The page must not keep its own."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")
    for style in tr.STYLES:
        assert f'value="{style}"' not in html, f"{style} is hardcoded in index.html"
    assert "t.styles" in js and "style-pick" in js


def test_the_command_prompt_names_every_style():
    from djai.intent import SYSTEM_PROMPT

    for style in tr.STYLE_CHOICES:
        assert f'"{style}"' in SYSTEM_PROMPT, f"{style} cannot be asked for by name"


@pytest.mark.parametrize("text, style", [
    ("roll it out into the next one", "loop_roll_out"),
    ("do a loop roll", "loop_roll_out"),
    ("stutter it and slam", "beat_repeat_in"),
    ("backspin out of this one", "backspin"),
    ("double drop them", "double_drop"),
    ("swap on the drop", "drop_swap"),
    ("slam straight into the next one", "cut"),
    ("trail off into a delay", "echo_out"),
])
def test_style_words_reach_every_style_without_the_model(text, style):
    """The keyword fallback, used when the model is offline or unusable."""
    import re

    from djai.intent import _KEYWORD_RULES, _style_from_text

    assert _style_from_text(text) == style
    pattern, action = _KEYWORD_RULES[0]
    assert action == "set_transition_style"
    assert re.search(pattern, text), f"{text!r} would not be read as a style request"


# --- choosing a track whose drop can be lined up -----------------------------------


def _ranked(session):
    from djai.selector import rank_candidates

    current = session.analysis_for(session.live_deck)
    return [c.track for c in rank_candidates(session.crate, current, session.played, 0.0)]


def _setup_drops(session):
    """Live track with a drop ahead; the top-ranked candidate's drop has no
    runway, the runner-up's is usable."""
    live = session.engine.deck(session.live_deck).track.analysis
    first, second = _ranked(session)[:2]
    live.hot_cues = [_drop_cue(live, 80)]
    first.hot_cues = [_drop_cue(first, _late_bar(first, 0.85))]
    second.hot_cues = [_drop_cue(second, 20)]
    return first, second


def test_a_drop_style_cues_the_best_track_whose_drop_can_be_lined_up(session):
    first, second = _setup_drops(session)
    session.transition_style = "double_drop"
    assert session.cue_next(origin="test")
    assert session._cued.analysis.track_id == second.track_id

    cued = [e for e in log_events(session) if e["event"] == "track_cued"]
    assert "passed over" in cued[-1]["action"], "the log must say why"


def test_other_styles_keep_the_selectors_first_choice(session):
    first, _ = _setup_drops(session)
    session.transition_style = "bass_swap"
    assert session.cue_next(origin="test")
    assert session._cued.analysis.track_id == first.track_id


def test_a_drop_style_with_no_usable_drop_still_cues_a_track(session):
    first, second = _setup_drops(session)
    second.hot_cues = []
    session.transition_style = "drop_swap"
    assert session.cue_next(origin="test")
    assert session._cued.analysis.track_id == first.track_id

    cued = [e for e in log_events(session) if e["event"] == "track_cued"]
    assert "no track's drop" in cued[-1]["action"]


# --- the envelope CSV shows the loop and the rate -----------------------------------


def _csv_rows(result):
    """In-transition rows exactly as written to disk, not the in-memory copy."""
    with gzip.open(result.envelope_path, "rt", newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh) if row["in_transition"] == "1"]


def test_the_envelope_csv_carries_deck_a_motion_columns_at_the_end(renders):
    start = ENVELOPE_COLUMNS.index("a_rate")
    assert ENVELOPE_COLUMNS[start:start + 3] == ["a_rate", "a_loop_beats", "a_loop_start_bar"]
    # Only ever appended to: the filter columns (Phase 2) come after these.
    assert ENVELOPE_COLUMNS[start + 3:] == ["a_filter", "b_filter", "reverb_send", "riser_gain"]
    rows = _csv_rows(renders["bass_swap"])
    assert rows
    assert all(float(row["a_rate"]) == 1.0 for row in rows)
    assert all(float(row["a_loop_beats"]) == 0.0 for row in rows)
    assert all(row["a_loop_start_bar"] == "" for row in rows), "no loop reads as blank"


def test_the_csv_shows_the_roll_halving_from_a_fixed_loop_in_point(renders):
    rows = _csv_rows(renders["loop_roll_out"])
    lengths = []
    for row in rows:
        beats = float(row["a_loop_beats"])
        if beats > 0 and (not lengths or lengths[-1] != beats):
            lengths.append(beats)
    assert lengths == [16.0, 8.0, 4.0, 2.0]
    starts = {row["a_loop_start_bar"] for row in rows if float(row["a_loop_beats"]) > 0}
    assert len(starts) == 1, f"the loop-in point moved: {sorted(starts)}"


def test_the_csv_shows_the_stutter_on_a_beat_line(renders):
    rows = _csv_rows(renders["beat_repeat_in"])
    looping = [row for row in rows if float(row["a_loop_beats"]) > 0]
    assert looping, "the stutter never appeared in the CSV"
    assert {float(row["a_loop_beats"]) for row in looping} == {0.25}
    for row in looping:
        beat = float(row["a_loop_start_bar"]) * 4
        assert beat == pytest.approx(round(beat), abs=1e-3)


def test_the_csv_shows_the_record_going_backwards(renders):
    rates = [float(row["a_rate"]) for row in _csv_rows(renders["backspin"])]
    assert rates[0] == 1.0
    assert min(rates) < 0.0, "the backspin never reversed deck A"


@pytest.mark.parametrize("style", ["drop_swap", "double_drop"])
def test_the_drop_styles_leave_deck_a_straight_in_the_csv(renders, style):
    rows = _csv_rows(renders[style])
    assert all(float(row["a_rate"]) == 1.0 for row in rows)
    assert all(float(row["a_loop_beats"]) == 0.0 for row in rows)
