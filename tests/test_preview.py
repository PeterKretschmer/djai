"""The transition preview loop: render, measure, revise, commit.

THREADING CONTEXT: main thread (pytest), plus one real Scheduler worker thread
in the pre-roll tests. No audio device is opened anywhere here: the renderer
drives ``engine.callback`` directly, which is the same code the audio thread
runs.

The guarantees under test are not "the preview improves the mix" -- that is the
loop's job and the measurements say whether it did. They are the ones that must
hold when it does not work: the transition fires on time, with usable
parameters, whatever the renderer, the measurements or the model do.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from djai import analysis, config, phrase, preview, render as render_mod, revise
from djai import transition as tr
from djai.commands import StartTransition
from djai.deck import SAMPLE_RATE, load_track
from djai.engine import Engine
from djai.scheduler import Scheduler
from tests import synth


# --- a pair of synthetic tracks, analysed once ---------------------------------


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """Two beat-gridded tracks, with bass, so the band measurements have work."""
    folder = tmp_path_factory.mktemp("preview")
    # Same tempo on purpose: phase coherence is a control here, not a variable.
    synth.render_track(folder / "a.wav", bpm=128.0, root="A", minor=True, bars=56, seed=1)
    synth.render_track(folder / "b.wav", bpm=128.0, root="E", minor=True, bars=56, seed=2)
    a = analysis.analyze_file(folder / "a.wav")
    b = analysis.analyze_file(folder / "b.wav")
    return a, b


@pytest.fixture(scope="module")
def long_b(tmp_path_factory):
    """An incoming track long enough to enter it 20 bars in.

    transition.MIN_ENTRY_RUNWAY_BARS wants 48 bars of track after any entry, so
    a later-cue revision on the 56-bar pair could never be legal.
    """
    folder = tmp_path_factory.mktemp("preview_long")
    synth.render_track(folder / "b.wav", bpm=128.0, root="E", minor=True, bars=80, seed=3)
    b = analysis.analyze_file(folder / "b.wav")
    return b, load_track(b)


@pytest.fixture(scope="module")
def loaded(pair):
    a, b = pair
    return load_track(a), load_track(b)


@pytest.fixture(scope="module")
def state(pair, loaded):
    a, _b = pair
    la, _lb = loaded
    return preview.DeckAState(
        analysis=a, loaded=la, transition_at_frame=phrase.frame_at_bar(a, 8.0)
    )


def entry(track_b):
    return track_b.first_downbeat * SAMPLE_RATE


SANE = tr.TransitionParams(
    length_bars=8, curve="equal_power", low_swap_bar=2, low_swap_bars=1,
    deck_b_low_delay_bars=0, intensity=0.6, name="sane",
)

#: The brief's deliberately bad set: a long linear crossfade with no bass swap
#: at all, so both low bands run together for the whole blend.
BAD = tr.TransitionParams(
    length_bars=32, curve="linear", low_swap_bar=None, low_swap_bars=8,
    deck_b_low_delay_bars=0, intensity=1.0, name="bad",
)


# --- loudness maths, checked against a second implementation -------------------


def test_integrated_loudness_agrees_with_pyloudnorm():
    """The BS.1770 meter here is built from the spec, not borrowed.

    pyloudnorm is already a dependency and implements the same standard, so it
    is the independent check that the filter coefficients and the two gates are
    right. Anything under ~0.1 LU is arithmetic ordering.
    """
    pyloudnorm = pytest.importorskip("pyloudnorm")
    rng = np.random.default_rng(7)
    seconds = 8
    noise = rng.normal(0.0, 0.1, size=(seconds * SAMPLE_RATE, 2))
    # Something with structure as well as noise, so the gates have to work.
    t = np.arange(seconds * SAMPLE_RATE) / SAMPLE_RATE
    tone = 0.2 * np.sin(2 * np.pi * 220.0 * t)[:, None]
    signal = (noise + tone).astype(np.float64)
    signal[: SAMPLE_RATE] *= 0.0001          # a quiet stretch for the gate

    mine = preview.integrated_lufs(signal, SAMPLE_RATE)
    theirs = pyloudnorm.Meter(SAMPLE_RATE).integrated_loudness(signal)
    assert mine == pytest.approx(theirs, abs=0.1), f"{mine} vs {theirs}"


def test_peak_dbfs_is_full_scale_at_one():
    assert preview.peak_dbfs(np.array([[1.0, -1.0]])) == pytest.approx(0.0)
    assert preview.peak_dbfs(np.array([[0.5, 0.0]])) == pytest.approx(-6.02, abs=0.01)
    assert preview.peak_dbfs(np.zeros((10, 2))) == -120.0


def test_the_dip_is_measured_moment_by_moment():
    """A source that goes quiet is not a crossfade fault.

    Both curves drop together, so the dip is zero -- the earlier version of this
    measurement compared the mix's quietest moment against the sources' average
    over the whole window and scored every breakdown as a broken crossfade.
    """
    steady = np.full(40, 0.01)
    quiet_middle = steady.copy()
    quiet_middle[15:25] = 0.0001
    assert preview._loudness_dip(quiet_middle, [quiet_middle, quiet_middle]) == 0.0
    # The mix alone dropping IS a fault, and is measured in dB.
    dipped = steady.copy()
    dipped[20] = steady[20] / 10.0
    assert preview._loudness_dip(dipped, [steady, steady]) == pytest.approx(10.0, abs=0.1)


# --- the measurements on a real render -----------------------------------------


@pytest.fixture(scope="module")
def sane_measurements(state, pair, loaded):
    _a, b = pair
    _la, lb = loaded
    return preview.preview_transition(state, b, SANE, lb, entry(b))


def test_a_preview_returns_every_measurement_the_brief_asks_for(sane_measurements):
    for key in (
        "peak_dbfs", "integrated_lufs", "loudness_dip_db", "spectral_clash",
        "low_end_overlap_bars", "vocal_overlap_bars", "transient_density_ratio",
        "phase_coherence",
    ):
        assert key in sane_measurements, key
        value = sane_measurements[key]
        assert value is None or isinstance(value, (int, float)), key


def test_measurements_are_in_their_natural_ranges(sane_measurements):
    m = sane_measurements
    assert -120.0 <= m["peak_dbfs"] <= 0.0
    assert m["loudness_dip_db"] >= 0.0
    assert 0.0 <= m["spectral_clash"] <= 1.0
    assert 0.0 <= m["phase_coherence"] <= 1.0
    assert m["low_end_overlap_bars"] >= 0.0
    assert m["transient_density_ratio"] > 0.0


def test_two_beatmatched_decks_are_phase_coherent(sane_measurements):
    """Both tracks are on the same grid at the same tempo, so this is the
    control: if a clean render does not score near 1, the measurement is
    broken rather than the transition."""
    assert sane_measurements["phase_coherence"] > 0.9


def test_the_preview_renders_only_its_own_window(state, pair, loaded):
    """Not the whole track: the window is the transition plus the context bars."""
    a, b = pair
    la, lb = loaded
    result = render_mod.render_preview(
        a, b, la, lb, SANE,
        transition_at_frame_a=state.transition_at_frame,
        entry_frame_b=entry(b), context_bars=4.0,
    )
    bar_s = 4 * 60.0 / a.bpm
    expected = (SANE.length_bars + 8) * bar_s
    assert result.duration_s == pytest.approx(expected, rel=0.02)
    assert result.duration_s < a.duration_s / 2, "a preview is not a whole track"


def test_the_preview_reuses_the_real_engine(state, pair, loaded):
    """The same guard render_transition has: no private mixer, no private
    envelope. A preview that measured its own reimplementation would measure
    nothing about what is going to play."""
    import inspect

    src = inspect.getsource(render_mod.render_preview)
    assert "engine.callback(" in src
    assert "Scheduler(" in src
    assert "arm_transition_params" in src, "designed parameters, not a named preset"
    assert "gains_at" not in src


# --- the rules ------------------------------------------------------------------


def test_a_bad_parameter_set_is_measurably_improved(state, pair, loaded):
    """The brief's acceptance case, measured before and after.

    32 bars, linear, no bass swap: the two low bands run together for the whole
    blend and the level sags through the middle of it.
    """
    _a, b = pair
    _la, lb = loaded
    before = preview.preview_transition(state, b, BAD, lb, entry(b))
    revised, revisions, force_cut = revise.apply_rules(before, BAD)
    assert not force_cut
    assert revisions, "the rules had nothing to say about a deliberately bad set"
    after = preview.preview_transition(state, b, revised, lb, entry(b))

    print(
        f"\nbefore: dip={before['loudness_dip_db']} dB, "
        f"low_overlap={before['low_end_overlap_bars']} bars, "
        f"clash={before['spectral_clash']}\n"
        f"after:  dip={after['loudness_dip_db']} dB, "
        f"low_overlap={after['low_end_overlap_bars']} bars, "
        f"clash={after['spectral_clash']}\n"
        f"changed: {[(r.field, r.before, r.after) for r in revisions]}"
    )
    assert revised.curve == "equal_power", "a linear crossfade is what sags"
    assert revised.low_swap_bar is not None, "a blend with no bass swap got one"
    assert after["low_end_overlap_bars"] <= before["low_end_overlap_bars"]
    assert after["loudness_dip_db"] <= before["loudness_dip_db"] + 0.25


def test_a_good_parameter_set_is_left_alone(sane_measurements):
    _revised, revisions, force_cut = revise.apply_rules(sane_measurements, SANE)
    assert not force_cut
    assert revisions == [], f"a sane transition was revised anyway: {revisions}"


def test_each_rule_fires_on_its_own_measurement():
    """One measurement over its threshold, one correction, in isolation."""
    base = tr.TransitionParams(
        length_bars=16, curve="linear", low_swap_bar=8, low_swap_bars=4,
        deck_a_high_rolloff=False, deck_b_low_delay_bars=0, filter_sweep="none",
        intensity=1.0, vocal_aware=True, name="base",
    )
    clean = {
        "loudness_dip_db": 0.0, "low_end_overlap_bars": 0.0,
        "vocal_overlap_bars": 0.0, "spectral_clash": 0.0,
        "peak_dbfs": -6.0, "phase_coherence": 1.0,
    }

    def fire(**over):
        return revise.apply_rules({**clean, **over}, base)

    out, revs, _ = fire(loudness_dip_db=5.0)
    assert out.curve == "equal_power" and revs[0].rule == "loudness_dip"

    out, revs, _ = fire(low_end_overlap_bars=6.0)
    assert out.low_swap_bar == 4, "the bass hands over earlier"
    assert revs[0].rule == "low_overlap"

    out, revs, _ = fire(vocal_overlap_bars=3.0)
    assert out.deck_b_low_delay_bars == 2 and revs[0].rule == "vocal_overlap"

    out, revs, _ = fire(spectral_clash=0.95)
    assert out.deck_a_high_rolloff is True and revs[0].rule == "spectral_clash"

    out, revs, _ = fire(peak_dbfs=-0.2)
    assert out.intensity == pytest.approx(0.8) and revs[0].rule == "peak"


def test_a_vocal_clash_moves_the_entry_to_a_real_later_cue(long_b):
    b, _lb = long_b
    bar_s = 4 * 60.0 / b.bpm
    track_b = dataclasses.replace(
        b,
        hot_cues=[analysis.make_hot_cue(3, 20 * bar_s, "later")],
        mix_in=0.0,
        mix_out=b.duration_s,
    )
    params = tr.TransitionParams(vocal_aware=True, name="vocal")
    out, revs, _cut = revise.apply_rules(
        {"vocal_overlap_bars": 4.0, "phase_coherence": 1.0}, params, track_b
    )
    assert out.entry_point == "hot_cue_3"
    assert revs[0].rule == "vocal_overlap"


def test_a_vocal_clash_with_no_later_cue_falls_back_to_the_delay(pair):
    _a, b = pair
    track_b = dataclasses.replace(b, hot_cues=[])
    out, _revs, _cut = revise.apply_rules(
        {"vocal_overlap_bars": 4.0, "phase_coherence": 1.0},
        tr.TransitionParams(vocal_aware=True, name="vocal"), track_b,
    )
    assert out.entry_point == "mix_in"
    assert out.deck_b_low_delay_bars == 2


def test_a_vocal_clash_with_no_room_for_a_later_cue_uses_the_delay(pair):
    """The runway rule wins: a cue too near the end is not an entry."""
    _a, b = pair
    bar_s = 4 * 60.0 / b.bpm
    short = dataclasses.replace(
        b, hot_cues=[analysis.make_hot_cue(3, 20 * bar_s, "too late")],
        mix_in=0.0, mix_out=b.duration_s,
    )
    out, _revs, _cut = revise.apply_rules(
        {"vocal_overlap_bars": 4.0, "phase_coherence": 1.0},
        tr.TransitionParams(vocal_aware=True, name="vocal"), short,
    )
    assert out.entry_point == "mix_in"
    assert out.deck_b_low_delay_bars == 2


def test_a_vocal_clashing_pair_is_revised_to_zero_overlap(pair, loaded, long_b):
    """The brief's acceptance case, measured on a render both times.

    Deck A sings across the whole transition. Deck B sings for its first 16
    bars and has a hot cue at bar 20, after its vocal. Entering at the start
    clashes; the revision moves the entry to the cue and the clash is gone.
    """
    a, _b = pair
    la, _lb = loaded
    b, lb = long_b
    bar_s = 4 * 60.0 / b.bpm
    singing_a = dataclasses.replace(a, vocal_bars=[[0, 56]])
    singing_b = dataclasses.replace(
        b,
        vocal_bars=[[0, 16]],
        hot_cues=[analysis.make_hot_cue(2, b.first_downbeat + 20 * bar_s, "after vocal")],
        mix_in=b.first_downbeat,
        mix_out=b.duration_s,
    )
    state_v = preview.DeckAState(
        analysis=singing_a, loaded=la,
        transition_at_frame=phrase.frame_at_bar(singing_a, 8.0),
    )
    params = dataclasses.replace(SANE, vocal_aware=True)

    before = preview.preview_transition(state_v, singing_b, params, lb, entry(singing_b))
    assert before["vocal_overlap_bars"] > 0, "the fixture does not clash"

    revised, revisions, _cut = revise.apply_rules(before, params, singing_b)
    after = preview.preview_transition(state_v, singing_b, revised, lb, entry(singing_b))
    print(f"\nvocal_overlap_bars: {before['vocal_overlap_bars']} -> "
          f"{after['vocal_overlap_bars']}  "
          f"({[(r.field, r.before, r.after) for r in revisions]})")
    assert revised.entry_point == "hot_cue_2"
    assert after["vocal_overlap_bars"] == 0.0


def test_vocal_overlap_is_ignored_when_the_design_said_not_to_care():
    params = tr.TransitionParams(vocal_aware=False, name="dont-care")
    _out, revs, _cut = revise.apply_rules(
        {"vocal_overlap_bars": 8.0, "phase_coherence": 1.0}, params
    )
    assert revs == []


def test_a_grid_that_does_not_agree_forces_a_cut():
    params = tr.TransitionParams(name="x")
    out, revs, force_cut = revise.apply_rules(
        {"phase_coherence": config.PREVIEW_MIN_PHASE_COHERENCE - 0.01}, params
    )
    assert force_cut is True
    assert revs == [], "a grid fault is not fixed by changing the transition"
    assert out is params


def test_the_rules_never_mutate_what_they_are_given():
    params = tr.TransitionParams(curve="linear", intensity=1.0, name="orig")
    before = params.to_schema()
    revise.apply_rules(
        {"loudness_dip_db": 9.0, "peak_dbfs": 0.5, "phase_coherence": 1.0}, params
    )
    assert params.to_schema() == before


def test_the_rules_stay_inside_the_schema():
    """Whatever the measurements say, the result is still valid to the
    supervisor. A revision that produced an out-of-range value would be
    rejected downstream and silently become a rule-based preset."""
    params = tr.TransitionParams(
        length_bars=4, curve="equal_power", low_swap_bar=1, low_swap_bars=1,
        deck_b_low_delay_bars=16, intensity=0.0, deck_a_high_rolloff=True,
        filter_sweep="hp_out", name="edge",
    )
    for _ in range(10):
        params, revs, _cut = revise.apply_rules({
            "loudness_dip_db": 99.0, "low_end_overlap_bars": 99.0,
            "vocal_overlap_bars": 99.0, "spectral_clash": 1.0,
            "peak_dbfs": 0.0, "phase_coherence": 1.0,
        }, params)
        if not revs:
            break
    lo, hi = tr.LENGTH_BARS_RANGE
    assert lo <= params.length_bars <= hi
    assert tr.LOW_SWAP_BARS_RANGE[0] <= params.low_swap_bars <= tr.LOW_SWAP_BARS_RANGE[1]
    assert (tr.DECK_B_LOW_DELAY_RANGE[0] <= params.deck_b_low_delay_bars
            <= tr.DECK_B_LOW_DELAY_RANGE[1])
    assert tr.INTENSITY_RANGE[0] <= params.intensity <= tr.INTENSITY_RANGE[1]
    assert params.curve in tr.CURVES
    assert params.filter_sweep in tr.FILTER_SWEEPS


# --- the loop's guarantees ------------------------------------------------------


def test_a_worse_revision_is_reverted(state, pair, loaded, monkeypatch):
    """Measure, revise, re-measure worse, put it back.

    Driven with a stubbed preview so the regression is certain rather than
    hoped for; the loop's arithmetic is what is under test, not the DSP.
    """
    _a, b = pair
    _la, lb = loaded
    rounds = iter([
        {"loudness_dip_db": 5.0, "low_end_overlap_bars": 0.0,
         "vocal_overlap_bars": 0.0, "spectral_clash": 0.0,
         "peak_dbfs": -6.0, "phase_coherence": 1.0},
        # The revision made the dip worse.
        {"loudness_dip_db": 9.0, "low_end_overlap_bars": 0.0,
         "vocal_overlap_bars": 0.0, "spectral_clash": 0.0,
         "peak_dbfs": -6.0, "phase_coherence": 1.0},
    ])
    monkeypatch.setattr(
        preview, "preview_transition", lambda *a, **k: next(rounds)
    )
    original = tr.TransitionParams(curve="linear", name="orig")
    outcome = revise.run_preview(
        state, b, lb, entry(b), original, budget_ms=10_000
    )
    assert outcome.status == "reverted"
    assert outcome.params is original, "the worse revision was kept"
    assert outcome.params.curve == "linear"
    assert any("worse" in e for e in outcome.events)


def test_a_better_revision_is_kept(state, pair, loaded, monkeypatch):
    _a, b = pair
    _la, lb = loaded
    rounds = iter([
        {"loudness_dip_db": 5.0, "low_end_overlap_bars": 4.0,
         "vocal_overlap_bars": 0.0, "spectral_clash": 0.0,
         "peak_dbfs": -6.0, "phase_coherence": 1.0},
        {"loudness_dip_db": 1.0, "low_end_overlap_bars": 0.0,
         "vocal_overlap_bars": 0.0, "spectral_clash": 0.0,
         "peak_dbfs": -6.0, "phase_coherence": 1.0},
        {"loudness_dip_db": 1.0, "low_end_overlap_bars": 0.0,
         "vocal_overlap_bars": 0.0, "spectral_clash": 0.0,
         "peak_dbfs": -6.0, "phase_coherence": 1.0},
    ])
    monkeypatch.setattr(preview, "preview_transition", lambda *a, **k: next(rounds))
    outcome = revise.run_preview(
        state, b, lb, entry(b),
        tr.TransitionParams(curve="linear", low_swap_bar=None, name="orig"),
        budget_ms=10_000,
    )
    assert outcome.status == "revised"
    assert outcome.params.curve == "equal_power"
    assert outcome.revisions


def test_a_render_failure_commits_the_original(state, pair, loaded, monkeypatch):
    _a, b = pair
    _la, lb = loaded

    def boom(*a, **k):
        raise preview.PreviewError("render", "deliberate")

    monkeypatch.setattr(preview, "preview_transition", boom)
    original = tr.TransitionParams(curve="linear", name="orig")
    outcome = revise.run_preview(state, b, lb, entry(b), original, budget_ms=10_000)
    assert outcome.status == "failed"
    assert outcome.params is original
    assert outcome.force_cut is False


def test_an_unexpected_exception_also_commits_the_original(
    state, pair, loaded, monkeypatch
):
    """Not just PreviewError: anything at all. A preview must never be the
    reason a transition does not happen."""
    _a, b = pair
    _la, lb = loaded

    def boom(*a, **k):
        raise ZeroDivisionError("something nobody predicted")

    monkeypatch.setattr(preview, "preview_transition", boom)
    original = tr.TransitionParams(name="orig")
    outcome = revise.run_preview(state, b, lb, entry(b), original, budget_ms=10_000)
    assert outcome.status == "failed"
    assert outcome.params is original


def test_a_tiny_budget_commits_immediately_and_says_so(state, pair, loaded):
    """100 ms is less than one render, so the loop must not start one."""
    _a, b = pair
    _la, lb = loaded
    original = tr.TransitionParams(name="orig")
    started = time.perf_counter()
    outcome = revise.run_preview(
        state, b, lb, entry(b), original, budget_ms=100.0,
        clock=_clock_that_jumps(0.2),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert outcome.status == "budget"
    assert outcome.params is original
    assert any("budget" in e for e in outcome.events)
    assert elapsed_ms < 100, "a blown budget must not cost real time"


def test_a_budget_too_small_for_one_render_never_starts_one(
    state, pair, loaded, monkeypatch
):
    """With a real clock, not a jumping one: 100 ms is less than any render.

    Measured before this check existed: the loop started its first render
    anyway and came back ~1.4 s later, fourteen times over budget.
    """
    _a, b = pair
    _la, lb = loaded
    rendered = {"n": 0}

    def counting(*a, **k):
        rendered["n"] += 1
        return _bad_measurements()

    monkeypatch.setattr(preview, "preview_transition", counting)
    original = tr.TransitionParams(name="orig")
    started = time.perf_counter()
    outcome = revise.run_preview(state, b, lb, entry(b), original, budget_ms=100.0)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert rendered["n"] == 0
    assert outcome.status == "budget" and outcome.params is original
    assert any("exceeded" in e for e in outcome.events)
    assert elapsed_ms < 100


def _clock_that_jumps(seconds: float):
    """A clock that is already past the budget on its second reading."""
    readings = iter([0.0, seconds, seconds, seconds, seconds, seconds])

    def clock():
        try:
            return next(readings)
        except StopIteration:
            return seconds
    return clock


def test_a_budget_committed_outcome_still_summarises(state, pair, loaded, monkeypatch):
    """The unverified round has no measurements. The summary must not care."""
    _a, b = pair
    _la, lb = loaded
    monkeypatch.setattr(preview, "preview_transition",
                        lambda *a, **k: _bad_measurements())
    readings = iter([0.0, 0.0, 0.0] + [3.9] * 50)
    outcome = revise.run_preview(
        state, b, lb, entry(b), tr.TransitionParams(curve="linear", name="o"),
        budget_ms=4000.0, clock=lambda: next(readings),
    )
    assert outcome.status == "budget"
    assert outcome.rounds[-1]["measurements"] is None
    summary = outcome.summary()
    assert summary["final"]["loudness_dip_db"] == 5.0
    assert summary["revisions"]


def test_the_round_cap_is_hard(state, pair, loaded, monkeypatch):
    """Two rounds, never three, whatever is asked for and however bad it is."""
    _a, b = pair
    _la, lb = loaded
    calls = {"n": 0}

    def always_bad(*a, **k):
        calls["n"] += 1
        return {"loudness_dip_db": 9.0, "low_end_overlap_bars": 0.0,
                "vocal_overlap_bars": 0.0, "spectral_clash": 0.0,
                "peak_dbfs": -6.0, "phase_coherence": 1.0}

    monkeypatch.setattr(preview, "preview_transition", always_bad)
    outcome = revise.run_preview(
        state, b, lb, entry(b), tr.TransitionParams(curve="linear", name="o"),
        budget_ms=10_000, max_rounds=99,
    )
    assert len(outcome.rounds) <= config.PREVIEW_MAX_ROUNDS + 1
    assert calls["n"] <= config.PREVIEW_MAX_ROUNDS + 1


def test_preview_disabled_skips_without_touching_anything(
    state, pair, loaded, monkeypatch
):
    _a, b = pair
    _la, lb = loaded
    monkeypatch.setattr(config, "PREVIEW_ENABLED", False)

    def boom(*a, **k):
        raise AssertionError("preview ran while disabled")

    monkeypatch.setattr(preview, "preview_transition", boom)
    original = tr.TransitionParams(name="orig")
    outcome = revise.run_preview(state, b, lb, entry(b), original)
    assert outcome.status == "skipped"
    assert outcome.params is original


# --- the model pass -------------------------------------------------------------


class _Ollama:
    """An intent engine stand-in. `raw` is what the model 'returns'."""

    def __init__(self, raw, reason="", explode=False):
        self.raw, self.reason, self.explode = raw, reason, explode
        self.calls = 0
        self.available = True

    def revise_transition(self, context, timeout_s=None):
        self.calls += 1
        self.context = context
        if self.explode:
            raise RuntimeError("ollama fell over")
        return self.raw, self.reason


class _Supervisor:
    def __init__(self, result, reason=""):
        self.result, self.reason = result, reason

    def validate_transition_params(self, raw, track_b=None, track_a=None):
        return self.result, self.reason


def _bad_measurements():
    return {"loudness_dip_db": 5.0, "low_end_overlap_bars": 0.0,
            "vocal_overlap_bars": 0.0, "spectral_clash": 0.0,
            "peak_dbfs": -6.0, "phase_coherence": 1.0}


def test_the_model_never_sees_audio(state, pair, loaded, monkeypatch):
    """The whole mechanism: measurements as text, never samples.

    llama3.1:8b cannot process audio. This asserts that nothing array-shaped or
    array-sized reaches the call at all.
    """
    _a, b = pair
    _la, lb = loaded
    monkeypatch.setattr(preview, "preview_transition",
                        lambda *a, **k: _bad_measurements())
    engine = _Ollama(None, "unavailable")
    revise.run_preview(
        state, b, lb, entry(b), tr.TransitionParams(curve="linear", name="o"),
        intent_engine=engine, supervisor=_Supervisor(None, "n/a"),
        budget_ms=10_000,
    )
    assert engine.calls == 1
    sent = engine.context

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            assert len(node) < 64, "a long sequence in the model's context"
            for v in node:
                walk(v)
        else:
            assert not isinstance(node, (np.ndarray, bytes, bytearray)), node

    walk(sent)
    assert set(sent) == {"params", "measurements", "rules_applied"}


def test_a_model_revision_that_fails_validation_is_dropped(
    state, pair, loaded, monkeypatch
):
    _a, b = pair
    _la, lb = loaded
    monkeypatch.setattr(preview, "preview_transition",
                        lambda *a, **k: _bad_measurements())
    outcome = revise.run_preview(
        state, b, lb, entry(b), tr.TransitionParams(curve="linear", name="o"),
        intent_engine=_Ollama({"length_bars": 999}),
        supervisor=_Supervisor(None, "length_bars 999 is outside 4.0..32.0"),
        budget_ms=10_000,
    )
    # The rules' answer stands.
    assert outcome.params.curve == "equal_power"
    assert any("rejected by the supervisor" in e for e in outcome.events)


def test_a_model_that_falls_over_leaves_the_rules_in_charge(
    state, pair, loaded, monkeypatch
):
    """Ollama stopped mid-session: rule-based revision still runs."""
    _a, b = pair
    _la, lb = loaded
    monkeypatch.setattr(preview, "preview_transition",
                        lambda *a, **k: _bad_measurements())
    outcome = revise.run_preview(
        state, b, lb, entry(b), tr.TransitionParams(curve="linear", name="o"),
        intent_engine=_Ollama(None, explode=True),
        supervisor=_Supervisor(None), budget_ms=10_000,
    )
    assert outcome.params.curve == "equal_power", "the rules still revised it"
    assert any("raised" in e for e in outcome.events)


def test_a_hanging_model_is_abandoned_inside_the_budget(
    state, pair, loaded, monkeypatch
):
    """httpx's timeout is per phase, so it is not a deadline. This is."""
    _a, b = pair
    _la, lb = loaded
    monkeypatch.setattr(preview, "preview_transition",
                        lambda *a, **k: _bad_measurements())

    class Hangs(_Ollama):
        def revise_transition(self, context, timeout_s=None):
            time.sleep(5.0)
            return {"curve": "linear"}, ""

    started = time.perf_counter()
    outcome = revise.run_preview(
        state, b, lb, entry(b),
        tr.TransitionParams(curve="linear", length_bars=4, name="o"),
        intent_engine=Hangs(None), supervisor=_Supervisor(None),
        budget_ms=3000.0,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    limit_ms = config.PREVIEW_LLM_TIMEOUT_S * 1000.0
    assert elapsed_ms < limit_ms + 300, f"the loop waited {elapsed_ms:.0f} ms on the model"
    assert outcome.params.curve == "equal_power", "the rules' answer stands"
    assert any("abandoned" in e for e in outcome.events)


def test_the_model_is_not_asked_when_nothing_is_wrong(
    state, pair, loaded, monkeypatch
):
    _a, b = pair
    _la, lb = loaded
    fine = {**_bad_measurements(), "loudness_dip_db": 0.5}
    monkeypatch.setattr(preview, "preview_transition", lambda *a, **k: fine)
    engine = _Ollama(None)
    outcome = revise.run_preview(
        state, b, lb, entry(b), tr.TransitionParams(name="o"),
        intent_engine=engine, supervisor=_Supervisor(None), budget_ms=10_000,
    )
    assert engine.calls == 0
    assert outcome.status == "committed"


def test_no_model_at_all_still_revises(state, pair, loaded, monkeypatch):
    _a, b = pair
    _la, lb = loaded
    monkeypatch.setattr(preview, "preview_transition",
                        lambda *a, **k: _bad_measurements())
    outcome = revise.run_preview(
        state, b, lb, entry(b), tr.TransitionParams(curve="linear", name="o"),
        intent_engine=None, supervisor=None, budget_ms=10_000,
    )
    assert outcome.params.curve == "equal_power"


def test_an_accepted_model_revision_is_recorded_as_the_models(
    state, pair, loaded, monkeypatch
):
    _a, b = pair
    _la, lb = loaded
    monkeypatch.setattr(preview, "preview_transition",
                        lambda *a, **k: _bad_measurements())
    validated = tr.TransitionParams(
        curve="equal_power", length_bars=12, intensity=0.4, name="llm"
    )
    outcome = revise.run_preview(
        state, b, lb, entry(b), tr.TransitionParams(curve="linear", name="o"),
        intent_engine=_Ollama({"anything": True}),
        supervisor=_Supervisor(validated), budget_ms=10_000,
    )
    rules = [r for r in outcome.revisions if r.rule != "llm"]
    from_model = [r for r in outcome.revisions if r.rule == "llm"]
    assert rules and from_model, "the log must say which pass made which change"
    assert any("accepted" in e for e in outcome.events)


# --- logging --------------------------------------------------------------------


class _Log:
    def __init__(self):
        self.records = []

    def write(self, event, **fields):
        self.records.append((event, fields))
        return fields


def test_every_preview_logs_what_it_did(state, pair, loaded, monkeypatch):
    _a, b = pair
    _la, lb = loaded
    monkeypatch.setattr(preview, "preview_transition",
                        lambda *a, **k: _bad_measurements())
    session_log = _Log()
    revise.run_preview(
        state, b, lb, entry(b), tr.TransitionParams(curve="linear", name="o"),
        session_log=session_log, budget_ms=10_000,
    )
    assert len(session_log.records) == 1
    event, fields = session_log.records[0]
    assert event == "preview"
    for key in ("status", "elapsed_ms", "budget_ms", "original", "committed",
                "revisions", "rounds", "events"):
        assert key in fields, key
    assert fields["original"]["curve"] == "linear"
    assert fields["committed"]["curve"] == "equal_power"
    assert fields["rounds"][0]["measurements"]["loudness_dip_db"] == 5.0
    assert fields["revisions"][0]["rule"] == "loudness_dip"


def test_a_broken_logger_does_not_break_a_transition(
    state, pair, loaded, monkeypatch
):
    _a, b = pair
    _la, lb = loaded
    monkeypatch.setattr(preview, "preview_transition",
                        lambda *a, **k: _bad_measurements())

    class Broken:
        def write(self, *a, **k):
            raise OSError("disk full")

    outcome = revise.run_preview(
        state, b, lb, entry(b), tr.TransitionParams(name="o"),
        session_log=Broken(), budget_ms=10_000,
    )
    assert outcome.params is not None


# --- the pre-roll hook ----------------------------------------------------------


def test_the_pre_roll_hook_runs_off_the_tick_loop():
    """The tick must hand the work over, not do it.

    A hook that sleeps for half a second would, if it ran inline, make the tick
    that released it half a second late -- which at 2 ms per tick is a quarter
    of a second of transitions firing behind the beat.
    """
    engine = Engine(blocksize=2048)
    seen = []
    released = threading_event()

    def slow_hook(cmd):
        time.sleep(0.25)
        seen.append(cmd)
        released.set()

    scheduler = Scheduler(
        engine, on_pre_roll=slow_hook, pre_roll_frames=100_000
    )
    scheduler.start()
    try:
        cmd = StartTransition(
            from_deck="a", to_deck="b", total_frames=1000,
            execute_at=50_000, origin="test",
        )
        scheduler.submit(cmd)
        started = time.perf_counter()
        scheduler.tick(0)              # inside the horizon: announce it
        tick_ms = (time.perf_counter() - started) * 1000.0
        assert tick_ms < 25, f"the tick blocked for {tick_ms:.0f} ms"
        assert released.wait(3.0), "the hook never ran"
        assert seen == [cmd]
    finally:
        scheduler.stop()


def threading_event():
    import threading
    return threading.Event()


def test_a_command_is_pre_rolled_once_and_only_once():
    engine = Engine(blocksize=2048)
    seen = []
    scheduler = Scheduler(
        engine, on_pre_roll=seen.append, pre_roll_frames=100_000
    )
    cmd = StartTransition(
        from_deck="a", to_deck="b", total_frames=1000,
        execute_at=50_000, origin="test",
    )
    scheduler.submit(cmd)
    for now in range(0, 40_000, 4_000):
        scheduler._announce_pre_roll(now)
    drained = []
    while not scheduler._pre_roll_q.empty():
        drained.append(scheduler._pre_roll_q.get_nowait())
    assert drained == [cmd], f"announced {len(drained)} times"


def test_a_command_outside_the_horizon_is_not_pre_rolled():
    engine = Engine(blocksize=2048)
    scheduler = Scheduler(engine, on_pre_roll=lambda c: None, pre_roll_frames=1_000)
    scheduler.submit(StartTransition(
        from_deck="a", to_deck="b", total_frames=1000,
        execute_at=500_000, origin="test",
    ))
    scheduler._announce_pre_roll(0)
    assert scheduler._pre_roll_q.empty()


def test_a_failing_hook_does_not_stop_the_scheduler():
    """The command still fires. That is the only thing that matters."""
    engine = Engine(blocksize=2048)

    def boom(cmd):
        raise RuntimeError("the preview exploded")

    scheduler = Scheduler(engine, on_pre_roll=boom, pre_roll_frames=100_000)
    scheduler.start()
    try:
        cmd = StartTransition(
            from_deck="a", to_deck="b", total_frames=1000,
            execute_at=10, origin="test",
        )
        scheduler.submit(cmd)
        scheduler.tick(0)
        time.sleep(0.2)
        released = scheduler.tick(1000)
        assert released == [cmd], "the transition did not fire"
    finally:
        scheduler.stop()


def test_a_100ms_budget_transition_fires_on_its_frame(state, pair, loaded):
    """The brief's case end to end: real scheduler, real hook, real loop.

    The preview gives up inside its budget and logs why; the transition is
    released on exactly the frame it was scheduled for, not a tick later.
    """
    _a, b = pair
    _la, lb = loaded
    engine = Engine(blocksize=2048)
    outcomes = []
    done = threading_event()
    session_log = _Log()

    def hook(cmd):
        outcomes.append(revise.run_preview(
            state, b, lb, entry(b), SANE, session_log=session_log, budget_ms=100.0,
        ))
        done.set()

    scheduler = Scheduler(engine, on_pre_roll=hook, pre_roll_frames=200_000)
    scheduler.start()
    try:
        cmd = StartTransition(from_deck="a", to_deck="b", total_frames=4096,
                              execute_at=100_000, origin="test")
        scheduler.submit(cmd)
        scheduler._stop.set()              # drive ticks by hand from here
        scheduler._thread.join(timeout=1.0)
        assert scheduler.tick(0) == []
        assert done.wait(5.0), "the hook never ran"
        assert scheduler.tick(99_999) == []
        assert scheduler.tick(100_000) == [cmd]
    finally:
        scheduler._thread = None
        scheduler.stop()

    outcome = outcomes[0]
    assert outcome.status == "budget"
    assert outcome.elapsed_ms < 100.0
    event, fields = session_log.records[0]
    assert event == "preview" and fields["status"] == "budget"
    assert any("exceeded" in e for e in fields["events"])


def test_the_scheduler_is_unchanged_without_a_hook():
    """The default path allocates no thread and runs no extra code."""
    engine = Engine(blocksize=2048)
    scheduler = Scheduler(engine)
    scheduler.start()
    try:
        assert scheduler._pre_roll_thread is None
        cmd = StartTransition(
            from_deck="a", to_deck="b", total_frames=1000,
            execute_at=10, origin="test",
        )
        scheduler.submit(cmd)
        assert scheduler.tick(1000) == [cmd]
    finally:
        scheduler.stop()
