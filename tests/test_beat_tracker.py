"""Beat This! grids and the confidence measured against them.

THREADING CONTEXT: main thread (pytest). Offline analysis only. Everything but
the last test runs without PyTorch: detections are built here with known
tempo, phase and bar 1, so each score is checked against a grid known to be
right or known to be wrong in one specific way.
"""

from __future__ import annotations

import numpy as np
import pytest

from djai import analysis as an
from djai import beat_tracker as bt
from tests import synth

BPM = 124.0
PERIOD = 60.0 / BPM
DURATION = 200.0
FIRST_DOWNBEAT = 0.35 + PERIOD  # bar 1 falls on the second beat line


def detections(bpm=BPM, first_downbeat=FIRST_DOWNBEAT, duration=DURATION,
               jitter_s=0.008, gap=(90.0, 110.0), seed=3):
    """What a tracker reports: beats on 20 ms frames, jittered, none in a gap."""
    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    t0 = first_downbeat - np.floor(first_downbeat / period) * period
    beats = t0 + period * np.arange(int((duration - t0) / period) + 1)
    phase = int(round((first_downbeat - t0) / period)) % 4
    is_down = (np.arange(beats.size) % 4) == phase
    noisy = beats + rng.uniform(-jitter_s, jitter_s, beats.size)
    noisy = np.round(noisy / 0.02) * 0.02
    keep = ~((noisy > gap[0]) & (noisy < gap[1])) if gap else np.ones(beats.size, bool)
    return noisy[keep], noisy[keep & is_down]


def grid(bpm=BPM, first_downbeat=FIRST_DOWNBEAT, duration=DURATION):
    period = 60.0 / bpm
    t0 = first_downbeat - np.floor(first_downbeat / period) * period
    beats = t0 + period * np.arange(int((duration - t0) / period) + 1)
    phase = int(round((first_downbeat - t0) / period)) % 4
    return beats, beats[phase::4]


# --- fitting a constant grid ------------------------------------------------------


def test_the_fit_recovers_tempo_phase_and_bar_one_through_frames_and_a_gap():
    beats, downs = detections()
    bpm, first = bt.fit_constant_grid(beats, downs)
    assert bpm == pytest.approx(BPM, abs=0.02)
    period = 60.0 / bpm
    assert (first - FIRST_DOWNBEAT) / period == pytest.approx(
        round((first - FIRST_DOWNBEAT) / period), abs=0.05
    ), "on a beat line"
    assert round((first - FIRST_DOWNBEAT) / period) % 4 == 0, "and on the bar line"


def test_too_few_beats_fit_nothing():
    assert bt.fit_constant_grid(np.arange(5) * PERIOD, np.array([0.0])) is None


# --- confidence -----------------------------------------------------------------


def test_the_right_grid_is_trusted_and_a_breakdown_does_not_count_against_it():
    beats, downs = detections()
    g, gd = grid()
    assert bt.grid_confidence(g, gd, beats, downs, drift_ms=0.0) > 0.9


@pytest.mark.parametrize("name, bpm, first", [
    ("twice the tempo", BPM * 2, FIRST_DOWNBEAT),
    ("half the tempo", BPM / 2, FIRST_DOWNBEAT),
    ("0.5% fast", BPM * 1.005, FIRST_DOWNBEAT),
    ("1% slow", BPM * 0.99, FIRST_DOWNBEAT),
    ("half a beat late", BPM, FIRST_DOWNBEAT + PERIOD / 2),
    ("bar 1 a beat late", BPM, FIRST_DOWNBEAT + PERIOD),
    ("bar 1 two beats late", BPM, FIRST_DOWNBEAT + 2 * PERIOD),
])
def test_a_wrong_grid_is_not_trusted(name, bpm, first):
    beats, downs = detections()
    g, gd = grid(bpm, first)
    assert bt.grid_confidence(g, gd, beats, downs, drift_ms=0.0) < 0.25, name


def test_onset_drift_alone_withdraws_trust():
    beats, downs = detections()
    g, gd = grid()
    assert bt.grid_confidence(g, gd, beats, downs, drift_ms=float("nan")) > 0.9
    assert bt.grid_confidence(g, gd, beats, downs, drift_ms=60.0) == pytest.approx(0.4)
    assert bt.grid_confidence(g, gd, beats, downs, drift_ms=-150.0) == 0.0


SR, HOP = 22050, 512


def onsets(times, weight=1.0, oenv=None):
    frames_per_s = SR / HOP
    n = int(DURATION * frames_per_s)
    oenv = np.zeros(n) if oenv is None else oenv
    idx = np.round(np.asarray(times) * frames_per_s).astype(int)
    idx = idx[(idx >= 0) & (idx < n)]
    pulse = np.zeros(n)
    pulse[idx] = weight
    return oenv + np.convolve(pulse, np.hanning(5), mode="same")


def test_onset_drift_is_near_zero_at_the_right_tempo_and_walks_at_a_wrong_one():
    true_beats, _ = grid()
    oenv = onsets(true_beats)
    right, _ = grid()
    wrong, _ = grid(BPM * 1.0003)  # ~40 ms apart by the last window
    assert abs(bt.onset_drift_ms(oenv, SR, HOP, right, DURATION)) <= 12.0
    assert abs(bt.onset_drift_ms(oenv, SR, HOP, wrong, DURATION)) >= 25.0


def test_loud_sixteenths_in_one_section_are_not_read_as_drift():
    """Found after the rollout: a reach of a sixteenth note let the end window
    lock onto hi-hats and report a steady grid as drifting by one."""
    true_beats, _ = grid()
    oenv = onsets(true_beats)
    late = true_beats[true_beats > DURATION - 60]
    sixteenths = (late[:, None] + PERIOD / 4 * np.arange(1, 4)).ravel()
    oenv = onsets(sixteenths, weight=1.4, oenv=oenv)
    assert abs(bt.onset_drift_ms(oenv, SR, HOP, true_beats, DURATION)) <= 12.0


# --- analysis ------------------------------------------------------------------


@pytest.fixture(scope="module")
def track(tmp_path_factory):
    path = tmp_path_factory.mktemp("bt") / "steady.wav"
    return synth.render_track(path, bpm=BPM, bars=48, seed=11)


def _fake_tracker(monkeypatch, beats, downs):
    monkeypatch.setattr(bt, "available", lambda: True)
    monkeypatch.setattr(bt, "detect", lambda y, sr: (beats, downs))


def test_analysis_builds_the_grid_from_the_tracker_and_scores_it(track, monkeypatch):
    monkeypatch.setattr(an, "BEAT_TRACKER", "beat_this")
    period = 60.0 / BPM
    true = period * np.arange(int(48 * 4))
    _fake_tracker(monkeypatch, true, true[::4])

    ta = an.analyze_file(track)

    assert ta.bpm == pytest.approx(BPM, abs=0.02)
    assert ta.first_downbeat == pytest.approx(0.0, abs=0.01)
    assert ta.grid_confidence > 0.8
    assert not ta.grid_manually_corrected, "a detector's grid, not a person's"
    assert not ta.tempo_ambiguous


def test_a_person_grid_is_kept_but_measured_against_the_tracker(track, monkeypatch):
    monkeypatch.setattr(an, "BEAT_TRACKER", "beat_this")
    period = 60.0 / BPM
    true = period * np.arange(int(48 * 4))
    _fake_tracker(monkeypatch, true, true[::4])

    doubled = an.analyze_file(track, grid=(BPM * 2, 0.0))

    assert doubled.bpm == pytest.approx(BPM * 2)
    assert doubled.grid_manually_corrected
    assert doubled.grid_confidence < 0.25, "a doubled grid cuts rather than blends"


def test_auto_falls_back_to_librosa_when_beat_this_is_missing(track, monkeypatch):
    monkeypatch.setattr(an, "BEAT_TRACKER", "auto")
    monkeypatch.setattr(bt, "available", lambda: False)
    monkeypatch.setattr(bt, "detect", lambda y, sr: pytest.fail("must not be called"))
    ta = an.analyze_file(track)
    assert ta.bpm == pytest.approx(BPM, abs=0.5)


def test_asking_for_beat_this_when_it_is_missing_is_an_error(track, monkeypatch):
    monkeypatch.setattr(an, "BEAT_TRACKER", "beat_this")
    monkeypatch.setattr(bt, "available", lambda: False)
    with pytest.raises(RuntimeError, match="not installed"):
        an.analyze_file(track)


def test_playback_never_imports_torch():
    import subprocess
    import sys

    code = ("import sys, djai.cli, djai.engine, djai.ui_server; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("False")


@pytest.mark.skipif(not bt.available(), reason="Beat This! is not installed")
def test_the_real_model_grids_a_steady_track(track, monkeypatch):
    monkeypatch.setattr(an, "BEAT_TRACKER", "beat_this")
    ta = an.analyze_file(track)
    assert ta.bpm == pytest.approx(BPM, abs=0.1)
    assert ta.grid_confidence >= 0.25


# --- clipped detector input (Phase 0A root cause) ------------------------------
#
# A loudness-war MP3 decodes above full scale -- 99 of the 105 tracks in the
# reference crate do -- and Beat This! degrades on it. Measured 2026-09-21:
# "Don't Let Me Down" (peak 1.05) fitted no grid at all raw and 80.003 BPM
# scaled; "girl$" (peak 1.68) fitted 72.46 raw, a half-time octave error, and
# 144.98 scaled against a true 145.000.


def test_audio_above_full_scale_is_scaled_into_range():
    y = np.array([1.7, -0.85, 0.0], dtype=np.float64)
    out = bt._in_range(y)
    assert float(np.abs(out).max()) == pytest.approx(1.0)
    # a pure gain: the shape, and so every zero crossing, is untouched
    assert out[1] / out[0] == pytest.approx(y[1] / y[0])


def test_audio_already_in_range_is_passed_through_untouched():
    y = np.array([0.5, -0.25, 0.0], dtype=np.float64)
    out = bt._in_range(y)
    assert out is y  # identity: cannot perturb a grid fitted from valid input


def test_the_detector_is_never_handed_audio_above_full_scale(monkeypatch):
    """The fix has to hold at the boundary, not merely exist as a helper."""
    seen = {}

    class FakeModel:
        def __call__(self, y, sr):
            seen["peak"] = float(np.abs(y).max())
            return np.array([0.0, 0.5]), np.array([0.0])

    monkeypatch.setattr(bt, "_model", FakeModel())
    clipped = (1.77 * np.sin(np.linspace(0, 40, 4096))).astype(np.float32)
    assert float(np.abs(clipped).max()) > 1.0
    bt.detect(clipped, 22050)
    assert seen["peak"] <= 1.0


def test_an_empty_signal_does_not_divide_by_zero():
    assert bt._in_range(np.zeros(0)).size == 0
    assert float(np.abs(bt._in_range(np.zeros(16))).max()) == 0.0
