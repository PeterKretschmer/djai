"""Tempo detection, grid confidence, and where the master clock comes from.

THREADING CONTEXT: main thread (pytest). Synthetic audio, so the true tempo is
known rather than assumed.

Context for why this was rewritten. The old chain was: librosa's beat tracker
(whose `start_bpm` defaults to 120), then a refinement that tried only octave
and triplet multiples of that answer, then a tempo prior centred on 128 to
break ties. The errors seen on real tracks were neither octaves nor triplets --
160 BPM read as 119 (x0.744), 105 read as 131 (x1.25) -- so refinement could
not reach the right answer from the wrong one, and the prior made the wrong
answer stable.
"""

from __future__ import annotations

import numpy as np
import pytest

from djai import analysis
from tests import synth


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """One track per tempo, across the whole search range."""
    folder = tmp_path_factory.mktemp("tempo")
    out = {}
    for bpm in (75.0, 98.0, 105.0, 120.0, 128.0, 140.0, 160.0, 174.0):
        path = folder / f"t{int(bpm)}.wav"
        synth.render_track(path, bpm=bpm, bars=32, seed=int(bpm))
        out[bpm] = path
    return out


# --- detection -----------------------------------------------------------------


@pytest.mark.parametrize("bpm", [98.0, 120.0, 128.0, 140.0, 160.0])
def test_the_detected_tempo_is_the_rendered_one(rendered, bpm):
    ta = analysis.analyze_file(rendered[bpm])
    assert ta.bpm == pytest.approx(bpm, abs=0.5), (
        f"rendered {bpm}, detected {ta.bpm:.2f}"
    )


def test_a_128_bpm_track_reports_128(rendered):
    """Named in the acceptance list, so it gets its own test."""
    ta = analysis.analyze_file(rendered[128.0])
    assert ta.bpm == pytest.approx(128.0, abs=0.5)


def test_nothing_is_pulled_toward_120(rendered):
    """The failure mode this replaced: everything drifting to the prior."""
    detected = {}
    for bpm, path in rendered.items():
        detected[bpm] = analysis.analyze_file(path).bpm

    near = [b for b, got in detected.items()
            if abs(got - 120.0) <= 1.0 and abs(b - 120.0) > 1.0]
    assert not near, f"tempos pulled to 120 from {near}: {detected}"

    # And the spread is preserved rather than compressed toward the middle.
    got = np.array(sorted(detected.values()))
    assert got.min() < 100.0 and got.max() > 155.0


def test_no_start_bpm_prior_is_passed_anywhere():
    """A prior is what turns a hard track into a confidently wrong one.

    Checked against the parse tree rather than the text, so the prose that
    explains why it was removed does not read as the thing itself.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(analysis))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg in ("start_bpm", "prior")
    ]
    assert not offenders, f"a tempo prior is passed at line(s) {offenders}"


def test_the_search_covers_the_whole_range():
    assert analysis.TEMPO_SEARCH_MIN <= 70.0
    assert analysis.TEMPO_SEARCH_MAX >= 180.0


def test_a_half_tempo_grid_agrees_worse_than_the_real_one(rendered):
    import librosa

    y, sr = librosa.load(str(rendered[128.0]), sr=analysis.ANALYSIS_SR, mono=True)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=analysis._HOP)
    true = analysis.grid_agreement(oenv, sr, 128.0)
    half = analysis.grid_agreement(oenv, sr, 64.0)
    assert true > half, f"half-tempo scored {half:.3f} vs {true:.3f}"


def test_agreement_alone_cannot_rule_out_double_tempo(rendered):
    """Documenting the limitation the alternation check exists to cover.

    On a track with eighth-note content every beat of a doubled grid still
    lands on an onset, so it scores at least as well as the real tempo. Any
    detector resting on agreement alone will take the double.
    """
    import librosa

    y, sr = librosa.load(str(rendered[128.0]), sr=analysis.ANALYSIS_SR, mono=True)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=analysis._HOP)
    assert analysis.grid_agreement(oenv, sr, 256.0) >= analysis.grid_agreement(
        oenv, sr, 128.0
    )


def test_a_slow_track_may_read_at_double_time(rendered):
    """A known limitation, asserted so it cannot regress quietly.

    Onset agreement cannot separate a grid from its double on material with
    eighth-note content: both land on every onset. A 75 BPM track therefore
    reads as 150, and for that material 150 is a defensible reading of the
    pulse rather than nonsense.

    What bounds it: the search range is 70-180, so only tracks under 90 can
    double, and the decks beat-match by ratio, so a doubled reading still
    mixes in phase. It costs bar counts and LLM context, not sync.
    """
    ta = analysis.analyze_file(rendered[75.0])
    assert ta.bpm == pytest.approx(75.0, abs=0.5) or ta.bpm == pytest.approx(
        150.0, abs=0.5
    ), f"detected {ta.bpm:.2f}, which is neither 75 nor its double"


def test_the_doubling_is_bounded_by_the_search_range(rendered):
    """Only tracks under 90 can double, because 180 is the ceiling."""
    assert analysis.TEMPO_SEARCH_MAX <= 180.0
    for bpm in (98.0, 120.0, 128.0, 140.0, 160.0):
        assert bpm * 2 > analysis.TEMPO_SEARCH_MAX, (
            f"{bpm} could double to {bpm * 2} inside the range"
        )


def test_a_genuinely_fast_track_is_not_dragged_down(rendered):
    """The other side of the tie-break: 160 must not become 80."""
    ta = analysis.analyze_file(rendered[160.0])
    assert ta.bpm == pytest.approx(160.0, abs=0.5)


def test_and_so_detection_lands_on_the_real_tempo(rendered):
    """The two together: agreement finds the pulse, alternation picks the beat."""
    assert analysis.analyze_file(rendered[128.0]).bpm == pytest.approx(128.0, abs=0.5)


# --- confidence ------------------------------------------------------------------


def test_a_real_grid_scores_higher_than_a_wrong_one(rendered):
    import librosa

    y, sr = librosa.load(str(rendered[128.0]), sr=analysis.ANALYSIS_SR, mono=True)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=analysis._HOP)
    right = analysis.grid_agreement(oenv, sr, 128.0)
    for wrong in (101.0, 113.0, 137.0):
        assert right > analysis.grid_agreement(oenv, sr, wrong), (
            f"a {wrong} grid scored as well as the real 128"
        )


def test_noise_scores_low_confidence_rather_than_reporting_a_tempo(tmp_path):
    """The acceptance case: corrupted audio must score low, not report 120.

    Confidence used to come from the variance of the detected beat intervals,
    and a grid laid down at a made-up tempo is perfectly even -- so noise
    scored *high*. It is now onset agreement, which noise cannot fake.
    """
    import soundfile as sf

    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.2, size=(analysis.ANALYSIS_SR * 25, 2)).astype(np.float32)
    path = tmp_path / "corrupt.wav"
    sf.write(str(path), noise, analysis.ANALYSIS_SR)

    ta = analysis.analyze_file(path)
    assert ta.grid_confidence < 0.30, (
        f"noise scored {ta.grid_confidence:.3f}; it must not look like a grid"
    )


def test_a_clean_track_scores_well_above_noise(rendered, tmp_path):
    import soundfile as sf

    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.2, size=(analysis.ANALYSIS_SR * 25, 2)).astype(np.float32)
    path = tmp_path / "noise.wav"
    sf.write(str(path), noise, analysis.ANALYSIS_SR)

    clean = analysis.analyze_file(rendered[128.0]).grid_confidence
    junk = analysis.analyze_file(path).grid_confidence
    assert clean > junk * 1.5, f"clean {clean:.3f} vs noise {junk:.3f}"


def test_confidence_stays_in_range(rendered):
    for path in rendered.values():
        conf = analysis.analyze_file(path).grid_confidence
        assert 0.0 <= conf <= 1.0


# --- the master clock -------------------------------------------------------------


def test_master_tempo_comes_from_the_playing_deck():
    """Never a constant. The clock is whatever is actually playing."""
    from djai.commands import IMMEDIATE, LoadTrack
    from djai.deck import SAMPLE_RATE, LoadedTrack
    from djai.engine import Engine
    from tests.test_engine import drive, loud_track

    engine = Engine(blocksize=512)
    assert engine.master_bpm == 0.0, "nothing playing, no clock"

    for bpm in (98.0, 128.0, 174.0):
        base = loud_track(220.0, bpm, 20.0)
        object.__setattr__(base.analysis, "bpm", bpm)
        track = LoadedTrack(analysis=base.analysis, audio=base.audio)
        engine.submit(
            LoadTrack(deck="a", track=track, start_frame=0, rate=1.0, play=True,
                      master=True, execute_at=IMMEDIATE, origin="test")
        )
        drive(engine, 2, 512)
        assert engine.master_bpm == pytest.approx(bpm, rel=1e-3), (
            f"deck plays {bpm}, master clock says {engine.master_bpm:.2f}"
        )
        assert engine.master_deck == "a"


def test_master_tempo_follows_the_deck_rate_not_just_its_tempo():
    """A stretched deck drives the clock at its stretched tempo."""
    from djai.commands import IMMEDIATE, LoadTrack
    from djai.deck import LoadedTrack
    from djai.engine import Engine
    from tests.test_engine import drive, loud_track

    engine = Engine(blocksize=512)
    base = loud_track(220.0, 124.0, 20.0)
    object.__setattr__(base.analysis, "bpm", 124.0)
    track = LoadedTrack(analysis=base.analysis, audio=base.audio)
    engine.submit(
        LoadTrack(deck="a", track=track, start_frame=0, rate=1.04, play=True,
                  master=True, execute_at=IMMEDIATE, origin="test")
    )
    drive(engine, 2, 512)
    assert engine.master_bpm == pytest.approx(124.0 * 1.04, rel=1e-3)


def test_no_fixed_master_tempo_constant_exists():
    """The clock must never be a number written down somewhere."""
    import inspect

    from djai import config, engine

    src = inspect.getsource(engine)
    assert "_beats_per_frame: float = 0.0" in src, (
        "the clock should start unset, not at a tempo"
    )
    for name in dir(config):
        if "MASTER" in name and "BPM" in name:
            raise AssertionError(f"config.{name} pins a master tempo")
