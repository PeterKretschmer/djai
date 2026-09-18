"""The offline renderer, and the transition acceptance criteria measured on it.

THREADING CONTEXT: main thread (pytest). No audio device: the renderer drives
the engine callback directly, which is the same code the audio thread runs.

These tests assert the Phase 1 acceptance criteria against real rendered
output, not against the shape function in isolation.
"""

from __future__ import annotations

import csv
import gzip
import math

import numpy as np
import pytest
import soundfile as sf

from djai import transition
from djai.deck import SAMPLE_RATE
from djai.render import ENVELOPE_COLUMNS, GRID_COLUMNS, render_transition
from tests import synth
from djai import analysis


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """Two synthetic tracks close enough in tempo to be mixed."""
    folder = tmp_path_factory.mktemp("rendertracks")
    synth.render_track(folder / "a.wav", bpm=126.0, root="A", minor=True, bars=48, seed=1)
    synth.render_track(folder / "b.wav", bpm=128.0, root="E", minor=True, bars=48, seed=2)
    return analysis.analyze_file(folder / "a.wav"), analysis.analyze_file(folder / "b.wav")


@pytest.fixture(scope="module")
def rendered(pair, tmp_path_factory):
    a, b = pair
    out = tmp_path_factory.mktemp("render") / "t"
    return render_transition(a, b, out)


def test_criterion_no_sample_reaches_full_scale(rendered):
    """A rendered blend must not contain a squared-off sample.

    Full-scale samples are what a hard-clipped master looks like, and the
    stage that used to sit at the end of the chain was exactly that.
    """
    mix = rendered.mix
    peak = float(np.max(np.abs(mix)))
    assert peak < 1.0, f"rendered peak {peak:.4f}"
    assert peak <= 0.99, "the acceptance bar is 0.99"
    assert int(np.sum(np.abs(mix) >= 1.0)) == 0

    # And the file on disk agrees with the array.
    written, _sr = sf.read(str(rendered.wav_path), dtype="float32")
    assert float(np.max(np.abs(written))) <= 0.99


def read_rows(path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- the renderer produces what it promises ----------------------------------


def test_writes_wav_and_both_csvs(rendered):
    assert rendered.wav_path.exists()
    assert rendered.envelope_path.exists()
    assert rendered.grid_path.exists()
    for path in (rendered.envelope_path, rendered.grid_path):
        assert path.name.endswith(".csv.gz"), f"{path.name} should be gzipped"
        with path.open("rb") as fh:
            assert fh.read(2) == b"\x1f\x8b", f"{path.name} is not actually gzip"

    audio, sr = sf.read(str(rendered.wav_path))
    assert sr == SAMPLE_RATE
    assert audio.shape[1] == 2
    assert audio.shape[0] == rendered.mix.shape[0]
    assert np.all(np.isfinite(audio))

    env = read_rows(rendered.envelope_path)
    assert list(env[0]) == ENVELOPE_COLUMNS
    grid = read_rows(rendered.grid_path)
    assert list(grid[0]) == GRID_COLUMNS


def test_grid_has_both_decks_beats_and_downbeats(rendered):
    grid = read_rows(rendered.grid_path)
    decks = {r["deck"] for r in grid}
    kinds = {r["kind"] for r in grid}
    assert decks == {"a", "b"}
    assert kinds == {"beat", "downbeat"}
    for deck in ("a", "b"):
        downs = [r for r in grid if r["deck"] == deck and r["kind"] == "downbeat"]
        assert len(downs) >= transition.TRANSITION_BARS


def test_render_reuses_the_real_engine_not_a_reimplementation():
    """A regression guard: if render.py ever grows its own mixer, this fails."""
    import inspect

    from djai import render

    src = inspect.getsource(render)
    assert "engine.callback(" in src, "renderer must drive the real callback"
    assert "Scheduler(" in src, "renderer must release via the real scheduler"
    assert "gains_at" not in src, "renderer must not compute its own envelope"


# --- Phase 1 acceptance criteria, measured -----------------------------------


def envelope_during(rendered):
    return [r for r in read_rows(rendered.envelope_path) if r["in_transition"] == "1"]


def test_in_transition_flag_marks_only_the_transition(rendered):
    rows = read_rows(rendered.envelope_path)
    during = [r for r in rows if r["in_transition"] == "1"]
    assert during, "no blocks flagged as in-transition"
    # Contiguous, and pre-roll/post-roll are excluded.
    idx = [int(r["block"]) for r in during]
    assert idx == list(range(idx[0], idx[-1] + 1))
    assert idx[0] > 0, "pre-roll should not be flagged"
    assert idx[-1] < len(rows) - 1, "post-roll should not be flagged"
    expected = rendered.transition_frames / rendered.blocksize
    assert len(during) == pytest.approx(expected, abs=2)


def test_criterion_gains_change_every_block(rendered):
    during = envelope_during(rendered)
    assert len(during) > 100

    for col in ("a_gain", "b_gain"):
        vals = [float(r[col]) for r in during]
        changed = sum(
            1 for i in range(1, len(vals)) if abs(vals[i] - vals[i - 1]) > 1e-9
        )
        longest, run = 1, 1
        for i in range(1, len(vals)):
            run = run + 1 if abs(vals[i] - vals[i - 1]) <= 1e-9 else 1
            longest = max(longest, run)
        assert changed >= len(vals) - 3, (
            f"{col} changed in only {changed}/{len(vals) - 1} blocks"
        )
        assert longest <= 3, f"{col} held constant for {longest} blocks"


def test_criterion_downbeats_coincide_within_10ms(rendered):
    t0 = rendered.transition_start_frame / SAMPLE_RATE
    t1 = t0 + rendered.transition_frames / SAMPLE_RATE
    grid = read_rows(rendered.grid_path)

    def downs(deck):
        return [
            float(r["time_s"])
            for r in grid
            if r["deck"] == deck and r["kind"] == "downbeat"
            and t0 <= float(r["time_s"]) <= t1
        ]

    a_db, b_db = downs("a"), downs("b")
    assert a_db and b_db
    bar_s = 4 * 60.0 / rendered.bpm_a
    offsets = []
    for ta in a_db:
        nearest = min(b_db, key=lambda tb: abs(tb - ta))
        if abs(nearest - ta) < bar_s / 2:
            offsets.append(abs(nearest - ta) * 1000.0)
    assert len(offsets) >= len(a_db) - 1
    assert max(offsets) <= 10.0, f"worst downbeat offset {max(offsets):.2f} ms"


def test_criterion_exactly_one_low_band_attenuated(rendered):
    for r in envelope_during(rendered):
        a_low, b_low = float(r["a_low"]), float(r["b_low"])
        lows = sorted((a_low, b_low))
        assert lows[0] == pytest.approx(0.0, abs=1e-6), (
            f"both decks carry low band at block {r['block']}"
        )
        assert lows[1] == pytest.approx(1.0, abs=1e-6), (
            f"neither deck carries a full low band at block {r['block']}"
        )


def test_criterion_summed_level_within_3db_through_the_midpoint(rendered):
    """The crossfade must not build level. Measured against the equal-power
    prediction from each deck's own gain, which isolates the gain law from the
    tracks' own dynamics."""
    during = envelope_during(rendered)
    mid = [
        r for r in during
        if 0.35 <= float(r["bar"]) / rendered.bars <= 0.65
    ]
    assert mid
    for r in mid:
        power = float(r["a_gain"]) ** 2 + float(r["b_gain"]) ** 2
        assert 10 * math.log10(power) == pytest.approx(0.0, abs=3.0), (
            f"combined power {10 * math.log10(power):+.2f} dB at block {r['block']}"
        )


def test_render_output_never_clips_flat(rendered):
    """Constant power should keep the master limiter out of the signal."""
    during = envelope_during(rendered)
    clipped = sum(1 for r in during if float(r["peak"]) >= 0.9799)
    assert clipped / len(during) < 0.05, (
        f"{clipped}/{len(during)} blocks pinned at the limiter ceiling"
    )


def test_transition_length_follows_the_configured_default(rendered):
    seconds = rendered.transition_frames / SAMPLE_RATE
    expected = transition.TRANSITION_BARS * 4 * 60.0 / rendered.bpm_a
    assert seconds == pytest.approx(expected, rel=1e-3)
    assert rendered.bars == transition.TRANSITION_BARS
