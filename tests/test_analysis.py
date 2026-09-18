"""Analysis and cache tests.

THREADING CONTEXT: main thread (pytest). These are slow (librosa on real
audio); the crate is rendered once per session and shared.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from djai import analysis
from djai.analysis import (
    TrackAnalysis,
    analyze_folder,
    load_cached,
    load_crate,
    track_hash,
)
from tests import synth

#: A small crate: enough tracks to exercise selection, few enough to analyse
#: quickly. Short (16 bars) because tempo/key accuracy is checked elsewhere.
SPECS = synth.CRATE_SPEC[:3]


@pytest.fixture(scope="module")
def crate_dir(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("tracks")
    for i, spec in enumerate(SPECS):
        synth.render_track(
            folder / f"{spec['name']}.wav",
            bpm=spec["bpm"],
            root=spec["root"],
            minor=spec["minor"],
            bars=24,
            seed=i,
        )
    return folder


def test_analyze_writes_one_valid_sidecar_per_track(crate_dir, tmp_path):
    cache = tmp_path / "cache"
    results, analyzed, skipped = analyze_folder(crate_dir, cache)

    assert analyzed == len(SPECS)
    assert skipped == 0
    sidecars = sorted(cache.glob("*.json"))
    assert len(sidecars) == len(SPECS)

    for p in sidecars:
        data = json.loads(p.read_text(encoding="utf-8"))
        for field in (
            "track_id", "path", "bpm", "key_name", "camelot",
            "grid_confidence", "analysis_version",
        ):
            assert field in data, f"{p.name} is missing {field!r}"
        for field in analysis.ARRAY_FIELDS:
            assert field not in data, f"{field} belongs in the array file, not the JSON"

        npz = p.with_suffix(".npz")
        assert npz.exists(), "every sidecar has its array file"
        with np.load(npz) as arrays:
            assert int(arrays["analysis_version"]) == analysis.ANALYSIS_VERSION
            for field in analysis.ARRAY_FIELDS:
                assert arrays[field].dtype == np.float32, f"{field} is not float32"
            beats, downbeats = arrays["beats"], arrays["downbeats"]
        assert data["bpm"] > 0
        assert len(beats) > 8
        assert len(downbeats) > 0, "no downbeats estimated"
        assert 0.0 <= data["grid_confidence"] <= 1.0
        assert p.stem == data["track_id"], "sidecar must be named by content hash"


def test_rerunning_analyze_does_no_work(crate_dir, tmp_path):
    """The second acceptance criterion for `analyze`: it must be a no-op."""
    cache = tmp_path / "cache"
    analyze_folder(crate_dir, cache)
    mtimes = {p: p.stat().st_mtime_ns for p in cache.iterdir()}
    assert any(p.suffix == ".npz" for p in mtimes), "array files are part of the cache"

    started = time.time()
    _, analyzed, skipped = analyze_folder(crate_dir, cache)
    elapsed = time.time() - started

    assert analyzed == 0, "re-analysed a cached track"
    assert skipped == len(SPECS)
    assert elapsed < 2.0, f"a cache hit should be near-instant, took {elapsed:.2f}s"
    assert {p: p.stat().st_mtime_ns for p in cache.iterdir()} == mtimes


def test_force_reanalyses(crate_dir, tmp_path):
    cache = tmp_path / "cache"
    analyze_folder(crate_dir, cache)
    _, analyzed, _ = analyze_folder(crate_dir, cache, force=True)
    assert analyzed == len(SPECS)


def test_cache_key_follows_content_not_filename(crate_dir, tmp_path):
    """A renamed file must hit the cache; an edited one must miss it."""
    src = sorted(crate_dir.glob("*.wav"))[0]
    original = track_hash(src)

    renamed = tmp_path / "totally_different_name.wav"
    renamed.write_bytes(src.read_bytes())
    assert track_hash(renamed) == original

    edited = tmp_path / "edited.wav"
    data = bytearray(src.read_bytes())
    data[1000:1200] = bytes(200)
    edited.write_bytes(bytes(data))
    assert track_hash(edited) != original


def test_a_stale_analysis_version_is_a_cache_miss(crate_dir, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    analyze_folder(crate_dir, cache)
    sidecar = sorted(cache.glob("*.json"))[0]
    data = json.loads(sidecar.read_text(encoding="utf-8"))

    # Below every upgradable version, so there is no path forward and it is a
    # genuine miss rather than an in-place upgrade.
    data["analysis_version"] = min(analysis.UPGRADABLE_FROM + (analysis.ANALYSIS_VERSION,)) - 1
    sidecar.write_text(json.dumps(data), encoding="utf-8")
    assert load_cached(data["track_id"], cache) is None


def test_corrupt_sidecars_do_not_break_loading(crate_dir, tmp_path):
    cache = tmp_path / "cache"
    analyze_folder(crate_dir, cache)
    (cache / "garbage.json").write_text("{not json", encoding="utf-8")

    crate = load_crate(cache)
    assert len(crate) == len(SPECS), "a corrupt sidecar should be skipped, not fatal"


def test_load_crate_skips_tracks_whose_audio_vanished(crate_dir, tmp_path):
    cache = tmp_path / "cache"
    results, _, _ = analyze_folder(crate_dir, cache)
    missing = results[0]
    data = json.loads((cache / f"{missing.track_id}.json").read_text(encoding="utf-8"))
    data["path"] = str(tmp_path / "gone.wav")
    (cache / f"{missing.track_id}.json").write_text(json.dumps(data), encoding="utf-8")

    assert len(load_crate(cache)) == len(SPECS) - 1


def test_detected_bpm_matches_the_rendered_tempo(crate_dir, tmp_path):
    cache = tmp_path / "cache"
    results, _, _ = analyze_folder(crate_dir, cache)
    by_title = {t.title: t for t in results}
    for spec in SPECS:
        got = by_title[spec["name"]].bpm
        assert got == pytest.approx(spec["bpm"], abs=0.5), (
            f"{spec['name']}: wanted {spec['bpm']} BPM, got {got:.2f}"
        )


def test_beat_grid_is_uniform_and_matches_the_reported_bpm(crate_dir, tmp_path):
    cache = tmp_path / "cache"
    results, _, _ = analyze_folder(crate_dir, cache)
    for ta in results:
        intervals = [b - a for a, b in zip(ta.beats, ta.beats[1:])]
        assert max(intervals) - min(intervals) < 1e-4, "grid is not uniform"
        assert intervals[0] == pytest.approx(60.0 / ta.bpm, rel=1e-4)


def test_downbeats_are_a_subset_of_the_beat_grid(crate_dir, tmp_path):
    cache = tmp_path / "cache"
    results, _, _ = analyze_folder(crate_dir, cache)
    for ta in results:
        beats = set(ta.beats)
        assert all(d in beats for d in ta.downbeats)
        # Every 4th beat, so consecutive downbeats are one bar apart.
        gaps = [b - a for a, b in zip(ta.downbeats, ta.downbeats[1:])]
        if gaps:
            assert gaps[0] == pytest.approx(4 * 60.0 / ta.bpm, rel=1e-3)


def test_grid_confidence_separates_tight_from_drifting(tmp_path):
    """A machine-tight grid must score higher than a wandering one."""
    tight = tmp_path / "tight.wav"
    loose = tmp_path / "loose.wav"
    synth.render_track(tight, bpm=126.0, bars=32, drift=0.0, seed=1)
    synth.render_track(loose, bpm=126.0, bars=32, drift=0.03, seed=1)

    a = analysis.analyze_file(tight)
    b = analysis.analyze_file(loose)
    assert a.grid_confidence > b.grid_confidence, (
        f"tight={a.grid_confidence:.3f} loose={b.grid_confidence:.3f}"
    )


def test_analysis_round_trips_through_json(crate_dir, tmp_path):
    cache = tmp_path / "cache"
    results, _, _ = analyze_folder(crate_dir, cache)
    original = results[0]
    restored = TrackAnalysis.from_dict(json.loads(original.to_json()))
    assert restored.track_id == original.track_id
    assert restored.bpm == original.bpm
    assert restored.beats == original.beats
    assert restored.camelot == original.camelot


def test_mono_and_resampled_files_are_handled(tmp_path):
    mono = tmp_path / "mono48k.wav"
    synth.render_track(mono, bpm=124.0, bars=16, sr=48000, stereo=False, seed=3)
    ta = analysis.analyze_file(mono)
    assert ta.bpm == pytest.approx(124.0, abs=0.5)

    from djai.deck import CHANNELS, SAMPLE_RATE, load_track

    loaded = load_track(ta)
    assert loaded.audio.shape[1] == CHANNELS, "mono must be widened to stereo"
    assert loaded.audio.dtype.name == "float32"
    expected = ta.duration_s * SAMPLE_RATE
    assert loaded.n_frames == pytest.approx(expected, rel=0.01), "not resampled to 44.1k"

