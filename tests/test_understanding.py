"""SPEC §2 (Phase 2.2): track understanding, v9 sidecars, corrections, agreement.

THREADING CONTEXT: main thread (pytest). One synthetic track is analysed from
audio (module fixture); everything else runs on hand-built analyses.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from djai import analysis as an
from djai import cli, understanding as u
from tests import synth
from tests.test_integration import make_analysis

UNCERTAINTY_KEYS = {
    "grid", "tempo", "key", "sections", "vocals", "intensity", "energy",
    "mix_in", "mix_out", "embedding",
}

#: intro 16 quiet, drop 32 loud, breakdown 16 quiet, drop 32 loud, outro 16 quiet.
LAYOUT = [("intro", 16, 0.10), ("drop", 32, 0.60), ("breakdown", 16, 0.12),
          ("drop", 32, 0.60), ("outro", 16, 0.08)]


def shaped(tmp_path, bpm: float = 124.0, vocals=((24, 40),)) -> an.TrackAnalysis:
    """An analysis whose per-beat energy follows LAYOUT, with matching sections."""
    ta = make_analysis("shaped", bpm, "8A", 0.5, tmp_path / "shaped.wav")
    rms, sections, bar = [], [], 0
    for label, n, level in LAYOUT:
        rms += [level] * (4 * n)
        sections.append({"label": label, "start_bar": bar, "end_bar": bar + n})
        bar += n
    n_beats = min(len(ta.beats), len(rms))
    ta.beats, ta.beat_rms = ta.beats[:n_beats], rms[:n_beats]
    ta.downbeats = ta.beats[::4]
    ta.sections = sections
    ta.vocal_bars = [list(v) for v in vocals]
    ta.vocal_fraction = sum(b - a for a, b in vocals) / bar
    ta.intensity, ta.intensity_features = 0.7, {"percussive": 0.4, "flux": 3.0, "density": 5.0}
    ta.mix_in_bar, ta.mix_out_bar = 16.0, 96.0
    return ta


# --- energy -------------------------------------------------------------------------


def test_energy_is_reported_per_bar_per_phrase_and_whole_track(tmp_path):
    ta = shaped(tmp_path)
    bars = u.bar_energy(ta)
    assert bars.size == 112
    assert bars[20] - bars[4] == pytest.approx(20 * np.log10(0.60 / 0.10), abs=1e-6)
    phrases = u.phrase_energy(ta)
    assert phrases.size == 4  # 112 bars in 32-bar phrases, the last one short
    assert u.track_energy_db(ta) == pytest.approx(float(bars.mean()))
    assert u.energy_uncertainty(ta) == 0.0  # every beat of every bar agrees


# --- section confidence ---------------------------------------------------------------


def test_clear_sections_are_confident_and_a_borderline_drop_is_not(tmp_path):
    ta = shaped(tmp_path)
    confs = u.section_confidences(ta)
    assert min(confs) >= 0.7, confs
    # A "drop" that is 2.9 dB under the loudest section only just made the rule.
    weak = shaped(tmp_path)
    for i in range(4 * 64, 4 * 96):
        weak.beat_rms[i] = 0.60 * 10 ** (-2.9 / 20)
    assert u.section_confidences(weak)[3] < confs[3] - 0.3


# --- mix regions ------------------------------------------------------------------------


def test_mix_regions_are_ranked_and_carry_their_components(tmp_path):
    ta = shaped(tmp_path)
    u.derive(ta)
    for regions, weights in ((ta.mix_in_regions, u.MIX_IN_WEIGHTS),
                             (ta.mix_out_regions, u.MIX_OUT_WEIGHTS)):
        assert 1 <= len(regions) <= u.MAX_REGIONS
        q = [r["quality"] for r in regions]
        assert q == sorted(q, reverse=True)
        for r in regions:
            assert set(r["components"]) == set(weights)
            assert r["quality"] == pytest.approx(
                sum(weights[k] * v for k, v in r["components"].items()), abs=1e-3)
    # The drop after the intro is where to be in; the outro is where to leave.
    assert ta.mix_in_regions[0]["bar"] == 16
    assert ta.mix_out_regions[0]["bar"] == 96
    # The vocal in bars 24-40 counts against mixing in at bar 32 or 40.
    at32 = next((r for r in u.mix_regions(ta, "in") if r["bar"] == 32), None)
    assert at32 is None or at32["components"]["vocal_clear"] < 1.0


# --- embedding ----------------------------------------------------------------------------


def test_the_embedding_is_documented_deterministic_and_discriminates(tmp_path):
    a = shaped(tmp_path)
    assert "Not learned" in u.EMBEDDING_METHOD
    ea = u.embedding(a)
    assert len(ea) == u.EMBEDDING_DIM and all(0.0 <= x <= 1.0 for x in ea)
    assert ea == u.embedding(shaped(tmp_path))
    assert u.embedding_similarity(ea, ea) == 1.0
    near = shaped(tmp_path, bpm=125.0)
    far = shaped(tmp_path, bpm=160.0)
    far.camelot, far.intensity = "3B", 0.1
    assert u.embedding_similarity(ea, u.embedding(near)) > u.embedding_similarity(
        ea, u.embedding(far))


# --- uncertainty ----------------------------------------------------------------------------


def test_every_derived_feature_carries_an_uncertainty(tmp_path):
    ta = shaped(tmp_path)
    ta.uncertainty = {"key": 0.2, "vocals": 0.1}  # as analyze would have measured
    u.derive(ta)
    assert set(ta.uncertainty) == UNCERTAINTY_KEYS
    assert all(v is not None and 0.0 <= v <= 1.0 for v in ta.uncertainty.values())
    assert ta.uncertainty["key"] == 0.2 and ta.uncertainty["vocals"] == 0.1
    ta.tempo_ambiguous = True
    u.derive(ta)
    assert ta.uncertainty["tempo"] >= 0.5


# --- v9 sidecars: old caches load, upgrades need no audio ---------------------------------


def test_a_v8_sidecar_upgrades_in_place_without_reading_audio(tmp_path, monkeypatch):
    fresh = shaped(tmp_path)
    old = dataclasses.replace(fresh, analysis_version=8, mix_in_regions=[],
                              mix_out_regions=[], embedding=[], uncertainty={})
    old.sections = [dict(s) for s in fresh.sections]
    path = an.write_sidecar(old, tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("mix_in_regions", "mix_out_regions", "embedding", "uncertainty",
                "structure_manually_corrected"):
        raw.pop(key)  # a genuine v8 file never had them
    path.write_text(json.dumps(raw), encoding="utf-8")

    def no_audio(*_a, **_k):
        raise AssertionError("an upgrade must not decode audio")

    monkeypatch.setattr(an.librosa, "load", no_audio)
    data, status = an.read_sidecar(path)
    assert status == "upgraded" and data["analysis_version"] == 9
    loaded = an.TrackAnalysis.from_dict(data)
    u.derive(fresh)
    assert loaded.mix_in_regions == fresh.mix_in_regions
    assert loaded.embedding == fresh.embedding
    assert [s["confidence"] for s in loaded.sections] == [s["confidence"] for s in fresh.sections]
    assert loaded.uncertainty["key"] is None and loaded.uncertainty["vocals"] is None, (
        "measured from audio only; never invented"
    )
    # Rewritten once: the next read is a plain v9 hit.
    assert an.read_sidecar(path)[1] == "ok"


@pytest.mark.parametrize("version", [5, 6, 7])
def test_older_caches_still_load(tmp_path, version):
    ta = dataclasses.replace(shaped(tmp_path), analysis_version=version, sections=[],
                             vocal_bars=[], vocal_fraction=None, intensity=None,
                             intensity_features={})
    path = an.write_sidecar(ta, tmp_path)
    if version == 5:  # v5 held the arrays inline
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.update(beats=ta.beats, downbeats=ta.downbeats, beat_rms=ta.beat_rms)
        path.write_text(json.dumps(raw), encoding="utf-8")
    data, status = an.read_sidecar(path)
    assert status == "upgraded" and data["analysis_version"] == an.ANALYSIS_VERSION
    assert data["uncertainty"]["sections"] is None  # no sections yet: none claimed


# --- manual corrections ------------------------------------------------------------------------


def test_structure_validation_rejects_nonsense():
    assert u.validate_structure([{"label": "chorus", "start_bar": 0, "end_bar": 8}], None)
    assert u.validate_structure([{"label": "drop", "start_bar": 8, "end_bar": 8}], None)
    assert u.validate_structure(
        [{"label": "drop", "start_bar": 8, "end_bar": 16},
         {"label": "intro", "start_bar": 0, "end_bar": 8}], None)
    assert u.validate_structure(None, [[5, 3]])
    assert u.validate_structure([{"label": "intro", "start_bar": 0, "end_bar": 8}], [[2, 6]]) is None


@pytest.fixture(scope="module")
def real_track(tmp_path_factory):
    folder = tmp_path_factory.mktemp("understand")
    layout, vocal = synth.STRUCTURED_LAYOUTS[0]
    path = folder / "struct.wav"
    labels, vocals = synth.structured_track(path, synth.STRUCTURED_BPMS[0], layout, vocal, seed=3)
    return folder, path, labels, vocals


def test_analysis_measures_key_and_vocal_uncertainty(real_track, tmp_path):
    folder, *_ = real_track
    results, _, _ = an.analyze_folder(folder, tmp_path / "cache")
    ta = results[0]
    assert ta.analysis_version == 9
    assert set(ta.uncertainty) == UNCERTAINTY_KEYS
    assert ta.uncertainty["key"] is not None and ta.uncertainty["vocals"] is not None
    assert all("confidence" in s for s in ta.sections)


def test_manual_corrections_survive_a_forced_reanalysis(real_track, tmp_path):
    folder, *_ = real_track
    cache = tmp_path / "cache"
    ta = an.analyze_folder(folder, cache)[0][0]
    mine = [{"label": "intro", "start_bar": 0, "end_bar": 8},
            {"label": "drop", "start_bar": 8, "end_bar": 40},
            {"label": "outro", "start_bar": 40, "end_bar": 48}]
    u.correct_structure(ta, mine, [[10, 20]])
    an.write_sidecar(ta, cache)

    again = an.analyze_folder(folder, cache, force=True)[0][0]
    assert again.structure_manually_corrected
    assert [(s["label"], s["start_bar"], s["end_bar"]) for s in again.sections] == [
        (s["label"], s["start_bar"], s["end_bar"]) for s in mine]
    assert again.vocal_bars == [[10, 20]]
    assert all(s["confidence"] == 1.0 for s in again.sections)
    assert again.uncertainty["vocals"] == 0.0

    # And the other path that re-measures structure: an entry missing intensity.
    again.intensity = None
    an.complete_measurements(again, next(folder.glob("*.wav")))
    assert again.vocal_bars == [[10, 20]] and again.sections[1]["label"] == "drop"


# --- agreement with a person's annotations --------------------------------------------------------


def test_agreement_is_exact_on_a_perfect_annotation_and_lists_disagreements(tmp_path):
    ta = shaped(tmp_path)
    perfect = {"sections": [dict(s) for s in ta.sections], "vocal_bars": [[24, 40]]}
    r = u.structure_agreement(ta, perfect)
    assert r["label_accuracy"] == 1.0 and r["boundary_recall"] == 1.0
    assert r["vocal_precision"] == 1.0 and r["vocal_recall"] == 1.0
    assert r["label_disagreements"] == []

    moved = [dict(s) for s in ta.sections]
    moved[1]["end_bar"] = moved[2]["start_bar"] = 52  # breakdown starts 4 bars early
    r = u.structure_agreement(ta, {"sections": moved})
    assert r["missed_boundaries"] == [52] and r["false_boundaries"] == [48]
    assert r["label_disagreements"] == [
        {"start_bar": 48, "end_bar": 52, "ours": "breakdown", "theirs": "drop"}]
    assert r["label_accuracy"] == pytest.approx(108 / 112, abs=1e-4)


def test_the_annotate_command_reports_then_applies(real_track, tmp_path, capsys):
    folder, path, labels, vocals = real_track
    cache = tmp_path / "cache"
    ta = an.analyze_folder(folder, cache)[0][0]
    # The synthetic truth, as a person would write it down.
    sections, bar = [], 0
    for label in labels:
        if sections and sections[-1]["label"] == label:
            sections[-1]["end_bar"] += 1
        else:
            sections.append({"label": label, "start_bar": bar, "end_bar": bar + 1})
        bar += 1
    ranges, start = [], None
    for i, v in enumerate(list(vocals) + [False]):  # per-bar flags -> ranges
        if v and start is None:
            start = i
        elif not v and start is not None:
            ranges.append([start, i])
            start = None
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "one.json").write_text(json.dumps(
        {"track": ta.title, "sections": sections, "vocal_bars": ranges}))
    (notes / "bad.json").write_text(json.dumps(
        {"track": ta.title, "sections": [{"label": "chorus", "start_bar": 0, "end_bar": 4}]}))
    report = tmp_path / "report.json"

    assert cli.main(["annotate", str(notes), "--cache", str(cache), "--apply",
                     "--report", str(report)]) == 0
    out = capsys.readouterr().out
    assert "bad.json: section" in out, "an unusable annotation is reported, not applied"
    assert "1 track(s) compared (SPEC §2 asks for 10 or more)" in out
    rows = json.loads(report.read_text())
    assert len(rows) == 1 and rows[0]["label_accuracy"] is not None
    stored = an.load_cached(ta.track_id, cache)
    assert stored.structure_manually_corrected
    assert [s["label"] for s in stored.sections] == [s["label"] for s in sections]


def test_analyze_fills_in_the_audio_only_uncertainties_of_an_upgraded_entry(real_track, tmp_path):
    folder, *_ = real_track
    cache = tmp_path / "cache"
    ta = an.analyze_folder(folder, cache)[0][0]
    sections = [dict(s) for s in ta.sections]
    ta.uncertainty = {k: v for k, v in ta.uncertainty.items() if k not in ("key", "vocals")}
    an.write_sidecar(ta, cache)  # as a v8 -> v9 upgrade leaves it
    ta = an.load_cached(ta.track_id, cache)  # the stored (float32) grid
    assert ta.uncertainty.get("key") is None

    again = an.analyze_folder(folder, cache)[0][0]  # not forced
    assert again.uncertainty["key"] is not None and again.uncertainty["vocals"] is not None
    assert again.bpm == ta.bpm and again.beats == ta.beats, "tempo and grid untouched"
    assert [(s["label"], s["start_bar"], s["end_bar"]) for s in again.sections] == [
        (s["label"], s["start_bar"], s["end_bar"]) for s in sections]
    # And now it is complete, so the next run decodes nothing.
    assert an.analyze_folder(folder, cache)[0][0].uncertainty == again.uncertainty
