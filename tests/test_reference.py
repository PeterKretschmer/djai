"""Reference-set measurement: markers, aggregation, and failure handling.

THREADING CONTEXT: main thread (pytest). No audio device.

These cover the deterministic half of :mod:`djai.reference` -- parsing,
aggregation, the profile schema, and what happens when a marker cannot be
measured. The acoustic estimators are deliberately *not* asserted for accuracy
here: measured against constructed ground truth they were out by up to +38 bars
and -15 BPM, so there is no accuracy contract to pin down yet. See
docs/tuning.md.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from djai import reference
from djai.reference import (
    TransitionMeasurement,
    build_profile,
    parse_markers,
    parse_timestamp,
    write_profile,
)


# --- markers -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("204", 204.0),
        ("204.5", 204.5),
        ("3:24", 204.0),
        ("3:24.5", 204.5),
        ("1:03:24", 3804.0),
        ("0:00", 0.0),
    ],
)
def test_parse_timestamp_formats(text, expected):
    assert parse_timestamp(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "abc", "3:xx", "--", "1:2:3:4"])
def test_parse_timestamp_rejects_junk(text):
    assert parse_timestamp(text) is None


def as_tuples(markers):
    return [(m.set_name, m.start_s, m.end_s) for m in markers]


def test_parse_markers_centres_only(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text(
        "# a comment\n"
        "\n"
        "set_one.mp3  3:24  7:10.5\n"
        "  12:00   # trailing comment\n"
        "set_two.wav\n"
        "1:30\n"
        "205\n",
        encoding="utf-8",
    )
    markers = parse_markers(p)
    assert not any(m.has_bounds for m in markers)
    assert as_tuples(markers) == [
        ("set_one.mp3", 204.0, None),
        ("set_one.mp3", 430.5, None),
        ("set_one.mp3", 720.0, None),
        ("set_two.wav", 90.0, None),
        ("set_two.wav", 205.0, None),
    ]


def test_parse_markers_detects_start_end_pairs(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text(
        "# file        start   end\n"
        "set.mp3       2:12    3:00\n"
        "set.mp3       14:24   15:30\n"
        "set.mp3       1:03:00 1:04:15\n",
        encoding="utf-8",
    )
    markers = parse_markers(p)
    assert all(m.has_bounds for m in markers)
    assert as_tuples(markers) == [
        ("set.mp3", 132.0, 180.0),
        ("set.mp3", 864.0, 930.0),
        ("set.mp3", 3780.0, 3855.0),
    ]
    assert markers[0].length_s == pytest.approx(48.0)
    assert markers[0].centre_s == pytest.approx(156.0)


def test_two_far_apart_timestamps_are_two_markers_not_a_blend(tmp_path):
    """A pair further apart than any plausible blend is two separate markers."""
    p = tmp_path / "m.txt"
    p.write_text("set.mp3 2:00 40:00\n", encoding="utf-8")
    markers = parse_markers(p)
    assert len(markers) == 2
    assert not any(m.has_bounds for m in markers)


def test_parse_markers_accepts_commas_and_tabs(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("set.wav,3:24\nset.wav\t7:10\n", encoding="utf-8")
    assert as_tuples(parse_markers(p)) == [
        ("set.wav", 204.0, None),
        ("set.wav", 430.0, None),
    ]


def test_parse_markers_ignores_timestamps_before_any_file(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("3:24\nset.wav 1:00\n", encoding="utf-8")
    assert as_tuples(parse_markers(p)) == [("set.wav", 60.0, None)]


def test_a_renamed_set_falls_back_to_the_only_audio_file(tmp_path):
    """Markers naming a file that was since renamed still measure, loudly."""
    sr = 22050
    y = np.random.default_rng(0).standard_normal(sr * 5).astype(np.float32) * 0.1
    sf.write(str(tmp_path / "Actual Name.wav"), y, sr)
    markers = tmp_path / "m.txt"
    markers.write_text("old_name.mp3 1:00 1:30\n", encoding="utf-8")

    measurements, _ = reference.analyze_sets(tmp_path, markers, progress=False)
    assert len(measurements) == 1
    # It resolved to the real file, then failed for the honest reason.
    assert measurements[0].reason and "no audio file named" not in measurements[0].reason


def test_parse_markers_on_an_empty_file(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("# nothing here\n\n", encoding="utf-8")
    assert parse_markers(p) == []


# --- aggregation -------------------------------------------------------------


def fake(length_bars=16.0, grid=16, bpm_delta=0.0, ok=True, **kw):
    base = dict(
        set_name="s", marker_s=100.0, bpm_before=128.0,
        bpm_after=128.0 + bpm_delta, bpm_delta=bpm_delta, bpm_delta_pct=0.0,
        length_s=30.0, length_bars=length_bars, nearest_grid=grid,
        grid_error_bars=0.0, level_before_db=-12.0, level_after_db=-12.0,
        level_change_db=0.0, level_peak_db=0.0, low_dip_db=-1.0,
        low_bump_db=1.0,
        low_trajectory_db=[0.0] * reference.TRAJECTORY_POINTS,
        ok=ok,
    )
    base.update(kw)
    return TransitionMeasurement(**base)


def test_profile_medians_and_iqr():
    ms = [fake(length_bars=b) for b in (8, 12, 16, 20, 32)]
    prof = build_profile(ms)
    s = prof["measures"]["length_bars"]
    assert s["n"] == 5
    assert s["median"] == pytest.approx(16.0)
    assert s["iqr"] == [pytest.approx(12.0), pytest.approx(20.0)]
    assert s["min"] == pytest.approx(8.0)
    assert s["max"] == pytest.approx(32.0)


def test_profile_counts_grid_alignment():
    ms = [fake(grid=16), fake(grid=16), fake(grid=8), fake(grid=None)]
    prof = build_profile(ms)
    assert prof["grid_alignment"]["16"] == 2
    assert prof["grid_alignment"]["8"] == 1
    assert prof["grid_alignment"]["none"] == 1
    assert prof["grid_alignment"]["32"] == 0


def test_profile_separates_rejected_from_measured():
    ms = [fake(), fake(ok=False, reason="decode failed"), fake()]
    prof = build_profile(ms)
    assert prof["n_transitions"] == 2
    assert prof["n_rejected"] == 1
    assert len(prof["transitions"]) == 2
    assert prof["rejected"][0]["reason"] == "decode failed"


def test_profile_low_trajectory_is_elementwise_median():
    ms = [
        fake(low_trajectory_db=[0.0] * reference.TRAJECTORY_POINTS),
        fake(low_trajectory_db=[2.0] * reference.TRAJECTORY_POINTS),
        fake(low_trajectory_db=[4.0] * reference.TRAJECTORY_POINTS),
    ]
    prof = build_profile(ms)
    traj = prof["low_trajectory_db_median"]
    assert len(traj) == reference.TRAJECTORY_POINTS
    assert all(v == pytest.approx(2.0) for v in traj)


def test_profile_on_no_measurements_is_still_valid():
    prof = build_profile([])
    assert prof["n_transitions"] == 0
    assert prof["measures"]["length_bars"] == {"n": 0}
    assert prof["low_trajectory_db_median"] == []


def test_profile_round_trips_through_json(tmp_path):
    prof = build_profile([fake(), fake(length_bars=32.0, grid=32)])
    out = write_profile(prof, tmp_path / "reference_profile.json")
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["n_transitions"] == 2
    assert loaded["measures"]["length_bars"]["median"] == pytest.approx(24.0)
    assert "generated_at" in loaded


# --- failure handling --------------------------------------------------------


def test_missing_audio_file_is_reported_not_raised(tmp_path):
    """With no audio at all there is nothing to fall back to."""
    markers = tmp_path / "m.txt"
    markers.write_text("nope.wav 1:00\n", encoding="utf-8")
    measurements, prof = reference.analyze_sets(tmp_path, markers, progress=False)
    assert len(measurements) == 1
    assert not measurements[0].ok
    assert "no audio file named" in measurements[0].reason
    assert prof["n_transitions"] == 0


def write_noise(path, seconds=200, sr=22050, seed=0):
    rng = np.random.default_rng(seed)
    sf.write(str(path), (rng.standard_normal(sr * seconds) * 0.1).astype(np.float32), sr)


# --- resolving a markers filename to an audio file ---------------------------


def test_resolves_an_exact_name(tmp_path):
    p = tmp_path / "set.mp3"
    p.touch()
    got, note = reference.resolve_set_file("set.mp3", [p])
    assert got == p and note is None


def test_resolves_a_renamed_file_on_shared_words(tmp_path):
    """`peak_hour_tech_house.mp3` plainly means the long descriptive filename."""
    a = tmp_path / "Peak Hour Tech House Set (John Summit, Chris Lake).mp3"
    b = tmp_path / "Four Tet @ Lightning in a Bottle 2025.mp3"
    a.touch()
    b.touch()

    got, note = reference.resolve_set_file("peak_hour_tech_house.mp3", [a, b])
    assert got == a
    assert note and "matched" in note

    got, note = reference.resolve_set_file("four_tet_lib_2025.mp3", [a, b])
    assert got == b
    assert note and "matched" in note


def test_resolution_refuses_when_nothing_matches(tmp_path):
    a = tmp_path / "Some Set.mp3"
    b = tmp_path / "Another Set.mp3"
    a.touch()
    b.touch()
    got, _ = reference.resolve_set_file("completely_unrelated_xyz.mp3", [a, b])
    assert got is None


def test_resolution_falls_back_to_a_lone_file(tmp_path):
    only = tmp_path / "Whatever.mp3"
    only.touch()
    got, note = reference.resolve_set_file("does_not_match.mp3", [only])
    assert got == only
    assert note and "only audio file" in note


def test_a_folder_of_markers_files_is_read_whole(tmp_path):
    write_noise(tmp_path / "Alpha Set.wav", seed=1)
    write_noise(tmp_path / "Beta Set.wav", seed=2)
    (tmp_path / "markers Alpha.txt").write_text(
        "alpha_set.wav 60 100\n", encoding="utf-8"
    )
    (tmp_path / "markers Beta.txt").write_text(
        "beta_set.wav 60 100\n", encoding="utf-8"
    )

    measurements, prof = reference.analyze_sets(tmp_path, tmp_path, progress=False)
    assert len(measurements) == 2
    assert {m.set_name for m in measurements} == {"Alpha Set", "Beta Set"}


def test_profile_reports_each_set_separately():
    """Pooling two DJs can produce a median that describes neither."""
    ms = [fake(set_name="a", length_bars=26.0) for _ in range(5)]
    ms += [fake(set_name="b", length_bars=37.0) for _ in range(5)]
    prof = build_profile(ms)

    assert set(prof["by_set"]) == {"a", "b"}
    assert prof["by_set"]["a"]["n"] == 5
    assert prof["by_set"]["a"]["measures"]["length_bars"]["median"] == pytest.approx(26.0)
    assert prof["by_set"]["b"]["measures"]["length_bars"]["median"] == pytest.approx(37.0)
    # The pooled median sits in the gap between them.
    pooled = prof["measures"]["length_bars"]["median"]
    assert 26.0 < pooled < 37.0


def test_bounded_measurement_uses_the_given_length_verbatim(tmp_path):
    """Length is arithmetic when boundaries are given, not inferred."""
    sr = 22050
    rng = np.random.default_rng(1)
    y = (rng.standard_normal(sr * 200) * 0.1).astype(np.float32)
    sf.write(str(tmp_path / "s.wav"), y, sr)

    marker = reference.Marker("s.wav", 80.0, 128.0)  # exactly 48 s
    m = reference.measure_bounded(tmp_path / "s.wav", marker)
    assert m.ok, m.reason
    assert m.bounded is True
    assert m.length_s == pytest.approx(48.0)
    assert m.marker_s == pytest.approx(104.0)


def test_a_marker_too_close_to_the_start_is_rejected_cleanly(tmp_path):
    sr = 22050
    y = np.random.default_rng(0).standard_normal(sr * 20).astype(np.float32) * 0.1
    sf.write(str(tmp_path / "short.wav"), y, sr)
    m = reference.measure_transition(tmp_path / "short.wav", 5.0)
    assert not m.ok
    assert m.reason and "too short" in m.reason


def test_measure_transition_never_raises_on_garbage(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")
    m = reference.measure_transition(bad, 60.0)
    assert not m.ok
    assert m.reason


def test_analyze_sets_with_no_markers(tmp_path):
    markers = tmp_path / "m.txt"
    markers.write_text("", encoding="utf-8")
    measurements, prof = reference.analyze_sets(tmp_path, markers, progress=False)
    assert measurements == []
    assert prof["n_transitions"] == 0


# --- the window has to be able to hold a blend -------------------------------


def test_window_is_wider_than_the_longest_blend_it_claims_to_measure():
    """A 32-bar blend at 120 BPM is 64 s; the reference stretches must sit
    outside it or every measurement is taken from inside the blend."""
    longest_blend_s = 32 * 4 * 60.0 / 120.0
    usable = 2 * reference.WINDOW_S - 2 * (reference.REF_A[1] - reference.REF_A[0])
    assert usable > longest_blend_s


def test_reference_never_sources_audio():
    """It reads what it is given and nothing else.

    Checked against the imports rather than the source text, so the module's
    own docstring saying it never downloads does not trip the assertion.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(reference))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    for forbidden in ("urllib", "requests", "httpx", "socket", "ftplib", "yt_dlp"):
        assert forbidden not in imported, f"reference.py imports {forbidden}"
