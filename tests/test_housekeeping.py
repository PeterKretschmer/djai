"""Phase 0: disk footprint.

THREADING CONTEXT: main thread (pytest). The recorder's writer thread and the
stretch worker's thread are real; tests wait on them explicitly.

Covers every Phase 0 behaviour: recordings off by default and FLAC with a
retention cap, render scratch folders and gzipped CSVs, session log collapsing
and rotation, the float32 array cache and its in-place upgrade, stretched
copies freed after a transition, and `clean` asking before it deletes.
"""

from __future__ import annotations

import builtins
import gc
import gzip
import json
import time
import types
import weakref
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from djai import analysis as an
from djai import cli, config
from djai import engine as engine_module
from djai import housekeeping as hk
from djai import supervisor as sup
from djai.commands import LoadTrack
from djai.deck import CHANNELS, SAMPLE_RATE
from djai.engine import SessionRecorder
from tests import synth
from tests.test_engine import drive, loud_track
from tests.test_integration import make_analysis
from tests.test_integration import session as _session_fixture

session = _session_fixture

BLOCK = 512


# --- recordings -----------------------------------------------------------------


def test_recording_is_off_unless_asked_for():
    assert config.RECORD_ENABLED is False, "recording must default to off"
    args = cli.build_parser().parse_args(["play"])
    assert args.record is False
    assert cli.build_parser().parse_args(["play", "--record"]).record is True
    # The old flag still parses and still wins, so old scripts keep working.
    args = cli.build_parser().parse_args(["play", "--record", "--no-record"])
    assert args.record and args.no_record


def test_a_recording_is_flac_and_the_oldest_beyond_the_cap_are_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RECORD_RETENTION", 5)
    for day in range(1, 8):
        (tmp_path / f"set_2026-01-0{day}_120000.flac").write_bytes(b"old")
    legacy = [tmp_path / "set_2025-12-01_120000.wav", tmp_path / "set_2025-12-02_120000.wav"]
    for p in legacy:
        p.write_bytes(b"legacy")

    path = tmp_path / "set_2099-01-01_120000.flac"
    rec = SessionRecorder(path, ring_seconds=1.0)
    rec.start()
    block = np.full((BLOCK, CHANNELS), 0.25, dtype=np.float32)
    for _ in range(20):
        rec.capture(block, BLOCK)
    deadline = time.time() + 5.0
    while rec.frames_written < 20 * BLOCK and time.time() < deadline:
        time.sleep(0.02)
    rec.stop()

    flacs = sorted(p.name for p in tmp_path.glob("set_*.flac"))
    assert len(flacs) == 5, flacs
    assert path.name in flacs, "the new recording is never the one deleted"
    assert flacs[0] == "set_2026-01-04_120000.flac", "the oldest went first"
    assert all(p.exists() for p in legacy), "WAV recordings are never auto-deleted"

    info = sf.info(str(path))
    assert info.format == "FLAC"
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE and audio.shape[0] >= 20 * BLOCK * 0.95


def test_a_killed_flac_recording_keeps_all_but_its_last_fraction_of_a_second(tmp_path):
    """No close(), as if the process died mid-set.

    Unlike a WAV, a FLAC whose process dies never gets its length written, and
    libsndfile still holds frames it has not encoded yet. Measured: an unclosed
    file reads back in chunks up to its last encoded frame and then stops with
    a seek error, which is exactly where a tolerant reader or ffmpeg stops.
    What is lost is the encoder's buffer -- 8,192 frames, a fifth of a second
    -- not the set. Bounded here at half a second.
    """
    path = tmp_path / "set_2099-01-01_120000.flac"
    rec = SessionRecorder(path, ring_seconds=2.0)
    rec.start()
    block = np.full((BLOCK, CHANNELS), 0.4, dtype=np.float32)
    for _ in range(80):
        rec.capture(block, BLOCK)
    deadline = time.time() + 5.0
    while rec.frames_written < 80 * BLOCK and time.time() < deadline:
        time.sleep(0.02)
    rec._file.flush()

    recovered = 0
    try:
        with sf.SoundFile(str(path)) as fh:
            assert fh.samplerate == SAMPLE_RATE
            while True:
                chunk = fh.read(4096, dtype="float32", always_2d=True)
                if chunk.shape[0] == 0:
                    break
                recovered += chunk.shape[0]
    except RuntimeError:
        pass  # the missing end of an unfinished stream; everything before it read
    written = 80 * BLOCK
    assert recovered >= written - SAMPLE_RATE // 2, f"recovered {recovered} of {written}"
    rec.stop()


def test_retention_ignores_a_cap_below_one(tmp_path):
    (tmp_path / "set_2026-01-01_120000.flac").write_bytes(b"x")
    assert hk.enforce_recording_retention(tmp_path, keep=0) == []
    assert (tmp_path / "set_2026-01-01_120000.flac").exists()


# --- renders ----------------------------------------------------------------------


def _fake_render_setup(monkeypatch, tmp_path):
    a = make_analysis("alpha", 124.0, "8A", 0.5, tmp_path / "alpha.wav")
    b = make_analysis("beta", 125.0, "9A", 0.5, tmp_path / "beta.wav")
    monkeypatch.setattr(cli, "load_crate", lambda cache: [a, b])

    def fake_render(track_a, track_b, out, **kwargs):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        wav = out.with_suffix(".wav")
        wav.write_bytes(b"RIFF")
        return types.SimpleNamespace(
            duration_s=1.0, rate_b=1.0, wav_path=wav,
            envelope_path=out.parent / f"{out.stem}_envelope.csv.gz",
            grid_path=out.parent / f"{out.stem}_grid.csv.gz",
        )

    monkeypatch.setattr("djai.render.render_transition", fake_render)
    monkeypatch.chdir(tmp_path)


def test_a_render_scratch_folder_is_removed_on_exit(monkeypatch, tmp_path, capsys):
    _fake_render_setup(monkeypatch, tmp_path)
    args = cli.build_parser().parse_args(["render", "alpha", "beta"])
    assert cli.cmd_render(args) == 0
    renders = tmp_path / "renders"
    leftovers = list(renders.iterdir()) if renders.exists() else []
    assert leftovers == [], f"render left files behind: {leftovers}"
    assert "--keep-renders" in capsys.readouterr().out


def test_keep_renders_keeps_them(monkeypatch, tmp_path):
    _fake_render_setup(monkeypatch, tmp_path)
    args = cli.build_parser().parse_args(["render", "alpha", "beta", "--keep-renders"])
    assert cli.cmd_render(args) == 0
    assert (tmp_path / "renders" / "alpha__beta.wav").exists()


def test_an_explicit_out_path_is_always_kept(monkeypatch, tmp_path):
    _fake_render_setup(monkeypatch, tmp_path)
    out = tmp_path / "mine" / "take"
    args = cli.build_parser().parse_args(["render", "alpha", "beta", "--out", str(out)])
    assert cli.cmd_render(args) == 0
    assert out.with_suffix(".wav").exists()


# --- session log -----------------------------------------------------------------


def _lines(log: sup.SessionLog) -> list[dict]:
    log._file.flush()
    return [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines() if line]


def test_repeated_interventions_within_a_second_are_one_entry_with_a_count(tmp_path):
    log = sup.SessionLog(tmp_path)
    for i in range(10):
        log.write("drift_nudge", deck="a", drift_ms=float(i))
    entries = _lines(log)
    assert len(entries) == 1
    assert entries[0]["count"] == 10
    assert entries[0]["drift_ms"] == 9.0, "the entry carries the latest values"
    assert "first_ts" in entries[0]
    log.close()


def test_the_first_intervention_is_on_disk_immediately(tmp_path):
    """A reader mid-run must never miss an event that is still collapsing."""
    log = sup.SessionLog(tmp_path)
    log.write("drift_nudge", deck="a", drift_ms=1.0)
    assert [e["event"] for e in _lines(log)] == ["drift_nudge"]
    log.close()


def test_collapsing_is_per_deck_per_type_and_broken_by_other_events(tmp_path):
    log = sup.SessionLog(tmp_path)
    log.write("drift_nudge", deck="a")
    log.write("drift_nudge", deck="b")          # other deck: its own entry
    log.write("drift_nudge", deck="b")          # collapses into that one
    log.write("command_rejected", trigger="x")  # never collapsible
    log.write("drift_nudge", deck="b")          # a new run after it
    events = [(e["event"], e.get("deck"), e.get("count", 1)) for e in _lines(log)]
    assert events == [
        ("drift_nudge", "a", 1),
        ("drift_nudge", "b", 2),
        ("command_rejected", None, 1),
        ("drift_nudge", "b", 1),
    ]
    log.close()


def test_repeats_after_the_window_start_a_new_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "COLLAPSE_WINDOW_S", -1.0)
    log = sup.SessionLog(tmp_path)
    for _ in range(3):
        log.write("drift_nudge", deck="a")
    assert len(_lines(log)) == 3
    log.close()


def test_the_log_rotates_gzipped_and_keeps_only_the_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "LOG_ROTATE_BYTES", 2_000)
    monkeypatch.setattr(sup, "LOG_ROTATE_KEEP", 2)
    log = sup.SessionLog(tmp_path)
    for i in range(200):
        log.write("track_cued", track=f"track {i}", action="x" * 40)
    log.close()

    name = log.path.name
    rotated = sorted(p.name for p in tmp_path.glob(f"{name}.*.gz"))
    assert rotated == [f"{name}.1.gz", f"{name}.2.gz"], rotated
    assert log.path.stat().st_size < 2_000 + 200
    with gzip.open(tmp_path / f"{name}.1.gz", "rt", encoding="utf-8") as fh:
        assert all(json.loads(line)["event"] == "track_cued" for line in fh if line.strip())


# --- analysis cache -----------------------------------------------------------------


def _realistic_analysis(tmp_path, beats=520) -> an.TrackAnalysis:
    """About four minutes at 128 BPM: the shape of a real library entry."""
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"placeholder")
    rng = np.random.default_rng(3)
    beat_times = (np.arange(beats) * 60.0 / 128.0 + 0.137).tolist()
    ta = an.TrackAnalysis(
        track_id="0123456789abcdef", path=str(audio), title="track",
        duration_s=beats * 60.0 / 128.0, bpm=128.0, beats=beat_times,
        downbeats=beat_times[::4], key_name="A minor", camelot="8A",
        beat_rms=rng.uniform(0.01, 0.3, beats).tolist(), energy=0.2,
        grid_confidence=0.5, mix_in=15.0, mix_out=220.0,
    )
    ta.hot_cues = an.auto_hot_cues(ta)
    return ta


def _v5_text(ta: an.TrackAnalysis) -> str:
    """Exactly how a v5 sidecar was written: everything inline, indent=1."""
    d = {k: v for k, v in json.loads(ta.to_json()).items()}
    d["analysis_version"] = 5
    return json.dumps(d, indent=1)


def test_a_cache_entry_is_json_scalars_plus_float32_arrays(tmp_path):
    ta = _realistic_analysis(tmp_path)
    json_path = an.write_sidecar(ta, tmp_path / "cache")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for field in an.ARRAY_FIELDS:
        assert field not in data
    assert data["hot_cues"] == ta.hot_cues

    loaded = an.load_cached(ta.track_id, tmp_path / "cache")
    assert loaded is not None
    for field in an.ARRAY_FIELDS:
        original = np.asarray(getattr(ta, field))
        restored = np.asarray(getattr(loaded, field))
        assert restored.shape == original.shape
        # float32: exact to well under a millisecond on a four-minute timestamp.
        assert np.max(np.abs(restored - original)) < 1e-4


def test_the_array_cache_is_at_least_60_percent_smaller_than_v5_json(tmp_path):
    ta = _realistic_analysis(tmp_path)
    before = len(_v5_text(ta).encode("utf-8"))
    json_path = an.write_sidecar(ta, tmp_path / "cache")
    after = json_path.stat().st_size + json_path.with_suffix(".npz").stat().st_size
    assert after <= before * 0.40, f"v5 {before} bytes, v6 {after} bytes"


def test_a_v5_sidecar_is_upgraded_in_place_without_audio(tmp_path, monkeypatch):
    ta = _realistic_analysis(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{ta.track_id}.json").write_text(_v5_text(ta), encoding="utf-8")
    monkeypatch.setattr(an, "analyze_file", lambda *a, **k: pytest.fail("audio was read"))

    crate = an.load_crate(cache)
    assert len(crate) == 1
    assert (cache / f"{ta.track_id}.npz").exists()
    rewritten = json.loads((cache / f"{ta.track_id}.json").read_text(encoding="utf-8"))
    assert rewritten["analysis_version"] == an.ANALYSIS_VERSION
    assert "beats" not in rewritten
    assert crate[0].bpm == ta.bpm
    assert np.allclose(crate[0].beats, ta.beats, atol=1e-4)


def test_an_array_file_at_the_wrong_version_is_not_read(tmp_path):
    ta = _realistic_analysis(tmp_path)
    cache = tmp_path / "cache"
    json_path = an.write_sidecar(ta, cache)
    with np.load(json_path.with_suffix(".npz")) as data:
        arrays = {k: data[k] for k in an.ARRAY_FIELDS}
    np.savez_compressed(
        json_path.with_suffix(".npz"), analysis_version=np.int32(an.ANALYSIS_VERSION - 1),
        **arrays,
    )
    assert an.load_cached(ta.track_id, cache) is None


def test_a_sidecar_whose_array_file_is_missing_is_not_read(tmp_path):
    ta = _realistic_analysis(tmp_path)
    cache = tmp_path / "cache"
    json_path = an.write_sidecar(ta, cache)
    json_path.with_suffix(".npz").unlink()
    assert an.load_cached(ta.track_id, cache) is None


# --- stretched buffers ------------------------------------------------------------


def test_stretched_copies_no_deck_uses_are_freed(monkeypatch):
    made: dict[str, weakref.ref] = {}

    def fake_stretch(track, rate):
        audio = np.ones((SAMPLE_RATE, CHANNELS), dtype=np.float32)
        made[track.analysis.track_id] = weakref.ref(audio)
        return types.SimpleNamespace(analysis=track.analysis, audio=audio, is_stretched=True)

    monkeypatch.setattr(engine_module, "time_stretch", fake_stretch)
    worker = engine_module._StretchWorker(cache_size=4)
    try:
        tracks = []
        for tid in ("live", "cued", "gone"):
            track = loud_track(220.0, 120.0, 2.0)
            track.analysis.track_id = tid
            tracks.append(track)
            worker.request(track, 1.03)
        deadline = time.time() + 5.0
        while worker.completed < 3 and time.time() < deadline:
            time.sleep(0.01)
        assert worker.completed == 3

        assert worker.retain({"live", "cued"}) == 1
        gc.collect()
        assert made["gone"]() is None, "the unused stretched buffer is still alive"
        assert made["live"]() is not None and made["cued"]() is not None
    finally:
        worker.stop()


def test_nothing_stretched_is_ever_written_to_disk():
    import inspect

    source = inspect.getsource(engine_module._StretchWorker)
    for writer in ("open(", "np.save", "write_bytes", "sf.write", "tofile"):
        assert writer not in source, f"the stretch worker writes to disk via {writer}"


def test_the_autopilot_releases_stretches_when_a_transition_completes(session, monkeypatch):
    calls: list[set] = []
    monkeypatch.setattr(
        session.engine, "release_stretches", lambda keep: calls.append(set(keep)) or 0
    )
    incoming = session.engine.deck(session.cued_deck())
    other = loud_track(330.0, 124.0, 30.0)
    other.analysis.track_id = "incoming"
    session.engine.submit(LoadTrack(deck=incoming.name, track=other, play=True))
    drive(session.engine, 2, BLOCK)
    session.engine.deck(session.live_deck).playing = False
    session._transition_armed = True

    session._autopilot_tick()
    assert calls, "a completed transition must release unused stretched copies"
    assert "incoming" in calls[-1], "the new live deck's track is kept"


# --- clean ----------------------------------------------------------------------


def _clean_tree(root: Path) -> dict[str, Path]:
    paths = {
        "cache": root / "cache", "logs": root / "logs",
        "recordings": root / "recordings", "renders": root / "renders",
    }
    for p in paths.values():
        p.mkdir()
    (paths["renders"] / "a__b.wav").write_bytes(b"x" * 1000)
    (paths["renders"] / "_tmp_123_20260101_000000").mkdir()
    (paths["renders"] / "_tmp_123_20260101_000000" / "x.wav").write_bytes(b"x" * 500)
    (paths["recordings"] / "set_2026-01-01_120000.wav").write_bytes(b"x" * 2000)
    (paths["logs"] / "session_20260101_120000.jsonl").write_text("{}\n", encoding="utf-8")
    (paths["cache"] / "old.json").write_text(json.dumps({"analysis_version": 3}), encoding="utf-8")
    (paths["cache"] / "orphan.npz").write_bytes(b"PK")
    return paths


def _clean_args(paths):
    return cli.build_parser().parse_args([
        "clean", "--cache", str(paths["cache"]), "--logs", str(paths["logs"]),
        "--record-dir", str(paths["recordings"]), "--renders", str(paths["renders"]),
    ])


def test_clean_lists_everything_with_sizes_and_deletes_nothing_without_yes(
    tmp_path, monkeypatch, capsys
):
    paths = _clean_tree(tmp_path)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "no")
    assert cli.cmd_clean(_clean_args(paths)) == 0
    out = capsys.readouterr().out
    for name in ("a__b.wav", "_tmp_123", "set_2026-01-01_120000.wav", "old.json", "orphan.npz"):
        assert name in out, f"{name} was not listed"
    assert "MB" in out and "Nothing deleted" in out
    assert (paths["renders"] / "a__b.wav").exists()
    assert (paths["recordings"] / "set_2026-01-01_120000.wav").exists()
    assert (paths["logs"] / "session_20260101_120000.jsonl").exists()


def test_clean_deletes_nothing_when_there_is_no_one_to_ask(tmp_path, monkeypatch):
    paths = _clean_tree(tmp_path)

    def no_terminal(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", no_terminal)
    assert cli.cmd_clean(_clean_args(paths)) == 0
    assert (paths["renders"] / "a__b.wav").exists()


def test_clean_deletes_what_it_listed_on_yes_and_nothing_else(tmp_path, monkeypatch):
    paths = _clean_tree(tmp_path)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "yes")
    assert cli.cmd_clean(_clean_args(paths)) == 0
    assert list(paths["renders"].iterdir()) == []
    assert not (paths["recordings"] / "set_2026-01-01_120000.wav").exists()
    assert not (paths["cache"] / "old.json").exists()
    assert not (paths["cache"] / "orphan.npz").exists()
    assert (paths["logs"] / "session_20260101_120000.jsonl").exists(), "current logs stay"


def test_analyze_after_clean_is_a_no_op(tmp_path, monkeypatch):
    library = tmp_path / "library"
    library.mkdir()
    for i, spec in enumerate(synth.CRATE_SPEC[:2]):
        synth.render_track(
            library / f"{spec['name']}.wav", bpm=spec["bpm"], root=spec["root"],
            minor=spec["minor"], bars=16, seed=i,
        )
    paths = _clean_tree(tmp_path)
    an.analyze_folder(library, paths["cache"])

    monkeypatch.setattr(builtins, "input", lambda prompt="": "yes")
    assert cli.cmd_clean(_clean_args(paths)) == 0

    entries = {p: p.stat().st_mtime_ns for p in paths["cache"].iterdir()}
    assert len(entries) == 4, "both tracks' JSON and array files survive clean"
    _, analyzed, skipped = an.analyze_folder(library, paths["cache"])
    assert analyzed == 0 and skipped == 2
    assert {p: p.stat().st_mtime_ns for p in paths["cache"].iterdir()} == entries


def test_the_same_track_under_two_names_does_not_rewrite_the_cache(tmp_path):
    """Regression from the real library: 97 files, 96 distinct contents.

    Both names hash to one entry, and the path refresh used to re-point it at
    whichever file came last, rewriting the cache on every single run.
    """
    library = tmp_path / "library"
    library.mkdir()
    spec = synth.CRATE_SPEC[0]
    original = library / "a_track.wav"
    synth.render_track(
        original, bpm=spec["bpm"], root=spec["root"], minor=spec["minor"], bars=16, seed=0
    )
    (library / "b_same_track_renamed.wav").write_bytes(original.read_bytes())

    cache = tmp_path / "cache"
    an.analyze_folder(library, cache)
    entries = {p: p.stat().st_mtime_ns for p in cache.iterdir()}
    assert len(entries) == 2, "one track, one JSON and one array file"

    _, analyzed, skipped = an.analyze_folder(library, cache)
    assert analyzed == 0 and skipped == 2
    assert {p: p.stat().st_mtime_ns for p in cache.iterdir()} == entries


def test_a_track_that_really_moved_still_gets_its_path_refreshed(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    spec = synth.CRATE_SPEC[0]
    old = library / "old_name.wav"
    synth.render_track(
        old, bpm=spec["bpm"], root=spec["root"], minor=spec["minor"], bars=16, seed=0
    )
    cache = tmp_path / "cache"
    (first,), _, _ = an.analyze_folder(library, cache)

    new = library / "new_name.wav"
    old.rename(new)
    an.analyze_folder(library, cache)
    reloaded = an.load_cached(first.track_id, cache)
    assert reloaded is not None and Path(reloaded.path) == new.resolve()

# --- what `clean` offers, and what it must never offer -------------------------


def _reclaimable(tmp_path):
    from djai import housekeeping as hk

    return hk.find_reclaimable(
        renders_dir=tmp_path / "renders", record_dir=tmp_path / "rec",
        log_dir=tmp_path / "logs", cache_dir=tmp_path / "cache",
        keep_recordings=3, keep_logs=3,
    )


def _stem_folder(cache: Path, track_id: str) -> Path:
    folder = cache / "stems" / track_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "drums.flac").write_bytes(b"x" * 2048)
    (folder / "stems.json").write_text("{}", encoding="utf-8")
    return folder


def test_stems_of_a_track_still_in_the_cache_are_never_offered(tmp_path):
    """Re-separating is minutes of GPU time; clean must not invite that."""
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    import json as _json

    from djai import analysis as an

    (cache / "abc123.json").write_text(
        _json.dumps({"analysis_version": an.ANALYSIS_VERSION}), encoding="utf-8"
    )
    folder = _stem_folder(cache, "abc123")
    assert folder not in [item.path for item in _reclaimable(tmp_path)]


def test_stems_of_a_track_that_is_gone_are_offered(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    folder = _stem_folder(cache, "deadbeef")
    items = {item.path: item for item in _reclaimable(tmp_path)}
    assert folder in items
    assert "no longer in the cache" in items[folder].reason
    assert items[folder].size > 0
