"""Pre-flight: does it fail for the right reason, and say which?

THREADING CONTEXT: main thread (pytest). Device enumeration and the Ollama probe
are stubbed -- the point of these tests is the reporting, not the hardware.

The acceptance criterion names three failures specifically: missing device,
undecodable track, Ollama down. Each must exit non-zero AND say what is wrong,
because the output has to be actionable by someone standing in a booth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from djai import preflight
from djai.analysis import TrackAnalysis
from djai.deck import CHANNELS, SAMPLE_RATE


def make_analysis(tid: str, path: Path, grid: float = 1.0) -> TrackAnalysis:
    period = 60.0 / 125.0
    n = int(30 / period)
    return TrackAnalysis(
        track_id=tid,
        path=str(path),
        title=tid,
        duration_s=30.0,
        bpm=125.0,
        beats=[i * period for i in range(n)],
        downbeats=[i * period for i in range(0, n, 4)],
        key_name="A minor",
        camelot="8A",
        beat_rms=[0.5] * n,
        energy=0.5,
        grid_confidence=grid,
        mix_in=8.0,
        mix_out=24.0,
        mix_in_bar=2.0,
        mix_out_bar=6.0,
        mix_points_estimated=False,
    )


@pytest.fixture
def crate(tmp_path):
    """Three real, decodable WAV files and their analyses."""
    out = []
    for i, tid in enumerate(("one", "two", "three")):
        p = tmp_path / f"{tid}.wav"
        t = np.arange(SAMPLE_RATE * 2, dtype=np.float32) / SAMPLE_RATE
        tone = (0.4 * np.sin(2 * np.pi * (220 + 110 * i) * t)).astype(np.float32)
        sf.write(str(p), np.repeat(tone[:, None], CHANNELS, axis=1), SAMPLE_RATE)
        out.append(make_analysis(tid, p))
    return out


# --- devices ------------------------------------------------------------------


def test_missing_device_fails_and_says_so(monkeypatch):
    monkeypatch.setattr("djai.engine.list_output_devices", lambda: [])
    check = preflight.check_devices()
    assert not check.ok
    assert "no output device" in check.detail


def test_devices_that_enumerate_but_will_not_open_fail(monkeypatch):
    monkeypatch.setattr(
        "djai.engine.list_output_devices",
        lambda: [(0, "Ghost Interface", "WASAPI", 2)],
    )
    monkeypatch.setattr("djai.engine._can_open", lambda d, b: False)
    check = preflight.check_devices()
    assert not check.ok
    assert "none opens" in check.detail
    assert "Ghost Interface" in check.detail, "say WHICH device"


def test_a_working_device_passes_and_warns_about_no_cue(monkeypatch):
    monkeypatch.setattr(
        "djai.engine.list_output_devices",
        lambda: [(0, "Speakers", "WASAPI", 2)],
    )
    monkeypatch.setattr("djai.engine._can_open", lambda d, b: True)
    check = preflight.check_devices()
    assert check.ok
    assert any("cue" in w for w in check.warnings)


def test_a_four_channel_device_does_not_warn_about_cue(monkeypatch):
    monkeypatch.setattr(
        "djai.engine.list_output_devices",
        lambda: [(3, "Scarlett 4i4", "WASAPI", 4)],
    )
    monkeypatch.setattr("djai.engine._can_open", lambda d, b: True)
    check = preflight.check_devices()
    assert check.ok
    assert not any("cue" in w for w in check.warnings)


def test_enumeration_blowing_up_is_a_failure_not_a_crash(monkeypatch):
    def boom():
        raise OSError("PortAudio exploded")

    monkeypatch.setattr("djai.engine.list_output_devices", boom)
    check = preflight.check_devices()
    assert not check.ok and "PortAudio exploded" in check.detail


# --- crate --------------------------------------------------------------------


def test_an_empty_cache_fails_with_the_command_to_fix_it(tmp_path):
    check, crate = preflight.check_crate(tmp_path / "nothing")
    assert not check.ok and crate == []
    assert "djai analyze" in check.detail


def test_a_track_whose_file_has_gone_fails(tmp_path, crate, monkeypatch):
    crate[1].path = str(tmp_path / "vanished.wav")
    monkeypatch.setattr("djai.preflight.load_crate", lambda d: crate)
    check, _ = preflight.check_crate(tmp_path)
    assert not check.ok
    assert "no longer exist" in check.detail and "two" in check.detail


def test_a_healthy_crate_passes(tmp_path, crate, monkeypatch):
    monkeypatch.setattr("djai.preflight.load_crate", lambda d: crate)
    check, loaded = preflight.check_crate(tmp_path)
    assert check.ok and len(loaded) == 3
    assert any("only 3 tracks" in w for w in check.warnings)


# --- grid ---------------------------------------------------------------------


def test_a_weak_beat_grid_fails_and_names_the_worst(tmp_path, crate):
    crate[2].grid_confidence = 0.11
    check = preflight.check_grid(crate, threshold=0.5)
    assert not check.ok
    assert "three" in check.detail and "0.11" in check.detail


def test_a_confident_grid_passes(crate):
    check = preflight.check_grid(crate, threshold=0.5)
    assert check.ok and "all 3" in check.detail


def test_estimated_mix_points_warn_without_failing(crate):
    crate[0].mix_points_estimated = True
    check = preflight.check_grid(crate, threshold=0.5)
    assert check.ok
    assert any("estimated" in w for w in check.warnings)


# --- decode -------------------------------------------------------------------


def test_an_undecodable_track_fails_and_names_it(tmp_path, crate):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"this is not a wav file")
    crate[1].path = str(broken)

    check = preflight.check_decode(crate, limit=0)
    assert not check.ok
    assert "will not decode" in check.detail
    assert "two" in check.detail, "say WHICH track"


def test_decodable_tracks_pass(crate):
    check = preflight.check_decode(crate, limit=0)
    assert check.ok and "3 track(s) decode" in check.detail


def test_a_silent_track_warns_without_failing(tmp_path, crate):
    silent = tmp_path / "silent.wav"
    sf.write(str(silent), np.zeros((SAMPLE_RATE, CHANNELS), np.float32), SAMPLE_RATE)
    crate[0].path = str(silent)
    check = preflight.check_decode(crate, limit=0)
    assert check.ok
    assert any("silence" in w for w in check.warnings)


def test_a_decode_limit_says_it_only_checked_some(crate):
    check = preflight.check_decode(crate, limit=1)
    assert check.ok
    assert any("only checked 1 of 3" in w for w in check.warnings)


def test_decode_reports_progress(crate):
    seen = []
    preflight.check_decode(crate, limit=0, progress=lambda i, n, t: seen.append(t))
    assert seen == ["one", "two", "three"]


# --- ollama -------------------------------------------------------------------


class FakeIntent:
    def __init__(self, ok, detail):
        self._ok = ok
        self._detail = detail
        self.model = "llama3.1:8b"
        self.base_url = "http://localhost:11434"
        self.closed = False

    def warmup(self):
        return self._ok, self._detail

    def close(self):
        self.closed = True


def test_ollama_down_fails_and_says_how_to_proceed(monkeypatch):
    monkeypatch.setattr(
        "djai.intent.IntentEngine",
        lambda *a, **k: FakeIntent(False, "connection refused"),
    )
    check = preflight.check_ollama(required=True)
    assert not check.ok
    assert "connection refused" in check.detail
    assert "--no-llm" in check.detail, "tell the operator the way out"


def test_ollama_down_is_not_a_failure_when_it_is_not_required(monkeypatch):
    monkeypatch.setattr(
        "djai.intent.IntentEngine",
        lambda *a, **k: FakeIntent(False, "connection refused"),
    )
    check = preflight.check_ollama(required=False)
    assert check.ok
    assert any("rule-based" in w for w in check.warnings)


def test_ollama_up_passes(monkeypatch):
    monkeypatch.setattr(
        "djai.intent.IntentEngine", lambda *a, **k: FakeIntent(True, "ready in 2.1s")
    )
    check = preflight.check_ollama()
    assert check.ok and "ready" in check.detail


def test_a_raising_warmup_is_a_failure_not_a_crash(monkeypatch):
    class Exploding(FakeIntent):
        def warmup(self):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(
        "djai.intent.IntentEngine", lambda *a, **k: Exploding(False, "")
    )
    check = preflight.check_ollama(required=True)
    assert not check.ok and "kaboom" in check.detail


# --- disk ---------------------------------------------------------------------


def test_a_full_disk_fails_with_the_numbers(tmp_path, monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(
        preflight.shutil, "disk_usage", lambda p: _shutil._ntuple_diskusage(100, 99, 1)
    )
    check = preflight.check_disk(tmp_path, min_mb=3000)
    assert not check.ok
    assert "below the 3,000 MB minimum" in check.detail


def test_enough_disk_passes(tmp_path):
    check = preflight.check_disk(tmp_path, min_mb=1)
    assert check.ok and "free" in check.detail


def test_an_unwritable_log_dir_fails(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise PermissionError("read-only volume")

    monkeypatch.setattr(Path, "mkdir", boom)
    check = preflight.check_disk(tmp_path / "logs", min_mb=1)
    assert not check.ok and "read-only volume" in check.detail


# --- the whole run ------------------------------------------------------------


@pytest.fixture
def healthy(monkeypatch, crate):
    monkeypatch.setattr(
        "djai.engine.list_output_devices", lambda: [(0, "Speakers", "WASAPI", 4)]
    )
    monkeypatch.setattr("djai.engine._can_open", lambda d, b: True)
    monkeypatch.setattr("djai.preflight.load_crate", lambda d: crate)
    monkeypatch.setattr(
        "djai.intent.IntentEngine", lambda *a, **k: FakeIntent(True, "ready")
    )
    return crate


def test_a_healthy_rig_passes_every_check(tmp_path, healthy):
    report = preflight.run_preflight(tmp_path, tmp_path / "logs")
    assert report.ok, [str(c) for c in report.failures]
    assert [c.name for c in report.checks] == list(preflight.CHECK_NAMES)


def test_each_named_failure_is_reported_specifically(tmp_path, healthy, monkeypatch):
    """The acceptance criterion, one failure at a time."""
    # 1. missing device
    monkeypatch.setattr("djai.engine.list_output_devices", lambda: [])
    report = preflight.run_preflight(tmp_path, tmp_path / "logs")
    assert not report.ok
    assert [c.name for c in report.failures] == ["devices"]
    assert "no output device" in report.failures[0].detail


def test_an_undecodable_track_fails_the_whole_run(tmp_path, healthy, monkeypatch):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"nope")
    healthy[0].path = str(broken)
    report = preflight.run_preflight(tmp_path, tmp_path / "logs")
    assert not report.ok
    assert "decode" in [c.name for c in report.failures]
    assert "one" in dict((c.name, c.detail) for c in report.failures)["decode"]


def test_ollama_down_fails_the_whole_run_unless_no_llm(tmp_path, healthy, monkeypatch):
    monkeypatch.setattr(
        "djai.intent.IntentEngine", lambda *a, **k: FakeIntent(False, "refused")
    )
    report = preflight.run_preflight(tmp_path, tmp_path / "logs")
    assert not report.ok and [c.name for c in report.failures] == ["ollama"]

    relaxed = preflight.run_preflight(
        tmp_path, tmp_path / "logs", require_ollama=False
    )
    assert relaxed.ok, [str(c) for c in relaxed.failures]


def test_an_empty_crate_does_not_pretend_the_later_checks_ran(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "djai.engine.list_output_devices", lambda: [(0, "Speakers", "WASAPI", 2)]
    )
    monkeypatch.setattr("djai.engine._can_open", lambda d, b: True)
    monkeypatch.setattr("djai.preflight.load_crate", lambda d: [])
    monkeypatch.setattr(
        "djai.intent.IntentEngine", lambda *a, **k: FakeIntent(True, "ready")
    )
    report = preflight.run_preflight(tmp_path, tmp_path / "logs")
    assert not report.ok
    by_name = {c.name: c for c in report.checks}
    assert "not reached" in by_name["grid"].detail
    assert "not reached" in by_name["decode"].detail


# --- the command --------------------------------------------------------------


def test_the_command_exits_non_zero_on_failure(tmp_path, healthy, monkeypatch, capsys):
    from djai import cli

    monkeypatch.setattr("djai.engine.list_output_devices", lambda: [])
    args = cli.build_parser().parse_args(
        ["preflight", "--cache", str(tmp_path), "--logs", str(tmp_path / "logs"),
         "--record-dir", str(tmp_path / "rec")]
    )
    assert cli.cmd_preflight(args) == 1
    out = capsys.readouterr().out
    assert "PREFLIGHT FAILED" in out and "no output device" in out


def test_the_command_exits_zero_when_healthy(tmp_path, healthy, capsys):
    from djai import cli

    args = cli.build_parser().parse_args(
        ["preflight", "--cache", str(tmp_path), "--logs", str(tmp_path / "logs"),
         "--record-dir", str(tmp_path / "rec")]
    )
    assert cli.cmd_preflight(args) == 0
    assert "PREFLIGHT PASSED" in capsys.readouterr().out


def test_the_command_honours_no_llm(tmp_path, healthy, monkeypatch, capsys):
    from djai import cli

    monkeypatch.setattr(
        "djai.intent.IntentEngine", lambda *a, **k: FakeIntent(False, "refused")
    )
    argv = [
        "preflight", "--cache", str(tmp_path), "--logs", str(tmp_path / "logs"),
        "--record-dir", str(tmp_path / "rec"),
    ]
    assert cli.cmd_preflight(cli.build_parser().parse_args(argv)) == 1
    assert cli.cmd_preflight(cli.build_parser().parse_args(argv + ["--no-llm"])) == 0

