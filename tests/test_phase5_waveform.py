"""Phase 5: coloured multi-level waveforms, and the page that draws them.

THREADING CONTEXT: main thread (pytest). The band maths runs directly; the
UI server is exercised through FastAPI's test client; the page's view maths
(static/wavemath.js) runs under Node, the same file the browser loads.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from djai import waveform as wf
from djai.deck import SAMPLE_RATE
from tests.test_integration import session as _session_fixture

session = _session_fixture

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"


# --- the bands -----------------------------------------------------------------


def _three_tones() -> np.ndarray:
    """Two seconds each of 60 Hz, 1 kHz and 8 kHz: one per band, in order."""
    t = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
    parts = [0.5 * np.sin(2 * np.pi * f * t) for f in (60.0, 1000.0, 8000.0)]
    mono = np.concatenate(parts).astype(np.float32)
    return np.stack([mono, mono], axis=1)


@pytest.fixture(scope="module")
def tones():
    return wf.compute(_three_tones(), SAMPLE_RATE)


def test_each_band_carries_only_its_own_frequencies(tones):
    pps, fine = tones.levels[0]
    seg = int(2 * pps)
    margin = int(0.1 * pps)             # skip the filters' settling at each edge
    means = np.array([
        [fine[band, s * seg + margin:(s + 1) * seg - margin].mean() for s in range(3)]
        for band in range(3)
    ])
    for s in range(3):
        own = means[s, s]
        others = [means[b, s] for b in range(3) if b != s]
        assert own > 5 * max(others), f"segment {s}: bands {means[:, s]}"


def test_the_bands_line_up_with_time(tones):
    """The mid band switches on where the 1 kHz tone starts: at 2.00 s."""
    pps, fine = tones.levels[0]
    mid = fine[1].astype(float)
    onset = int(np.argmax(mid > 0.5 * mid.max()))
    assert onset / pps == pytest.approx(2.0, abs=0.01)


def test_the_levels_have_the_documented_densities(tones):
    fine_pps = SAMPLE_RATE / wf.FINE_BUCKET
    for (pps, arr), factor in zip(tones.levels, wf.LEVEL_FACTORS):
        assert pps == pytest.approx(fine_pps / factor)
        assert arr.dtype == np.uint8 and arr.shape[0] == 3
        assert arr.shape[1] == pytest.approx(6.0 * pps, abs=2)
    assert tones.overview.shape == (3, wf.OVERVIEW_POINTS)
    assert tones.duration_s == pytest.approx(6.0)
    assert int(tones.levels[0][1].max()) == 255, "normalised to the loudest band"


def test_the_cache_round_trips_and_checks_its_version(tones, tmp_path):
    wf.save(tones, tmp_path, "abc")
    back = wf.load(tmp_path, "abc")
    assert back is not None
    for (p1, a1), (p2, a2) in zip(tones.levels, back.levels):
        assert p1 == pytest.approx(p2) and np.array_equal(a1, a2)
    assert np.array_equal(back.overview, tones.overview)

    # Another version is never read, whatever its arrays hold.
    path = wf.cache_file(tmp_path, "old")
    with path.open("wb") as fh:
        np.savez_compressed(fh, version=np.array(wf.WAVEFORM_VERSION + 1),
                            sample_rate=np.array(SAMPLE_RATE), duration_s=np.array(6.0),
                            pps=np.array([1.0]), overview=tones.overview,
                            level0=tones.levels[0][1])
    assert wf.load(tmp_path, "old") is None
    wf.cache_file(tmp_path, "junk").write_bytes(b"not a zip")
    assert wf.load(tmp_path, "junk") is None


# --- the server ---------------------------------------------------------------


@pytest.fixture
def ui(session):
    from fastapi.testclient import TestClient

    from djai.ui_server import UIServer

    server = UIServer(session, intent_engine=None)
    return server, TestClient(server._app)


def test_the_waveform_payload_describes_bands_and_sections(ui, session):
    server, client = ui
    deck = session.engine.deck("a")
    payload = client.get("/api/waveform/a").json()
    assert payload["track_id"] == deck.track.analysis.track_id
    meta = payload["bands"]
    assert meta["version"] == wf.WAVEFORM_VERSION
    assert meta["bands"] == ["low", "mid", "high"]
    assert [lv["level"] for lv in meta["levels"]] == [0, 1, 2]
    assert isinstance(payload["sections"], list)
    for s in payload["sections"]:
        assert s["end_s"] > s["start_s"]
    # Cached to disk, under its own folder.
    assert wf.cache_file(session.cache_dir, deck.track.analysis.track_id).exists()


@pytest.mark.parametrize("level", ["overview", "0", "1", "2"])
def test_every_band_level_is_served_as_binary(ui, level):
    server, client = ui
    meta = client.get("/api/waveform/a").json()["bands"]
    r = client.get(f"/api/waveform/a/bands/{level}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    points = int(r.headers["X-Points"])
    assert len(r.content) == 3 * points
    expected = (meta["overview_points"] if level == "overview"
                else meta["levels"][int(level)]["points"])
    assert points == expected
    assert float(r.headers["X-Points-Per-Second"]) > 0


def test_a_bad_band_level_is_a_404(ui):
    server, client = ui
    assert client.get("/api/waveform/a/bands/9").status_code == 404
    assert client.get("/api/waveform/z/bands/0").status_code == 404


def test_the_page_reports_its_frame_timing(ui):
    server, client = ui
    ack = server.handle_action({"type": "frame_stats", "fps": 60.1, "p50_ms": 16.6,
                                "p95_ms": 17.9, "p99_ms": 21.0, "max_ms": 33.0,
                                "draw_p95_ms": 2.4, "frames": 600, "decks_playing": 2})
    assert ack["ok"]
    frames = server.state()["ui_frames"]
    assert frames["fps"] == pytest.approx(60.1)
    assert frames["decks_playing"] == 2


# --- the page -----------------------------------------------------------------


def test_the_page_draws_on_animation_frames_with_zoom():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert html.index("/static/wavemath.js") < html.index("/static/app.js")
    for d in ("a", "b"):
        assert f'id="wz-in-{d}"' in html and f'id="wz-out-{d}"' in html
    for needle in ("requestAnimationFrame", "WaveMath.zoomSpans", "WaveMath.view",
                   "WaveMath.inView", "/bands/", "frame_stats", "djaiFrameStats",
                   "SECTION_COLOUR", "hot_cues", "phrases", "playheadNow"):
        assert needle in js, needle


# --- the view maths, run under Node ---------------------------------------------------

NODE = shutil.which("node")


def _node(script: str) -> dict:
    path = json.dumps(str(STATIC / "wavemath.js"))
    code = f"const W = require({path});\n{script}"
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_zoom_runs_from_the_whole_track_to_exactly_one_bar():
    result = _node("""
      const spans = W.zoomSpans(215.3, 128);
      console.log(JSON.stringify({spans, bar: W.barSeconds(128)}));
    """)
    spans, bar = result["spans"], result["bar"]
    assert spans[0] == pytest.approx(215.3)
    assert spans[-1] == bar == pytest.approx(1.875)
    assert all(a > b for a, b in zip(spans, spans[1:])), "each step zooms in"
    assert all(a / b <= 2.0 + 1e-9 for a, b in zip(spans, spans[1:])), "no step more than 2x"


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_the_grid_is_aligned_at_one_bar_zoom():
    """At one bar, four beats are exactly a quarter of the width apart, and
    every line sits where its time says it should."""
    result = _node("""
      const bpm = 128, first = 0.37, width = 1000, dur = 200;
      const beats = []; for (let i = 0; first + i * 60 / bpm < dur; i++) beats.push(first + i * 60 / bpm);
      const spans = W.zoomSpans(dur, bpm);
      const bar = spans[spans.length - 1];
      const playhead = first + 20 * 60 / bpm + 0.1;
      const view = W.view(dur, bar, playhead);
      const [i0, i1] = W.inView(beats, view);
      const xs = beats.slice(i0, i1).map((t) => W.x(t, view, width));
      const inverse = xs.map((x, k) => W.t(x, view, width) - beats[i0 + k]);
      const full = W.view(dur, spans[0], playhead);
      console.log(JSON.stringify({
        view, xs, inverse, times: beats.slice(i0, i1), full,
        finest: W.levelFor(width / bar, [{points_per_second: 25}, {points_per_second: 100}, {points_per_second: 400}]),
        coarse: W.levelFor(width / dur, [{points_per_second: 25}, {points_per_second: 100}, {points_per_second: 400}]),
        bucket: W.bucket(500, view, width, 400.9),
      }));
    """)
    view, xs, times = result["view"], result["xs"], result["times"]
    width, span = 1000.0, 1.875
    assert view[1] - view[0] == pytest.approx(span)
    assert len(xs) == 4, "one bar shows four beats"
    for x, t in zip(xs, times):
        assert x == pytest.approx((t - view[0]) / span * width, abs=1e-9)
        assert 0.0 <= x <= width
    gaps = np.diff(xs)
    assert np.allclose(gaps, width / 4, atol=1e-9), gaps
    assert max(abs(v) for v in result["inverse"]) < 1e-9, "pixel -> time is the inverse"
    assert result["full"] == [0, 200], "zoomed out is the whole track"
    assert result["finest"] == 2, "one bar needs the finest level"
    assert result["coarse"] == 0, "the whole track needs only the coarsest"
    p0, p1 = result["bucket"]
    assert p1 > p0
    assert p0 == int(np.floor((view[0] + 500 / width * span) * 400.9))
