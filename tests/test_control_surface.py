"""Phase 5.2 (SPEC §12): the control surface.

No decision logic lives in the UI: these tests check that it shows what the
system is doing and that one action takes over, never that it decides.

THREADING CONTEXT: main thread (pytest), except the polling test, which runs
``UIServer.state()`` on a second thread at 20 Hz as uvicorn would.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
from fastapi.testclient import TestClient

from djai import cli
from djai.commands import CancelTransition
from djai.deck import SAMPLE_RATE
from djai.ui_server import STATE_HZ, UIServer
from tests.test_live_robustness import _events, _run, _to_cue_window, live  # noqa: F401
from tests.test_song_cue import BAR, _drive


def _ui(s, intent_engine=None):
    server = UIServer(s, intent_engine=intent_engine)
    return server, TestClient(server._app)


def test_state_carries_the_set_panel(live):
    s = live
    server, _ = _ui(s)
    _run(s, 8 * BAR / SAMPLE_RATE)
    st = server.state()
    assert st["mode"] == "autonomous" and st["mc"] is False
    assert len(st["energy_history"]) >= 4, "a reading per finished bar"
    assert all(isinstance(v, float) for v in st["energy_history"])
    assert {"title": "Shaky Grid", "why": "grid 0.02"} in st["quarantined"]
    assert st["plan"] is None
    cli.handle_override(s, "plan")
    assert server.state()["plan"]["near"], "the upcoming plan, as titles"
    json.dumps(server.state())


def test_explain_reads_the_log_not_the_present(live):
    s = live
    assert s.cue_next(origin="test")
    _drive(s, 4096)
    assert s.arm_transition(origin="test") is not None
    cued = _events(s, "track_cued")[-1]
    armed = _events(s, "transition_armed")[-1]
    before = cli.handle_override(s, "explain")
    assert cued["decision_id"] in before and cued["action"] in before
    assert armed["style_rule"] in before
    # Change the present: the explanation is what was logged, so it does not.
    object.__setattr__(s._cued.analysis, "title", "Renamed After The Fact")
    s.energy_direction = 1.0
    assert cli.handle_override(s, "explain") == before
    assert "Renamed" not in before


def test_a_parse_failure_is_visible(live):
    import httpx

    from djai.intent import IntentEngine

    s = live
    server, _ = _ui(s)
    reply = server.handle_text("blorp the wibble")
    assert reply["parse_failure"], reply
    assert server.state()["last_parse_failure"]["text"] == "blorp the wibble"

    engine = IntentEngine()
    engine._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    engine.available = False
    server, _ = _ui(s, engine)
    reply = server.handle_text("blorp the wibble")
    assert reply["parse_failure"] and "unavailable" in reply["parse_failure"], reply
    assert server.handle_text("more energy")["parse_failure"] is None


def test_cueing_from_the_suggestions_panel_takes_one_action(live):
    s = live
    server, client = _ui(s)
    pick = server.state()["suggestions"][0]
    r = client.post(f"/override/cue?arg={pick['track_id']}%20next")   # the CUE button
    assert r.json()["ok"], r.json()
    assert s.cue_queue[0]["track_id"] == pick["track_id"]


def _bars_until(s, done) -> float:
    start = s.engine.frames_played
    for _ in range(int(4 * BAR / 512)):
        if done():
            break
        _drive(s, 512, block=512)
    assert done(), "never landed"
    return (s.engine.frames_played - start) / BAR


def test_ui_overrides_land_within_one_bar(live, capsys):
    s = live
    server, client = _ui(s)
    landed = {}

    client.post("/override/mc?arg=on")
    landed["mc"] = _bars_until(s, lambda: s.engine._master_gain < 1.0)
    client.post("/override/mc?arg=off")
    _drive(s, int(2 * BAR))

    assert s.cue_next(origin="test")
    _drive(s, 4096)
    assert s.arm_transition(origin="test") is not None
    client.post("/override/freeze")
    landed["hold"] = _bars_until(s, lambda: not s._transition_armed and s.frozen)

    client.post("/override/resume")
    _to_cue_window(s)
    s.freeze(False)
    _run(s, 200, until=lambda: s.engine.transition_active)
    cancels = []
    real = s.engine.submit
    s.engine.submit = lambda c: (cancels.append(c) if isinstance(c, CancelTransition)
                                 else None) or real(c)
    client.post("/override/cancel")
    landed["cancel"] = _bars_until(s, lambda: not s.engine.transition_active)
    assert cancels

    client.post("/override/mode?arg=assisted")
    landed["mode"] = _bars_until(s, lambda: server.state()["mode"] == "assisted")
    for what, bars in landed.items():
        assert bars <= 1.0, f"{what} took {bars:.2f} bars"
    with capsys.disabled():
        print("\n    UI overrides landed in: " +
              ", ".join(f"{k} {v:.2f} bars" for k, v in landed.items()))


def test_ui_polling_has_no_timing_impact(live, capsys):
    """The callback's own time, with and without a 20 Hz state poll beside it."""
    s = live
    server, _ = _ui(s)
    block = 512
    buf = np.zeros((block, 2), dtype=np.float32)

    def blocks(n: int) -> np.ndarray:
        took = np.empty(n)
        for i in range(n):
            s.scheduler.tick(s.engine.frames_played + block)
            t0 = time.perf_counter()
            s.engine.callback(buf, block, None, None)
            took[i] = time.perf_counter() - t0
            time.sleep(0.001)          # the room a real stream leaves between blocks
        return took * 1000

    blocks(200)
    quiet = blocks(1500)
    stop = threading.Event()

    def poll():
        while not stop.wait(1.0 / STATE_HZ):
            json.dumps(server.state())
    t = threading.Thread(target=poll, daemon=True)
    t.start()
    try:
        polled = blocks(1500)
    finally:
        stop.set()
        t.join()
    budget = block / SAMPLE_RATE * 1000
    q99, p99 = np.percentile(quiet, 99), np.percentile(polled, 99)
    with capsys.disabled():
        print(f"\n    callback p99 {q99:.3f} ms quiet, {p99:.3f} ms under a 20 Hz poll; "
              f"worst {polled.max():.3f} ms; budget {budget:.1f} ms")
    assert polled.max() < budget / 2, f"worst block {polled.max():.2f} ms"
    assert p99 < q99 + 0.5, f"p99 {q99:.3f} -> {p99:.3f} ms"
    assert s.engine.underruns == 0
