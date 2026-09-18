"""The web UI: state feed, panic path, and isolation from the audio engine.

THREADING CONTEXT: main thread (pytest). FastAPI's ``TestClient`` drives the app
in-process, so uvicorn's daemon thread never starts; the audio callback is
driven directly. No device, no browser, no sockets on the machine.

The property that matters most here is a negative one: nothing the UI does --
not a disconnect, not a dead socket, not a wedged model -- may perturb audio,
and the panic buttons must land whether or not any of that is working.
"""

from __future__ import annotations

import inspect
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from djai import ui_server
from djai.commands import IMMEDIATE, SetGain, StartTransition
from djai.deck import CHANNELS, SAMPLE_RATE, TransportState
from djai.ui_server import STATE_HZ, WAVEFORM_POINTS, UIServer
from tests.test_engine import drive
from tests.test_hardening import idle_session  # noqa: F401 - a fixture
from tests.test_integration import log_events
from tests.test_integration import session as _session_fixture

#: Reuse the integration Session wholesale: a real engine, real supervisor and
#: real scheduler wired to synthetic audio, with no device and no threads.
session = _session_fixture

BLOCK = 512


@pytest.fixture
def ui(session):
    """A server bound to that Session, with no local model."""
    server = UIServer(session, intent_engine=None)
    return server, TestClient(server._app), session


# --- state feed --------------------------------------------------------------


def test_state_carries_everything_the_strip_shows(ui):
    server, _client, _session = ui
    s = server.state()

    for key in ("master_bpm", "live_deck", "decks", "drift_ms", "queue",
                "transition", "engine", "llm", "notices"):
        assert key in s
    assert set(s["decks"]) == {"a", "b"}
    for key in ("title", "track_id", "bpm", "camelot", "position_s",
                "position_bars", "gain", "eq", "rate", "mix_in_s", "mix_out_s",
                "stretched", "live", "playing"):
        assert key in s["decks"]["a"]
    assert len(s["decks"]["a"]["eq"]) == 3
    json.dumps(s)  # it has to survive the wire


def test_state_follows_the_playhead(ui):
    server, _client, session = ui
    first = server.state()["decks"]["a"]["position_s"]
    drive(session.engine, 40, BLOCK)
    assert server.state()["decks"]["a"]["position_s"] > first


def test_state_endpoint_and_socket_payload_agree(ui):
    server, client, _session = ui
    r = client.get("/api/state")
    assert r.status_code == 200
    assert set(r.json()) == set(server.state())


def test_a_state_frame_is_cheap_enough_for_20hz(ui):
    """The acceptance bar is 100 ms; the frame itself must not eat that budget.

    This is also the isolation argument: the feed runs on uvicorn's thread and
    holds the GIL for however long it takes, so 'fast' is what keeps it from
    showing up as an underrun.
    """
    server, _client, _session = ui
    server.state()  # warm
    started = time.perf_counter()
    for _ in range(200):
        server.state()
    per_frame_ms = (time.perf_counter() - started) / 200 * 1000
    budget_ms = 1000.0 / STATE_HZ
    assert per_frame_ms < budget_ms / 5, (
        f"{per_frame_ms:.2f} ms per frame against a {budget_ms:.0f} ms budget"
    )


def test_drift_is_reported_per_deck(ui):
    server, _client, session = ui
    session.supervisor.check_drift()
    drive(session.engine, 4, BLOCK)
    session.supervisor.check_drift()

    drift = server.state()["drift_ms"]
    assert set(drift) == {"a", "b"}
    live = drift[session.live_deck]
    assert live is not None, "the live deck should have a measurable offset"
    assert abs(live) < 50.0, f"{live:.1f} ms of drift on a deck that is in sync"


def test_notices_reach_the_ui_without_starving_the_repl(ui):
    """Fan-out, not a shared queue: both consumers see every line."""
    server, _client, session = ui
    session.notify("[test] hello")
    session.notify("[test] world")

    ui_saw = server.state()["notices"]
    repl_saw = session.drain_notices()
    for line in ("[test] hello", "[test] world"):
        assert line in ui_saw
        assert line in repl_saw
    assert server.state()["notices"] == [], "notices should drain once per frame"


def test_a_broken_notice_listener_cannot_silence_the_repl(ui):
    _server, _client, session = ui
    session.add_notice_listener(lambda _msg: 1 / 0)
    session.notify("[test] still gets through")
    assert "[test] still gets through" in session.drain_notices()


# --- waveform ----------------------------------------------------------------


def test_waveform_has_peaks_and_the_musical_landmarks(ui):
    server, _client, _session = ui
    w = server.waveform("a")

    assert w["track_id"]
    assert 0 < len(w["peaks"]) <= WAVEFORM_POINTS
    assert all(0.0 <= p <= 1.0 for p in w["peaks"])
    assert w["downbeats"] and w["phrases"]
    assert w["duration_s"] > 0
    assert w["mix_out"] > w["mix_in"] >= 0


def test_waveform_is_computed_once_per_track(ui):
    server, _client, _session = ui
    first = server.waveform("a")
    assert server.waveform("a") is first


def test_waveform_endpoint_rejects_an_unknown_deck(ui):
    _server, client, _session = ui
    assert client.get("/api/waveform/a").status_code == 200
    assert client.get("/api/waveform/z").status_code == 404


def test_peaks_are_normalised_bounded_and_safe_on_silence():
    rng = np.random.default_rng(0)
    for scale in (0.01, 0.9):
        audio = (rng.standard_normal((SAMPLE_RATE * 3, CHANNELS)) * scale)
        peaks = ui_server._peaks(audio.astype(np.float32))
        assert len(peaks) <= WAVEFORM_POINTS
        assert max(peaks) == pytest.approx(1.0, abs=1e-3)
        assert min(peaks) >= 0.0

    assert ui_server._peaks(np.zeros((0, CHANNELS), np.float32)) == []
    # Digital silence must not divide by zero.
    assert set(ui_server._peaks(np.zeros((SAMPLE_RATE, CHANNELS), np.float32))) == {0.0}


# --- the panic path ----------------------------------------------------------


def test_panic_endpoints_put_a_command_on_the_engine_queue(ui):
    _server, client, session = ui
    for action in ("cut", "killbass", "stop"):
        before = session.engine.queue.qsize()
        r = client.post(f"/panic/{action}")
        assert r.status_code == 200 and r.json()["ok"] is True
        assert session.engine.queue.qsize() == before + 1
        assert len(session.scheduler) == 0, "panic must not go via the scheduler"


def test_panic_rejects_an_unknown_action(ui):
    _server, client, _session = ui
    assert client.post("/panic/launch_missiles").json()["ok"] is False


def test_the_panic_path_references_no_socket_no_scheduler_no_model():
    """A structural guard: the bypass has to stay a bypass.

    Checked against the compiled names rather than the source text so the
    docstring's prose ('no scheduler, no validation') does not fool it.
    """
    names = set(UIServer.panic.__code__.co_names)
    for forbidden in ("intent_engine", "interpret", "scheduler", "validate",
                      "supervisor", "clients"):
        assert forbidden not in names, f"panic touches {forbidden}"
    assert {"Cut", "KillBass", "Stop", "submit"} <= names


def test_panic_cuts_audio_in_under_100ms_with_a_transition_in_flight(ui):
    """The acceptance case, and the one that is easy to get wrong."""
    _server, client, session = ui
    engine = session.engine

    assert session.cue_next(0.0)
    drive(engine, 2, BLOCK)  # the audio thread applies the LoadTrack
    engine.submit(
        StartTransition(
            from_deck="a",
            to_deck="b",
            total_frames=SAMPLE_RATE * 40,
            execute_at=IMMEDIATE,
        )
    )
    drive(engine, 4, BLOCK)
    assert engine.transition_active
    assert float(np.max(np.abs(drive(engine, 8, BLOCK)))) > 0.1

    started = time.perf_counter()
    assert client.post("/panic/cut").json()["ok"] is True
    request_ms = (time.perf_counter() - started) * 1000

    buf = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    blocks = 0
    for _ in range(32):
        engine.callback(buf, BLOCK, None, None)
        blocks += 1
        if float(np.max(np.abs(buf))) < 1e-6:
            break
    else:
        pytest.fail("the mix never went silent after CUT")

    audio_ms = blocks * BLOCK / SAMPLE_RATE * 1000
    assert audio_ms < 100.0, f"cut took {audio_ms:.1f} ms of audio mid-transition"
    assert request_ms < 100.0, f"the POST itself took {request_ms:.1f} ms"
    assert not engine.transition_active, "CUT must abandon the running transition"


def test_the_transition_is_reported_in_frames_not_the_configured_bar_count(ui):
    """The canvas needs the real length, not TRANSITION_BARS.

    The engine derives ``transition_progress_bars`` from the *configured*
    TRANSITION_BARS whatever the actual length is, so a held or stretched blend
    would draw in the wrong place if the UI trusted a bar count.
    """
    _server, _client, session = ui
    server = _server
    engine = session.engine

    assert server.state()["transition"]["active"] is False

    assert session.cue_next(0.0)
    drive(engine, 2, BLOCK)
    total = SAMPLE_RATE * 40
    engine.submit(
        StartTransition(from_deck="a", to_deck="b", total_frames=total,
                        execute_at=IMMEDIATE)
    )
    drive(engine, 20, BLOCK)

    tr = server.state()["transition"]
    assert tr["active"] is True
    assert tr["from_deck"] == "a" and tr["to_deck"] == "b"
    assert tr["total_s"] == pytest.approx(40.0, abs=0.01)
    # 20 blocks of 512 have gone by, and the elapsed count must track them.
    assert tr["elapsed_s"] == pytest.approx(20 * BLOCK / SAMPLE_RATE, abs=0.05)
    assert 0.0 < tr["elapsed_s"] < tr["total_s"]


def test_panic_still_works_with_no_model_and_no_socket(ui):
    server, client, session = ui
    assert server.intent_engine is None
    assert server.clients == 0
    assert client.post("/panic/killbass").json()["ok"] is True
    drive(session.engine, 2, BLOCK)
    assert session.engine.deck_a.eq_low.target == pytest.approx(0.0)


def test_panic_is_recorded_in_the_session_log(ui):
    _server, client, session = ui
    client.post("/panic/cut")
    session.session_log._file.flush()
    events = [
        json.loads(line)
        for line in session.session_log.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(e["event"] == "panic" and e["trigger"] == "ui:cut" for e in events)


# --- isolation from the audio path -------------------------------------------


def test_killing_the_browser_mid_set_does_not_affect_audio(ui):
    """Open a socket, drop it without a close handshake, keep playing."""
    server, client, session = ui
    engine = session.engine

    with client.websocket_connect("/ws") as sock:
        assert "decks" in json.loads(sock.receive_text())
        assert server.clients == 1
        level_during = float(np.max(np.abs(drive(engine, 20, BLOCK))))
    # Socket gone.
    underruns_before = engine.underruns
    position_before = engine.deck_a.position

    level_after = float(np.max(np.abs(drive(engine, 40, BLOCK))))

    assert engine.underruns == underruns_before
    assert engine.deck_a.position > position_before
    assert level_after == pytest.approx(level_during, rel=0.05)
    assert server.clients == 0

    # And the UI can come back without a restart.
    with client.websocket_connect("/ws") as sock:
        assert "decks" in json.loads(sock.receive_text())
    assert server.clients == 0


def test_the_socket_sends_state_immediately_on_connect(ui):
    _server, client, _session = ui
    with client.websocket_connect("/ws") as sock:
        payload = json.loads(sock.receive_text())
    assert payload["decks"]["a"]["title"]
    assert "master_bpm" in payload


def test_the_feed_keeps_every_frame_inside_the_100ms_bar(ui):
    """Pacing regression guard.

    The bound is deliberately loose -- this asserts the loop is paced against a
    deadline rather than sleeping a fixed interval *after* each send, which on
    Windows (15.625 ms timer granularity) stretched a nominal 50 ms period to a
    measured 62.5 ms and put the worst gap at 96 ms, the whole budget.
    """
    _server, client, _session = ui
    nominal = 1.0 / STATE_HZ
    with client.websocket_connect("/ws") as sock:
        sock.receive_text()  # the first frame is sent without waiting
        started = time.perf_counter()
        for _ in range(10):
            sock.receive_text()
        elapsed = time.perf_counter() - started
    mean_gap = elapsed / 10
    assert mean_gap < 2 * nominal, (
        f"{mean_gap * 1000:.1f} ms mean gap against a {nominal * 1000:.0f} ms period"
    )


def test_garbage_on_the_socket_is_ignored_rather_than_fatal(ui):
    _server, client, _session = ui
    with client.websocket_connect("/ws") as sock:
        sock.receive_text()
        sock.send_text("{not json")
        sock.send_text(json.dumps({"text": "state"}))
        for _ in range(20):
            msg = json.loads(sock.receive_text())
            if msg.get("type") == "reply":
                assert "deck a" in msg["text"]
                break
        else:
            pytest.fail("the socket stopped answering after malformed input")


def test_the_ui_module_never_writes_to_the_audio_path():
    """It may read engine state. It may not drive a deck."""
    src = inspect.getsource(ui_server)
    for forbidden in (".read(", ".attach(", "set_rate(", "set_gain(",
                      "set_eq(", ".position =", "callback("):
        assert forbidden not in src, f"ui_server touches the audio path: {forbidden}"


def test_typed_text_takes_the_same_route_as_the_repl(ui):
    server, _client, session = ui
    reply = server.handle_text("state")
    assert reply["type"] == "reply" and "deck a" in reply["text"]

    before = session.engine.queue.qsize()
    reply = server.handle_text("cut")
    assert reply.get("panic") is True
    assert session.engine.queue.qsize() == before + 1

    assert server.handle_text("")["text"] == ""
    # No model configured: prose is answered, not dropped or crashed on.
    assert "model" in server.handle_text("play something harder")["text"].lower()


# --- served files ------------------------------------------------------------


def test_the_page_is_served_with_the_controls_the_phase_requires(ui):
    _server, client, _session = ui
    body = client.get("/").text
    for needed in ("wave-a", "wave-b", "btn-cut", "btn-kill", "btn-stop",
                   "queue-list", "s-drift", "s-llm", "app.js"):
        assert needed in body, f"missing {needed} in index.html"


def test_the_canvases_cannot_grow_without_bound(ui):
    """Layout regression guard, from a bug seen in the browser.

    The renderer writes the measured height back into the canvas ``height``
    attribute, which *is* the element's min-content size. Under a bare ``1fr``
    grid track -- whose automatic minimum is min-content -- that closes a
    feedback loop and the decks grew until they covered the queue and chat.
    Taking the canvas out of flow is what breaks the loop.
    """
    _server, client, _session = ui
    css = client.get("/static/style.css").text
    assert "minmax(0, 1fr)" in css, "grid tracks must cap their automatic minimum"
    assert "position: absolute" in css, "the canvas must be out of flow"
    body = client.get("/").text
    assert body.count("wave-wrap") >= 2, "each canvas needs its sizing wrapper"


def test_static_assets_are_served(ui):
    _server, client, _session = ui
    for path in ("/static/app.js", "/static/style.css"):
        r = client.get(path)
        assert r.status_code == 200 and len(r.text) > 500


def test_the_browser_panic_buttons_use_http_not_the_socket():
    """The separation only holds if the client side respects it too."""
    js = (ui_server.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    panic_fn = js[js.index("async function panic("):js.index("$(\"btn-cut\")")]
    assert "fetch(" in panic_fn
    assert "ws." not in panic_fn and "WebSocket" not in panic_fn


# --- the controller surface --------------------------------------------------


def _track(session, tid):
    return next(t for t in session.crate if t.track_id == tid)


def test_the_library_lists_the_crate_with_its_warnings(ui):
    server, client, session = ui
    rows = server.library()
    assert {r["track_id"] for r in rows} == {t.track_id for t in session.crate}

    row = next(r for r in rows if r["track_id"] == "mid")
    for key in ("title", "bpm", "camelot", "duration_s", "grid_confidence",
                "estimated", "weak_grid", "warn"):
        assert key in row
    # This crate is clean, so nothing should be flagged.
    assert row["warn"] is False

    served = client.get("/api/library").json()["tracks"]
    assert len(served) == len(rows)
    json.dumps(served)


def test_a_weak_grid_or_estimated_mix_points_raise_the_row_warning(ui):
    server, _client, session = ui
    weak = _track(session, "opener")
    object.__setattr__(weak, "grid_confidence", 0.01)
    estimated = _track(session, "banger")
    object.__setattr__(estimated, "mix_points_estimated", True)

    rows = {r["track_id"]: r for r in server.library()}
    assert rows["opener"]["weak_grid"] and rows["opener"]["warn"]
    assert rows["banger"]["estimated"] and rows["banger"]["warn"]
    assert rows["mid"]["warn"] is False


def test_a_drop_on_a_stopped_deck_loads_at_the_mix_in_point(ui):
    server, _client, session = ui
    engine = session.engine
    assert engine.deck("b").track is None

    out = server.load_deck("b", "mid")
    assert out["ok"] and out["when"] == "now"
    drive(engine, 2, BLOCK)

    deck = engine.deck("b")
    assert deck.track is not None and deck.track.analysis.track_id == "mid"
    # Parked, not running: the operator presses PLAY.
    assert deck.playing is False
    expected = _track(session, "mid").mix_in * SAMPLE_RATE
    assert abs(deck.position - expected) < BLOCK * 4


def test_a_drop_on_a_playing_deck_waits_for_the_next_phrase(ui):
    server, _client, session = ui
    engine = session.engine
    live = engine.deck("a")
    assert live.playing and live.track is not None
    before = live.track.analysis.track_id

    out = server.load_deck("a", "banger")
    assert out["ok"] and out["when"] == "next phrase"

    # The audio must not be interrupted: a few blocks later the deck is still
    # playing the track it was playing.
    drive(engine, 6, BLOCK)
    assert live.playing
    assert live.track.analysis.track_id == before

    pending = server.state()["pending_loads"]["a"]
    assert pending["title"] == "banger"
    assert pending["bars_until"] > 0


def test_a_queued_drop_stops_being_announced_once_it_lands(ui):
    server, _client, session = ui
    engine = session.engine
    server.load_deck("a", "banger")
    assert "a" in server.state()["pending_loads"]

    # Jump engine time past the scheduled moment; the badge must clear rather
    # than sit over a track that is already playing.
    engine.frames_played = server._pending["a"]["execute_at"] + 1
    assert server.state()["pending_loads"] == {}


def test_play_and_pause_act_on_one_deck_only(ui):
    server, _client, session = ui
    engine = session.engine
    server.load_deck("b", "mid")
    drive(engine, 2, BLOCK)
    assert engine.deck("a").playing and not engine.deck("b").playing

    assert server.set_transport("b", True)["ok"]
    drive(engine, 2, BLOCK)
    assert engine.deck("b").playing
    assert engine.deck("a").playing, "the other deck must be untouched"

    assert server.set_transport("b", False)["ok"]
    drive(engine, 2, BLOCK)
    assert not engine.deck("b").playing
    assert engine.deck("a").playing


def test_pausing_a_deck_in_the_ui_never_starts_the_other(ui):
    """The reported symptom, at the path it was reported on."""
    server, _client, session = ui
    engine = session.engine
    live = engine.deck(session.live_deck)
    other = engine.deck(session.cued_deck())
    assert live.transport is TransportState.PLAYING
    before = other.transport

    server.set_transport(live.name, False)
    drive(engine, 4, BLOCK)
    assert live.transport is TransportState.PAUSED

    for _ in range(40):
        session._autopilot_tick()
        drive(engine, 8, BLOCK)
        assert other.transport is not TransportState.PLAYING, (
            "pausing one deck started the other"
        )
    assert other.transport is before


def test_the_state_feed_reports_the_transport_state(ui):
    server, _client, session = ui
    s = server.state()
    assert s["decks"][session.live_deck]["transport"] == "playing"
    assert s["decks"][session.cued_deck()]["transport"] == "empty"
    server.set_transport(session.live_deck, False)
    drive(session.engine, 4, BLOCK)
    assert server.state()["decks"][session.live_deck]["transport"] == "paused"


def test_pausing_holds_the_playhead_where_it_stopped(ui):
    server, _client, session = ui
    engine = session.engine
    server.set_transport("a", False)
    drive(engine, 2, BLOCK)
    where = engine.deck("a").position
    drive(engine, 20, BLOCK)
    assert engine.deck("a").position == where


def test_pausing_mid_transition_restores_the_other_decks_eq(ui):
    server, _client, session = ui
    engine = session.engine
    server.load_deck("b", "mid")
    drive(engine, 2, BLOCK)
    server.set_transport("b", True)
    drive(engine, 2, BLOCK)

    engine.submit(
        StartTransition(
            from_deck="a", to_deck="b", total_frames=SAMPLE_RATE * 20,
            execute_at=IMMEDIATE, origin="test",
        )
    )
    drive(engine, 20, BLOCK)
    assert engine.transition_active
    # Part-way through a bass swap the incoming deck's low band is ducked.
    assert engine.deck_state("b").eq[0] < 1.0

    out = server.set_transport("a", False)
    assert out["restored"] == "b"
    drive(engine, 40, BLOCK)

    assert not engine.transition_active
    assert engine.deck_state("b").eq[0] == pytest.approx(1.0, abs=0.02)
    # Level too: a deck stranded part-way down the crossfade would otherwise
    # keep playing quietly with nothing left running to finish the fade.
    assert engine.deck_state("b").gain == pytest.approx(1.0, abs=0.02)

    events = [e for e in log_events(session) if e["event"] == "ui_transport"]
    assert events and events[-1]["eq_restored_on"] == "b"


def test_a_manual_load_holds_the_automation_and_resume_clears_it(ui):
    server, _client, session = ui
    assert not server.automation_held and not session.frozen

    server.load_deck("b", "mid")
    assert server.automation_held and session.frozen

    server.hold_automation(False)
    assert not server.automation_held and not session.frozen


def test_a_manual_pause_holds_the_automation(ui):
    server, _client, session = ui
    server.set_transport("a", False)
    assert server.automation_held and session.frozen


def test_the_hold_banner_does_not_outlive_a_repl_resume(ui):
    server, _client, session = ui
    server.load_deck("b", "mid")
    assert server.automation_held

    # The operator types `resume` in the terminal instead of clicking.
    session.freeze(False)
    assert server.automation_held is False
    assert server.state()["automation_held"] is False


def test_cue_parks_the_deck_back_at_its_mix_in(ui):
    server, _client, session = ui
    engine = session.engine
    server.load_deck("b", "mid")
    drive(engine, 2, BLOCK)
    server.set_transport("b", True)
    drive(engine, 40, BLOCK)
    moved = engine.deck("b").position

    server.cue_to_mix_in("b")
    drive(engine, 2, BLOCK)
    assert engine.deck("b").position < moved
    expected = _track(session, "mid").mix_in * SAMPLE_RATE
    assert abs(engine.deck("b").position - expected) < BLOCK * 4
    assert not engine.deck("b").playing


def test_the_eq_goes_through_the_command_queue(ui):
    server, _client, session = ui
    engine = session.engine
    before = engine.queue.qsize()
    assert server.apply_eq("a", low=0.25)["ok"]
    assert engine.queue.qsize() == before + 1
    drive(engine, 2, BLOCK)
    assert engine.deck_state("a").eq[0] == pytest.approx(0.25, abs=0.05)


def test_controller_messages_are_dispatched_by_type(ui):
    server, _client, session = ui
    engine = session.engine

    assert server.handle_action(
        {"type": "load", "deck": "b", "track_id": "mid"}
    )["ok"]
    drive(engine, 2, BLOCK)
    assert engine.deck("b").track is not None

    assert server.handle_action(
        {"type": "transport", "deck": "b", "playing": True}
    )["ok"]
    assert server.handle_action({"type": "automation", "held": False})["ok"]
    assert server.handle_action({"type": "library"})["type"] == "library"


def test_a_bad_controller_message_is_refused_not_crashed_on(ui):
    server, _client, _session = ui
    for msg in (
        {"type": "nonsense"},
        {"type": "load", "deck": "c", "track_id": "mid"},
        {"type": "load", "deck": "a", "track_id": "not-in-the-crate"},
        {"type": "transport", "deck": "b", "playing": True},  # deck is empty
    ):
        out = server.handle_action(msg)
        assert out["ok"] is False and out["error"]


def test_the_socket_still_accepts_a_plain_chat_line(ui):
    """The controller types must not have displaced the chat contract."""
    _server, client, _session = ui
    with client.websocket_connect("/ws") as ws:
        ws.receive_text()  # first state frame
        ws.send_text(json.dumps({"text": "state"}))
        for _ in range(40):
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "reply":
                assert "deck a" in msg["text"]
                return
    raise AssertionError("no reply to a plain chat line")


def test_the_page_has_the_controller_layout(ui):
    _server, client, _session = ui
    body = client.get("/").text
    for needed in ("platter-a", "platter-b", "play-a", "play-b",
                   "cuept-a", "cuept-b", "eq-a-low", "eq-b-high",
                   "fader-a", "xfader", "lib-rows", "lib-filter",
                   "meter-a", "held", "btn-resume-auto"):
        assert needed in body, f"missing {needed} in index.html"


def test_the_channel_fader_sets_that_decks_level(ui):
    server, _client, session = ui
    engine = session.engine
    assert server.set_level("a", 0.3)["gain"] == pytest.approx(0.3)
    drive(engine, 60, BLOCK)
    assert engine.deck_state("a").gain == pytest.approx(0.3, abs=0.02)


def test_the_fader_is_clamped_to_unity(ui):
    """Band gains may boost to 2.0; channel level may not. That is where a mix
    clips, and there is no headroom above unity to make it back."""
    server, _client, session = ui
    assert server.set_level("a", 4.0)["gain"] == 1.0
    assert server.set_level("a", -2.0)["gain"] == 0.0
    drive(session.engine, 60, BLOCK)
    assert session.engine.deck_state("a").gain == pytest.approx(0.0, abs=0.02)


def test_the_crossfader_is_equal_power(ui):
    server, _client, session = ui
    out = server.set_crossfader(0.5)
    assert out["a"] == pytest.approx(0.7071, abs=0.001)
    assert out["b"] == pytest.approx(0.7071, abs=0.001)
    # Ends are hard ends, not a dip either side.
    assert server.set_crossfader(0.0)["a"] == pytest.approx(1.0)
    assert server.set_crossfader(1.0)["b"] == pytest.approx(1.0)
    drive(session.engine, 60, BLOCK)
    assert session.engine.deck_state("b").gain == pytest.approx(1.0, abs=0.02)


def test_a_fader_move_takes_an_in_flight_blend_over_cleanly(ui):
    """Manual and automatic must not both be driving the same gains."""
    server, _client, session = ui
    engine = session.engine
    server.load_deck("b", "mid")
    drive(engine, 2, BLOCK)
    server.set_transport("b", True)
    drive(engine, 2, BLOCK)
    engine.submit(
        StartTransition(
            from_deck="a", to_deck="b", total_frames=SAMPLE_RATE * 20,
            execute_at=IMMEDIATE, origin="test",
        )
    )
    drive(engine, 20, BLOCK)
    assert engine.transition_active
    assert engine.deck_state("b").eq[0] < 1.0

    assert server.set_level("a", 0.5)["handed_over"] is True
    drive(engine, 60, BLOCK)

    assert not engine.transition_active
    # No half-swapped bass left behind on either deck.
    assert engine.deck_state("a").eq[0] == pytest.approx(1.0, abs=0.02)
    assert engine.deck_state("b").eq[0] == pytest.approx(1.0, abs=0.02)
    assert engine.deck_state("a").gain == pytest.approx(0.5, abs=0.02)
    assert server.automation_held


def test_a_fader_move_outside_a_blend_hands_nothing_over(ui):
    server, _client, session = ui
    assert session.engine.transition_active is False
    assert server.set_level("a", 0.8)["handed_over"] is False


def test_the_supervisor_still_vets_a_gain_command(ui):
    """The new command must not be a hole in the validator."""
    _server, _client, session = ui
    sup = session.supervisor
    assert sup.validate(SetGain(deck="a", gain=0.5, origin="test")) is None
    for bad in (
        SetGain(deck="a", gain=1.5, origin="test"),
        SetGain(deck="a", gain=-0.1, origin="test"),
        SetGain(deck="z", gain=0.5, origin="test"),
    ):
        rejection = sup.validate(bad)
        assert rejection is not None and rejection.reason


# --- the master beat clock ----------------------------------------------------


@pytest.fixture
def idle_ui(idle_session):
    """A server on a session that has never been told to start anything."""
    return UIServer(idle_session, intent_engine=None), idle_session


def test_a_deck_started_from_the_ui_takes_the_master_clock(idle_ui):
    """Regression, from a set that fell apart in about ninety seconds.

    On an idle start nothing has claimed the beat clock, so `master_bpm` is 0.
    A deck loaded and played from the UI used to leave it that way: the
    supervisor then measured the deck against a clock that never ticked, read
    unbounded drift, and hard-resynced it roughly every two seconds. The track
    looped around bars 1 and 2 and never progressed.
    """
    server, session = idle_ui
    engine = session.engine
    assert engine.master_bpm == 0.0, "nothing has claimed the clock yet"

    server.load_deck("b", "mid")
    drive(engine, 2, BLOCK)
    server.set_transport("b", True)
    drive(engine, 4, BLOCK)

    expected = session.crate[1].bpm
    assert engine.master_bpm == pytest.approx(expected, rel=0.01), (
        "a deck playing on its own must drive the master clock"
    )
    assert engine.master_deck == "b"


def test_the_supervisor_does_not_fight_a_ui_started_deck(idle_ui):
    """The symptom, measured: the playhead has to actually advance."""
    server, session = idle_ui
    engine = session.engine

    server.load_deck("b", "mid")
    drive(engine, 2, BLOCK)
    server.set_transport("b", True)

    deck = engine.deck("b")
    start = deck.position
    for _ in range(60):
        drive(engine, 20, BLOCK)
        session.supervisor.check_drift()

    assert session.supervisor.interventions == 0, "the clock was never in dispute"
    advanced = (deck.position - start) / SAMPLE_RATE
    assert advanced > 10.0, f"the deck only advanced {advanced:.1f}s"


def test_a_drop_onto_the_live_deck_retunes_the_master_clock(idle_ui):
    """A new track on the mix deck changes its tempo; the clock must follow."""
    server, session = idle_ui
    engine = session.engine

    server.load_deck("a", "opener")
    drive(engine, 2, BLOCK)
    server.set_transport("a", True)
    drive(engine, 4, BLOCK)
    assert engine.master_bpm == pytest.approx(session.crate[0].bpm, rel=0.01)

    # Now drop a different tempo onto the same, playing deck.
    server.load_deck("a", "banger")
    session.scheduler.tick(engine.frames_played + 10**9)   # let it land
    drive(engine, 4, BLOCK)
    assert engine.master_bpm == pytest.approx(session.crate[2].bpm, rel=0.01)


def test_a_second_deck_does_not_steal_the_clock(idle_ui):
    """Playing both decks by hand must not hand the clock back and forth."""
    server, session = idle_ui
    engine = session.engine

    server.load_deck("a", "opener")
    drive(engine, 2, BLOCK)
    server.set_transport("a", True)
    drive(engine, 4, BLOCK)
    assert engine.master_deck == "a"

    server.load_deck("b", "banger")
    drive(engine, 2, BLOCK)
    server.set_transport("b", True)
    drive(engine, 4, BLOCK)
    assert engine.master_deck == "a", "the running mix keeps the clock"


def test_stopping_a_server_that_never_started_is_harmless(ui):
    server, _client, _session = ui
    server.stop()
    server.stop()


def test_a_left_open_tab_cannot_hold_up_the_shutdown(ui, monkeypatch):
    """Graceful first, then forced.

    uvicorn's graceful shutdown waits for open connections, and the state feed
    is a websocket a forgotten browser tab holds indefinitely. That turned
    "type stop" into a wait long enough to reach for Ctrl+C.
    """
    server, _client, _session = ui

    class _StuckThread:
        def __init__(self):
            self.joins = []
            self._alive = True

        def join(self, timeout=None):
            self.joins.append(timeout)
            # Only the forced pass gets it to exit.
            if getattr(server._server, "force_exit", False):
                self._alive = False

        def is_alive(self):
            return self._alive

    class _Server:
        should_exit = False
        force_exit = False

    fake = _Server()
    server._server = fake
    server._thread = _StuckThread()
    thread = server._thread

    server.stop()

    assert fake.should_exit is True, "graceful shutdown is still tried first"
    assert fake.force_exit is True, "a stuck thread must then be forced"
    assert len(thread.joins) == 2, "one graceful join, one after forcing"
    assert sum(thread.joins) <= 5.0, "shutdown must stay bounded"
    assert server._thread is None


def test_a_dropped_browser_connection_is_not_reported_as_an_error(ui):
    """A closed tab used to print a traceback into the middle of the REPL."""
    server, _client, _session = ui

    seen = []

    class _Loop:
        def get_exception_handler(self):
            return lambda loop, ctx: seen.append(ctx)

        def set_exception_handler(self, fn):
            self.installed = fn

        def default_exception_handler(self, ctx):
            seen.append(ctx)

    loop = _Loop()
    server._quieten_dropped_connections(loop)
    handler = loop.installed

    handler(loop, {"exception": ConnectionResetError(10054, "forcibly closed")})
    handler(loop, {"exception": ConnectionAbortedError()})
    assert seen == [], "a dropped client is noise, not an error"

    # Anything else still gets through to whoever was handling it.
    real = {"exception": ValueError("something actually wrong")}
    handler(loop, real)
    assert seen == [real]


def test_the_quietening_is_installed_only_once(ui):
    server, _client, _session = ui

    class _Loop:
        installs = 0

        def get_exception_handler(self):
            return None

        def set_exception_handler(self, fn):
            type(self).installs += 1

    loop = _Loop()
    server._quieten_dropped_connections(loop)
    server._quieten_dropped_connections(loop)
    assert _Loop.installs == 1


def test_one_failing_teardown_step_does_not_skip_the_others():
    """The step a Ctrl+C used to skip was the one that closes the recording."""
    from djai import cli

    done = []

    def ok(name):
        return lambda: done.append(name)

    def interrupted():
        raise KeyboardInterrupt

    failed = cli.run_shutdown_steps(
        ("the web UI", interrupted),
        ("the session", ok("session")),
        ("the model client", ok("model")),
    )
    assert done == ["session", "model"], "later steps must still run"
    assert failed == ["the web UI"]


def test_a_missing_teardown_step_is_skipped_quietly():
    from djai import cli

    done = []
    assert cli.run_shutdown_steps(
        ("the web UI", None),
        ("the session", lambda: done.append("session")),
    ) == []
    assert done == ["session"]


def test_the_faders_are_live_controls_in_the_page(ui):
    _server, client, _session = ui
    body = client.get("/").text
    for control in ('id="fader-a"', 'id="fader-b"', 'id="xfader"'):
        tail = body[body.index(control):body.index(control) + 260]
        assert "disabled" not in tail, f"{control} must be operable"
    js = (ui_server.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert '"level"' in js and '"crossfader"' in js

