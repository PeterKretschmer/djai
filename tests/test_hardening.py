"""Club hardening: cue output, session recording, manual override, --no-llm.

THREADING CONTEXT: main thread (pytest). No device is opened -- the cue paths
are driven by calling the two callbacks directly, which is exactly what the two
sound cards would do, minus the clocks.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import soundfile as sf

from djai import cli
from djai.commands import IMMEDIATE, LoadTrack, StartTransition
from djai.deck import CHANNELS, SAMPLE_RATE, LoadedTrack, TransportState
from djai.engine import Engine, SessionRecorder, _Ring
from tests.test_engine import drive, loud_track
from tests.test_integration import make_analysis
from tests.test_integration import session as _session_fixture

#: The integration Session: real engine, scheduler and supervisor on synthetic
#: audio, with no device and no threads.
session = _session_fixture

BLOCK = 512


def level(buf: np.ndarray) -> float:
    return float(np.max(np.abs(buf)))


# --- the ring -----------------------------------------------------------------


def test_the_ring_round_trips_exactly():
    ring = _Ring(1024)
    data = np.arange(300 * CHANNELS, dtype=np.float32).reshape(300, CHANNELS)
    ring.write(data, 300)
    out = np.zeros((300, CHANNELS), dtype=np.float32)
    assert ring.read_into(out, 300) == 300
    assert np.array_equal(out, data)
    assert ring.dropped == 0


def test_the_ring_wraps_without_losing_a_sample():
    ring = _Ring(256)
    out = np.zeros((100, CHANNELS), dtype=np.float32)
    expected = []
    for i in range(20):
        block = np.full((100, CHANNELS), float(i), dtype=np.float32)
        ring.write(block, 100)
        assert ring.read_into(out, 100) == 100
        expected.append(out[0, 0])
    assert expected == [float(i) for i in range(20)]


def test_the_ring_drops_rather_than_blocking_when_the_consumer_stalls():
    """The audio thread must never wait for a slow disk."""
    ring = _Ring(256)
    block = np.ones((100, CHANNELS), dtype=np.float32)
    for _ in range(10):
        ring.write(block, 100)  # nobody is reading
    assert ring.dropped > 0
    # And it is still usable afterwards.
    out = np.zeros((100, CHANNELS), dtype=np.float32)
    assert ring.read_into(out, 100) == 100


def test_reading_an_empty_ring_returns_nothing_rather_than_stale_audio():
    ring = _Ring(256)
    out = np.ones((64, CHANNELS), dtype=np.float32)
    assert ring.read_into(out, 64) == 0
    assert level(out) == 1.0, "the caller's buffer is left alone to fill itself"


def test_a_lapped_consumer_resyncs_instead_of_returning_spliced_audio():
    """Regression: a consumer left behind used to read overwritten slots.

    Its read index still pointed into the buffer, so `read_into` happily
    returned whatever now occupied those frames -- audio spliced together from
    two different points in the set, with no indication anything was wrong.
    """
    ring = _Ring(400)
    for i in range(10):                       # 1000 frames through a 400 ring
        ring.write(np.full((100, CHANNELS), float(i), dtype=np.float32), 100)

    assert ring.dropped > 0
    out = np.zeros((400, CHANNELS), dtype=np.float32)
    took = ring.read_into(out, 400)
    assert took == 400, "should hand back exactly what the buffer still holds"
    # The four most recent blocks, in order, and nothing older.
    assert [out[i * 100, 0] for i in range(4)] == [6.0, 7.0, 8.0, 9.0]
    assert ring.read_into(out, 400) == 0, "and nothing is left over"


# --- cue: one device, channels 3/4 --------------------------------------------


def cue_engine(**kwargs) -> Engine:
    engine = Engine(blocksize=BLOCK, **kwargs)
    engine.submit(LoadTrack(deck="a", track=loud_track(200.0, seconds=20.0), master=True))
    engine.submit(LoadTrack(deck="b", track=loud_track(3000.0, seconds=20.0)))
    engine.deck_a.gain.jump(1.0)
    engine.deck_b.gain.jump(0.0)   # incoming deck: fader down, as it would be
    engine.deck_b.playing = True
    drive(engine, 2, BLOCK)
    return engine


def test_cue_on_channels_3_and_4_carries_the_incoming_deck():
    """The acceptance case: cue plays deck B while master plays deck A."""
    engine = cue_engine(cue_channels=(3, 4))
    engine.cue_channels = (3, 4)   # start() would set this from the device
    engine.cue_deck = "b"

    out = np.zeros((BLOCK, 4), dtype=np.float32)
    engine.callback(out, BLOCK, None, None)

    master, cue = out[:, 0:2], out[:, 2:4]
    assert level(master) > 0.1, "master should carry deck A"
    assert level(cue) > 0.1, "cue should carry deck B even with its fader down"


def test_cue_is_independent_of_the_master_fader():
    """Pre-listen is pre-fader: that is the entire point of it."""
    engine = cue_engine(cue_channels=(3, 4))
    engine.cue_channels = (3, 4)
    engine.cue_deck = "b"
    engine.deck_b.gain.jump(0.0)

    out = np.zeros((BLOCK, 4), dtype=np.float32)
    engine.callback(out, BLOCK, None, None)
    # Deck B contributes nothing to the master mix but is fully present on cue.
    assert level(out[:, 2:4]) > 0.1


def test_cue_off_leaves_the_cue_pair_silent():
    engine = cue_engine(cue_channels=(3, 4))
    engine.cue_channels = (3, 4)
    engine.cue_deck = None

    out = np.zeros((BLOCK, 4), dtype=np.float32)
    engine.callback(out, BLOCK, None, None)
    assert level(out[:, 0:2]) > 0.1
    assert level(out[:, 2:4]) == 0.0


def test_cueing_a_deck_does_not_double_advance_its_position():
    """Cue reads the block the deck already rendered, not a second read.

    A second ``deck.read()`` would sound fine and quietly run the cued deck at
    double speed, putting it a bar out by the time the blend started.
    """
    engine = cue_engine(cue_channels=(3, 4))
    engine.cue_channels = (3, 4)
    out = np.zeros((BLOCK, 4), dtype=np.float32)

    engine.cue_deck = None
    before = engine.deck_b.position
    engine.callback(out, BLOCK, None, None)
    without = engine.deck_b.position - before

    engine.cue_deck = "b"
    before = engine.deck_b.position
    engine.callback(out, BLOCK, None, None)
    with_cue = engine.deck_b.position - before

    assert with_cue == pytest.approx(without, rel=1e-9), (
        f"cue advanced the deck {with_cue / without:.2f}x as far"
    )
    assert with_cue == pytest.approx(BLOCK * engine.deck_b.rate, rel=1e-6)


def test_a_narrow_device_falls_back_to_master_only_rather_than_refusing():
    """No pre-listen is survivable. No audio is not."""
    engine = Engine(blocksize=BLOCK, cue_channels=(3, 4))
    engine.cue_channels = None       # what start() does on a 2-channel device
    engine.cue_mode = "none"
    engine.cue_deck = "b"
    engine.submit(LoadTrack(deck="a", track=loud_track(200.0), master=True))
    engine.deck_a.gain.jump(1.0)

    out = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    engine.callback(out, BLOCK, None, None)
    assert level(out) > 0.1, "master must still play"


# --- cue: a second device ------------------------------------------------------


def test_cue_on_a_second_device_gets_the_incoming_deck():
    engine = cue_engine(cue_device=7)
    engine.cue_deck = "b"

    master = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    engine.callback(master, BLOCK, None, None)

    cue = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    engine._cue_callback(cue, BLOCK, None, None)

    assert level(master) > 0.1
    assert level(cue) > 0.1, "the second stream should have drained the ring"


def test_the_cue_stream_plays_silence_rather_than_stalling_when_starved():
    """The two devices have independent clocks; the cue one may run ahead."""
    engine = cue_engine(cue_device=7)
    engine.cue_deck = "b"

    cue = np.ones((BLOCK, CHANNELS), dtype=np.float32)
    engine._cue_callback(cue, BLOCK, None, None)   # nothing written yet
    assert level(cue) == 0.0
    assert engine.cue_underruns == 1


def test_a_starved_cue_stream_never_disturbs_the_master():
    engine = cue_engine(cue_device=7)
    engine.cue_deck = "b"
    cue = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    for _ in range(50):
        engine._cue_callback(cue, BLOCK, None, None)

    before = engine.underruns
    assert level(drive(engine, 20, BLOCK)) > 0.1
    assert engine.underruns == before


def test_a_broken_cue_ring_cannot_take_down_the_cue_stream():
    engine = cue_engine(cue_device=7)
    engine.cue_deck = "b"
    engine._cue_ring = "not a ring"
    cue = np.ones((BLOCK, CHANNELS), dtype=np.float32)
    engine._cue_callback(cue, BLOCK, None, None)
    assert level(cue) == 0.0


def test_the_two_cue_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="alternatives"):
        Engine(blocksize=BLOCK, cue_device=7, cue_channels=(3, 4))


def test_cue_channel_parsing():
    assert cli._parse_cue_channels("3,4") == (3, 4)
    assert cli._parse_cue_channels("3:4") == (3, 4)
    assert cli._parse_cue_channels(None) is None
    assert cli._parse_cue_channels("") is None
    for bad in ("3", "3,5", "0,1", "a,b", "4,3", "1,2,3"):
        with pytest.raises(ValueError):
            cli._parse_cue_channels(bad)


# --- session recording ---------------------------------------------------------


def test_the_recording_contains_what_the_master_played(tmp_path):
    path = tmp_path / "set.wav"
    # The ring has to hold the whole test: `drive` runs far faster than real
    # time, so the writer thread cannot keep up in wall-clock terms and a
    # smaller ring would legitimately drop frames (covered separately below).
    rec = SessionRecorder(path, ring_seconds=6.0)
    rec.start()

    engine = Engine(blocksize=BLOCK, recorder=rec)
    engine.submit(LoadTrack(deck="a", track=loud_track(440.0, seconds=10.0), master=True))
    engine.deck_a.gain.jump(1.0)
    played = drive(engine, 200, BLOCK)

    deadline = time.time() + 5.0
    while rec.frames_written < 200 * BLOCK and time.time() < deadline:
        time.sleep(0.02)
    rec.stop()

    assert path.exists()
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE and audio.shape[1] == CHANNELS
    assert audio.shape[0] >= 200 * BLOCK * 0.95, "most of the set should be on disk"
    assert level(audio) > 0.1
    # PCM_16 quantisation is the only difference we should see.
    n = min(audio.shape[0], played.shape[0])
    assert np.max(np.abs(audio[:n] - played[:n])) < 1e-3


def test_capture_never_blocks_the_callback_when_the_writer_stalls(tmp_path):
    """A stuck disk costs recorded frames, never audio."""
    rec = SessionRecorder(tmp_path / "set.wav", ring_seconds=0.05)
    rec.start()
    engine = Engine(blocksize=BLOCK, recorder=rec)
    engine.submit(LoadTrack(deck="a", track=loud_track(440.0, seconds=30.0), master=True))
    engine.deck_a.gain.jump(1.0)

    rec._stop.set()          # freeze the writer thread
    rec._thread.join(timeout=2)

    before = engine.underruns
    out = drive(engine, 300, BLOCK)
    assert engine.underruns == before, "a stalled writer must not cost an underrun"
    assert level(out) > 0.1, "and must not cost audio"
    assert rec.dropped_frames > 0, "frames should have been dropped from the FILE"


def test_a_recorder_that_cannot_write_does_not_take_the_session_down(tmp_path):
    rec = SessionRecorder(tmp_path / "set.wav", ring_seconds=1.0)
    rec.start()

    class Exploding:
        def write(self, data):
            raise OSError("disk full")

        def close(self):
            pass

    rec._file = Exploding()
    engine = Engine(blocksize=BLOCK, recorder=rec)
    engine.submit(LoadTrack(deck="a", track=loud_track(440.0, seconds=10.0), master=True))
    engine.deck_a.gain.jump(1.0)
    drive(engine, 40, BLOCK)

    deadline = time.time() + 3.0
    while rec.error is None and time.time() < deadline:
        time.sleep(0.02)
    assert rec.error is not None and "disk full" in rec.error
    assert level(drive(engine, 20, BLOCK)) > 0.1, "audio carries on regardless"
    rec.stop()


def test_a_killed_process_still_leaves_a_playable_file(tmp_path):
    """No close(), as if the process had been killed mid-set."""
    path = tmp_path / "set.wav"
    rec = SessionRecorder(path, ring_seconds=2.0)
    rec.start()
    block = (np.ones((BLOCK, CHANNELS), dtype=np.float32) * 0.4)
    for _ in range(80):
        rec.capture(block, BLOCK)
    deadline = time.time() + 5.0
    while rec.frames_written < 80 * BLOCK and time.time() < deadline:
        time.sleep(0.02)
    rec._file.flush()

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE and audio.shape[0] > 0
    rec.stop()


def test_recording_is_off_when_no_path_is_given(session):
    assert session.recorder is None
    assert session.engine.recorder is None
    assert session.state_blob()["recording"] is None


# --- manual override -----------------------------------------------------------


def test_freeze_stops_the_autopilot_cueing(session):
    from tests.test_integration import seek_near_mix_out

    reply = session.freeze(True)
    assert "frozen" in reply.lower() and session.frozen

    seek_near_mix_out(session)
    session._autopilot_tick()
    assert not session.has_cued_track(), "frozen means nothing gets cued"
    assert len(session.scheduler) == 0

    session.freeze(False)
    session._autopilot_tick()
    assert session.has_cued_track(), "resume means the autopilot works again"


def test_freeze_gates_the_scheduler_including_silence_recovery(session):
    """Held automation makes no move at all -- not even to fill silence.

    This reverses the earlier rule, which let recovery run while frozen on the
    reasoning that freezing meant "stop choosing" rather than "go quiet". The
    hold now gates the scheduler at the point transitions are planned, so an
    operator who has taken manual control is never raced by the autopilot.
    """
    session.freeze(True)
    live = session.engine.deck(session.live_deck)
    live.ended = True
    live.playing = False

    session._autopilot_tick()
    session._autopilot_tick()
    drive(session.engine, 4, BLOCK)
    assert level(drive(session.engine, 20, BLOCK)) < 0.01, "frozen must not start a track"
    assert not session.has_cued_track(), "frozen must not even cue"

    # Releasing the hold hands it straight back.
    session.freeze(False)
    session._autopilot_tick()
    session._autopilot_tick()
    drive(session.engine, 4, BLOCK)
    assert level(drive(session.engine, 20, BLOCK)) > 0.1, "resume must recover"


def test_freeze_does_not_stop_the_drift_supervisor(session):
    session.freeze(True)
    before = session.supervisor.interventions
    session.supervisor.check_drift()
    assert session.supervisor.interventions >= before  # ran, did not raise


def test_force_next_overrides_the_selector(session):
    reply = session.force_next("banger")
    assert "banger" in reply

    assert session.cue_next(0.0)
    drive(session.engine, 2, BLOCK)
    assert session._cued is not None
    assert session._cued.title == "banger"


def test_a_forced_track_is_used_once_and_then_forgotten(session):
    """A one-off override must not silently govern the rest of the night."""
    session.force_next("banger")
    session.cue_next(0.0)
    assert session._forced_next is None


def test_force_next_rejects_a_track_that_is_not_in_the_crate(session):
    reply = session.force_next("nonexistent")
    assert "No track" in reply
    assert session._forced_next is None


def test_force_transition_arms_a_blend(session):
    from tests.test_integration import seek_near_mix_out

    seek_near_mix_out(session)
    session.cue_next(0.0)
    drive(session.engine, 2, BLOCK)

    reply = session.force_transition()
    assert "Blending into" in reply
    assert len(session.scheduler) > 0


def test_force_transition_says_so_when_one_is_already_running(session):
    session.cue_next(0.0)
    drive(session.engine, 2, BLOCK)
    session.engine.submit(
        StartTransition(from_deck="a", to_deck="b",
                        total_frames=SAMPLE_RATE * 30, execute_at=IMMEDIATE)
    )
    drive(session.engine, 2, BLOCK)
    assert "already running" in session.force_transition()


def test_set_cue_reports_when_there_is_no_cue_output(session):
    reply = session.set_cue("b")
    assert "No cue output" in reply
    assert session.engine.cue_deck is None


def test_set_cue_routes_a_deck_when_cue_exists(session):
    session.engine.cue_mode = "device 7"
    assert "deck b" in session.set_cue("b")
    assert session.engine.cue_deck == "b"
    assert "Cue off" in session.set_cue(None)
    assert session.engine.cue_deck is None


def test_the_override_words_are_matched_before_the_model(session):
    """They exist for when the automation is wrong; routing them through it
    would be routing them through the thing that is wrong."""
    assert cli.handle_override(session, "freeze") is not None
    assert session.frozen
    assert cli.handle_override(session, "resume") is not None
    assert not session.frozen
    assert cli.handle_override(session, "force banger") is not None
    assert cli.handle_override(session, "cue off") is not None
    assert cli.handle_override(session, "go") is not None

    # Prose stays prose: these must reach the model, not be swallowed.
    for prose in ("play something harder", "more energy", "", "forcefully drop it"):
        assert cli.handle_override(session, prose) is None


def test_force_without_an_argument_explains_itself(session):
    assert "force <track>" in cli.handle_override(session, "force")


def test_override_is_recorded_in_the_session_log(session):
    session.freeze(True)
    session.session_log._file.flush()
    events = [
        json.loads(line)
        for line in session.session_log.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(e["event"] == "automation_frozen" for e in events)


def test_state_reports_the_override_flags(session):
    session.freeze(True)
    session.force_next("banger")
    blob = session.state_blob()
    assert blob["frozen"] is True
    assert blob["forced_next"] == "banger"
    text = session.format_state()
    assert "AUTOMATION FROZEN" in text and "banger" in text


# --- --no-llm ------------------------------------------------------------------


def test_no_llm_makes_zero_network_calls(monkeypatch, session):
    """The acceptance criterion: zero Ollama calls, whatever is typed."""
    from djai.intent import IntentEngine

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("--no-llm must never reach the network")

    engine = IntentEngine()
    monkeypatch.setattr(engine._client, "post", forbidden)
    engine.available = False   # what cmd_play does for --no-llm

    for text in ("play something harder", "more energy", "take it down",
                 "hold the blend", "skip that", "give me a banger"):
        intent = engine.interpret(text, session.model_state())
        assert intent.ok, "every line must still resolve to something"
    assert calls == [], "no HTTP request should have been attempted"
    engine.close()


def test_no_llm_still_understands_the_keyword_commands(session):
    from djai.intent import IntentEngine

    engine = IntentEngine()
    engine.available = False
    state = session.model_state()

    assert engine.interpret("more energy", state).action == "set_energy"
    assert engine.interpret("next track", state).action == "next_track"
    engine.close()


def test_no_llm_never_calls_warmup(monkeypatch, tmp_path):
    """A cold 8B load is the thing --no-llm exists to avoid."""
    from djai.intent import IntentEngine

    warmed = []
    monkeypatch.setattr(
        IntentEngine, "warmup", lambda self, *a, **k: warmed.append(1) or (True, "x")
    )
    args = cli.build_parser().parse_args(
        ["play", "--cache", str(tmp_path), "--no-llm"]
    )
    assert args.no_llm is True
    # cmd_play returns early on an empty crate, before any model work.
    assert cli.cmd_play(args) == 1
    assert warmed == []


# --- idle start and cross-deck triggering -------------------------------------


@pytest.fixture
def idle_session(tmp_path, monkeypatch):
    """A Session that was never told to start anything. Both decks EMPTY."""
    crate = []
    for tid, bpm, cam, energy in (
        ("opener", 124.0, "8A", 0.2),
        ("mid", 125.0, "9A", 0.5),
        ("banger", 126.0, "8B", 1.0),
    ):
        p = tmp_path / f"{tid}.wav"
        p.write_bytes(b"placeholder")
        crate.append(make_analysis(tid, bpm, cam, energy, p))

    def fake_load(analysis):
        base = loud_track(200.0 + 500 * len(analysis.title), analysis.bpm, 240.0)
        return LoadedTrack(analysis=analysis, audio=base.audio)

    monkeypatch.setattr(cli, "load_track", fake_load)
    # Same two measures as the integration fixture, for the same reasons: no
    # background phase-vocoding of four-minute synthetic tracks, and a shutdown
    # so the engine's stretch worker does not outlive the test.
    monkeypatch.setattr("djai.config.TIME_STRETCH_ENABLED", False)
    s = cli.Session(crate, log_dir=tmp_path / "logs")
    yield s
    s.shutdown()


def test_an_idle_session_stays_silent_with_both_decks_empty(idle_session):
    """A minute after start, nothing has been chosen, decoded or played."""
    session = idle_session
    for name in ("a", "b"):
        assert session.engine.deck(name).transport is TransportState.EMPTY

    peak = 0.0
    blocks_per_second = SAMPLE_RATE / BLOCK
    for _ in range(60):
        session._autopilot_tick()
        peak = max(peak, level(drive(session.engine, int(blocks_per_second), BLOCK)))

    assert peak == 0.0, "the idle session put audio on the master output"
    for name in ("a", "b"):
        assert session.engine.deck(name).transport is TransportState.EMPTY
    assert not session.has_cued_track(), "idle must not cue a track"
    assert not session.played, "idle must not consume the crate"


def test_nothing_is_selected_or_planned_while_the_decks_are_empty(
    idle_session, monkeypatch
):
    """The guard is that the selector is never even reached."""
    called: list[str] = []
    monkeypatch.setattr(
        cli, "select_next",
        lambda *a, **k: called.append("select_next") or None,
    )
    monkeypatch.setattr(
        cli.phrase, "plan_transition",
        lambda *a, **k: called.append("plan_transition") or None,
    )
    for _ in range(50):
        idle_session._autopilot_tick()
        drive(idle_session.engine, 8, BLOCK)
    assert called == []


def test_pausing_the_live_deck_never_starts_the_other_one(session):
    """The diagnostic symptom: a pause must not read as a hand-off cue."""
    engine = session.engine
    live = engine.deck(session.live_deck)
    other = engine.deck(session.cued_deck())
    assert live.transport is TransportState.PLAYING
    before = other.transport

    # Pause the live deck the way the UI does: re-load its own audio at the
    # current playhead, stopped.
    engine.submit(
        LoadTrack(
            deck=live.name, track=live.track, start_frame=int(live.position),
            rate=live.rate, play=False, execute_at=IMMEDIATE, origin="test:pause",
        )
    )
    drive(engine, 4, BLOCK)
    assert live.transport is TransportState.PAUSED

    for _ in range(40):
        session._autopilot_tick()
        drive(engine, 8, BLOCK)
        assert other.transport is not TransportState.PLAYING, (
            "pausing one deck started the other"
        )

    assert other.transport is before
    assert live.transport is TransportState.PAUSED


def test_a_paused_deck_and_an_ended_deck_are_different_events(session):
    """Same 'not advancing', opposite handling."""
    engine = session.engine
    live = engine.deck(session.live_deck)

    # Paused: the autopilot does nothing at all.
    engine.submit(
        LoadTrack(
            deck=live.name, track=live.track, start_frame=int(live.position),
            rate=live.rate, play=False, execute_at=IMMEDIATE, origin="test:pause",
        )
    )
    drive(engine, 4, BLOCK)
    session._autopilot_tick()
    session._autopilot_tick()
    assert not session.has_cued_track(), "a pause must not trigger an advance"

    # Ended: the autopilot recovers.
    live.ended = True
    live.playing = False
    assert live.transport is TransportState.LOADED_STOPPED
    session._autopilot_tick()
    session._autopilot_tick()
    drive(engine, 4, BLOCK)
    assert level(drive(engine, 20, BLOCK)) > 0.1, "an ended track must advance"


def test_the_play_command_has_an_autostart_flag_defaulting_off():
    assert cli.build_parser().parse_args(["play"]).autostart is False
    assert cli.build_parser().parse_args(["play", "--autostart"]).autostart is True


def test_autostart_still_opens_with_a_track(idle_session):
    """--autostart routes to the same start_first_track it always did."""
    session = idle_session
    assert session.start_first_track()
    drive(session.engine, 8, BLOCK)
    live = session.engine.deck(session.live_deck)
    assert live.transport is TransportState.PLAYING
    assert level(drive(session.engine, 20, BLOCK)) > 0.1


def test_the_play_command_accepts_the_hardening_flags():
    args = cli.build_parser().parse_args(
        ["play", "--no-llm", "--no-record", "--no-fallback", "--cue-device", "7"]
    )
    assert args.no_llm and args.no_record and args.no_fallback
    assert args.cue_device == 7

    args = cli.build_parser().parse_args(["play", "--cue-channels", "3,4"])
    assert args.cue_channels == "3,4"


def test_cue_device_and_cue_channels_are_refused_together(tmp_path, capsys):
    args = cli.build_parser().parse_args(
        ["play", "--cache", str(tmp_path), "--cue-device", "7",
         "--cue-channels", "3,4"]
    )
    # An empty crate would return 1 anyway, so give it one that loads.
    (tmp_path / "x").mkdir()
    assert cli.cmd_play(args) == 1

