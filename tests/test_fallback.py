"""The safety net: does the room still hear something when the engine stops?

THREADING CONTEXT: main thread (pytest). No device is opened -- the watchdog's
own-stream path is exercised through a stubbed sounddevice, and the callback is
driven directly. Timing assertions use the watchdog's real clock but a shortened
stall threshold, so the tests measure the logic, not the wall clock.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from djai.commands import LoadTrack
from djai.deck import CHANNELS, SAMPLE_RATE
from djai.engine import Engine
from djai.fallback import FallbackPlayer, Watchdog
from tests.test_engine import drive, loud_track

BLOCK = 512


def player(seconds: float = 2.0, freq: float = 300.0) -> FallbackPlayer:
    track = loud_track(freq, bpm=120.0, seconds=seconds)
    return FallbackPlayer.from_track(track)


def level(buf: np.ndarray) -> float:
    return float(np.max(np.abs(buf)))


# --- the player ---------------------------------------------------------------


def test_the_player_makes_sound():
    p = player()
    out = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    p.fill(out, BLOCK)
    assert level(out) > 0.1


def test_the_player_loops_forever_without_a_gap():
    """It has to outlast the set. A one-shot player is a delayed silence."""
    p = player(seconds=0.5)
    out = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    source_frames = p.audio.shape[0]
    blocks = int(source_frames / BLOCK * 3) + 10

    quiet = 0
    for _ in range(blocks):
        p.fill(out, BLOCK)
        if level(out) < 1e-6:
            quiet += 1
    assert p.loops >= 2, "should have wrapped at least twice"
    assert quiet == 0, f"{quiet} silent blocks while looping"


def test_the_player_never_allocates_in_fill():
    """Same rule as the audio callback: it runs on one."""
    p = player()
    out = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    p.fill(out, BLOCK)  # warm any lazy numpy setup

    import tracemalloc

    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for _ in range(50):
        p.fill(out, BLOCK)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    grown = sum(s.size_diff for s in after.compare_to(before, "lineno") if s.size_diff > 0)
    assert grown < 8192, f"fill() allocated {grown} bytes over 50 blocks"


def test_a_short_source_is_padded_once_up_front():
    """Padding at construction keeps the wrap logic out of the audio thread."""
    tiny = np.ones((128, CHANNELS), dtype=np.float32) * 0.5
    p = FallbackPlayer(tiny)
    assert p.audio.shape[0] > 128
    out = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    p.fill(out, BLOCK)
    assert level(out) > 0.1


def test_the_player_survives_a_broken_buffer():
    """The code that runs when everything else failed cannot itself throw."""
    p = player()
    p.audio = "not an array"  # the worst thing that could have happened to it
    out = np.ones((BLOCK, CHANNELS), dtype=np.float32)
    p.fill(out, BLOCK)
    assert level(out) == 0.0, "should fail to silence, not raise"


def test_the_player_fills_a_wide_buffer_without_leaking_into_the_cue_pair():
    p = player()
    out = np.ones((BLOCK, 4), dtype=np.float32)
    p.fill(out, BLOCK)
    assert level(out[:, 0:2]) > 0.1
    assert level(out[:, 2:4]) == 0.0


# --- takeover -----------------------------------------------------------------


@pytest.fixture
def engine_with_audio():
    engine = Engine(blocksize=BLOCK)
    engine.submit(
        LoadTrack(deck="a", track=loud_track(440.0, seconds=30.0), master=True)
    )
    engine.deck_a.gain.jump(1.0)
    drive(engine, 2, BLOCK)
    return engine


def test_takeover_is_audible_on_the_very_next_block(engine_with_audio):
    engine = engine_with_audio
    p = player(freq=300.0)
    dog = Watchdog(engine, p)

    assert level(drive(engine, 2, BLOCK)) > 0.1
    dog.take_over("test")

    out = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    engine.callback(out, BLOCK, None, None)
    assert level(out) > 0.1, "the very next block after takeover must have audio"
    assert p.frames_played == BLOCK, "the block came from the fallback, not the mix"


def test_takeover_keeps_producing_audio_indefinitely(engine_with_audio):
    engine = engine_with_audio
    dog = Watchdog(engine, player(seconds=1.0))
    dog.take_over("test")

    quiet = 0
    for _ in range(400):  # ~4.6 s at 512 frames
        out = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
        engine.callback(out, BLOCK, None, None)
        if level(out) < 1e-6:
            quiet += 1
    assert quiet == 0, f"{quiet} silent blocks after takeover"


def test_takeover_is_idempotent(engine_with_audio):
    dog = Watchdog(engine_with_audio, player())
    dog.take_over("first")
    dog.take_over("second")
    assert dog.reason == "first", "the first reason is the one that explains it"


def test_release_gives_the_output_back(engine_with_audio):
    engine = engine_with_audio
    dog = Watchdog(engine, player())
    dog.take_over("test")
    dog.release()
    assert engine.fallback is None
    assert not dog.tripped
    assert level(drive(engine, 2, BLOCK)) > 0.1


# --- the decision -------------------------------------------------------------


def test_a_stalled_callback_trips_the_watchdog(engine_with_audio):
    """The acceptance case: the engine thread stops, the room does not."""
    engine = engine_with_audio
    dog = Watchdog(engine, player(), stall_ms=50.0)

    assert dog.check() is None, "a healthy engine must not trip"

    # Stop driving the callback. This is what a dead engine thread looks like
    # from outside: the counters simply stop advancing.
    time.sleep(0.08)
    reason = dog.check()
    assert reason is not None and "stalled" in reason


def test_a_stall_is_not_reported_before_the_first_block():
    """An engine that has not started yet is not an engine that has died."""
    engine = Engine(blocksize=BLOCK)
    dog = Watchdog(engine, player(), stall_ms=1.0)
    time.sleep(0.02)
    assert dog.check() is None


def test_a_burst_of_underruns_trips_but_a_single_one_does_not(engine_with_audio):
    engine = engine_with_audio
    dog = Watchdog(engine, player(), underrun_burst=8)
    dog._last_underruns = engine.underruns

    engine.underruns += 1
    assert dog.check() is None, "one underrun is system noise"

    engine.underruns += 8
    reason = dog.check()
    assert reason is not None and "underrun" in reason


# --- handing the output back ----------------------------------------------------


def test_the_output_is_handed_back_once_the_engine_is_healthy_again(engine_with_audio):
    """A takeover freezes the mix, so a passing fault must not end the night.

    While the fallback holds the output the engine drains no commands and no
    deck advances: nothing is ever cued again. A 450 ms stall is far too small
    a fault to cost that, so a healthy engine gets the output back.
    """
    engine = engine_with_audio
    released: list[str] = []
    dog = Watchdog(engine, player(), stall_ms=50.0, recover_after_s=0.5,
                   on_release=released.append)
    dog.take_over("a passing stall")
    assert engine.fallback is not None

    dog._last_callbacks = engine.callbacks
    drive(engine, 2, BLOCK)
    dog._consider_recovery()          # first healthy look starts the clock
    assert engine.fallback is not None, "not on the strength of one look"
    # Wind that clock back rather than sleeping: a real sleep here measures the
    # machine's load, and this test is about the logic.
    dog._healthy_since -= 1.0
    drive(engine, 2, BLOCK)
    dog._consider_recovery()

    assert engine.fallback is None, "the engine has the output back"
    assert not dog.tripped and dog.recoveries == 1
    assert released and "healthy" in released[0]
    assert level(drive(engine, 2, BLOCK)) > 0.1, "and the mix is audible again"


def test_a_still_broken_engine_keeps_the_fallback(engine_with_audio):
    engine = engine_with_audio
    dog = Watchdog(engine, player(), stall_ms=20.0, recover_after_s=0.0)
    dog.take_over("stalled")
    dog._last_callbacks = engine.callbacks

    time.sleep(0.05)                  # no callbacks at all: still dead
    dog._consider_recovery()
    assert engine.fallback is not None
    assert dog.recoveries == 0


def test_a_dead_thread_keeps_the_fallback_however_healthy_the_callback_is(
    engine_with_audio,
):
    engine = engine_with_audio
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    dog = Watchdog(engine, player(), stall_ms=50.0, recover_after_s=0.0,
                   threads={"scheduler": dead})
    dog.take_over("scheduler thread died")

    dog._last_callbacks = engine.callbacks
    drive(engine, 2, BLOCK)
    dog._consider_recovery()
    dog._consider_recovery()
    assert engine.fallback is not None, "nothing would be cued even so"


def test_it_stops_handing_back_after_the_recovery_limit(engine_with_audio):
    """Flapping between two sources all night is its own fault."""
    engine = engine_with_audio
    dog = Watchdog(engine, player(), stall_ms=50.0, recover_after_s=0.0,
                   max_recoveries=2)
    for _ in range(3):
        dog.take_over("stall")
        dog._last_callbacks = engine.callbacks
        drive(engine, 2, BLOCK)
        dog._consider_recovery()
        dog._consider_recovery()
    assert dog.recoveries == 2
    assert engine.fallback is not None, "the third fault keeps the fallback"


def test_a_dead_scheduler_thread_trips_the_watchdog(engine_with_audio):
    """Nothing gets queued again, so the current track plays out into silence."""
    engine = engine_with_audio
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()

    dog = Watchdog(engine, player(), threads={"scheduler": dead})
    reason = dog.check()
    assert reason == "scheduler thread died"


def test_a_live_thread_does_not_trip_the_watchdog(engine_with_audio):
    stop = threading.Event()
    alive = threading.Thread(target=stop.wait, daemon=True)
    alive.start()
    try:
        dog = Watchdog(engine_with_audio, player(), threads={"scheduler": alive})
        assert dog.check() is None
    finally:
        stop.set()
        alive.join(timeout=1)


def test_killing_the_engines_thread_is_audible_within_500ms(engine_with_audio):
    """The acceptance case, measured end to end through the real watchdog thread.

    "Killing the engine thread" in this program means killing the thread that
    keeps music *coming* -- the scheduler. PortAudio's own thread carries on
    calling the callback, so nothing sounds wrong for one track and then the
    room goes quiet forever. The watchdog has to notice the dead thread, not
    wait for the silence.
    """
    engine = engine_with_audio
    p = player(freq=300.0)

    keep_running = threading.Event()
    scheduler = threading.Thread(target=keep_running.wait, daemon=True)
    scheduler.start()

    dog = Watchdog(engine, p, threads={"scheduler": scheduler}, poll_hz=20.0)

    stop = threading.Event()
    silent_blocks = [0]
    first_fallback_at = [None]

    def device():
        """PortAudio's thread: keeps calling the callback throughout."""
        out = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
        period = BLOCK / SAMPLE_RATE
        while not stop.is_set():
            engine.callback(out, BLOCK, None, None)
            if level(out) < 1e-6:
                silent_blocks[0] += 1
            if p.frames_played > 0 and first_fallback_at[0] is None:
                first_fallback_at[0] = time.perf_counter()
            time.sleep(period)

    dog.start()
    thread = threading.Thread(target=device, daemon=True)
    thread.start()
    time.sleep(0.4)
    assert not dog.tripped, "tripped while everything was healthy"

    keep_running.set()          # kill the scheduler
    scheduler.join(timeout=1)
    died_at = time.perf_counter()

    time.sleep(1.0)
    stop.set()
    thread.join(timeout=2)
    dog.stop()

    assert dog.tripped, f"watchdog never tripped (checks={dog.checks})"
    assert dog.reason == "scheduler thread died"
    assert first_fallback_at[0] is not None, "no audio ever came from the fallback"
    took_ms = (first_fallback_at[0] - died_at) * 1000
    assert took_ms < 500.0, f"fallback took {took_ms:.0f} ms to become audible"
    assert silent_blocks[0] == 0, f"{silent_blocks[0]} blocks of dead air"


def test_a_dead_callback_is_covered_by_the_watchdogs_own_stream(monkeypatch):
    """The other half: PortAudio itself stops calling. Nothing to route to."""
    engine = Engine(blocksize=BLOCK)
    engine.submit(
        LoadTrack(deck="a", track=loud_track(440.0, seconds=10.0), master=True)
    )
    engine.deck_a.gain.jump(1.0)

    made_audio = []

    class FakeStream:
        def __init__(self, **kwargs):
            self.active = True
            self._cb = kwargs["callback"]

        def start(self):
            out = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
            self._cb(out, BLOCK, None, None)
            made_audio.append(level(out))

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("djai.fallback.sd.OutputStream", FakeStream)
    engine._stream = object()   # it HAD a device
    monkeypatch.setattr(type(engine), "running", property(lambda self: False))

    p = player()
    dog = Watchdog(engine, p, stall_ms=100.0, poll_hz=20.0)
    dog.start()
    try:
        # Drive a couple of blocks so the stall clock is meaningful, then stop.
        out = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
        engine.callback(out, BLOCK, None, None)

        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline and not dog.tripped:
            time.sleep(0.02)
    finally:
        dog.stop()

    assert dog.tripped, "a callback that stopped being called must trip"
    assert made_audio and made_audio[0] > 0.1, "the own-stream produced no audio"


def test_the_engine_keeps_serving_the_fallback_even_if_mixing_would_throw(
    engine_with_audio,
):
    """Takeover must not depend on any of the machinery that just failed."""
    engine = engine_with_audio
    dog = Watchdog(engine, player())
    dog.take_over("test")

    # Break the mixer completely. The fallback branch returns before any of it.
    engine.deck_a = None
    engine._decks = None
    engine._mix = None

    out = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    engine.callback(out, BLOCK, None, None)
    assert level(out) > 0.1


def test_takeover_notifies_and_a_broken_listener_cannot_stop_it(engine_with_audio):
    seen = []
    dog = Watchdog(
        engine_with_audio, player(), on_takeover=lambda r: seen.append(r)
    )
    dog.take_over("because")
    assert seen == ["because"]

    dog2 = Watchdog(engine_with_audio, player(), on_takeover=lambda r: 1 / 0)
    dog2.take_over("still works")
    assert dog2.tripped and engine_with_audio.fallback is not None


def test_the_watchdog_opens_its_own_stream_when_the_engines_is_gone(monkeypatch):
    """The other failure: nothing is calling the callback at all."""
    engine = Engine(blocksize=BLOCK)
    engine.submit(
        LoadTrack(deck="a", track=loud_track(440.0, seconds=5.0), master=True)
    )
    engine.deck_a.gain.jump(1.0)
    drive(engine, 2, BLOCK)

    opened = {}

    class FakeStream:
        def __init__(self, **kwargs):
            opened.update(kwargs)
            self.active = True

        def start(self):
            opened["started"] = True

        def stop(self):
            opened["stopped"] = True

        def close(self):
            pass

    monkeypatch.setattr("djai.fallback.sd.OutputStream", FakeStream)
    # The engine HAD a stream and it is no longer active: routing the callback
    # is useless because nothing is calling it.
    engine._stream = object()
    monkeypatch.setattr(type(engine), "running", property(lambda self: False))

    p = player()
    dog = Watchdog(engine, p)
    dog.take_over("stream died")

    assert opened.get("started") is True, "should have opened its own stream"
    assert opened["samplerate"] == SAMPLE_RATE
    assert opened["channels"] == CHANNELS

    # And that stream produces audio.
    out = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    opened["callback"](out, BLOCK, None, None)
    assert level(out) > 0.1

    dog._close_own_stream()
    assert opened.get("stopped") is True


def test_no_own_stream_is_opened_when_the_engine_is_still_running(
    engine_with_audio, monkeypatch
):
    """Routing the existing callback is enough, and cheaper than a new stream."""
    def boom(**kwargs):
        raise AssertionError("should not have opened a stream")

    monkeypatch.setattr("djai.fallback.sd.OutputStream", boom)
    engine_with_audio._stream = object()
    monkeypatch.setattr(
        type(engine_with_audio), "running", property(lambda self: True)
    )
    dog = Watchdog(engine_with_audio, player())
    dog.take_over("engine broken but stream alive")
    assert dog.tripped


def test_an_engine_that_never_had_a_device_is_never_given_one(monkeypatch):
    """An offline render or a test is not a failed engine.

    Regression: this used to open the sound card, and PortAudio aborted the
    whole process rather than raising -- which no try/except can catch.
    """
    def boom(**kwargs):
        raise AssertionError("should not have touched the sound card")

    monkeypatch.setattr("djai.fallback.sd.OutputStream", boom)
    engine = Engine(blocksize=BLOCK)
    assert engine._stream is None
    dog = Watchdog(engine, player())
    dog.take_over("no device was ever opened")
    assert dog.tripped
    assert engine.fallback is not None
