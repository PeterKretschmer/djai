"""Phase 1.3 / SPEC §9-§10: the device goes away, and the room still hears music.

THREADING CONTEXT: main thread (pytest). The callback is driven directly and
no audio device is opened -- which is also the honest limit of these tests:
they exercise the *recovery contract* against a simulated stream failure, not
real hardware. A genuine unplug/replug needs a real interface, and nothing
here should be read as claiming that coverage.

What a lost device looks like from inside this program, and what each one must
do, is the whole subject:

* PortAudio stops calling the callback  -> the stall check trips
* the stream object reports not running -> the stream check trips
* a steering thread dies                -> that check trips
* underruns arrive in a burst           -> that check trips

and in every case the fallback takes the output, the audio stays continuous,
and the engine is handed back once it is healthy again.
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


@pytest.fixture
def engine():
    made = Engine(blocksize=BLOCK)
    made.submit(LoadTrack(deck="a", track=loud_track(200.0, bpm=128.0),
                          play=True, master=True, origin="t"))
    made.deck_a.gain.jump(0.8)
    drive(made, 8)
    yield made
    made.stop()


@pytest.fixture
def player():
    tone = (0.4 * np.sin(np.linspace(0, 2000, SAMPLE_RATE))).astype(np.float32)
    return FallbackPlayer(np.stack([tone, tone], axis=1), title="safety")


class DeadStream:
    """A stream object that reports itself stopped, as a lost device does."""

    active = False


# --- detection ----------------------------------------------------------------


def test_a_stalled_callback_is_detected(engine, player):
    """The device stopped calling us. Measured on a real machine once: 481 s
    of audio in 630 s of wall clock, no error, underruns still zero."""
    dog = Watchdog(engine, player, stall_ms=50.0)
    assert dog.check() is None
    engine.last_callback_at = time.monotonic() - 5.0
    reason = dog.check()
    assert reason is not None and "stalled" in reason


def test_a_stream_that_is_no_longer_running_is_detected(engine, player):
    dog = Watchdog(engine, player, stall_ms=50.0)
    engine._stream = DeadStream()
    reason = dog.check()
    assert reason is not None and "no longer running" in reason


def test_an_engine_with_no_stream_is_not_treated_as_failed(engine, player):
    """An offline render or a test has no stream and has not lost a device."""
    dog = Watchdog(engine, player, stall_ms=50.0)
    assert engine._stream is None
    assert dog.check() is None


def test_a_dead_steering_thread_is_detected(engine, player):
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    dog = Watchdog(engine, player, threads={"scheduler": dead}, stall_ms=50.0)
    reason = dog.check()
    assert reason is not None and "scheduler" in reason


def test_a_burst_of_underruns_is_detected(engine, player):
    dog = Watchdog(engine, player, stall_ms=50.0, underrun_burst=3)
    engine.underruns += 5
    reason = dog.check()
    assert reason is not None and "underrun" in reason


# --- taking over --------------------------------------------------------------


def test_the_fallback_takes_the_output_and_the_audio_keeps_coming(engine, player):
    """The point of all of it: whatever failed, the room does not hear silence."""
    dog = Watchdog(engine, player, stall_ms=50.0)
    engine.fallback = None
    dog.take_over("device lost")
    assert dog.tripped
    assert engine.fallback is not None

    out = drive(engine, 40)
    assert np.isfinite(out).all()
    assert float(np.abs(out).max()) > 0.01, "the fallback produced silence"
    engine.stop()


def test_the_takeover_is_announced_once(engine, player):
    said: list[str] = []
    dog = Watchdog(engine, player, stall_ms=50.0, on_takeover=said.append)
    dog.take_over("device lost")
    dog.take_over("device lost again")
    assert len(said) == 1, "a second takeover announced itself twice"


def test_audio_is_continuous_across_the_takeover(engine, player):
    """No gap at the seam: blocks either side must both carry audio."""
    dog = Watchdog(engine, player, stall_ms=50.0)
    before = drive(engine, 8)
    dog.take_over("device lost")
    after = drive(engine, 8)
    assert float(np.abs(before).max()) > 0.01
    assert float(np.abs(after).max()) > 0.01
    assert np.isfinite(after).all()
    engine.stop()


# --- handing back -------------------------------------------------------------


def test_the_engine_is_handed_back_once_it_is_healthy_again(engine, player):
    """Unplug, replug: the fallback must not keep the output for ever."""
    released: list[str] = []
    dog = Watchdog(engine, player, stall_ms=50.0, recover_after_s=0.0,
                   on_release=released.append)
    dog.take_over("device lost")
    assert dog.tripped

    drive(engine, 8)                      # the device is back: blocks flowing
    dog._consider_recovery()              # first call arms the healthy timer
    drive(engine, 8)
    dog._consider_recovery()

    assert not dog.tripped, "the fallback never handed back"
    assert engine.fallback is None
    assert released and "healthy" in released[0]
    engine.stop()


def test_an_engine_that_is_still_dead_is_not_handed_back_to(engine, player):
    dog = Watchdog(engine, player, stall_ms=50.0, recover_after_s=0.0)
    dog.take_over("device lost")
    engine._stream = DeadStream()
    drive(engine, 8)
    dog._consider_recovery()
    dog._consider_recovery()
    assert dog.tripped, "handed back to an engine whose stream is stopped"
    engine.stop()


def test_handing_back_is_bounded(engine, player):
    """A device that flaps must not be handed the set back for ever."""
    dog = Watchdog(engine, player, stall_ms=50.0, recover_after_s=0.0,
                   max_recoveries=1)
    for _ in range(3):
        dog.take_over("device lost")
        drive(engine, 8)
        dog._consider_recovery()
        dog._consider_recovery()
    assert dog.recoveries <= 1
    engine.stop()


# --- the sample rate is not negotiable ----------------------------------------


def test_the_engine_asks_for_one_sample_rate_only():
    """A device that silently ran at another rate would pitch the whole set.
    The engine names SAMPLE_RATE when it opens the stream and resamples every
    file to it at load, so there is one rate in the program and no other."""
    import inspect

    from djai import engine as engine_mod

    source = inspect.getsource(engine_mod.Engine.start)
    assert "samplerate=SAMPLE_RATE" in source
    assert engine_mod.SAMPLE_RATE == SAMPLE_RATE


def test_a_block_larger_than_the_engine_expected_is_survivable(engine):
    """PortAudio may hand over a different blocksize than requested."""
    for frames in (64, 256, 512, 1024):
        buf = np.zeros((frames, CHANNELS), dtype=np.float32)
        engine.callback(buf, frames, None, None)
        assert np.isfinite(buf).all(), f"{frames}-frame block produced bad audio"


def test_a_stalled_pre_roll_worker_cannot_grow_memory_without_limit():
    """The pre-roll queue is bounded: announcements past it are dropped, not
    held -- each one can pin a whole decoded track. The command still fires."""
    from djai.commands import SetEQ
    from djai.scheduler import Scheduler

    engine = Engine(blocksize=BLOCK)
    sch = Scheduler(engine, on_pre_roll=lambda cmd: None, pre_roll_frames=10**9)
    for i in range(100):                      # worker never started: nothing drains
        sch.submit(SetEQ(deck="a", low=1.0, execute_at=10**6 + i, origin="t"))
    sch.tick(0)                               # announces everything in range
    assert sch._pre_roll_q.qsize() <= sch._pre_roll_q.maxsize
    assert len(sch.tick(10**7)) == 100, "a dropped preview must not drop the command"
    engine.stop()
