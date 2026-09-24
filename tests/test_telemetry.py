"""Phase 1.1 / SPEC §1: the feature bus.

THREADING CONTEXT: main thread (pytest), plus a deliberately misbehaving
reader thread in the stalled-consumer test. The callback is driven directly.

What these tests are really defending is the first invariant -- nothing blocks
the audio callback -- so they measure the writer rather than trusting it: no
net allocation, no measurable timing cost, and no way for a reader to reach
back into the audio thread.
"""

from __future__ import annotations

import gc
import sys
import threading
import time

import numpy as np
import pytest

from djai import phrase, telemetry
from djai.commands import LoadTrack, SetEQ
from djai.deck import SAMPLE_RATE
from djai.engine import Engine
from tests.test_engine import drive, loud_track

BLOCK = 512


def running_engine(bpm: float = 128.0, blocks: int = 20) -> Engine:
    engine = Engine(blocksize=BLOCK)
    engine.submit(LoadTrack(deck="a", track=loud_track(200.0, bpm=bpm),
                            play=True, master=True, origin="t"))
    engine.deck_a.gain.jump(0.8)
    drive(engine, blocks)
    return engine


# --- what the row carries ------------------------------------------------------


def test_a_snapshot_reports_the_state_the_callback_saw():
    engine = running_engine()
    snap = engine.features.latest()
    assert snap is not None
    assert snap.seq == engine.features.published
    assert snap.frames_played == engine.frames_played
    assert snap.master_bpm == pytest.approx(128.0, abs=0.01)
    a = snap.a
    assert a.playing and a.native_bpm == pytest.approx(128.0)
    assert a.rate == pytest.approx(1.0)
    assert a.stretch_ratio == pytest.approx(1.0)
    assert a.gain == pytest.approx(0.8, abs=1e-6)
    assert a.tempo == pytest.approx(128.0, abs=0.01)
    engine.stop()


def test_the_bus_carries_every_field_the_spec_asks_for():
    """SPEC §1 names these by name; a missing one is a silent gap."""
    engine = running_engine()
    snap = engine.features.latest()
    for field in ("master_bpm", "transition_active", "frames_played"):
        assert hasattr(snap, field), field
    for field in ("bar", "bars_to_phrase_end", "seconds_to_downbeat", "energy",
                  "eq", "filter_pos", "loop_frames", "tempo", "phase",
                  "stretch_ratio"):
        assert hasattr(snap.a, field), field
    engine.stop()


def test_mixer_moves_show_up_in_the_next_snapshot():
    engine = running_engine()
    engine.submit(SetEQ(deck="a", low=0.25, origin="t"))
    drive(engine, 8)
    assert engine.features.latest().a.eq[0] == pytest.approx(0.25, abs=1e-6)
    engine.stop()


def test_derived_musical_time_agrees_with_the_phrase_module():
    """The bus derives bar position itself, from constants the row carries.

    That duplication is deliberate (the callback must not call into analysis),
    so it has to be pinned against the single source of truth it mirrors.
    """
    engine = running_engine(bpm=124.0, blocks=57)
    snap = engine.features.latest()
    analysis = engine.deck("a").track.analysis
    position = snap.a.position_frames
    assert snap.a.beat == pytest.approx(
        phrase.beat_at_frame(analysis, position), abs=1e-6)
    assert snap.a.bar == pytest.approx(
        phrase.bar_at_frame(analysis, position), abs=1e-6)
    engine.stop()


def test_an_empty_deck_reports_zeroes_rather_than_nonsense():
    engine = Engine(blocksize=BLOCK)
    drive(engine, 4)
    b = engine.features.latest().b
    assert not b.playing
    assert b.beat == 0.0 and b.bar == 0.0
    assert b.seconds_to_downbeat == 0.0
    assert b.energy == 0.0
    engine.stop()


# --- the ring ------------------------------------------------------------------


def test_the_ring_overwrites_the_oldest_and_counts_what_a_reader_lost():
    bus = telemetry.FeatureBus(capacity=8)
    engine = Engine(blocksize=BLOCK)
    for _ in range(20):
        bus.publish(time.monotonic(), 0, 0.0, 0.0, False, 0.0, 0.0, 1.0,
                    engine.deck_a, engine.deck_b)
    assert bus.sequence == 20
    assert bus._read(1) is None, "an overwritten row must not be served"
    got = bus.since(0)
    assert len(got) <= 8
    assert bus.dropped > 0, "a reader that fell behind was not told"
    assert [s.seq for s in got] == sorted(s.seq for s in got)
    engine.stop()


def test_since_returns_only_what_is_new():
    engine = running_engine(blocks=10)
    mark = engine.features.sequence
    drive(engine, 5)
    fresh = engine.features.since(mark)
    assert [s.seq for s in fresh] == list(range(mark + 1, mark + 6))
    engine.stop()


# --- the invariant: the writer is never slowed by anyone ----------------------


def test_publishing_allocates_nothing_that_survives_the_call():
    """No allocation in the real-time thread.

    Measured rather than asserted by inspection: a net growth in allocated
    blocks across a thousand publishes would mean the callback is handing work
    to the garbage collector.
    """
    engine = Engine(blocksize=BLOCK)
    bus = telemetry.FeatureBus(capacity=64)
    args = (time.monotonic(), 0, 0.0, 0.0, False, 0.0, 0.0, 1.0,
            engine.deck_a, engine.deck_b)
    for _ in range(200):                       # warm every code path first
        bus.publish(*args)
    gc.collect()
    before = sys.getallocatedblocks()
    for _ in range(1000):
        bus.publish(*args)
    gc.collect()
    growth = sys.getallocatedblocks() - before
    assert growth <= 8, f"publish leaked {growth} allocated blocks"
    engine.stop()


def test_a_stalled_consumer_cannot_slow_the_callback():
    """The point of the whole design: there is no backpressure to apply.

    A reader is parked mid-read for the length of the run. The writer must not
    notice -- it never looks at consumer state, so there is nothing to notice
    with.
    """
    engine = running_engine(blocks=4)
    stop = threading.Event()
    started = threading.Event()

    def stalled_reader() -> None:
        engine.features.latest()
        started.set()
        stop.wait(5.0)        # holds a snapshot and does nothing with it

    thread = threading.Thread(target=stalled_reader, daemon=True)
    thread.start()
    assert started.wait(2.0)

    before = engine.features.published
    t0 = time.perf_counter()
    drive(engine, 200)
    elapsed = time.perf_counter() - t0
    stop.set()
    thread.join(2.0)

    assert engine.features.published == before + 200, "the writer lost blocks"
    # 200 blocks of 512 frames is 2.3 s of audio; a blocked writer would show
    # up as wall time in the same order, not milliseconds.
    assert elapsed < 2.0, f"callback ran slowly with a stalled reader: {elapsed:.3f}s"
    engine.stop()


def test_the_bus_costs_the_callback_almost_nothing():
    """Reported as a number, not a claim. Compared against the same engine
    with publishing stubbed out, so only the bus is being measured."""
    engine = running_engine(blocks=8)

    def timed(n: int) -> float:
        buf = np.zeros((BLOCK, 2), dtype=np.float32)
        t0 = time.perf_counter()
        for _ in range(n):
            engine.callback(buf, BLOCK, None, None)
        return (time.perf_counter() - t0) / n

    with_bus = timed(400)
    real, engine.features.publish = engine.features.publish, lambda *a, **k: None
    without_bus = timed(400)
    engine.features.publish = real

    cost_us = (with_bus - without_bus) * 1e6
    block_us = BLOCK / SAMPLE_RATE * 1e6          # 11,610 us of audio per block
    print(f"\nfeature bus: {cost_us:.1f} us/block of {block_us:.0f} us "
          f"({100 * cost_us / block_us:.2f}% of the budget)")
    assert cost_us < 0.05 * block_us, (
        f"bus costs {cost_us:.1f} us/block, over 5% of the {block_us:.0f} us budget"
    )
    engine.stop()


def test_snapshot_latency_p99_is_under_50ms():
    """SPEC §1's number. Latency is snapshot age at the moment a consumer
    reads it, which is what a consumer actually experiences."""
    engine = running_engine(blocks=4)
    ages = []
    buf = np.zeros((BLOCK, 2), dtype=np.float32)
    for _ in range(500):
        engine.callback(buf, BLOCK, None, None)
        snap = engine.features.latest()
        ages.append(snap.age_seconds * 1000.0)
    p99 = float(np.percentile(ages, 99))
    print(f"\nsnapshot age: p50 {np.percentile(ages, 50):.3f} ms, "
          f"p99 {p99:.3f} ms, max {max(ages):.3f} ms")
    assert p99 < 50.0, f"p99 snapshot latency {p99:.1f} ms"
    engine.stop()
