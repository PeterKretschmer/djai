"""Phase 0B: the master clock follows the set, and the stretch stays sane.

The bug this pins down, in arithmetic: an incoming track is matched to master
before the blend, so its rate is ``master / native``. ``hand_master_to`` then
adopts ``native * rate`` -- which is ``master`` again. The hand-over could
never move the clock, so a set stayed at whatever tempo it opened at and every
later track was stretched back to it for the rest of the night.

THREADING CONTEXT: main thread (pytest). The callback is driven directly and
the scheduler is stepped in lockstep, so every timing assertion is exact and
no audio device is opened.
"""

from __future__ import annotations

import numpy as np
import pytest

from djai import config
from djai.commands import LoadTrack, SetPitch
from djai.deck import CHANNELS, SAMPLE_RATE, LoadedTrack
from djai.engine import Engine
from tests.test_engine import drive, loud_track


# --- the arithmetic that caused the lock ---------------------------------------


def test_the_hand_over_alone_cannot_move_the_clock():
    """Documents the root cause: it is a no-op by construction, not a bug in
    ``hand_master_to`` itself. This is why the glide has to exist."""
    engine = Engine(blocksize=512)
    track = loud_track(200.0, bpm=128.0)
    master = 120.0
    rate = master / track.analysis.bpm          # how a cue is always matched
    engine.submit(LoadTrack(deck="a", track=track, rate=rate, play=True,
                            master=True, origin="t"))
    drive(engine, 4)
    assert engine.master_bpm == pytest.approx(120.0)
    # the hand-over adopts native * rate == master: nothing moves
    assert engine.hand_master_to("a") == pytest.approx(120.0)
    assert engine.master_bpm == pytest.approx(120.0)
    engine.stop()


# --- the glide -----------------------------------------------------------------


def glide_engine(bpm: float, master: float) -> tuple[Engine, LoadedTrack, float]:
    engine = Engine(blocksize=512)
    track = loud_track(200.0, bpm=bpm)
    rate = master / bpm
    engine.submit(LoadTrack(deck="a", track=track, rate=rate, play=True,
                            master=True, origin="t"))
    drive(engine, 4)
    return engine, track, rate


@pytest.mark.parametrize("native, master", [
    (128.0, 120.0), (120.0, 128.0), (124.0, 120.0), (132.0, 126.0),
])
def test_a_pitch_step_to_unity_lands_the_clock_on_the_tracks_own_bpm(native, master):
    """What the scheduled glide's last step does, checked at the engine."""
    engine, _track, rate = glide_engine(native, master)
    assert engine.master_bpm == pytest.approx(master, abs=0.01)
    engine.submit(SetPitch(deck="a", rate=1.0, origin="master_glide"))
    drive(engine, 4)
    assert engine.master_bpm == pytest.approx(native, abs=0.01)
    engine.stop()


def test_the_glide_is_monotonic_and_never_overshoots():
    """No audible tempo jump: each step moves the same way and stops on target."""
    engine, _track, rate = glide_engine(128.0, 120.0)
    seen = [engine.master_bpm]
    steps = 16
    for i in range(1, steps + 1):
        r = rate + (1.0 - rate) * i / steps
        engine.submit(SetPitch(deck="a", rate=(1.0 if i == steps else r),
                               origin="master_glide"))
        drive(engine, 2)
        seen.append(engine.master_bpm)
    assert seen[0] == pytest.approx(120.0, abs=0.01)
    assert seen[-1] == pytest.approx(128.0, abs=0.01)
    assert all(b <= a + 1e-6 for a, b in zip(seen, seen[1:])) or \
           all(b >= a - 1e-6 for a, b in zip(seen, seen[1:])), "glide reversed"
    assert max(seen) <= 128.0 + 0.01, "overshot the target tempo"
    engine.stop()


# --- stretch applied exactly once ----------------------------------------------


def stretched_copy(track: LoadedTrack, rate: float) -> LoadedTrack:
    """A stand-in for a Rubber Band copy: the right length and stretch_rate,
    without paying for the real tool in a unit test."""
    n = int(round(track.audio.shape[0] / rate))
    audio = np.zeros((n, CHANNELS), dtype=np.float32)
    audio[:] = 0.25
    return LoadedTrack(analysis=track.analysis, audio=audio,
                       stretch_rate=float(rate), source=track)


def test_a_stretched_buffer_is_played_at_step_one_so_the_stretch_is_applied_once():
    """rate/stretch == 1.0 is the whole invariant: the buffer already carries
    the tempo change, so the deck must not resample it as well."""
    engine = Engine(blocksize=512)
    src = loud_track(200.0, bpm=128.0)
    rate = 120.0 / 128.0
    copy = stretched_copy(src, rate)
    engine.submit(LoadTrack(deck="a", track=copy, rate=rate, play=True,
                            master=True, origin="t"))
    drive(engine, 4)
    deck = engine.deck("a")
    assert deck.track.is_stretched
    step = deck.rate / deck.track.stretch_rate
    assert step == pytest.approx(1.0, abs=1e-9)
    engine.stop()


def test_returning_to_unity_puts_the_source_back_under_the_playhead():
    """Otherwise the deck resamples the stretched copy by 1/stretch and the
    pitch is out by exactly the stretch that was meant to be inaudible."""
    engine = Engine(blocksize=512)
    src = loud_track(200.0, bpm=128.0)
    rate = 120.0 / 128.0
    copy = stretched_copy(src, rate)
    engine.submit(LoadTrack(deck="a", track=copy, rate=rate, play=True,
                            master=True, origin="t"))
    drive(engine, 4)
    assert engine.deck("a").track.is_stretched

    engine.submit(SetPitch(deck="a", rate=1.0, origin="master_glide"))
    drive(engine, 4)
    deck = engine.deck("a")
    assert not deck.track.is_stretched, "still on the stretched copy at unity"
    assert deck.track is src
    assert deck.rate == pytest.approx(1.0)
    engine.stop()


# --- the cap -------------------------------------------------------------------


def test_the_requested_rate_never_exceeds_the_cap(tmp_path):
    """Past the cap the stretcher refuses a key-locked copy and the deck
    resamples the whole rate -- 0.75 for a 160 BPM track under a 120 clock,
    five semitones down. The rate is capped before that can happen."""
    from djai import cli
    from tests.test_integration import make_analysis

    crate = [make_analysis("x", bpm, "8A", 0.5, tmp_path / f"{bpm}.wav")
             for bpm in (72.0, 91.0, 100.0, 128.0, 140.0, 160.0, 175.0)]
    session = cli.Session(crate, log_dir=tmp_path / "logs")
    try:
        for track in crate:
            rate = session._rate_for(track)
            assert abs(rate - 1.0) <= config.MAX_STRETCH_RATIO + 1e-9, (
                f"{track.bpm} BPM asked for rate {rate:.4f}, past the "
                f"+-{config.MAX_STRETCH_RATIO:.0%} cap"
            )
    finally:
        session.shutdown()


def test_the_clamped_rate_survives_the_supervisor(tmp_path):
    """The cap and the supervisor's check must agree at the boundary.

    `abs(1.08 - 1.0)` is 0.08000000000000007, and the supervisor compares with
    no epsilon. A rate clamped to the limit exactly was refused, and cueing
    stalled: 4 of 20 tracks reached a deck in the soak that caught this.
    """
    from djai import cli
    from djai.supervisor import MAX_RATE_DELTA
    from tests.test_integration import make_analysis

    crate = [make_analysis("x", bpm, "8A", 0.5, tmp_path / f"{bpm}.wav")
             for bpm in (72.0, 90.0, 100.0, 111.0, 128.0, 140.0, 160.0, 175.0)]
    session = cli.Session(crate, log_dir=tmp_path / "logs")
    try:
        for track in crate:
            rate = session._rate_for(track)
            assert abs(rate - 1.0) <= MAX_RATE_DELTA, (
                f"{track.bpm} BPM -> rate {rate!r}, delta {abs(rate - 1.0)!r}, "
                f"which the supervisor refuses"
            )
    finally:
        session.shutdown()


def test_half_and_double_time_is_preferred_to_clamping(tmp_path):
    """A 160 BPM track under a 120 clock counts perfectly well at 80."""
    from djai import cli
    from tests.test_integration import make_analysis

    crate = [make_analysis("a", 120.0, "8A", 0.5, tmp_path / "a.wav"),
             make_analysis("b", 240.0, "8A", 0.5, tmp_path / "b.wav")]
    session = cli.Session(crate, log_dir=tmp_path / "logs")
    try:
        # 120/240 = 0.5 -> doubled to 1.0, an exact half-time count
        assert session._rate_for(crate[1]) == pytest.approx(1.0)
    finally:
        session.shutdown()


# --- the manual lock -----------------------------------------------------------


def test_the_master_lock_is_off_by_default_and_suppresses_the_glide(tmp_path):
    from djai import cli
    from tests.test_integration import make_analysis

    crate = [make_analysis(n, bpm, "8A", 0.5, tmp_path / f"{n}.wav")
             for n, bpm in (("a", 120.0), ("b", 128.0))]
    session = cli.Session(crate, log_dir=tmp_path / "logs")
    try:
        assert session.master_tempo_locked is False
        session.engine.submit(LoadTrack(
            deck="a", track=loud_track(200.0, bpm=120.0), rate=120.0 / 128.0,
            play=True, master=True, origin="t"))
        drive(session.engine, 4)
        session.set_master_tempo_lock(True)
        assert session.master_tempo_locked is True
        before = list(session._ride_steps)
        session._schedule_master_glide(session.live_deck)
        assert session._ride_steps == before, "locked, but a glide was scheduled"
        assert "unlock" in session.set_master_tempo_lock(False).lower() or \
               "glide" in session.set_master_tempo_lock(False).lower()
    finally:
        session.shutdown()


# --- the chain -----------------------------------------------------------------


def test_the_sample_rate_is_the_same_at_every_stage():
    """One rate from decode to output; a mismatch anywhere is pitch and speed."""
    from djai import deck as deck_mod
    from djai import engine as engine_mod

    assert deck_mod.SAMPLE_RATE == SAMPLE_RATE
    assert engine_mod.SAMPLE_RATE == SAMPLE_RATE
    # and the rate the decks actually run at, end to end: a track decoded at
    # any file rate is resampled to SAMPLE_RATE by load_track, and the stretch
    # tool's output is rejected unless it comes back at the same rate.
    engine = Engine(blocksize=512)
    engine.submit(LoadTrack(deck="a", track=loud_track(200.0), play=True,
                            master=True, origin="t"))
    out = drive(engine, 8)
    # one second of output is exactly SAMPLE_RATE frames: no rate change anywhere
    assert out.shape[1] == CHANNELS
    seconds = out.shape[0] / SAMPLE_RATE
    assert engine.frames_played == pytest.approx(seconds * SAMPLE_RATE, rel=1e-9)
    engine.stop()


def test_nothing_reaches_full_scale_before_the_limiter_on_a_normal_mix():
    engine = Engine(blocksize=512)
    engine.submit(LoadTrack(deck="a", track=loud_track(200.0), play=True,
                            master=True, origin="t"))
    engine.submit(LoadTrack(deck="b", track=loud_track(3000.0), play=True,
                            origin="t"))
    engine.deck_a.gain.jump(0.5)
    engine.deck_b.gain.jump(0.5)
    peaks = []
    for _ in range(40):
        drive(engine, 5)
        peaks.append(float(engine.master_peak_in))
    assert max(peaks) < 1.0, f"pre-limiter peak reached {max(peaks):.4f}"
    engine.stop()


def test_a_track_solo_at_its_own_tempo_nulls_against_its_source():
    """The acceptance test for the glide's endpoint.

    Once the glide has landed, the deck is at rate 1.0 on the unstretched
    source, so the output is the source samples with nothing done to them. A
    null against the buffer is the strongest statement available that no
    resampling, no second stretch and no gain staging is left in the path.

    The comparison is aligned first: the buffer swap crossfades and the
    playhead bookkeeping puts the output a few tens of milliseconds from a
    naive read of ``deck.position``. The search is deliberately bounded, so a
    real timing error cannot hide inside it.
    """
    engine = Engine(blocksize=512)
    src = loud_track(200.0, bpm=128.0)
    rate = 120.0 / 128.0
    copy = stretched_copy(src, rate)
    engine.submit(LoadTrack(deck="a", track=copy, start_frame=0, rate=rate,
                            play=True, master=True, origin="t"))
    drive(engine, 2)
    engine.submit(SetPitch(deck="a", rate=1.0, origin="master_glide"))
    drive(engine, 4)
    deck = engine.deck("a")
    assert deck.track is src and not deck.track.is_stretched
    assert deck.rate == pytest.approx(1.0)

    engine.deck_a.gain.jump(1.0)
    drive(engine, 4)                      # let the fader settle
    start = int(deck.position)
    out = drive(engine, 20)
    ref = src.audio[start:start + out.shape[0]]
    n = min(ref.shape[0], out.shape[0])
    o = out[:n, 0].astype(np.float64)
    r = ref[:n, 0].astype(np.float64)

    reach = 4096                          # ~93 ms; more than any swap needs
    best, at = np.inf, 0
    for shift in range(-reach, reach + 1, 8):
        a = o[-shift:] if shift < 0 else o[:len(o) - shift]
        b = r[:len(r) + shift] if shift < 0 else r[shift:]
        m = min(len(a), len(b))
        if m < 5000:
            continue
        residual = float(np.abs(a[:m] - b[:m]).max())
        if residual < best:
            best, at = residual, shift
    assert best < 5e-3, (
        f"solo track does not null against its source: {best:.6f} at {at} samples"
    )
    assert abs(at) < reach, "alignment ran to the edge of the search"
    engine.stop()


def test_tempo_path_never_offers_a_rate_the_supervisor_refuses():
    """The producer/consumer boundary, pinned.

    A deck sitting exactly at the edge of the stretch range made
    ``_meet`` return ``playing / counted == 1 + limit`` exactly, whose float
    delta is 0.08000000000000007. ``tempo_path`` accepted it (it compares with
    +1e-9); the supervisor refused the resulting load (it does not). The
    autopilot then could not cue that track at all -- found in a 20-track soak
    where only 4 of 20 tracks reached a deck.
    """
    from djai.selector import tempo_path
    from djai.supervisor import MAX_RATE_DELTA

    # 97.2 is exactly 89.998 * 1.08: the boundary that produced it.
    path = tempo_path(97.2, 89.998)
    assert path.blendable
    assert abs(path.rate_b - 1.0) <= MAX_RATE_DELTA, repr(path.rate_b)

    # and a sweep across the edge from both sides
    for counted in (89.998, 100.0, 124.0, 128.0):
        for mult in (1.0 - config.MAX_STRETCH_RATIO, 1.0 + config.MAX_STRETCH_RATIO):
            p = tempo_path(counted * mult, counted)
            if p.blendable:
                assert abs(p.rate_b - 1.0) <= MAX_RATE_DELTA, (
                    f"counted={counted} playing={counted * mult} "
                    f"rate_b={p.rate_b!r}"
                )
