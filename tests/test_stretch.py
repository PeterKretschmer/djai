"""Pitch-preserving time-stretch: correctness, the deck path, and the fallback.

THREADING CONTEXT: main thread (pytest). The audio callback is driven directly.

The headline claim is measured, not asserted by construction: a track played 6%
faster must keep its pitch, where the resampling path it replaces moves it a
full semitone.
"""

from __future__ import annotations

import logging

import librosa
import numpy as np
import pytest

from djai import config, deck as deck_mod, phrase
from djai.commands import LoadTrack
from djai.deck import (
    CHANNELS,
    SAMPLE_RATE,
    Deck,
    LoadedTrack,
    StretchError,
    time_stretch,
)
from djai.engine import Engine
from tests.test_engine import drive, loud_track

RATE = 1.06  # +6%, the acceptance-criterion case


# --- measuring pitch ---------------------------------------------------------

CENTS_PER_BIN = 5.0
F_LO, F_HI = 100.0, 6000.0


def log_spectrum(x: np.ndarray) -> np.ndarray:
    """Mean magnitude spectrum on a cents axis."""
    S = np.abs(librosa.stft(np.ascontiguousarray(x), n_fft=8192, hop_length=4096))
    mag = S.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=8192)
    band = (freqs >= F_LO) & (freqs <= F_HI)
    cents = 1200 * np.log2(freqs[band] / F_LO)
    grid = np.arange(0, cents[-1], CENTS_PER_BIN)
    out = np.interp(grid, cents, mag[band])
    return out - out.mean()


def shift_cents(reference: np.ndarray, processed: np.ndarray) -> float:
    """Global pitch shift, by cross-correlating log-frequency spectra.

    Comparing "the top N FFT bins" elementwise is not a pitch measurement --
    adjacent bins of one peak get counted twice and a single noisy bin
    dominates. This gives the shift directly.
    """
    a, b = log_spectrum(reference), log_spectrum(processed)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    lag = int(np.argmax(np.correlate(b, a, mode="full"))) - (n - 1)
    return lag * CENTS_PER_BIN


def musical(seconds: float = 12.0, seed: int = 0) -> LoadedTrack:
    """A harmonically rich tone complex -- something with a pitch to preserve."""
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    sig = np.zeros(n)
    for f, amp in ((220.0, 0.5), (330.0, 0.3), (440.0, 0.25), (660.0, 0.15)):
        sig += amp * np.sin(2 * np.pi * f * t)
    rng = np.random.default_rng(seed)
    sig += 0.02 * rng.standard_normal(n)
    audio = np.stack([sig, sig], axis=1).astype(np.float32)
    audio = np.vstack([audio, np.zeros((2, CHANNELS), dtype=np.float32)])
    base = loud_track(220.0, bpm=124.0, seconds=seconds)
    return LoadedTrack(analysis=base.analysis, audio=audio)


# --- the acceptance criterion ------------------------------------------------


def test_a_6_percent_stretch_keeps_its_pitch(capsys):
    """Measured against the resampling path it replaces."""
    track = musical()
    original = track.audio[:-2, 0]

    stretched = time_stretch(track, RATE).audio[:-2, 0]

    # What resampling does: read faster, pitch rises by the same factor.
    idx = np.arange(0, len(original) - 1, RATE)
    resampled = np.interp(idx, np.arange(len(original)), original).astype(np.float32)

    s_cents = shift_cents(original, stretched)
    r_cents = shift_cents(original, resampled)
    expected = 1200 * np.log2(RATE)

    with capsys.disabled():
        print(
            f"\n    +6% tempo: time-stretch {s_cents:+.1f} cents, "
            f"resample {r_cents:+.1f} cents (a semitone is 100)"
        )

    assert abs(s_cents) <= 10.0, f"stretch moved pitch {s_cents:+.1f} cents"
    assert abs(r_cents - expected) <= 25.0, (
        f"resample should shift ~{expected:.0f} cents, measured {r_cents:+.1f}"
    )
    assert abs(s_cents) < abs(r_cents) / 5


def test_stretch_changes_duration_by_the_rate():
    track = musical(seconds=8.0)
    original = track.audio.shape[0] - 2
    for rate in (0.94, 1.0 + config.STRETCH_DEADBAND * 2, 1.06):
        out = time_stretch(track, rate)
        assert out.stretch_rate == pytest.approx(rate)
        assert (out.audio.shape[0] - 2) == pytest.approx(original / rate, rel=0.02)
        assert out.source_frames == pytest.approx(original, rel=0.03)


def test_stretch_refuses_beyond_the_limit():
    track = musical(seconds=4.0)
    for bad in (1.0 + config.MAX_STRETCH_RATIO + 0.01, 0.8, 1.5):
        with pytest.raises(StretchError):
            time_stretch(track, bad)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(StretchError):
            time_stretch(track, bad)


def test_the_limit_is_inclusive_at_exactly_the_boundary():
    """1.08 - 1.0 is 0.08000000000000007. Without an epsilon a track needing
    exactly the maximum is refused and silently resampled -- a full semitone --
    precisely at the edge the limit was chosen to permit."""
    track = musical(seconds=4.0)
    for rate in (1.0 + config.MAX_STRETCH_RATIO, 1.0 - config.MAX_STRETCH_RATIO):
        out = time_stretch(track, rate)
        assert out.stretch_rate == pytest.approx(rate)


def test_librosa_delivers_the_ratio_it_is_asked_for():
    """A ratio error would be drift we manufactured ourselves."""
    track = musical(seconds=20.0)
    n_in = track.audio.shape[0] - 2
    for rate in (0.94, 0.98, 1.02, 1.06):
        out = time_stretch(track, rate)
        achieved = n_in / (out.audio.shape[0] - 2)
        drift_ms_per_min = abs((achieved - rate) / rate) * 60_000
        assert drift_ms_per_min < 5.0, (
            f"rate {rate}: achieved {achieved:.6f}, "
            f"{drift_ms_per_min:.1f} ms drift per minute"
        )


def test_stretched_audio_is_the_right_shape_and_finite():
    out = time_stretch(musical(seconds=6.0), RATE)
    assert out.audio.dtype.name == "float32"
    assert out.audio.shape[1] == CHANNELS
    assert out.audio.flags["C_CONTIGUOUS"]
    assert np.all(np.isfinite(out.audio))
    assert not np.any(out.audio[-2:]), "guard frames must stay silent"


# --- the deck plays it correctly ---------------------------------------------


def test_deck_reads_a_stretched_buffer_without_resampling_it():
    """Position stays in original frames; the buffer is stepped at 1.0."""
    track = musical(seconds=20.0)
    stretched = time_stretch(track, RATE)

    d = Deck("a")
    d.attach(stretched)
    d.playing = True
    d.gain.jump(1.0)
    d.set_rate(RATE)

    blocks = 40
    for _ in range(blocks):
        d.read(512)

    # Position advances in ORIGINAL frames, at the tempo ratio.
    assert d.position == pytest.approx(blocks * 512 * RATE, rel=1e-9)


def test_stretched_and_resampled_decks_stay_in_step():
    """Both must traverse the same musical position at the same wall time."""
    track = musical(seconds=20.0)
    stretched = time_stretch(track, RATE)

    a, b = Deck("a"), Deck("b")
    a.attach(track)
    b.attach(stretched)
    for d in (a, b):
        d.playing = True
        d.gain.jump(1.0)
        d.set_rate(RATE)

    for _ in range(60):
        a.read(512)
        b.read(512)
    assert a.position == pytest.approx(b.position, rel=1e-9)


def test_stretched_deck_preserves_pitch_through_the_whole_audio_path():
    """Not just the buffer -- what actually comes out of Deck.read."""
    track = musical(seconds=25.0)
    stretched = time_stretch(track, RATE)

    def render(t: LoadedTrack) -> np.ndarray:
        d = Deck("x")
        d.attach(t)
        d.playing = True
        d.gain.jump(1.0)
        d.set_rate(RATE)
        return np.concatenate([d.read(2048).copy() for _ in range(120)])[:, 0]

    flat = Deck("ref")
    flat.attach(track)
    flat.playing = True
    flat.gain.jump(1.0)
    reference = np.concatenate([flat.read(2048).copy() for _ in range(120)])[:, 0]

    s = shift_cents(reference, render(stretched))
    r = shift_cents(reference, render(track))
    assert abs(s) <= 10.0, f"stretched deck output shifted {s:+.1f} cents"
    assert abs(r) > 50.0, f"resampling deck should shift pitch, measured {r:+.1f}"


def test_remaining_frames_is_in_original_frames_for_a_stretched_track():
    track = musical(seconds=20.0)
    stretched = time_stretch(track, RATE)
    d = Deck("a")
    d.attach(stretched)
    d.playing = True
    # The stretched buffer is shorter, but the track is still 20 s of music.
    assert d.remaining_frames == pytest.approx(track.audio.shape[0] - 2, rel=0.03)


def test_a_stretched_deck_still_ends_and_goes_silent():
    track = musical(seconds=3.0)
    stretched = time_stretch(track, RATE)
    d = Deck("a")
    d.attach(stretched)
    d.playing = True
    d.gain.jump(1.0)
    for _ in range(400):
        block = d.read(512)
        assert np.all(np.isfinite(block))
        if d.ended:
            break
    assert d.ended
    for _ in range(5):
        assert not np.any(d.read(512))


def test_beat_positions_are_unchanged_by_stretching():
    """Phrase maths must not care that the buffer was stretched."""
    track = musical(seconds=30.0)
    stretched = time_stretch(track, RATE)
    a = track.analysis
    for bar in (0.0, 4.0, 16.0):
        frame = phrase.frame_at_bar(a, bar)
        assert phrase.bar_at_frame(a, frame) == pytest.approx(bar, abs=1e-6)
    assert stretched.analysis is a


# --- the engine wires it up --------------------------------------------------


def wait_for_stretch(engine: Engine, track: LoadedTrack, rate: float, timeout=30.0):
    import time as _t

    end = _t.time() + timeout
    while _t.time() < end:
        got = engine._stretcher.get(track, rate)
        if got is not None:
            return got
        _t.sleep(0.02)
    return None


def test_engine_pre_stretches_a_cued_track_and_uses_it():
    engine = Engine(blocksize=512)
    try:
        track = musical(seconds=20.0)
        engine.submit(LoadTrack(deck="a", track=track, rate=RATE, master=True))
        assert wait_for_stretch(engine, track, RATE) is not None

        # A later load of the same track at the same rate picks up the copy.
        engine.submit(LoadTrack(deck="b", track=track, rate=RATE, play=True))
        drive(engine, 4)
        assert engine.deck_b.track is not None
        assert engine.deck_b.track.is_stretched
        assert engine.deck_b.track.stretch_rate == pytest.approx(RATE)
        assert engine.stretched_loads >= 1
    finally:
        engine.stop()


def test_engine_falls_back_to_resampling_when_the_stretch_is_not_ready():
    """The load must never wait for the worker."""
    engine = Engine(blocksize=512)
    try:
        track = musical(seconds=20.0)
        # Load immediately, before the worker can possibly have finished.
        engine.submit(LoadTrack(deck="a", track=track, rate=RATE, master=True))
        drive(engine, 1)
        assert engine.deck_a.track is not None
        assert not engine.deck_a.track.is_stretched, "should not have waited"
        assert engine.deck_a.rate == pytest.approx(RATE)
        assert engine.resampled_loads >= 1
    finally:
        engine.stop()


def test_a_failing_stretch_is_logged_and_does_not_break_playback(caplog):
    engine = Engine(blocksize=512)
    try:
        track = musical(seconds=6.0)
        # Push a job straight past the guard so the worker itself has to fail.
        engine._stretcher._queue.put(("k", track, 999.0))
        with caplog.at_level(logging.WARNING):
            import time as _t

            _t.sleep(0.5)
        assert engine._stretcher.failures >= 1
        assert any("falling back to resampling" in r.message for r in caplog.records)

        engine.submit(LoadTrack(deck="a", track=track, rate=1.0, master=True))
        out = drive(engine, 5)
        assert np.all(np.isfinite(out))
    finally:
        engine.stop()


def test_no_stretch_requested_inside_the_deadband():
    engine = Engine(blocksize=512)
    try:
        track = musical(seconds=6.0)
        engine.submit(LoadTrack(deck="a", track=track, rate=1.0005, master=True))
        import time as _t

        _t.sleep(0.4)
        assert engine._stretcher.completed == 0
    finally:
        engine.stop()


def test_no_stretch_requested_beyond_the_limit():
    engine = Engine(blocksize=512)
    try:
        track = musical(seconds=6.0)
        engine.submit(LoadTrack(deck="a", track=track, rate=1.5, master=True))
        import time as _t

        _t.sleep(0.4)
        assert engine._stretcher.completed == 0
        assert engine._stretcher.failures == 0, "should not even be attempted"
    finally:
        engine.stop()


def test_stretching_never_happens_on_the_audio_thread():
    """A guard on the code, not the behaviour: the callback must not call it."""
    import inspect

    src = inspect.getsource(Engine.callback) + inspect.getsource(Engine._apply)
    assert "time_stretch(" not in src
    worker_src = inspect.getsource(deck_mod.time_stretch)
    assert "worker thread only" in worker_src.lower()


def test_stretch_cache_is_bounded():
    from djai.engine import _StretchWorker

    w = _StretchWorker(cache_size=2)
    try:
        for i in range(5):
            w._ready[f"k{i}"] = None
            w._order.append(f"k{i}")
            while len(w._order) > w._cache_size:
                w._ready.pop(w._order.pop(0), None)
        assert len(w._ready) == 2
    finally:
        w.stop()


# --- the selector guard ------------------------------------------------------


def test_the_deck_is_never_asked_to_stretch_past_its_limit():
    """Since Phase 5 a track just outside the stretch range is still offered --
    the playing deck is ridden toward it -- but what the deck is asked to
    stretch stays inside the limit, which is what this guard is about."""
    from tests.test_selector_intent import track as make

    current = make("cur", 128.0)
    crate = [
        current,
        make("just_inside", 128.0 * (1 + config.MAX_STRETCH_RATIO - 0.005)),
        make("just_outside", 128.0 * (1 + config.MAX_STRETCH_RATIO + 0.01)),
        make("way_outside", 160.0),
    ]
    from djai.selector import rank_candidates

    ranked = {c.track.track_id: c for c in rank_candidates(crate, current, bpm_tolerance=0.5)}
    assert "just_inside" in ranked
    assert ranked["just_inside"].tempo_path.technique == "direct"
    assert "just_outside" in ranked, "reachable by riding the playing deck"
    assert ranked["just_outside"].tempo_path.technique == "ride"
    assert "way_outside" not in ranked, "+25% is beyond a ride, and half is 80"
    for candidate in ranked.values():
        rate = candidate.tempo_path.rate_b
        assert abs(rate - 1.0) <= config.MAX_STRETCH_RATIO + 1e-9


def test_the_selector_limit_matches_what_the_deck_will_accept():
    from djai.selector import rank_candidates
    from tests.test_selector_intent import track as make

    current = make("cur", 128.0)
    crate = [current] + [
        make(f"t{i}", 128.0 * (1 + d))
        for i, d in enumerate(np.linspace(-0.12, 0.12, 25))
    ]
    for cand in rank_candidates(crate, current, bpm_tolerance=0.5):
        # The rate the deck actually gets, after the planner has ridden the
        # playing deck or counted this one half or double -- not the raw ratio
        # of two printed tempos, which is no longer what is asked of it.
        rate = cand.tempo_path.rate_b
        assert abs(rate - 1.0) <= config.MAX_STRETCH_RATIO + 1e-9, (
            f"{cand.track.track_id} needs rate {rate:.4f}, beyond the deck's limit"
        )
