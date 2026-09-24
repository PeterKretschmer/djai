"""Deck tests: resampling accuracy, EQ reconstruction, real-time safety.

THREADING CONTEXT: main thread (pytest).
"""

from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pytest

from djai.analysis import TrackAnalysis
from djai.deck import CHANNELS, SAMPLE_RATE, Deck, LoadedTrack


def make_track(n_frames: int = SAMPLE_RATE * 4, bpm: float = 128.0) -> LoadedTrack:
    t = np.arange(n_frames) / SAMPLE_RATE
    sig = (
        0.4 * np.sin(2 * np.pi * 60.0 * t)  # low band
        + 0.4 * np.sin(2 * np.pi * 900.0 * t)  # mid band
        + 0.4 * np.sin(2 * np.pi * 8000.0 * t)  # high band
    )
    audio = np.stack([sig, sig], axis=1).astype(np.float32)
    audio = np.vstack([audio, np.zeros((2, CHANNELS), dtype=np.float32)])
    period = 60.0 / bpm
    beats = np.arange(0.0, n_frames / SAMPLE_RATE, period)
    analysis = TrackAnalysis(
        track_id="test", path="<synthetic>", title="test",
        duration_s=n_frames / SAMPLE_RATE, bpm=bpm,
        beats=[float(b) for b in beats],
        downbeats=[float(b) for b in beats[::4]],
        key_name="A minor", camelot="8A",
        beat_rms=[0.5] * len(beats), energy=0.5, grid_confidence=1.0,
    )
    return LoadedTrack(analysis=analysis, audio=audio)


def render(deck: Deck, n_blocks: int, block: int = 512) -> np.ndarray:
    return np.concatenate([deck.read(block).copy() for _ in range(n_blocks)])


#: Frames to discard before spectral checks, so the IIR filters have settled
#: from zero initial conditions.
SETTLE = 8192


def spectrum(out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = out[SETTLE:, 0]
    assert x.size > SETTLE, "not enough frames rendered for a spectral check"
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return np.fft.rfftfreq(x.size, 1 / SAMPLE_RATE), spec


def peak_near(freqs: np.ndarray, spec: np.ndarray, f: float) -> float:
    band = (freqs > f * 0.9) & (freqs < f * 1.1)
    return float(np.max(spec[band]))


def test_flat_eq_preserves_the_magnitude_spectrum():
    """LR4 branches sum to an allpass, so a flat EQ must not colour the signal.

    The check is on magnitude, not samples: an allpass sum reproduces every
    band's level while rotating phase, which is the accepted trade for bands
    that can actually be killed.
    """
    deck = Deck("a")
    track = make_track()
    deck.attach(track)
    deck.playing = True
    deck.gain.jump(1.0)

    out = render(deck, 300)
    # Measure the source through the identical window and slicing so that
    # windowing artefacts (Hann scalloping loss is up to 1.4 dB) cancel out.
    freqs, spec_out = spectrum(out)
    _, spec_src = spectrum(track.audio[: out.shape[0]])

    for f in (60.0, 900.0, 8000.0):
        ratio = peak_near(freqs, spec_out, f) / peak_near(freqs, spec_src, f)
        assert 0.97 < ratio < 1.03, (
            f"flat EQ colours {f} Hz by {20 * np.log10(ratio):+.2f} dB"
        )


def test_kill_bass_removes_low_band_and_keeps_the_rest():
    deck = Deck("a")
    deck.attach(make_track())
    deck.playing = True
    deck.gain.jump(1.0)
    deck.eq_low.jump(0.0)

    freqs, spec = spectrum(render(deck, 300))
    assert peak_near(freqs, spec, 60.0) < 0.02 * peak_near(freqs, spec, 900.0), (
        "60 Hz not killed"
    )
    assert peak_near(freqs, spec, 8000.0) > 0.3 * peak_near(freqs, spec, 900.0), (
        "high band damaged"
    )


def test_rate_change_shifts_pitch_by_the_expected_ratio():
    deck = Deck("a")
    deck.attach(make_track())
    deck.playing = True
    deck.gain.jump(1.0)
    deck.set_rate(1.05)

    freqs, spec = spectrum(render(deck, 300))
    band = (freqs > 800) & (freqs < 1000)
    peak = float(freqs[band][np.argmax(spec[band])])
    assert abs(peak - 900.0 * 1.05) < 3.0, f"expected ~945 Hz, got {peak:.1f}"


def test_position_advances_at_the_rate():
    deck = Deck("a")
    deck.attach(make_track())
    deck.playing = True
    deck.gain.jump(1.0)
    deck.set_rate(1.016)
    for _ in range(10):
        deck.read(512)
    assert deck.position == pytest.approx(10 * 512 * 1.016, rel=1e-9)


def test_silent_when_no_track_attached():
    deck = Deck("a")
    assert not np.any(deck.read(512))


def test_track_end_produces_silence_not_garbage():
    """The contract past the end: `ended` is flagged and every later read is
    exactly silent -- never looped samples, never NaN, never a runaway filter."""
    deck = Deck("a")
    deck.attach(make_track(n_frames=1000))
    deck.playing = True
    deck.gain.jump(1.0)
    deck.read(512)
    final = deck.read(512).copy()  # frames 512..1023; the track ends at 1000
    assert deck.ended
    assert np.all(np.isfinite(final))

    for _ in range(20):
        assert not np.any(deck.read(512)), "reads past the end must be silent"


def test_gain_ramp_has_no_discontinuity():
    """A cut must ramp across the block, not step (which would click)."""
    deck = Deck("a")
    deck.attach(make_track())
    deck.playing = True
    deck.gain.jump(1.0)
    deck.read(512)
    deck.set_gain(0.0)
    block = deck.read(512).copy()
    # First sample still near full level, last sample silent, monotone envelope.
    assert abs(block[0, 0]) > 1e-4 or abs(block[1, 0]) > 1e-4
    assert abs(block[-1, 0]) < 1e-6


def test_hot_path_does_not_grow_allocations():
    """read() must not allocate proportionally to the number of blocks.

    scipy's sosfilt has no out= parameter so two block-sized arrays per call
    are unavoidable; what must NOT happen is unbounded growth.
    """
    deck = Deck("a")
    deck.attach(make_track())
    deck.playing = True
    deck.gain.jump(1.0)
    for _ in range(50):  # warm up: build the ramp basis, settle filter state
        deck.read(512)

    gc.collect()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    for _ in range(500):
        deck.read(512)
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    grew = sum(s.size_diff for s in snap_after.compare_to(snap_before, "filename"))
    # 500 blocks x 512 frames x 2ch x 4 bytes = 2 MB per retained buffer.
    assert grew < 512 * 1024, f"hot path retained {grew} bytes over 500 blocks"


@pytest.mark.parametrize("rate", [1.0, 1.07, 0.93])
@pytest.mark.parametrize("block", [1024, 333])
def test_a_deck_reaching_the_end_of_its_audio_fades_rather_than_clicks(rate, block):
    """Phase 3 regression: a file whose last sample is not silent stopped dead.

    Found by the Phase 2.3 fuzz once a live deck was allowed to run to the end
    of a pure-tone buffer at full gain: one sample from 0.07 to 0.
    """
    n = SAMPLE_RATE * 2
    t = np.arange(n) / SAMPLE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)
    base = make_track(n_frames=n)
    track = LoadedTrack(analysis=base.analysis, audio=np.stack([tone, tone], axis=1))
    deck = Deck("a")
    deck.attach(track)
    deck.playing = True
    deck.rate = rate
    deck.gain.jump(1.0)
    chunks = []
    while not deck.ended:
        chunks.append(deck.read(block)[:, 0].astype(np.float64).copy())
    chunks.append(deck.read(block)[:, 0].astype(np.float64).copy())
    x = np.concatenate(chunks)[SAMPLE_RATE:]   # past the deck's own start-up
    d2 = np.abs(x[2:] - 2 * x[1:-1] + x[:-2])
    tone_d2 = 0.5 * (2 * np.pi * 110.0 * rate / SAMPLE_RATE) ** 2
    assert d2.max() < 20 * tone_d2, f"end-of-audio step {d2.max():.2e}"
    assert abs(x[-block:]).max() == 0.0, "and silent after the end"


@pytest.mark.parametrize("offset", [0, 128, 384, 768])
def test_a_filter_move_across_the_detent_is_spread_over_two_blocks(offset):
    """Phase 3 regression, from the Phase 2.3 fuzz at payload 438.

    A one-block jump from low-pass -0.197 to high-pass +0.927 faded one side
    out and the other in within a few dozen samples: 6.05e-3 second difference
    on a 105 Hz tone, over the fuzz's 5e-3 bar. It now reaches the centre in
    the first block and the target in the second.
    """
    n = SAMPLE_RATE * 20
    t = np.arange(n) / SAMPLE_RATE
    tone = (0.2 * np.sin(2 * np.pi * 105.0 * t)).astype(np.float32)
    track = LoadedTrack(analysis=make_track(n_frames=n).analysis,
                        audio=np.stack([tone, tone], axis=1))
    deck = Deck("a")
    deck.attach(track)
    deck.playing = True
    deck.rate = 1.0732
    deck.gain.jump(0.875)
    deck.eq_high.jump(0.831)
    deck.set_filter(-0.1971, 0.306)
    out = [deck.read(1024)[:, 0].astype(np.float64).copy() for _ in range(40)]
    if offset:
        out.append(deck.read(offset)[:, 0].astype(np.float64).copy())
    deck.set_filter(0.9271, 0.306)
    first = deck.read(1024)[:, 0].astype(np.float64).copy()
    assert deck.filter_pos.cur == 0.0, "the first block stops at the centre"
    out += [first] + [deck.read(1024)[:, 0].astype(np.float64).copy() for _ in range(3)]
    assert deck.filter_pos.cur == pytest.approx(0.9271), "and the second arrives"
    x = np.concatenate(out)[-6 * 1024:]
    d2 = float(np.abs(x[2:] - 2 * x[1:-1] + x[:-2]).max())
    assert d2 < 2e-3, f"detent crossing step {d2:.2e}"
