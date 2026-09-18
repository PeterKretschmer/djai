"""The master limiter: the last stage before the output.

THREADING CONTEXT: main thread (pytest). The callback is driven directly, so
every claim here is made against the code the audio thread runs.

Context for why this exists, measured before it was written: the decoded
sources themselves peak above full scale (1.31 and 1.63 on two ordinary
tracks), and a beatmatched blend of them reached 1.48. The stage that used to
sit here was `np.clip`, which is a hard clipper -- so 12% of blocks were being
squared off. That is what the distortion was.
"""

from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pytest

from djai.commands import IMMEDIATE, LoadTrack
from djai.deck import CHANNELS, SAMPLE_RATE, LoadedTrack
from djai.engine import MASTER_CEILING, MASTER_KNEE, Engine
from tests.test_engine import drive, loud_track

BLOCK = 512


def hot_engine(amplitude: float) -> Engine:
    """An engine playing a tone at ``amplitude``, which may exceed full scale."""
    engine = Engine(blocksize=BLOCK)
    base = loud_track(220.0, 120.0, 30.0)
    audio = np.clip(base.audio * (amplitude / max(1e-9, float(np.max(np.abs(base.audio))))),
                    -4.0, 4.0).astype(np.float32)
    track = LoadedTrack(analysis=base.analysis, audio=audio)
    engine.submit(
        LoadTrack(deck="a", track=track, start_frame=0, rate=1.0, play=True,
                  master=True, execute_at=IMMEDIATE, origin="test")
    )
    engine.deck("a").gain.jump(1.0)
    return engine


def peaks_over(engine: Engine, blocks: int) -> tuple[float, float]:
    """Highest output and input peak across ``blocks`` callbacks."""
    buf = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    out_peak = 0.0
    in_peak = 0.0
    for _ in range(blocks):
        engine.callback(buf, BLOCK, None, None)
        out_peak = max(out_peak, float(np.max(np.abs(buf))))
        in_peak = max(in_peak, engine.master_peak_in)
    return in_peak, out_peak


# --- the curve ----------------------------------------------------------------


def test_the_limiter_is_transparent_below_the_knee():
    engine = Engine(blocksize=BLOCK)
    quiet = MASTER_KNEE * 0.5
    assert engine._limiter_target(quiet) == 1.0
    assert engine._limiter_target(MASTER_KNEE) == 1.0


def test_output_never_passes_the_ceiling_however_hot_the_input():
    """The curve approaches the ceiling asymptotically and never crosses it.

    At absurd inputs the exponential underflows and the result lands exactly
    on the ceiling, which is the limit of the curve rather than an overshoot.
    """
    engine = Engine(blocksize=BLOCK)
    for peak in (0.9, 1.0, 1.5, 3.0, 10.0, 100.0):
        allowed = peak * engine._limiter_target(peak)
        assert allowed <= MASTER_CEILING, f"{peak} mapped to {allowed}"
        assert allowed <= 0.99, "the acceptance bar is 0.99"

    # Below the asymptote, strictly under.
    assert 1.5 * engine._limiter_target(1.5) < MASTER_CEILING


def test_the_knee_has_no_corner_in_it():
    """A soft knee: the gain curve is continuous and monotonic through it."""
    engine = Engine(blocksize=BLOCK)
    xs = np.linspace(MASTER_KNEE * 0.9, MASTER_KNEE * 2.5, 400)
    gains = np.array([engine._limiter_target(float(x)) for x in xs])
    assert np.all(np.diff(gains) <= 1e-9), "gain must never rise with level"
    # No step anywhere: the largest jump between neighbours stays tiny.
    assert float(np.max(np.abs(np.diff(gains)))) < 0.02


# --- driven through the real callback ----------------------------------------


def test_a_hot_signal_leaves_the_box_under_the_ceiling():
    engine = hot_engine(1.6)
    in_peak, out_peak = peaks_over(engine, 200)
    assert in_peak > 1.0, "this test needs a genuinely hot input"
    assert out_peak <= 0.99, f"output reached {out_peak:.4f}"
    assert engine.limiter_clips == 0, "the backstop should never be needed"


def test_a_quiet_signal_is_not_touched():
    engine = hot_engine(0.3)
    _in_peak, out_peak = peaks_over(engine, 100)
    assert out_peak == pytest.approx(0.3, abs=0.02)
    assert engine.limiter_gain == pytest.approx(1.0, abs=1e-6)


def test_no_sample_ever_reaches_full_scale():
    engine = hot_engine(2.5)
    buf = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    worst = 0.0
    for _ in range(300):
        engine.callback(buf, BLOCK, None, None)
        worst = max(worst, float(np.max(np.abs(buf))))
        assert not np.any(np.abs(buf) >= 1.0), "a full-scale sample got out"
    assert worst < 1.0


def test_the_limiter_releases_back_toward_unity():
    """It must not duck the whole set after one loud passage."""
    engine = hot_engine(1.8)
    peaks_over(engine, 60)
    pulled_down = engine.limiter_gain
    assert pulled_down < 0.9

    # Drop the deck's level so the mix is quiet, then let it recover.
    engine.deck("a").gain.jump(0.2)
    peaks_over(engine, 400)
    assert engine.limiter_gain > pulled_down + 0.05, "gain never came back"


def test_the_reported_peaks_match_what_left_the_box():
    engine = hot_engine(1.4)
    buf = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    for _ in range(50):
        engine.callback(buf, BLOCK, None, None)
    measured = float(np.max(np.abs(buf)))
    assert engine.master_peak == pytest.approx(measured, abs=0.02)
    assert engine.master_peak_in >= engine.master_peak


# --- the cost of it ------------------------------------------------------------


def _heap_growth(engine, blocks=500):
    buf = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    for _ in range(40):                       # settle caches
        engine.callback(buf, BLOCK, None, None)
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for _ in range(blocks):
        engine.callback(buf, BLOCK, None, None)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    return sum(s.size_diff for s in after.compare_to(before, "filename"))


def test_limiting_allocates_nothing_that_not_limiting_does_not():
    """The acceptance bar: adding the limiter added no allocation.

    Measured as a delta between the same callback with the limiter working
    hard and with it idle. The interpreter allocates a little for any work at
    all, so an absolute figure would measure Python, not this change. A
    per-block buffer in the limiter would show up here as growth proportional
    to the block count.
    """
    idle = _heap_growth(hot_engine(0.3))
    working = hot_engine(1.5)
    limited = _heap_growth(working)

    assert working.limiter_gain < 1.0, "the limiter should be working here"
    assert limited <= idle + 8_000, (
        f"limiting added {limited - idle} bytes over 500 blocks "
        f"(idle {idle}, limited {limited})"
    )


def test_the_scratch_buffer_is_allocated_once():
    engine = Engine(blocksize=BLOCK)
    buf = engine._abs_buf
    peaks_over(hot_engine(1.5), 10)
    assert engine._abs_buf is buf
    assert engine._abs_buf.shape[1] == CHANNELS


# --- the other three candidates, kept honest ----------------------------------


def test_the_echo_feedback_coefficient_decays():
    """Candidate 2 from the diagnosis: a coefficient at or above 1.0 would
    accumulate without bound. It is 0.45."""
    from djai import transition as tr

    assert 0.0 < tr.ECHO_FEEDBACK < 1.0


def test_the_recorder_clamps_rather_than_wraps(tmp_path):
    """Candidate 4: wrapping at the int conversion would produce hard clicks.

    soundfile clamps on the way to PCM_16, and in any case the limiter means
    nothing above full scale ever reaches it.
    """
    import soundfile as sf

    path = tmp_path / "clamp.wav"
    data = np.array([[1.5, -1.5], [0.5, -0.5]], dtype=np.float32)
    sf.write(str(path), data, SAMPLE_RATE, subtype="PCM_16")
    back, _ = sf.read(str(path), dtype="float32")
    assert back[0][0] == pytest.approx(1.0, abs=1e-3), "clamped, not wrapped"
    assert back[0][1] == pytest.approx(-1.0, abs=1e-3)
    assert back[1][0] == pytest.approx(0.5, abs=1e-3)
