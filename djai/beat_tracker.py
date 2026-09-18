"""Beat This! beat and downbeat tracking, and the grid confidence built on it.

THREADING CONTEXT: offline analysis only -- the ``analyze`` and ``import``
subcommands. Never imported by playback, and PyTorch is imported lazily inside
:func:`detect`, so a session that never analyses never loads it.

Why this exists (measured 2026-09-17 on the 24 weakest-grid tracks plus 8
strong ones and one known-bad grid):

* The librosa grids drifted against the audio's own onsets by a median 48 ms
  start to end, 19 of 33 by more than 40 ms, and put section starts on bar 1
  less often than chance (0.19 vs 0.25). Several were at twice, half, 2:3 or
  3:4 of the real tempo -- including three of the highest-confidence tracks.
  Grids fitted to Beat This! drifted a median 4 ms, none over 40 ms.
* The old confidence (onset agreement) could not see any of that. The
  confidence here asks two independent witnesses whether the constant grid is
  right: Beat This!'s detections, and the onsets themselves.

Beat This! (CPJKU, ISMIR 2024) is MIT licensed, code and weights.
"""

from __future__ import annotations

import importlib.util
import math
import threading

import numpy as np

#: The Beat This! checkpoint: the full model. The small one ("small0") scored
#: within a point of it in the paper, if memory or CPU-only speed ever matter.
CHECKPOINT = "final0"

#: A detected beat within this of a grid line agrees with it. Tighter than the
#: MIREX 70 ms: 70 ms against a partner deck is an audible flam.
BEAT_TOLERANCE_S = 0.035
#: The same for downbeats. Looser, because downbeat placement is the harder
#: call and a bar line off by one beat is already ~470 ms away at 128 BPM.
BAR_TOLERANCE_S = 0.070
#: Grid lines further than this many detected beats from any detection are in
#: a beatless passage (a breakdown), and not held against the grid.
COVER_BEATS = 1.5
#: Onset phase drift from the start of the track to its end that scores 0.
DRIFT_ZERO_MS = 100.0
#: Shifts searched when lining a grid window up with the onsets: at most this
#: far, and never more than an eighth of a beat. A reach of a sixteenth note
#: lets one window lock onto hi-hat sixteenths and read a steady grid as
#: drifting by exactly that much -- seen after the rollout on two tracks whose
#: beats and bars agreed with Beat This! at 0.98-1.00 (-104 ms at 125 BPM,
#: -152 ms at 96). Drift beyond an eighth of a beat already fails the beat
#: check at BEAT_TOLERANCE_S.
_DRIFT_SEARCH_S = 0.12
_DRIFT_SEARCH_BEATS = 0.125
_DRIFT_STEP_S = 0.004
#: Length of the start and end windows the drift is measured over.
_DRIFT_WINDOW_S = 30.0

_model = None
_model_lock = threading.Lock()


def available() -> bool:
    """Are Beat This! and PyTorch installed? Imports neither."""
    return (
        importlib.util.find_spec("beat_this") is not None
        and importlib.util.find_spec("torch") is not None
    )


def detect(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Beat and downbeat times, in seconds, for mono audio ``y`` at ``sr``.

    Loads the model once per process, on the GPU when PyTorch can see one.
    Blocking: about a second per track on an RTX 3060, including the model's
    own resampling; the first call also pays the model load.
    """
    global _model
    import torch
    from beat_this.inference import Audio2Beats

    with _model_lock:
        if _model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _model = Audio2Beats(checkpoint_path=CHECKPOINT, device=device, dbn=False)
        beats, downbeats = _model(np.asarray(y, dtype=np.float32), sr)
    return np.asarray(beats, dtype=np.float64), np.asarray(downbeats, dtype=np.float64)


def _longest_steady_run(beats: np.ndarray) -> np.ndarray:
    """The longest stretch of consecutive beats with no gap or extra beat."""
    ibi = np.diff(beats)
    median = float(np.median(ibi))
    steady = np.abs(ibi - median) < 0.25 * median
    best = cur = best_end = 0
    for i, ok in enumerate(steady):
        cur = cur + 1 if ok else 0
        if cur > best:
            best, best_end = cur, i + 1
    return beats[best_end - best: best_end + 1]


def fit_constant_grid(
    beats: np.ndarray, downbeats: np.ndarray
) -> tuple[float, float] | None:
    """``(bpm, first_downbeat_s)`` of the constant grid that fits the detections.

    Tempo and phase are a least-squares line through the longest run of steady
    beats -- Beat This! reports beats on 20 ms frames, so a median interval is
    only good to about 2%, while a line through a few hundred beats is good to
    a few hundredths of a BPM. Bar 1 is the grid phase most detected downbeats
    fall on. None when there is too little to fit.
    """
    beats = np.asarray(beats, dtype=np.float64)
    if beats.size < 8:
        return None
    run = _longest_steady_run(beats)
    if run.size < 8:
        return None
    period, t0 = np.polyfit(np.arange(run.size), run, 1)
    if not (math.isfinite(period) and period > 0):
        return None
    t0 = float(t0) - math.floor(float(t0) / period) * period
    downbeats = np.asarray(downbeats, dtype=np.float64)
    if downbeats.size:
        phases = np.round((downbeats - t0) / period).astype(np.int64) % 4
        phase = int(np.bincount(phases, minlength=4).argmax())
    else:
        phase = 0
    return 60.0 / float(period), t0 + phase * float(period)


def _near(ref: np.ndarray, times: np.ndarray, tol: float) -> np.ndarray:
    """For each of ``times``, is some ``ref`` time within ``tol``? Both sorted."""
    if ref.size == 0 or times.size == 0:
        return np.zeros(times.size, dtype=bool)
    i = np.clip(np.searchsorted(ref, times), 1, max(ref.size - 1, 1))
    lo = np.abs(ref[i - 1] - times)
    hi = np.abs(ref[np.minimum(i, ref.size - 1)] - times)
    return np.minimum(lo, hi) <= tol


def _precision_recall(grid: np.ndarray, detected: np.ndarray, tol: float) -> tuple[float, float]:
    """Share of (covered) grid lines with a detection, and of detections on a line."""
    if detected.size < 4 or grid.size == 0:
        return 0.0, 0.0
    detected_period = float(np.median(np.diff(detected)))
    covered = grid[_near(detected, grid, COVER_BEATS * detected_period)]
    if covered.size == 0:
        return 0.0, 0.0
    return (
        float(_near(detected, covered, tol).mean()),
        float(_near(grid, detected, tol).mean()),
    )


def _above_harmonic(share: float) -> float:
    """Rescale an agreement share so a harmonic grid scores nothing.

    A grid at twice the real tempo has every other line on a beat, and one at
    half the tempo catches every other beat: 50% agreement comes free with the
    wrong tempo. So 0.5 maps to 0 and 1.0 to 1. Without this, every doubled and
    halved grid in the trial scored about 0.67 and cleared the cut threshold.
    """
    return float(np.clip((share - 0.5) / 0.5, 0.0, 1.0))


def onset_drift_ms(
    oenv: np.ndarray, sr: int, hop: int, beats: np.ndarray, duration_s: float
) -> float:
    """How far the grid's best phase against the onsets walks, start to end.

    A right tempo holds the phase steady; one 0.1% out walks it about 200 ms
    over three minutes. This witness is the audio itself, independent of any
    beat tracker. NaN when the track is too short or too sparse to measure.
    """
    beats = np.asarray(beats, dtype=np.float64)
    if oenv.size == 0 or beats.size < 16 or duration_s < 2 * _DRIFT_WINDOW_S:
        return float("nan")
    onset_t = np.arange(oenv.size) * hop / sr
    margin = min(20.0, 0.1 * duration_s)
    windows = (
        (margin, margin + _DRIFT_WINDOW_S),
        (duration_s - margin - _DRIFT_WINDOW_S, duration_s - margin),
    )
    period = float(np.median(np.diff(beats)))
    reach = min(_DRIFT_SEARCH_S, _DRIFT_SEARCH_BEATS * period)
    shifts = np.arange(-reach, reach + 1e-9, _DRIFT_STEP_S)
    best = []
    for lo, hi in windows:
        g = beats[(beats >= lo) & (beats < hi)]
        if g.size < 8:
            return float("nan")
        scores = [float(np.interp(g + s, onset_t, oenv).sum()) for s in shifts]
        best.append(float(shifts[int(np.argmax(scores))]))
    return (best[1] - best[0]) * 1000.0


def grid_confidence(
    beats: np.ndarray,
    downbeats: np.ndarray,
    detected_beats: np.ndarray,
    detected_downbeats: np.ndarray,
    drift_ms: float,
) -> float:
    """0..1: can a blend hold phase and bar position against this grid?

    The weakest of three answers:

    * **beats** -- precision and recall of the grid against the detected beats,
      each rescaled by :func:`_above_harmonic`;
    * **bars** -- the same for downbeats;
    * **drift** -- 1 at no onset phase drift, 0 at :data:`DRIFT_ZERO_MS`. Left
      out when it cannot be measured (NaN), since absent evidence is not
      evidence against the grid.

    Trial result: 24 of 24 weak tracks and 8 of 8 strong ones clear 0.25 on a
    grid fitted to Beat This!, and 264 of 264 corrupted grids (twice and half
    tempo, +-0.5%, +1%, half a beat late, bar 1 one or two beats late) do not.
    """
    beats = np.asarray(beats, dtype=np.float64)
    downbeats = np.asarray(downbeats, dtype=np.float64)
    detected_beats = np.asarray(detected_beats, dtype=np.float64)
    detected_downbeats = np.asarray(detected_downbeats, dtype=np.float64)
    pb, rb = _precision_recall(beats, detected_beats, BEAT_TOLERANCE_S)
    pd, rd = _precision_recall(downbeats, detected_downbeats, BAR_TOLERANCE_S)
    parts = [
        _above_harmonic(pb), _above_harmonic(rb),
        _above_harmonic(pd), _above_harmonic(rd),
    ]
    if math.isfinite(drift_ms):
        parts.append(float(np.clip(1.0 - abs(drift_ms) / DRIFT_ZERO_MS, 0.0, 1.0)))
    return float(min(parts))
