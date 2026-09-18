"""Measure real DJ transitions from recorded sets, to tune our parameters.

THREADING CONTEXT: main thread only, fully offline. Decodes audio and runs
librosa; never called from the audio thread, the scheduler thread, or the
monitor thread.

This is measurement, not training. It reads sets *you* supply, looks at a window
around each marked transition, and reduces them to a handful of numbers whose
medians become defaults in :mod:`djai.config`. It never downloads or sources
audio.

WHAT CAN AND CANNOT BE MEASURED FROM A MIXED RECORDING
------------------------------------------------------
A recorded set is one stereo file: the two tracks are already summed and cannot
be pulled apart without stem separation, which is out of scope. So the
per-track low-band trajectories are not recoverable. What *is* recoverable --
and what actually constrains our parameters -- is the **combined** low-band
trajectory through the blend: whether the low end dips, holds, or doubles. That
is precisely the question our complementary bass swap answers, so it is the
measurement we take, and it is reported as such rather than dressed up as two
separate curves.

Everything else is measured directly: blend length in bars, how that length sits
against 4/8/16/32-bar grids, overall level change, and BPM delta.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from djai.analysis import AUDIO_EXTENSIONS, _refine_tempo, _fine_tempo_and_phase

log = logging.getLogger(__name__)

#: Everything is analysed at this rate. Matches djai.analysis.
ANALYSIS_SR = 22050
_HOP = 512
BEATS_PER_BAR = 4

#: Seconds either side of a marker that get decoded. Must comfortably exceed the
#: longest blend being measured: at 45 s a 32-bar blend (60 s at 128 BPM) was
#: wider than the window, so the "before" and "after" reference stretches both
#: landed *inside* the blend and every measurement came out wrong.
#: There is a real tension here. Too small and the reference stretches fall
#: inside the blend (at 45 s a 32-bar blend was wider than the window). Too
#: large and they fall inside the *neighbouring* transitions, since sets often
#: run a blend every 2-3 minutes -- at 120 s every measurement was rejected for
#: exactly that reason. 75 s sits between: wide enough for a ~60 s blend, narrow
#: enough to stay inside one pair of tracks.
WINDOW_S = 75.0

#: The steady stretches used as "this is track A" and "this is track B"
#: references, in seconds from the marker. Taken at the far ends. Used only
#: when a marker gives no boundaries; with boundaries the references are placed
#: relative to them instead (see REF_PAD_S / REF_LEN_S).
REF_A = (-WINDOW_S, -WINDOW_S + 17.0)
REF_B = (WINDOW_S - 17.0, WINDOW_S)

#: With explicit boundaries: leave this much clear either side of the blend,
#: then take this much as the steady reference. Both tempo estimates and both
#: level readings come from these stretches, which by construction contain one
#: track only.
REF_PAD_S = 5.0
REF_LEN_S = 30.0

#: 10-90% width of the fitted logistic, as a multiple of its scale parameter.
_LOGISTIC_10_90 = 4.394449

#: Fits worse than this are not trusted and the transition is rejected.
MIN_FIT_R2 = 0.80

#: The low-band envelope is smoothed over roughly this long -- about a bar. At
#: 0.1 s the envelope just tracked individual kicks and the "dip" was measuring
#: the gap between them, not the blend.
LOW_SMOOTH_S = 2.0

#: Crossover used for the low-band trajectory. Matches deck.EQ_LOW_XOVER so the
#: measurement and the thing being tuned are talking about the same band.
LOW_XOVER_HZ = 250.0

#: The low-band trajectory is resampled to this many points so transitions of
#: different lengths can be aggregated elementwise.
TRAJECTORY_POINTS = 16

#: Grids a transition length is tested against.
CANDIDATE_GRIDS = (4, 8, 16, 32)

#: A length counts as "on" a grid within this many bars.
GRID_TOLERANCE_BARS = 1.0


# --- markers -----------------------------------------------------------------

_TIME_RE = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$")


def parse_timestamp(text: str) -> float | None:
    """Parse ``h:mm:ss``, ``mm:ss``, ``mm:ss.s`` or plain seconds."""
    m = _TIME_RE.match(text.strip())
    if not m:
        return None
    a, b, c = m.groups()
    parts = [p for p in (a, b) if p is not None]
    seconds = float(c)
    if len(parts) == 1:
        seconds += int(parts[0]) * 60
    elif len(parts) == 2:
        seconds += int(parts[0]) * 3600 + int(parts[1]) * 60
    return seconds


#: Longest plausible blend. Two timestamps on a line further apart than this are
#: read as two separate markers rather than one transition's start and end.
MAX_BLEND_S = 300.0


@dataclass(frozen=True)
class Marker:
    """One marked transition. ``end_s`` is None when only a centre was given."""

    set_name: str
    start_s: float
    end_s: float | None = None

    @property
    def has_bounds(self) -> bool:
        return self.end_s is not None and self.end_s > self.start_s

    @property
    def centre_s(self) -> float:
        return (self.start_s + self.end_s) / 2.0 if self.has_bounds else self.start_s

    @property
    def length_s(self) -> float:
        return (self.end_s - self.start_s) if self.has_bounds else 0.0


def parse_markers(path: Path) -> list[Marker]:
    """Read a markers file.

    Forgiving on purpose -- these are hand-written. Blank lines and ``#``
    comments are ignored. A line may be:

    * ``<file> <start> <end>``  -- one transition, boundaries given
    * ``<file> <time> [...]``   -- one or more approximate centres
    * ``<file>``                -- sets the current file
    * ``<time>`` / ``<s> <e>``  -- belongs to the current file

    Times may be ``h:mm:ss``, ``mm:ss``, ``mm:ss.s``, or seconds; separators may
    be whitespace, commas or tabs.

    **Start/end pairs are detected, not declared**: if every data line carries
    exactly two increasing timestamps less than :data:`MAX_BLEND_S` apart, the
    file is read as boundaries. Otherwise every timestamp is its own marker.
    Boundaries are worth detecting because they turn blend length from an
    inference into arithmetic.
    """
    rows: list[tuple[str, list[float]]] = []
    current: str | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [f for f in re.split(r"[,\t ]+", line) if f]
        times = [parse_timestamp(f) for f in fields]

        if times and times[0] is None:
            current = fields[0]
            values = [t for t in times[1:] if t is not None]
        else:
            if current is None:
                log.warning("markers: timestamp %r before any filename", fields[0])
                continue
            values = [t for t in times if t is not None]
        for f, t in zip(fields, times):
            if t is None and f != current:
                log.debug("markers: not a timestamp: %r", f)
        if values:
            rows.append((current, values))

    paired = bool(rows) and all(
        len(v) == 2 and v[1] > v[0] and (v[1] - v[0]) <= MAX_BLEND_S for _, v in rows
    )
    out: list[Marker] = []
    for name, values in rows:
        if paired:
            out.append(Marker(name, values[0], values[1]))
        else:
            out.extend(Marker(name, v) for v in values)
    return out


# --- one measured transition -------------------------------------------------


@dataclass
class TransitionMeasurement:
    """What one marked transition looked like."""

    set_name: str
    marker_s: float
    bpm_before: float
    bpm_after: float
    bpm_delta: float
    bpm_delta_pct: float
    #: 10-90% width of the A->B mix fraction.
    length_s: float
    length_bars: float
    nearest_grid: int | None
    grid_error_bars: float
    #: Level of the outgoing and incoming steady sections, and the peak between.
    level_before_db: float
    level_after_db: float
    level_change_db: float
    level_peak_db: float
    #: Combined low band through the blend, relative to the surrounding steady
    #: level. Negative means the low end dipped.
    low_dip_db: float
    low_bump_db: float
    low_trajectory_db: list[float] = field(default_factory=list)
    #: R^2 of the A->B logistic fit. With explicit boundaries this is a quality
    #: signal only -- it does not set the length.
    fit_r2: float = 0.0
    #: True when start/end came from the markers file rather than being inferred.
    bounded: bool = False
    ok: bool = True
    reason: str | None = None


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}


def resolve_set_file(
    name: str, audio_files: list[Path], hint: str | None = None
) -> tuple[Path | None, str | None]:
    """Find the audio file a markers line refers to.

    Exact name or stem first. Failing that, match on shared word tokens: a
    markers file saying ``peak_hour_tech_house.mp3`` plainly means
    ``Peak Hour Tech House Set (John Summit, ...).mp3``, and refusing to see
    that helps nobody. ``hint`` is the markers file's own stem, used when the
    line's filename resolves nothing.

    Returns ``(path, note)``; ``note`` is set when the match was inexact, so the
    caller can say what it did rather than resolving silently.
    """
    by_name: dict[str, Path] = {}
    for p in audio_files:
        by_name.setdefault(p.name, p)
        by_name.setdefault(p.stem, p)
    for key in (name, Path(name).name, Path(name).stem):
        if key in by_name:
            return by_name[key], None

    best: tuple[float, Path | None] = (0.0, None)
    runner_up = 0.0
    for candidate in (name, hint or ""):
        wanted = _tokens(Path(candidate).stem)
        if not wanted:
            continue
        for p in audio_files:
            have = _tokens(p.stem)
            score = len(wanted & have) / len(wanted)
            if score > best[0]:
                runner_up, best = best[0], (score, p)
            elif score > runner_up:
                runner_up = score
        if best[0] >= 0.5 and best[0] > runner_up:
            return best[1], (
                f"{name!r} not found; matched {best[1].name!r} on name "
                f"({best[0]:.0%} of words)"
            )

    if len(audio_files) == 1:
        return audio_files[0], (
            f"{name!r} not found; using the only audio file present: "
            f"{audio_files[0].name!r}"
        )
    return None, None


def _read_window(path: Path, centre_s: float, half_width_s: float) -> np.ndarray:
    """Decode one mono window at ANALYSIS_SR. Reads only what it needs."""
    with sf.SoundFile(str(path)) as fh:
        sr = fh.samplerate
        start = int(max(0.0, centre_s - half_width_s) * sr)
        stop = int(min(fh.frames, int((centre_s + half_width_s) * sr)))
        if stop <= start:
            return np.zeros(0, dtype=np.float32)
        fh.seek(start)
        block = fh.read(stop - start, dtype="float32", always_2d=True)
    mono = block.mean(axis=1)
    if sr != ANALYSIS_SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR)
    return mono.astype(np.float32)


def _tempo_of(y: np.ndarray) -> float:
    """Octave-resolved tempo of a segment, reusing the analysis machinery."""
    if y.size < ANALYSIS_SR * 4:
        return 0.0
    oenv = librosa.onset.onset_strength(y=y, sr=ANALYSIS_SR, hop_length=_HOP)
    raw, _ = librosa.beat.beat_track(
        onset_envelope=oenv, sr=ANALYSIS_SR, hop_length=_HOP, trim=False
    )
    coarse = _refine_tempo(oenv, ANALYSIS_SR, float(np.atleast_1d(raw)[0]))
    bpm, _t0 = _fine_tempo_and_phase(oenv, ANALYSIS_SR, coarse)
    return float(bpm)


def _db(x: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(x, 1e-12))


def _mix_fraction(
    y: np.ndarray, sr: int, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """How much of the window sounds like "after" rather than "before".

    Timbre (MFCC) and harmony (chroma) together: a blend changes both, and
    either alone is easy to fool. Similarity is taken against a steady stretch
    from each end, then rescaled so the curve reads 0 across the "before"
    reference and 1 across the "after" one -- an absolute anchor, rather than
    the global percentiles used previously, which stretched noise to full scale
    and roughly doubled every measured width.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=_HOP)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=_HOP)
    n = min(mfcc.shape[1], chroma.shape[1])
    feat = np.vstack([mfcc[:, :n], chroma[:, :n] * 10.0])
    feat = (feat - feat.mean(axis=1, keepdims=True)) / (
        feat.std(axis=1, keepdims=True) + 1e-9
    )
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=_HOP) + offsets[0]

    def ref(lo: float, hi: float) -> np.ndarray | None:
        sel = (times >= lo) & (times < hi)
        return feat[:, sel].mean(axis=1) if sel.sum() > 4 else None

    a_ref, b_ref = ref(*REF_A), ref(*REF_B)
    if a_ref is None or b_ref is None:
        return np.zeros(0), times

    def sim(v: np.ndarray) -> np.ndarray:
        num = v @ feat
        den = np.linalg.norm(v) * np.linalg.norm(feat, axis=0) + 1e-9
        return (num / den + 1.0) / 2.0  # cosine -> 0..1

    sa, sb = sim(a_ref), sim(b_ref)
    frac = sb / (sa + sb + 1e-9)

    in_a = (times >= REF_A[0]) & (times < REF_A[1])
    in_b = (times >= REF_B[0]) & (times < REF_B[1])
    lo = float(np.median(frac[in_a])) if in_a.sum() else float(frac.min())
    hi = float(np.median(frac[in_b])) if in_b.sum() else float(frac.max())
    if hi - lo < 1e-6:
        return np.zeros(0), times
    frac = np.clip((frac - lo) / (hi - lo), 0.0, 1.0)

    win = max(3, int(2.0 * sr / _HOP) | 1)
    return np.convolve(frac, np.ones(win) / win, mode="same"), times


def _fit_logistic(times: np.ndarray, frac: np.ndarray) -> tuple[float, float, float]:
    """Fit ``1/(1+exp(-(t-t0)/w))`` to the mix fraction.

    Returns ``(centre_s, width_10_90_s, r2)``. Fitting beats thresholding a
    noisy curve: a single stray frame crossing 0.1 early would otherwise set the
    whole width.
    """
    from scipy.optimize import curve_fit

    def logistic(t, t0, w):
        return 1.0 / (1.0 + np.exp(-(t - t0) / np.maximum(w, 1e-3)))

    span = float(times[-1] - times[0])
    try:
        popt, _ = curve_fit(
            logistic,
            times,
            frac,
            p0=[float(times[len(times) // 2]), 5.0],
            bounds=([times[0], 0.2], [times[-1], span / 2.0]),
            maxfev=20000,
        )
    except Exception:
        return 0.0, 0.0, 0.0

    pred = logistic(times, *popt)
    ss_res = float(np.sum((frac - pred) ** 2))
    ss_tot = float(np.sum((frac - frac.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(popt[0]), float(popt[1]) * _LOGISTIC_10_90, r2


def measure_bounded(
    set_path: Path, marker: Marker, set_name: str | None = None
) -> TransitionMeasurement:
    """Measure a transition whose start and end are given.

    With boundaries there is nothing to infer: length is arithmetic, and the
    tempo and level references sit in stretches that provably contain one track
    each. The mix-fraction fit is still computed, but only as a quality signal
    (``fit_r2``) -- it no longer decides anything.
    """
    name = set_name or set_path.stem
    length_s = marker.length_s
    centre = marker.centre_s
    half = length_s / 2.0 + REF_PAD_S + REF_LEN_S

    blank = dict(
        set_name=name, marker_s=round(centre, 3), bpm_before=0.0, bpm_after=0.0,
        bpm_delta=0.0, bpm_delta_pct=0.0, length_s=round(length_s, 3),
        length_bars=0.0, nearest_grid=None, grid_error_bars=0.0,
        level_before_db=0.0, level_after_db=0.0, level_change_db=0.0,
        level_peak_db=0.0, low_dip_db=0.0, low_bump_db=0.0,
    )
    try:
        y = _read_window(set_path, centre, half)
    except Exception as exc:
        return TransitionMeasurement(**blank, ok=False, reason=f"decode failed: {exc}")

    sr = ANALYSIS_SR
    need = int((2 * half - 1.0) * sr)
    if y.size < need:
        return TransitionMeasurement(
            **blank, ok=False,
            reason="not enough audio around the blend (near start/end of set?)",
        )

    t = np.arange(y.size) / sr - half  # seconds relative to the blend centre
    a_lo, a_hi = -length_s / 2 - REF_PAD_S - REF_LEN_S, -length_s / 2 - REF_PAD_S
    b_lo, b_hi = length_s / 2 + REF_PAD_S, length_s / 2 + REF_PAD_S + REF_LEN_S

    def seg(lo: float, hi: float) -> np.ndarray:
        return y[(t >= lo) & (t < hi)]

    before, after = seg(a_lo, a_hi), seg(b_lo, b_hi)
    bpm_before, bpm_after = _tempo_of(before), _tempo_of(after)
    bpm_delta = bpm_after - bpm_before
    bpm_pct = (bpm_delta / bpm_before * 100.0) if bpm_before > 0 else 0.0

    bpm_ref = bpm_after if bpm_after > 0 else bpm_before
    bar_s = BEATS_PER_BAR * 60.0 / bpm_ref if bpm_ref > 0 else 0.0
    length_bars = length_s / bar_s if bar_s > 0 else 0.0

    nearest_grid: int | None = None
    grid_error = float("inf")
    for g in CANDIDATE_GRIDS:
        err = abs(length_bars - g)
        if err < grid_error:
            grid_error, nearest_grid = err, g
    if grid_error > GRID_TOLERANCE_BARS:
        nearest_grid = None

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0

    level_before, level_after = rms(before), rms(after)
    level_peak = rms(seg(-length_s / 2, length_s / 2))

    low_dip, low_bump, traj = _low_band(y, sr, t, -length_s / 2, length_s / 2,
                                        (a_lo, a_hi), (b_lo, b_hi))

    frac, times = _mix_fraction_bounded(y, sr, t, a_lo, a_hi, b_lo, b_hi)
    _c, _w, r2 = _fit_logistic(times, frac) if frac.size else (0.0, 0.0, 0.0)

    blank.update(
        bpm_before=round(bpm_before, 3),
        bpm_after=round(bpm_after, 3),
        bpm_delta=round(bpm_delta, 3),
        bpm_delta_pct=round(bpm_pct, 3),
        length_bars=round(length_bars, 3),
        nearest_grid=nearest_grid,
        grid_error_bars=round(grid_error, 3),
        level_before_db=round(float(_db(level_before)), 2),
        level_after_db=round(float(_db(level_after)), 2),
        level_change_db=round(float(_db(level_after) - _db(level_before)), 2),
        level_peak_db=round(
            float(_db(level_peak) - _db(max(level_before, level_after))), 2
        ),
        low_dip_db=round(low_dip, 2),
        low_bump_db=round(low_bump, 2),
    )
    return TransitionMeasurement(
        **blank,
        low_trajectory_db=[round(float(v), 2) for v in traj],
        fit_r2=round(float(r2), 3),
        bounded=True,
    )


def _low_band(
    y: np.ndarray,
    sr: int,
    t: np.ndarray,
    blend_lo: float,
    blend_hi: float,
    ref_a: tuple[float, float],
    ref_b: tuple[float, float],
) -> tuple[float, float, np.ndarray]:
    """Combined low band through the blend, relative to the steady sections."""
    sos = butter(4, LOW_XOVER_HZ, btype="low", fs=sr, output="sos")
    low = sosfilt(sos, y)
    frame = max(1, int(LOW_SMOOTH_S * sr))
    n_frames = max(1, low.size // frame)
    env = np.array(
        [
            np.sqrt(np.mean(np.square(low[i * frame : (i + 1) * frame], dtype=np.float64)))
            for i in range(n_frames)
        ]
    )
    env_t = np.arange(n_frames) * LOW_SMOOTH_S + t[0]
    steady = ((env_t >= ref_a[0]) & (env_t < ref_a[1])) | (
        (env_t >= ref_b[0]) & (env_t < ref_b[1])
    )
    ref = float(np.median(env[steady])) if steady.sum() else 0.0
    in_blend = (env_t >= blend_lo) & (env_t <= blend_hi)
    if ref <= 0 or in_blend.sum() < 2:
        return 0.0, 0.0, np.zeros(TRAJECTORY_POINTS)
    rel = np.asarray(_db(env[in_blend] / ref))
    traj = np.interp(
        np.linspace(0, 1, TRAJECTORY_POINTS), np.linspace(0, 1, rel.size), rel
    )
    return float(rel.min()), float(rel.max()), traj


def _mix_fraction_bounded(
    y: np.ndarray, sr: int, t: np.ndarray,
    a_lo: float, a_hi: float, b_lo: float, b_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix fraction with references placed against known boundaries."""
    global REF_A, REF_B
    saved = REF_A, REF_B
    REF_A, REF_B = (a_lo, a_hi), (b_lo, b_hi)
    try:
        return _mix_fraction(y, sr, np.array([t[0]]))
    finally:
        REF_A, REF_B = saved


def measure_transition(
    set_path: Path, marker_s: float, set_name: str | None = None
) -> TransitionMeasurement:
    """Measure one transition from an approximate centre only.

    Blend length here is *inferred*, and validation against constructed ground
    truth showed that inference is unreliable (see docs/tuning.md). Prefer
    markers with explicit boundaries, which route to :func:`measure_bounded`.
    """
    name = set_name or set_path.name
    blank = dict(
        set_name=name, marker_s=marker_s, bpm_before=0.0, bpm_after=0.0,
        bpm_delta=0.0, bpm_delta_pct=0.0, length_s=0.0, length_bars=0.0,
        nearest_grid=None, grid_error_bars=0.0, level_before_db=0.0,
        level_after_db=0.0, level_change_db=0.0, level_peak_db=0.0,
        low_dip_db=0.0, low_bump_db=0.0,
    )
    try:
        y = _read_window(set_path, marker_s, WINDOW_S)
    except Exception as exc:
        return TransitionMeasurement(**blank, ok=False, reason=f"decode failed: {exc}")

    if y.size < ANALYSIS_SR * 30:
        return TransitionMeasurement(
            **blank, ok=False, reason="window too short (near start/end of set?)"
        )

    sr = ANALYSIS_SR
    offsets = np.array([-WINDOW_S])

    # --- tempo either side, from the reference stretches only ---
    # Taken at the far ends of the window so neither segment overlaps the blend.
    def segment(lo: float, hi: float) -> np.ndarray:
        i0 = int(max(0.0, lo + WINDOW_S) * sr)
        i1 = int(min(2 * WINDOW_S, hi + WINDOW_S) * sr)
        return y[i0:i1]

    bpm_before = _tempo_of(segment(*REF_A))
    bpm_after = _tempo_of(segment(*REF_B))
    bpm_delta = bpm_after - bpm_before
    bpm_pct = (bpm_delta / bpm_before * 100.0) if bpm_before > 0 else 0.0

    # --- blend width ---
    frac, times = _mix_fraction(y, sr, offsets)
    if frac.size == 0:
        return TransitionMeasurement(
            **blank, ok=False, reason="not enough steady audio either side"
        )

    centre, length_s, r2 = _fit_logistic(times, frac)
    if r2 < MIN_FIT_R2:
        return TransitionMeasurement(
            **blank, ok=False,
            reason=f"no clean A->B crossover (logistic fit R2={r2:.2f})",
        )
    if length_s <= 0:
        return TransitionMeasurement(**blank, ok=False, reason="degenerate fit")
    t_lo, t_hi = centre - length_s / 2.0, centre + length_s / 2.0

    bpm_ref = bpm_after if bpm_after > 0 else bpm_before
    bar_s = BEATS_PER_BAR * 60.0 / bpm_ref if bpm_ref > 0 else 0.0
    length_bars = length_s / bar_s if bar_s > 0 else 0.0

    nearest_grid: int | None = None
    grid_error = float("inf")
    for g in CANDIDATE_GRIDS:
        err = abs(length_bars - g)
        if err < grid_error:
            grid_error, nearest_grid = err, g
    if grid_error > GRID_TOLERANCE_BARS:
        nearest_grid = None

    # --- levels ---
    def rms_between(lo: float, hi: float) -> float:
        sel = (np.arange(y.size) / sr - WINDOW_S >= lo) & (
            np.arange(y.size) / sr - WINDOW_S < hi
        )
        seg = y[sel]
        return float(np.sqrt(np.mean(np.square(seg, dtype=np.float64)))) if seg.size else 0.0

    level_before = rms_between(*REF_A)
    level_after = rms_between(*REF_B)
    level_peak = rms_between(t_lo, t_hi)

    # --- combined low band through the blend ---
    sos = butter(4, LOW_XOVER_HZ, btype="low", fs=sr, output="sos")
    low = sosfilt(sos, y)
    frame = int(LOW_SMOOTH_S * sr)
    n_frames = max(1, low.size // frame)
    low_rms = np.array(
        [
            np.sqrt(np.mean(np.square(low[i * frame : (i + 1) * frame], dtype=np.float64)))
            for i in range(n_frames)
        ]
    )
    low_times = np.arange(n_frames) * LOW_SMOOTH_S - WINDOW_S
    steady = ((low_times >= REF_A[0]) & (low_times < REF_A[1])) | (
        (low_times >= REF_B[0]) & (low_times < REF_B[1])
    )
    low_ref = float(np.median(low_rms[steady])) if steady.sum() else 0.0
    in_blend = (low_times >= t_lo) & (low_times <= t_hi)
    if low_ref > 0 and in_blend.sum() >= 2:
        rel_db = _db(low_rms[in_blend] / low_ref)
        traj = np.interp(
            np.linspace(0, 1, TRAJECTORY_POINTS),
            np.linspace(0, 1, rel_db.size),
            rel_db,
        )
        low_dip = float(np.min(rel_db))
        low_bump = float(np.max(rel_db))
    else:
        traj = np.zeros(TRAJECTORY_POINTS)
        low_dip = low_bump = 0.0

    blank.update(
        bpm_before=round(bpm_before, 3),
        bpm_after=round(bpm_after, 3),
        bpm_delta=round(bpm_delta, 3),
        bpm_delta_pct=round(bpm_pct, 3),
        length_s=round(length_s, 3),
        length_bars=round(length_bars, 3),
        nearest_grid=nearest_grid,
        grid_error_bars=round(grid_error, 3),
        level_before_db=round(float(_db(level_before)), 2),
        level_after_db=round(float(_db(level_after)), 2),
        level_change_db=round(float(_db(level_after) - _db(level_before)), 2),
        level_peak_db=round(float(_db(level_peak) - _db(max(level_before, level_after))), 2),
        low_dip_db=round(low_dip, 2),
        low_bump_db=round(low_bump, 2),
    )
    return TransitionMeasurement(
        **blank, low_trajectory_db=[round(float(v), 2) for v in traj]
    )


# --- aggregation -------------------------------------------------------------


def _stats(values: Iterable[float]) -> dict:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 4),
        "iqr": [round(float(q1), 4), round(float(q3), 4)],
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def build_profile(measurements: list[TransitionMeasurement]) -> dict:
    """Reduce measured transitions to medians and IQRs."""
    good = [m for m in measurements if m.ok]
    grids: dict[str, int] = {str(g): 0 for g in CANDIDATE_GRIDS}
    grids["none"] = 0
    for m in good:
        grids[str(m.nearest_grid) if m.nearest_grid else "none"] += 1

    traj = [m.low_trajectory_db for m in good if len(m.low_trajectory_db) == TRAJECTORY_POINTS]
    traj_median = (
        [round(float(v), 2) for v in np.median(np.array(traj), axis=0)] if traj else []
    )

    def measures_for(rows: list[TransitionMeasurement]) -> dict:
        return {
            "length_bars": _stats(m.length_bars for m in rows),
            "length_s": _stats(m.length_s for m in rows),
            "bpm_delta_pct": _stats(abs(m.bpm_delta_pct) for m in rows),
            "level_peak_db": _stats(m.level_peak_db for m in rows),
            "low_dip_db": _stats(m.low_dip_db for m in rows),
        }

    # Per set as well as pooled: two DJs in different genres should be compared,
    # not silently averaged into a median that describes neither.
    by_set = {
        name: {
            "n": sum(1 for m in good if m.set_name == name),
            "median_bpm": round(
                float(
                    np.median(
                        [m.bpm_before for m in good if m.set_name == name and m.bpm_before > 0]
                        or [0.0]
                    )
                ),
                1,
            ),
            "measures": measures_for([m for m in good if m.set_name == name]),
        }
        for name in sorted({m.set_name for m in good})
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_transitions": len(good),
        "n_rejected": len(measurements) - len(good),
        "n_sets": len({m.set_name for m in good}),
        "window_s": WINDOW_S,
        "by_set": by_set,
        "measures": {
            "length_bars": _stats(m.length_bars for m in good),
            "length_s": _stats(m.length_s for m in good),
            "bpm_delta": _stats(m.bpm_delta for m in good),
            "bpm_delta_pct": _stats(abs(m.bpm_delta_pct) for m in good),
            "level_change_db": _stats(m.level_change_db for m in good),
            "level_peak_db": _stats(m.level_peak_db for m in good),
            "low_dip_db": _stats(m.low_dip_db for m in good),
            "low_bump_db": _stats(m.low_bump_db for m in good),
        },
        "grid_alignment": grids,
        "low_trajectory_db_median": traj_median,
        "rejected": [
            {"set": m.set_name, "marker_s": m.marker_s, "reason": m.reason}
            for m in measurements
            if not m.ok
        ],
        "transitions": [asdict(m) for m in good],
    }


def analyze_sets(
    sets_dir: Path, markers_path: Path, progress: bool = True
) -> tuple[list[TransitionMeasurement], dict]:
    """Measure every marked transition. Returns ``(measurements, profile)``."""
    sets_dir = Path(sets_dir)
    markers_path = Path(markers_path)

    # A folder of markers files is read whole -- one per set is the natural way
    # to keep them, and it is how they arrive.
    if markers_path.is_dir():
        marker_files = sorted(markers_path.glob("*.txt"))
    else:
        marker_files = [markers_path]

    markers: list[Marker] = []
    hints: list[str] = []
    for mf in marker_files:
        parsed = parse_markers(mf)
        markers.extend(parsed)
        hints.extend([mf.stem] * len(parsed))
    if not markers:
        return [], build_profile([])

    audio_files = [
        p
        for p in sorted(sets_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]

    resolved: dict[tuple[str, str], Path | None] = {}
    measurements: list[TransitionMeasurement] = []
    for i, (marker, hint) in enumerate(zip(markers, hints), 1):
        name = marker.set_name
        key = (name, hint)
        if key not in resolved:
            path, note = resolve_set_file(name, audio_files, hint=hint)
            resolved[key] = path
            if note and progress:
                print(f"[markers] {note}")
        path = resolved[key]
        if path is None:
            measurements.append(
                TransitionMeasurement(
                    set_name=name, marker_s=marker.centre_s, bpm_before=0.0,
                    bpm_after=0.0, bpm_delta=0.0, bpm_delta_pct=0.0,
                    length_s=marker.length_s, length_bars=0.0, nearest_grid=None,
                    grid_error_bars=0.0, level_before_db=0.0, level_after_db=0.0,
                    level_change_db=0.0, level_peak_db=0.0, low_dip_db=0.0,
                    low_bump_db=0.0, ok=False,
                    reason=f"no audio file named {name!r} under {sets_dir}",
                )
            )
            continue

        if marker.has_bounds:
            m = measure_bounded(path, marker, set_name=path.stem)
        else:
            m = measure_transition(path, marker.centre_s, set_name=path.stem)
        measurements.append(m)
        if progress:
            status = (
                f"{m.length_s:6.1f}s = {m.length_bars:5.1f} bars  "
                f"{m.bpm_before:6.1f}->{m.bpm_after:6.1f} BPM "
                f"({m.bpm_delta:+5.2f})  grid {m.nearest_grid or '-'}  "
                f"lvl {m.level_change_db:+5.2f} dB  low {m.low_dip_db:+6.2f} dB"
                if m.ok
                else f"SKIPPED: {m.reason}"
            )
            print(f"[{i}/{len(markers)}] {marker.centre_s:8.1f}s  {status}")

    return measurements, build_profile(measurements)


def write_profile(profile: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=1), encoding="utf-8")
    return out_path
