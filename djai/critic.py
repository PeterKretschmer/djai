"""How far a transition sits from the reference sets, and the search for closer.

THREADING CONTEXT: the preview's **worker thread**, never the audio thread.
Pure numbers in, pure numbers out: no audio leaves here, and every value the
model ever sees is a float in a line of text.

Why a distance and not thresholds
---------------------------------
The rules this replaces were one-step and fixed: "dip over 3 dB, switch the
curve". They could only say *worse than a constant*, so a blend already inside
the range real DJs use still got revised, and one that was far outside got a
single nudge and stopped. In a soak they left dips above the limit (4.87 ->
4.46 dB) and called it done.

What the reference sets actually do is a distribution, measured in
``reference_profile.json``: 26 transitions from two sets, each as a median and
an interquartile range. A blend inside that range is not "under the limit", it
is *what the DJs did*, and it scores zero. Outside it, the distance grows with
how far outside, so a search has a gradient to follow and the worst measure is
the one worth fixing.

Measuring like for like
-----------------------
The profile's measures come from :mod:`djai.reference`, on real sets, with 30 s
of steady audio either side of the blend. A preview render carries only its
context bars, so this module calls reference's own measurement code on the
rendered mix -- same definitions, same low-band crossover, same smoothing --
with :data:`CONTEXT_BARS` of context. Measured on the reference sets, shrinking
the steady window to 15 s moves the medians by at most 0.7 dB, while 7.5 s
widens every IQR; eight bars is about 15 s at 128 BPM, which is why that is the
context a critic render asks for.

The search itself does not render: a render of a 24-bar blend costs 893 ms and
its measurement another 667 ms, so a 4 s budget buys two. Candidates are scored
by :class:`Surrogate`, which combines each deck's per-block band energies --
computed once -- with the envelope a candidate would run. The winner is then
rendered and measured for real by the caller, and only committed if that real
measurement is actually closer. A surrogate that is wrong costs a wasted
render, never a bad transition.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from djai import config, transition as tr

log = logging.getLogger(__name__)

#: Where the profile lives by default: written by ``python -m djai reference``
#: into the project root, which is the package's parent. Resolved from the
#: module rather than the working directory, so a session started from
#: somewhere else still scores against the reference sets instead of silently
#: falling back to the old fixed thresholds.
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent.parent / "reference_profile.json"

#: Context each side of the blend in a critic render. Eight bars is ~15 s at
#: 128 BPM; see the module docstring for why 15 and not 30 or 7.5.
CONTEXT_BARS: float = 8.0

#: Seconds dropped between the blend and the steady segment measured against
#: it. The reference used 5 s because its blend boundaries were human marks;
#: here they are known exactly, so 1 s is enough and leaves more steady audio.
PAD_S: float = 1.0

#: The measures a preview can compare with the profile. ``length_bars`` is the
#: shape of the blend; the rest are what it does to the sound.
MEASURE_KEYS: tuple[str, ...] = (
    "length_bars",
    "level_change_db",
    "level_peak_db",
    "low_dip_db",
    "low_bump_db",
    # Measured by the preview rather than taken from the render, and scored
    # against the configured limit rather than the reference sets -- see
    # STAND_IN_RANGES. Without them the search optimises the blend's average
    # level while a 4.5 dB hole in the middle of it goes unpunished, which is
    # exactly what a 20-minute soak showed: 10 of 13 dips over the 3 dB limit
    # while every reference measure sat inside its range.
    "loudness_dip_db",
    "low_end_overlap_bars",
)

#: Ranges for measures the reference sets were never measured for. The DJs'
#: own dips and low-end overlaps are not in reference_profile.json -- adding
#: them means re-running `djai reference` over the sets, which rewrites a file
#: this program is judged against -- so until then the configured limit stands
#: in for the distribution, and says so in every log line it produces.
#: What the search itself optimises: the measures the surrogate predicts well.
#: Validated against real renders on 20 candidates -- these five agree at
#: r +0.74 to +1.00 and the surrogate's best pick is the real best pick. The
#: preview's own dip and overlap are NOT here: K-weighted or not, they predict
#: at r +0.24 and +0.11, and searching on a number that cannot be predicted is
#: how a search ends up confidently wrong. They are scored, and they decide
#: whether the result is accepted; they just do not steer it.
SEARCH_KEYS: tuple[str, ...] = (
    "length_bars", "level_change_db", "level_peak_db", "low_dip_db", "low_bump_db",
)


def stand_in_ranges() -> dict[str, dict]:
    return {
        "loudness_dip_db": {
            "median": 0.0,
            "iqr": [0.0, float(config.PREVIEW_MAX_LOUDNESS_DIP_DB)],
            "stand_in": True,
        },
        "low_end_overlap_bars": {
            "median": 0.0,
            "iqr": [0.0, float(config.PREVIEW_MAX_LOW_OVERLAP_BARS)],
            "stand_in": True,
        },
    }

#: Per-measure weights. Feedback moves these (Phase 4's feedback loop); equal
#: is the honest starting point, since nothing has been learned yet.
DEFAULT_WEIGHTS: dict[str, float] = {key: 1.0 for key in MEASURE_KEYS}

#: A measure this many IQR widths outside the range is as bad as it gets: past
#: it, the distance stops growing so one silly number cannot drown the rest.
MAX_DISTANCE: float = 4.0

#: Low-band crossover and smoothing, taken from the reference measurement so
#: both sides of the comparison mean the same thing.
LOW_XOVER_HZ: float = 250.0
LOW_SMOOTH_S: float = 2.0


def load_profile(path: Path | str | None = None) -> dict | None:
    """The reference profile, or None when there is not one to compare with."""
    path = Path(path) if path is not None else DEFAULT_PROFILE_PATH
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("no reference profile at %s: %s", path, exc)
        return None
    if not isinstance(profile, dict) or not profile.get("measures"):
        log.warning("reference profile at %s has no measures", path)
        return None
    return profile


# --- measuring a render the way the reference sets were measured -----------------


def _db(x: float) -> float:
    return float(20.0 * math.log10(max(float(x), 1e-12)))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def _smooth_rms(values: np.ndarray, width: int) -> np.ndarray:
    """RMS of ``values`` in groups of ``width``, the reference's 2 s frames."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    width = max(1, min(int(width), values.size))
    n = values.size // width
    if n == 0:
        return np.array([_rms(values)])
    return np.sqrt(np.square(values[: n * width]).reshape(n, width).mean(axis=1))


def _mono(mix: np.ndarray) -> np.ndarray:
    return mix.mean(axis=1) if mix.ndim > 1 else mix


def measures_from_render(render: Any, sample_rate: int = 44100) -> dict[str, float]:
    """Reference-comparable measures of one rendered preview.

    Returns an empty dict when the render has too little steady audio either
    side to compare with -- a measure taken from nothing is worse than no
    measure, because it would score.
    """
    from scipy.signal import butter, sosfilt

    mono = _mono(np.asarray(render.mix, dtype=np.float64))
    t0 = int(render.transition_start_frame)
    t1 = t0 + int(render.transition_frames)
    pad = int(PAD_S * sample_rate)
    before = mono[: max(t0 - pad, 0)]
    after = mono[min(t1 + pad, mono.size):]
    blend = mono[t0:t1]
    if blend.size < sample_rate or before.size < sample_rate or after.size < sample_rate:
        return {}

    level_before, level_after = _rms(before), _rms(after)
    if level_before <= 0.0 or level_after <= 0.0:
        return {}
    out = {
        "length_bars": float(render.bars),
        "level_change_db": round(_db(level_after) - _db(level_before), 2),
        "level_peak_db": round(
            _db(_rms(blend)) - _db(max(level_before, level_after)), 2
        ),
    }

    sos = butter(4, LOW_XOVER_HZ, btype="low", fs=sample_rate, output="sos")
    low = np.asarray(sosfilt(sos, mono))
    frame = max(1, int(LOW_SMOOTH_S * sample_rate))
    n = max(1, low.size // frame)
    env = np.array([_rms(low[i * frame: (i + 1) * frame]) for i in range(n)])
    starts = np.arange(n) * frame
    steady = (starts < max(t0 - pad, 0)) | (starts >= min(t1 + pad, mono.size))
    inside = (starts >= t0) & (starts < t1)
    ref = float(np.median(env[steady])) if steady.any() else 0.0
    if ref > 0 and inside.sum() >= 2:
        rel = 20.0 * np.log10(np.maximum(env[inside], 1e-12) / ref)
        out["low_dip_db"] = round(float(rel.min()), 2)
        out["low_bump_db"] = round(float(rel.max()), 2)
    return out


# --- distance from the profile ----------------------------------------------------


def distance(
    measures: dict[str, Any],
    profile: dict | None,
    weights: dict[str, float] | None = None,
    keys: tuple[str, ...] | None = None,
) -> tuple[float, dict[str, dict[str, float]]]:
    """``(score, per-measure detail)``. Zero means "inside what the DJs did".

    Each measure is scored in interquartile widths outside the reference range,
    so the units are the reference's own spread rather than decibels, and
    measures with different units can be compared and summed.
    """
    if not profile:
        return 0.0, {}
    stats = dict(profile.get("measures", {}))
    for key, entry in stand_in_ranges().items():
        stats.setdefault(key, entry)
    weights = weights or DEFAULT_WEIGHTS
    detail: dict[str, dict[str, float]] = {}
    total = weight_sum = 0.0
    for key in (keys or MEASURE_KEYS):
        value = measures.get(key)
        entry = stats.get(key) or {}
        iqr = entry.get("iqr")
        if not isinstance(value, (int, float)) or not iqr or len(iqr) != 2:
            continue
        lo, hi = float(iqr[0]), float(iqr[1])
        width = max(hi - lo, 1e-6)
        if value < lo:
            d = (lo - value) / width
        elif value > hi:
            d = (value - hi) / width
        else:
            d = 0.0
        d = min(float(d), MAX_DISTANCE)
        w = float(weights.get(key, 1.0))
        detail[key] = {"value": float(value), "low": lo, "high": hi,
                       "median": float(entry.get("median", (lo + hi) / 2)),
                       "distance": round(d, 3), "weight": w}
        total += w * d
        weight_sum += w
    score = total / weight_sum if weight_sum else 0.0
    return round(score, 4), detail


def as_text(detail: dict[str, dict[str, float]], score: float) -> str:
    """The critic's verdict as numbers in lines of text, for the local model.

    The model is text-only and cannot be handed arrays, so each measure is one
    line: what it is, what the reference range is, and how far outside it sits.
    """
    lines = [f"distance {score:.3f} (0 = inside the reference range)"]
    for key, d in sorted(detail.items(), key=lambda kv: -kv[1]["distance"]):
        lines.append(
            f"{key} {d['value']:.2f} reference {d['low']:.2f}..{d['high']:.2f} "
            f"median {d['median']:.2f} distance {d['distance']:.2f}"
        )
    return "\n".join(lines)


def worst(detail: dict[str, dict[str, float]]) -> str | None:
    """The measure furthest outside the reference range, if any is."""
    outside = {k: d["distance"] for k, d in detail.items() if d["distance"] > 0}
    return max(outside, key=outside.get) if outside else None


# --- a cheap stand-in for a render ------------------------------------------------


class Surrogate:
    """Predicts a candidate's measures from each deck's own per-block energies.

    Built once per transition: for every output block of the window, how loud
    each deck is, full band and below the crossover, taken from the decks' own
    audio through the render's frame anchors. A candidate's envelope then says
    how much of each deck is audible in each block, and the reference's
    statistics are computed from that -- no engine, no render.

    An earlier version used one steady level per deck instead of per-block
    energies. It was worthless where it mattered: against real renders its
    low_dip_db was out by 10.9 dB on average and its rank correlation was
    -0.12, because a dip is mostly the tracks arriving at a breakdown, not the
    curve. Per-block energies are the fix, and they cost one filter per deck.

    Decks are summed in power, which is what incoherent music does on average
    and is wrong where the two are phase-locked. The winner is rendered for
    real before anything is committed, so that error costs a render, not a
    transition.
    """

    def __init__(self, render: Any, loaded_a: Any, loaded_b: Any,
                 params: Any = None, sample_rate: int = 44100):
        #: Set by :meth:`calibrate`; until then the prediction stands alone.
        self.offsets: dict[str, float] = {}
        self._base_params = params
        self.sample_rate = sample_rate
        self.block = int(render.blocksize)
        self.t0 = int(render.transition_start_frame)
        self.frames = int(render.transition_frames)
        self.bars = float(render.bars)
        self.bpm_a = float(render.bpm_a)
        self.rate_b = float(render.rate_b) or 1.0
        mono = _mono(np.asarray(render.mix, dtype=np.float64))
        self.n_blocks = max(1, mono.size // self.block)

        starts = np.arange(self.n_blocks) * self.block
        a_frames = float(render.a_start_deck_frame) + starts
        b_frames = (float(render.b_anchor_deck_frame)
                    + (starts - int(render.b_anchor_output_frame)) * self.rate_b)
        self.a_full, self.a_low = self._deck_energies(loaded_a, a_frames)
        self.b_full, self.b_low = self._deck_energies(loaded_b, b_frames)

        pad_blocks = int(PAD_S * sample_rate / self.block)
        first = max(0, self.t0 // self.block - pad_blocks)
        last = min(self.n_blocks, (self.t0 + self.frames) // self.block + pad_blocks)
        self.before = slice(0, first)
        self.after = slice(last, self.n_blocks)
        self.inside = slice(self.t0 // self.block,
                            (self.t0 + self.frames) // self.block)
        # The reference reads the low band in 2 s frames, so this must too: a
        # minimum taken per 46 ms block catches momentary nulls that a 2 s
        # window never sees, which pinned the predicted dip at -33 dB while
        # real renders measured -9.
        self.smooth_blocks = max(1, int(round(LOW_SMOOTH_S * sample_rate / self.block)))
        steady_low = np.concatenate([
            _smooth_rms(self.a_low[self.before], self.smooth_blocks),
            _smooth_rms(self.b_low[self.after], self.smooth_blocks),
        ])
        self.steady_low_ref = float(np.median(steady_low)) if steady_low.size else 0.0
        self.usable = bool(
            first > 4 and last < self.n_blocks - 4
            and self.a_full[self.before].size and self.b_full[self.after].size
            and self.steady_low_ref > 0
        )

    def _deck_energies(self, loaded: Any, frames: np.ndarray):
        """Per-block loudness and low-band RMS of one deck, at those frames.

        The loudness curve is K-weighted, because that is what the preview's
        dip is measured on: an unweighted RMS predicted it with a correlation
        of +0.01, which is no prediction at all.
        """
        from scipy.signal import butter, sosfilt

        from djai import preview as preview_mod

        # Slice first, convert second: a whole track is minutes of float32
        # stereo, and converting all of it to mono float64 to look at a
        # 60 s window cost 400 ms of the preview's budget.
        source = loaded.audio
        lo = int(max(0, np.floor(frames.min())))
        hi = int(min(source.shape[0], np.ceil(frames.max()) + self.block + 1))
        if hi - lo < self.block:
            zeros = np.zeros(frames.size)
            return zeros, zeros.copy()
        span = _mono(np.asarray(source[lo:hi], dtype=np.float64))
        try:
            weighted = _mono(preview_mod._k_weighted(span[:, None], self.sample_rate))
        except Exception:  # noqa: BLE001 - a flat level still predicts something
            weighted = span
        sos = butter(4, LOW_XOVER_HZ, btype="low", fs=self.sample_rate, output="sos")
        span_low = np.asarray(sosfilt(sos, span))
        # Block RMS by differences of a cumulative sum of squares: deck B's
        # block starts are not block-aligned (it plays at its own rate), so a
        # reshape will not do, and a Python loop over ~1300 blocks per deck
        # cost 387 ms of a 4 s budget.
        starts = np.clip(frames.astype(np.int64) - lo, 0, max(span.size - self.block, 0))
        ends = starts + self.block

        def block_rms(x: np.ndarray) -> np.ndarray:
            cumulative = np.concatenate(([0.0], np.cumsum(np.square(x))))
            return np.sqrt(
                np.maximum(cumulative[ends] - cumulative[starts], 0.0) / self.block
            )

        return block_rms(weighted), block_rms(span_low)

    def calibrate(self, real: dict[str, float]) -> dict[str, float]:
        """Line the surrogate up with one real measurement of the same params.

        The preview renders the original parameters for real anyway, so that
        render is a free anchor: the offset between it and the prediction for
        the same parameters is subtracted from every later prediction. It is
        the systematic part of the error -- a dip predicted 7 dB low stays 7 dB
        low across candidates -- and removing it is what lets a distance built
        on interquartile ranges rank candidates rather than measure an offset.
        """
        self.offsets = {}
        predicted = self.predict(self._base_params) if self._base_params else {}
        for key in MEASURE_KEYS:
            a, b = real.get(key), predicted.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                self.offsets[key] = float(a) - float(b)
        return dict(self.offsets)

    def predict(self, params) -> dict[str, float]:
        """The measures a candidate would land on, as far as this can tell."""
        if not self.usable:
            return {}
        total_frames = int(
            round(float(params.length_bars) * tr.BEATS_PER_BAR_T * 60.0
                  / max(self.bpm_a, 1e-6) * self.sample_rate)
        )
        env = tr.build_envelope_from_params(
            params, total_frames, self.block, self.bpm_a, self.sample_rate
        )
        if len(env) < 2:
            return {}
        # The candidate may be longer or shorter than the window that was
        # rendered; it starts where the rendered one did and is clipped to what
        # there is audio for.
        start = self.inside.start
        n = min(len(env), self.n_blocks - start)
        if n < 2:
            return {}
        idx = slice(start, start + n)
        a_gain = np.asarray(env[:n, tr.FROM_GAIN], dtype=np.float64)
        b_gain = np.asarray(env[:n, tr.TO_GAIN], dtype=np.float64)
        a_low_eq = np.asarray(env[:n, tr.FROM_LOW], dtype=np.float64)
        b_low_eq = np.asarray(env[:n, tr.TO_LOW], dtype=np.float64)

        blend_full = np.sqrt((a_gain * self.a_full[idx]) ** 2
                             + (b_gain * self.b_full[idx]) ** 2)
        blend_low = np.sqrt((a_low_eq * a_gain * self.a_low[idx]) ** 2
                            + (b_low_eq * b_gain * self.b_low[idx]) ** 2)
        before_rms = _rms(self.a_full[self.before])
        after_rms = _rms(self.b_full[self.after])
        if before_rms <= 0 or after_rms <= 0:
            return {}
        smoothed = _smooth_rms(blend_low, self.smooth_blocks)
        rel = 20.0 * np.log10(np.maximum(smoothed, 1e-12) / self.steady_low_ref)
        # The preview's own two measures, predicted the way it measures them:
        # the dip is how far the mix falls below the average of what the two
        # decks are playing, and the overlap is bars where both decks put real
        # energy under the crossover. Same definitions, cheaper arithmetic.
        sources = 0.5 * (np.square(self.a_full[idx]) + np.square(self.b_full[idx]))
        mixed = np.square(blend_full)
        window = max(1, int(round(0.4 * self.sample_rate / self.block)))
        source_db = 10.0 * np.log10(np.maximum(_smooth_rms(np.sqrt(sources), window), 1e-10))
        mix_db = 10.0 * np.log10(np.maximum(_smooth_rms(np.sqrt(mixed), window), 1e-10))
        live = source_db > -70.0
        dip = float(np.max(source_db[live] - mix_db[live])) if live.any() else 0.0

        low_a = a_low_eq * a_gain * self.a_low[idx]
        low_b = b_low_eq * b_gain * self.b_low[idx]
        blocks_per_bar = max(1, int(round(
            4 * 60.0 / max(self.bpm_a, 1e-6) * self.sample_rate / self.block
        )))
        overlap_bars = 0.0
        if low_a.size >= blocks_per_bar:
            bars = low_a.size // blocks_per_bar
            per_bar_a = _smooth_rms(low_a, blocks_per_bar)[:bars]
            per_bar_b = _smooth_rms(low_b, blocks_per_bar)[:bars]
            ref_a = float(np.percentile(per_bar_a, 90)) if per_bar_a.size else 0.0
            ref_b = float(np.percentile(per_bar_b, 90)) if per_bar_b.size else 0.0
            frac = config.PREVIEW_BAND_PRESENT_FRAC
            if ref_a > 0 and ref_b > 0:
                overlap_bars = float(np.sum(
                    (per_bar_a > frac * ref_a) & (per_bar_b > frac * ref_b)
                ))

        out = {
            "length_bars": float(params.length_bars),
            "level_change_db": round(_db(after_rms) - _db(before_rms), 2),
            "level_peak_db": round(
                _db(_rms(blend_full)) - _db(max(before_rms, after_rms)), 2
            ),
            "low_dip_db": round(float(rel.min()), 2),
            "low_bump_db": round(float(rel.max()), 2),
            "loudness_dip_db": round(max(0.0, dip), 2),
            "low_end_overlap_bars": round(overlap_bars, 1),
        }
        for key, offset in self.offsets.items():
            if key in out and key != "length_bars":
                out[key] = round(out[key] + offset, 2)
        return out


# --- the search -------------------------------------------------------------------


def candidates(params) -> Iterable[tuple[str, Any]]:
    """Single-parameter moves from ``params``, as ``(field, value)`` pairs.

    Deliberately a coordinate walk over a small grid rather than anything
    clever: the space is eight fields wide, the surrogate is approximate, and
    the winner is verified by a real render. What matters is that every move is
    one a DJ could name.
    """
    lo_len, hi_len = tr.LENGTH_BARS_RANGE
    length = float(params.length_bars)
    for factor in (0.5, 0.75, 1.25, 1.5):
        value = round(min(max(length * factor, lo_len), hi_len))
        if value != length:
            yield "length_bars", float(value)
    for curve in tr.CURVES:
        if curve != params.curve:
            yield "curve", curve
    lo_swap, hi_swap = tr.LOW_SWAP_BARS_RANGE
    for fraction in (0.25, 0.5, 0.75):
        bar = max(1, int(length * fraction))
        if bar != params.low_swap_bar:
            yield "low_swap_bar", bar
    if params.low_swap_bar is not None:
        yield "low_swap_bar", None
    for bars in {lo_swap, max(lo_swap, min(hi_swap, 2)), hi_swap}:
        if bars != params.low_swap_bars:
            yield "low_swap_bars", bars
    yield "deck_a_high_rolloff", not params.deck_a_high_rolloff
    for sweep in tr.FILTER_SWEEPS:
        if sweep != params.filter_sweep:
            yield "filter_sweep", sweep
    lo_i, hi_i = tr.INTENSITY_RANGE
    for factor in (0.8, 1.2):
        value = round(min(max(float(params.intensity) * factor, lo_i), hi_i), 3)
        if value != params.intensity:
            yield "intensity", value
    lo_d, hi_d = tr.DECK_B_LOW_DELAY_RANGE
    for delay in {lo_d, hi_d, max(lo_d, min(hi_d, params.deck_b_low_delay_bars + 2))}:
        if delay != params.deck_b_low_delay_bars:
            yield "deck_b_low_delay_bars", delay


@dataclasses.dataclass
class SearchResult:
    """What the search settled on, and the trail it left."""

    params: Any
    score: float
    start_score: float
    steps: list[dict]
    evaluations: int
    #: Candidates the supervisor would not allow. A search that proposes
    #: nothing because everything was refused looks identical to one that
    #: found nothing, and they need different fixes.
    refused: int = 0

    @property
    def improved(self) -> bool:
        return self.score < self.start_score - 1e-9


def search(
    params,
    profile: dict,
    predict: Callable[[Any], dict[str, float]],
    weights: dict[str, float] | None = None,
    max_passes: int = 3,
    max_evaluations: int = 400,
    supervisor: Any = None,
) -> SearchResult:
    """Coordinate descent on the distance from the reference distribution.

    ``predict`` maps parameters to measures -- the surrogate in the live path,
    a real render in tests. Stops when a pass finds nothing better, which on a
    blend already inside the reference range is the first pass.
    """
    best = params
    start_measures = predict(best)
    best_score, _ = distance(start_measures, profile, weights, keys=SEARCH_KEYS)
    start_score = best_score
    steps: list[dict] = []
    evaluations = 1
    refused = 0
    if best_score <= 0.0:
        # Already inside the reference range on every measure. A distance does
        # not go below zero, so there is nothing to search for, and the budget
        # is better spent not spending it.
        return SearchResult(params=best, score=best_score, start_score=start_score,
                            steps=steps, evaluations=evaluations, refused=refused)
    for _ in range(max_passes):
        improved_this_pass = False
        for field, value in candidates(best):
            if evaluations >= max_evaluations:
                break
            candidate = dataclasses.replace(best, **{field: value})
            if supervisor is not None:
                ok, _reason = _accepted(supervisor, candidate)
                if not ok:
                    refused += 1
                    continue
            measures = predict(candidate)
            evaluations += 1
            if not measures:
                continue
            score, _ = distance(measures, profile, weights, keys=SEARCH_KEYS)
            if score < best_score - 1e-6:
                steps.append({
                    "field": field,
                    "before": getattr(best, field),
                    "after": value,
                    "score": round(score, 4),
                    "was": round(best_score, 4),
                })
                best, best_score = candidate, score
                improved_this_pass = True
        if not improved_this_pass or best_score <= 0.0:
            break
    return SearchResult(
        params=best, score=round(best_score, 4), start_score=round(start_score, 4),
        steps=steps, evaluations=evaluations, refused=refused,
    )


def _accepted(supervisor: Any, params) -> tuple[bool, str]:
    """Would the supervisor allow these parameters? Never trust a search."""
    try:
        validated, reason = supervisor.validate_transition_params(params.to_schema())
    except Exception as exc:  # noqa: BLE001 - a refusal is safer than a crash
        return False, f"{type(exc).__name__}: {exc}"
    return (validated is not None), (reason or "")
