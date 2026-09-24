"""Track understanding (SPEC §2): what a track is, beyond BPM and key.

THREADING CONTEXT: offline and control thread only. Pure functions over a
:class:`~djai.analysis.TrackAnalysis`; no audio is read here.

Everything below is **derived from fields every sidecar already stores** --
the beat grid, per-beat RMS, sections, vocal bars, intensity features, grid
confidence -- which is what lets a v8 sidecar be brought to v9 in place, with
no audio decoded, and come out identical to a fresh analysis of the same
measurements. Two uncertainties need the audio (key and vocals); they are
measured by ``analyze`` and are ``None`` on an upgraded entry until it runs.

Every number here is a **heuristic**, and is named as one. Section-label
confidence is how far inside its labelling rule a section sits and how sharp
its boundaries are; mix-point quality is a weighted sum of inspectable
components; the embedding is a hand-built feature vector, not a trained model.
None has been fitted to human judgements, because none exist yet -- see
:func:`structure_agreement` and ``djai annotate`` for how that gets measured.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

BEATS_PER_BAR: int = 4
#: Mirrors djai.phrase.BARS_PER_PHRASE; not imported (phrase imports analysis).
BARS_PER_PHRASE: int = 32

SECTION_LABELS: tuple[str, ...] = ("intro", "build", "drop", "breakdown", "outro")

# --- section confidence ----------------------------------------------------------

#: The labelling rule's own thresholds (djai.analysis.detect_sections).
_DROP_LEVEL_DB: float = -3.0
_BUILD_SLOPE_DB_PER_BAR: float = 0.2
#: A boundary with this much level change across it (4 bars either side) is
#: as sharp as a boundary gets.
_SHARP_BOUNDARY_DB: float = 6.0

# --- mix regions -----------------------------------------------------------------

#: Same fraction of median per-beat RMS that djai.analysis._mix_points uses.
_ENERGY_FLOOR: float = 0.40
#: How many bars a blend runs over, for the runway, room and vocal windows.
_BLEND_BARS: int = 16
#: How many regions of each kind are kept, best first.
MAX_REGIONS: int = 3
#: ``coverage`` is how much of the track the point lets the room hear: what is
#: left after a mix-in, what has played before a mix-out. Without it a later
#: drop on a 32-bar line outranks the first one and half the track is skipped.
MIX_IN_WEIGHTS: dict[str, float] = {
    "sustain": 0.25, "runway": 0.10, "phrase": 0.15,
    "vocal_clear": 0.15, "section": 0.10, "grid": 0.10, "coverage": 0.15,
}
#: A transition *finishes* on the mix-out point (it is placement's anchor), so
#: the blend runs over the bars before it: that is the window ``steady`` and
#: ``vocal_clear`` read. A quiet outro after the point is where it belongs.
MIX_OUT_WEIGHTS: dict[str, float] = {
    "steady": 0.25, "phrase": 0.15, "vocal_clear": 0.15,
    "section": 0.15, "grid": 0.10, "coverage": 0.20,
}
#: Phrase-line score: a 16-bar line is as good as it gets (no extra credit for
#: 32, which on the real library pulled mix-ins a whole phrase later and cost
#: the room half the track), an 8-bar line is fine, anything else is off-grid.
_PHRASE_SCORE: tuple[tuple[int, float], ...] = ((16, 1.0), (8, 0.7))

# --- embedding -------------------------------------------------------------------

EMBEDDING_METHOD: str = (
    "hand-v1: 16 hand-chosen features, each scaled to about 0..1 -- log tempo, "
    "Camelot position on the circle (cos, sin), mode, intensity and its three "
    "components, vocal share, mean and spread of bar loudness, loudness shape by "
    "quarter (4), drop share. Not learned. Similarity = 1 - euclidean / 4."
)
EMBEDDING_DIM: int = 16

#: Mirrors djai.analysis intensity normalisation ranges.
_PERCUSSIVE_FULL: float = 0.5
_FLUX_RANGE: tuple[float, float] = (1.5, 5.0)
_DENSITY_FULL: float = 8.0


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# --- energy per bar, phrase and track --------------------------------------------


def _bar_beat_index(ta) -> tuple[np.ndarray, int]:
    """Beat index where each bar starts, counted from the first downbeat."""
    beats = np.asarray(ta.beats, dtype=np.float64)
    if beats.size == 0:
        return np.zeros(0, dtype=np.int64), 0
    first = int(np.searchsorted(beats, ta.first_downbeat - 1e-6))
    n_bars = max(0, (beats.size - first) // BEATS_PER_BAR)
    return first + BEATS_PER_BAR * np.arange(n_bars, dtype=np.int64), first


def bar_energy(ta) -> np.ndarray:
    """Mean per-beat RMS of each bar, in dB. Bar 0 is the first downbeat."""
    starts, _ = _bar_beat_index(ta)
    rms = np.asarray(ta.beat_rms, dtype=np.float64)
    if starts.size == 0 or rms.size == 0:
        return np.zeros(0)
    out = np.empty(starts.size)
    for i, s in enumerate(starts):
        seg = rms[s:s + BEATS_PER_BAR]
        out[i] = 20.0 * math.log10(float(seg.mean()) + 1e-9) if seg.size else -180.0
    return out


def phrase_energy(ta, bars: int = BARS_PER_PHRASE) -> np.ndarray:
    """Mean bar energy of each ``bars``-long phrase, in dB (last may be short)."""
    per_bar = bar_energy(ta)
    if per_bar.size == 0:
        return per_bar
    return np.array([per_bar[i:i + bars].mean() for i in range(0, per_bar.size, bars)])


def track_energy_db(ta) -> float:
    """Whole-track energy: the mean bar energy, in dB."""
    per_bar = bar_energy(ta)
    return float(per_bar.mean()) if per_bar.size else -180.0


def energy_uncertainty(ta) -> float | None:
    """How loosely a bar's energy is pinned down: mean within-bar CV of beat RMS."""
    starts, _ = _bar_beat_index(ta)
    rms = np.asarray(ta.beat_rms, dtype=np.float64)
    cvs = []
    for s in starts:
        seg = rms[s:s + BEATS_PER_BAR]
        if seg.size == BEATS_PER_BAR and seg.mean() > 0:
            cvs.append(float(seg.std() / seg.mean()))
    return round(_clip01(float(np.mean(cvs))), 4) if cvs else None


# --- section confidence ------------------------------------------------------------


def section_confidences(ta) -> list[float]:
    """0..1 per section: rule margin (half) and boundary sharpness (half).

    Heuristic. A drop 0.5 dB inside the drop threshold, or a boundary with
    barely any level change across it, is a label the rule only just made.
    """
    sections = ta.sections or []
    per_bar = bar_energy(ta)
    if not sections or per_bar.size == 0:
        return [0.0 for _ in sections]

    def mean_db(a: int, b: int) -> float:
        seg = per_bar[max(0, a):max(0, min(b, per_bar.size))]
        return float(seg.mean()) if seg.size else float(per_bar.mean())

    levels = [mean_db(s["start_bar"], s["end_bar"]) for s in sections]
    top = max(levels)
    median = float(np.median(per_bar))
    out = []
    for k, s in enumerate(sections):
        a, b = int(s["start_bar"]), int(s["end_bar"])
        rel = levels[k] - top
        label = s.get("label")
        if label == "drop":
            margin = (rel - _DROP_LEVEL_DB) / abs(_DROP_LEVEL_DB)
        elif label == "breakdown":
            margin = (_DROP_LEVEL_DB - rel) / abs(_DROP_LEVEL_DB)
        elif label == "build":
            seg = per_bar[max(0, a):min(b, per_bar.size)]
            slope = float(np.polyfit(np.arange(seg.size), seg, 1)[0]) if seg.size > 1 else 0.0
            margin = (slope - _BUILD_SLOPE_DB_PER_BAR) / _BUILD_SLOPE_DB_PER_BAR
        else:  # intro / outro: placed by position; confident when also quieter
            margin = (median - levels[k]) / 3.0 + 0.5
        sharp = []
        for edge in (a, b):
            if 0 < edge < per_bar.size:
                jump = abs(mean_db(edge, edge + 4) - mean_db(edge - 4, edge))
                sharp.append(_clip01(jump / _SHARP_BOUNDARY_DB))
        boundary = float(np.mean(sharp)) if sharp else 1.0
        out.append(round(0.5 * _clip01(margin) + 0.5 * boundary, 3))
    return out


# --- ranked mix regions -------------------------------------------------------------


def _vocal_overlap(ta, a: float, b: float) -> float:
    if b <= a:
        return 0.0
    hit = sum(max(0.0, min(b, float(hi)) - max(a, float(lo))) for lo, hi in ta.vocal_bars or [])
    return hit / (b - a)


def _near_boundary(ta, bar: int) -> float:
    edges = {int(s["start_bar"]) for s in ta.sections or []}
    if not edges:
        return 0.5  # unmeasured: neutral, neither helps nor hurts
    return 1.0 if any(abs(bar - e) <= 1 for e in edges) else 0.0


def _share_above(ta, first_beat: int, a_bar: int, b_bar: int, floor: float) -> float:
    rms = np.asarray(ta.beat_rms, dtype=np.float64)
    lo = first_beat + BEATS_PER_BAR * max(0, a_bar)
    hi = first_beat + BEATS_PER_BAR * max(0, b_bar)
    seg = rms[lo:min(hi, rms.size)]
    return float(np.mean(seg >= floor)) if seg.size else 0.0


def mix_regions(ta, kind: str) -> list[dict]:
    """The best places to mix in to (``"in"``) or out of (``"out"``) this track.

    Candidates are the 8-bar phrase lines, plus the track's current mix point
    so the anchor placement already uses is always scored too. Each carries
    its component scores, so a ranking can be inspected rather than trusted.
    """
    starts, first = _bar_beat_index(ta)
    n_bars = int(starts.size)
    rms = np.asarray(ta.beat_rms, dtype=np.float64)
    if n_bars < 8 or rms.size == 0:
        return []
    floor = _ENERGY_FLOOR * float(np.median(rms))
    grid = 1.0 if ta.grid_manually_corrected else _clip01(ta.grid_confidence)
    anchor = int(round(ta.mix_in_bar if kind == "in" else ta.mix_out_bar))
    if kind == "in":
        lo, hi, weights = 0, int(0.6 * n_bars), MIX_IN_WEIGHTS
    else:
        lo, hi, weights = int(0.4 * n_bars), n_bars - 1, MIX_OUT_WEIGHTS
    cands = sorted({b for b in range(lo, hi + 1) if b % 8 == 0} | ({anchor} if lo <= anchor <= hi else set()))

    out = []
    for bar in cands:
        phrase = next((v for step, v in _PHRASE_SCORE if bar % step == 0), 0.3)
        if kind == "in":
            comp = {
                "sustain": _share_above(ta, first, bar, bar + 8, floor),
                "runway": _clip01(bar / _BLEND_BARS),
                "phrase": phrase,
                "vocal_clear": 1.0 - _vocal_overlap(ta, bar - _BLEND_BARS, bar),
                "section": _near_boundary(ta, bar),
                "grid": grid,
                "coverage": _clip01(1.0 - bar / n_bars),
            }
        else:
            comp = {
                "steady": _share_above(ta, first, bar - _BLEND_BARS, bar, floor),
                "phrase": phrase,
                "vocal_clear": 1.0 - _vocal_overlap(ta, bar - _BLEND_BARS, bar),
                "section": _near_boundary(ta, bar),
                "grid": grid,
                "coverage": _clip01(bar / n_bars),
            }
        quality = sum(weights[k] * v for k, v in comp.items())
        seconds = float(ta.beats[starts[bar]]) if starts[bar] < len(ta.beats) else 0.0
        out.append({
            "bar": int(bar),
            "seconds": round(seconds, 4),
            "quality": round(quality, 4),
            "components": {k: round(v, 4) for k, v in comp.items()},
        })
    out.sort(key=lambda r: (-r["quality"], r["bar"] if kind == "in" else -r["bar"]))
    return out[:MAX_REGIONS]


# --- embedding ------------------------------------------------------------------------


def _camelot(code: str) -> tuple[int, str] | None:
    try:
        return int(code[:-1]), code[-1].upper()
    except (ValueError, IndexError, TypeError):
        return None


def embedding(ta) -> list[float]:
    """The :data:`EMBEDDING_METHOD` vector. Deterministic; about 0..1 per dim."""
    v = np.zeros(EMBEDDING_DIM)
    bpm = float(ta.bpm) if ta.bpm and ta.bpm > 0 else 120.0
    v[0] = _clip01(math.log2(bpm / 82.0))
    cam = _camelot(ta.camelot)
    if cam is not None:
        angle = 2 * math.pi * (cam[0] % 12) / 12.0
        v[1], v[2] = 0.5 + 0.5 * math.cos(angle), 0.5 + 0.5 * math.sin(angle)
        v[3] = 1.0 if cam[1] == "B" else 0.0
    else:
        v[1] = v[2] = 0.5
        v[3] = 0.5
    intensity = ta.intensity
    v[4] = intensity if intensity is not None and np.isfinite(intensity) else 0.5
    feats = ta.intensity_features or {}
    v[5] = _clip01(feats.get("percussive", 0.0) / _PERCUSSIVE_FULL)
    v[6] = _clip01((feats.get("flux", _FLUX_RANGE[0]) - _FLUX_RANGE[0]) / (_FLUX_RANGE[1] - _FLUX_RANGE[0]))
    v[7] = _clip01(feats.get("density", 0.0) / _DENSITY_FULL)
    v[8] = _clip01(ta.vocal_fraction or 0.0)
    per_bar = bar_energy(ta)
    if per_bar.size:
        mean = float(per_bar.mean())
        v[9] = _clip01((mean + 40.0) / 40.0)
        v[10] = _clip01(float(per_bar.std()) / 10.0)
        for q, chunk in enumerate(np.array_split(per_bar, 4)):
            v[11 + q] = _clip01(0.5 + (float(chunk.mean()) - mean) / 20.0) if chunk.size else 0.5
    n = max(1, per_bar.size)
    drop = sum(int(s["end_bar"]) - int(s["start_bar"]) for s in ta.sections or [] if s.get("label") == "drop")
    v[15] = _clip01(drop / n)
    return [round(float(x), 4) for x in v]


def embedding_similarity(a: list[float], b: list[float]) -> float:
    """1 for identical vectors, 0 at a distance of 4 (a quarter of the cube's diagonal)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return _clip01(1.0 - float(np.linalg.norm(np.subtract(a, b))) / 4.0)


# --- uncertainty and the whole derivation -----------------------------------------------


def _intensity_uncertainty(ta) -> float | None:
    feats = ta.intensity_features or {}
    if not feats:
        return None
    parts = [
        _clip01(feats.get("percussive", 0.0) / _PERCUSSIVE_FULL),
        _clip01((feats.get("flux", 0.0) - _FLUX_RANGE[0]) / (_FLUX_RANGE[1] - _FLUX_RANGE[0])),
        _clip01(feats.get("density", 0.0) / _DENSITY_FULL),
    ]
    # Components that disagree make the composite a compromise.
    return round(_clip01(2.0 * float(np.std(parts))), 4)


def derive(ta) -> None:
    """Fill every v9 field on ``ta`` from what it already holds. In place.

    Key and vocal uncertainty come from the audio and are kept if ``analyze``
    already measured them; everything else is recomputed, so running this
    twice gives the same answer.
    """
    confs = section_confidences(ta)
    for s, c in zip(ta.sections or [], confs):
        s["confidence"] = 1.0 if ta.structure_manually_corrected else c
    ta.mix_in_regions = mix_regions(ta, "in")
    ta.mix_out_regions = mix_regions(ta, "out")
    ta.embedding = embedding(ta)

    measured = dict(ta.uncertainty or {})
    unc: dict[str, float | None] = {}
    grid = 0.0 if ta.grid_manually_corrected else round(1.0 - _clip01(ta.grid_confidence), 4)
    unc["grid"] = grid
    unc["tempo"] = round(max(grid, 0.5 if ta.tempo_ambiguous else 0.0), 4)
    unc["key"] = measured.get("key")
    if ta.sections:
        lengths = [int(s["end_bar"]) - int(s["start_bar"]) for s in ta.sections]
        weighted = sum(s["confidence"] * n for s, n in zip(ta.sections, lengths)) / max(1, sum(lengths))
        unc["sections"] = round(1.0 - weighted, 4)
    else:
        unc["sections"] = None
    unc["vocals"] = 0.0 if ta.structure_manually_corrected else measured.get("vocals")
    unc["intensity"] = _intensity_uncertainty(ta)
    unc["energy"] = energy_uncertainty(ta)
    unc["mix_in"] = round(1.0 - ta.mix_in_regions[0]["quality"], 4) if ta.mix_in_regions else None
    unc["mix_out"] = round(1.0 - ta.mix_out_regions[0]["quality"], 4) if ta.mix_out_regions else None
    inputs = [unc[k] for k in ("tempo", "key", "sections", "vocals", "intensity", "energy")]
    known = [x for x in inputs if x is not None]
    unc["embedding"] = round(float(np.mean(known)), 4) if known else None
    ta.uncertainty = unc


# --- operator structure: corrections and agreement ----------------------------------------


def validate_structure(
    sections: list[dict] | None, vocal_bars: list[list[int]] | None
) -> str | None:
    """Why a hand-entered structure is unusable, or None if it is fine."""
    for s in sections or []:
        if not isinstance(s, dict) or s.get("label") not in SECTION_LABELS:
            return f"section {s!r}: label must be one of {', '.join(SECTION_LABELS)}"
        try:
            a, b = int(s["start_bar"]), int(s["end_bar"])
        except (KeyError, TypeError, ValueError):
            return f"section {s!r}: needs integer start_bar and end_bar"
        if a < 0 or b <= a:
            return f"section {s!r}: end_bar must be after start_bar, both >= 0"
    starts = [int(s["start_bar"]) for s in sections or []]
    if starts != sorted(starts):
        return "sections must be in order"
    for r in vocal_bars or []:
        if not (isinstance(r, (list, tuple)) and len(r) == 2 and 0 <= int(r[0]) < int(r[1])):
            return f"vocal range {r!r}: must be [start_bar, end_bar] with end after start"
    return None


def correct_structure(
    ta, sections: list[dict] | None = None, vocal_bars: list[list[int]] | None = None
) -> None:
    """Record a person's structure for this track. Authoritative from here on.

    Sets ``structure_manually_corrected``; re-analysis carries the corrected
    fields forward and never re-measures over them.
    """
    reason = validate_structure(sections, vocal_bars)
    if reason is not None:
        raise ValueError(reason)
    if sections is not None:
        ta.sections = [
            {"label": s["label"], "start_bar": int(s["start_bar"]), "end_bar": int(s["end_bar"])}
            for s in sections
        ]
    if vocal_bars is not None:
        ta.vocal_bars = [[int(a), int(b)] for a, b in vocal_bars]
        n_bars = max(1, bar_energy(ta).size)
        ta.vocal_fraction = round(sum(b - a for a, b in ta.vocal_bars) / n_bars, 4)
    ta.structure_manually_corrected = True
    derive(ta)


def _labels_per_bar(sections: list[dict], n: int) -> list[str | None]:
    out: list[str | None] = [None] * n
    for s in sections or []:
        for i in range(max(0, int(s["start_bar"])), min(n, int(s["end_bar"]))):
            out[i] = s["label"]
    return out


def _vocal_flags(ranges: list[list[int]], n: int) -> np.ndarray:
    f = np.zeros(n, dtype=bool)
    for a, b in ranges or []:
        f[max(0, int(a)):min(n, int(b))] = True
    return f


def structure_agreement(ta, annotation: dict, tolerance_bars: int = 2) -> dict[str, Any]:
    """How well the measured structure agrees with a person's annotation.

    Reported, not tuned: bar-label accuracy over the bars both cover, the
    share of annotated boundaries found within ``tolerance_bars``, the share
    of measured boundaries that are real, and vocal precision and recall per
    bar. Every disagreement is listed, not just counted.
    """
    truth = annotation.get("sections") or []
    n = max(
        [int(s["end_bar"]) for s in truth + list(ta.sections or [])] + [bar_energy(ta).size, 1]
    )
    ours, theirs = _labels_per_bar(ta.sections or [], n), _labels_per_bar(truth, n)
    both = [i for i in range(n) if ours[i] is not None and theirs[i] is not None]
    agree = [i for i in both if ours[i] == theirs[i]]
    wrong = [(i, ours[i], theirs[i]) for i in both if ours[i] != theirs[i]]

    def edges(secs):
        return sorted({int(s["start_bar"]) for s in secs} - {0})

    t_edges, o_edges = edges(truth), edges(ta.sections or [])
    found = [e for e in t_edges if any(abs(e - o) <= tolerance_bars for o in o_edges)]
    real = [o for o in o_edges if any(abs(e - o) <= tolerance_bars for e in t_edges)]

    report: dict[str, Any] = {
        "track": ta.title,
        "bars_compared": len(both),
        "label_accuracy": round(len(agree) / len(both), 4) if both else None,
        "boundary_recall": round(len(found) / len(t_edges), 4) if t_edges else None,
        "boundary_precision": round(len(real) / len(o_edges), 4) if o_edges else None,
        "missed_boundaries": [e for e in t_edges if e not in found],
        "false_boundaries": [o for o in o_edges if o not in real],
        "label_disagreements": _runs_of(wrong),
    }
    if "vocal_bars" in annotation:
        t_v = _vocal_flags(annotation["vocal_bars"], n)
        o_v = _vocal_flags(ta.vocal_bars, n)
        tp = int(np.sum(t_v & o_v))
        report["vocal_precision"] = round(tp / int(o_v.sum()), 4) if o_v.any() else None
        report["vocal_recall"] = round(tp / int(t_v.sum()), 4) if t_v.any() else None
    return report


def _runs_of(wrong: list[tuple[int, str | None, str | None]]) -> list[dict]:
    """Consecutive disagreeing bars, merged: ``{start_bar, end_bar, ours, theirs}``."""
    out: list[dict] = []
    for bar, o, t in wrong:
        if out and out[-1]["end_bar"] == bar and out[-1]["ours"] == o and out[-1]["theirs"] == t:
            out[-1]["end_bar"] = bar + 1
        else:
            out.append({"start_bar": bar, "end_bar": bar + 1, "ours": o, "theirs": t})
    return out
