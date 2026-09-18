"""Cue the planned transition before it plays, and measure what came out.

THREADING CONTEXT: a **worker thread**, never the audio thread and never the
scheduler's tick loop. Nothing here is real-time: it allocates freely, runs FFTs
and takes as long as it takes -- which is exactly why it must never be reached
from either of those two places. The transition fires on time whatever happens
in this module, including an exception; see :mod:`djai.revise` for the loop that
enforces that.

This is the headphone cue, done numerically. A DJ listens to the next mix before
letting it out of the booth; the model cannot listen, so the transition is
rendered offline through the real signal path and reduced to numbers that say
the same things an ear would:

* is it loud enough, and does it sag in the middle
* are both basslines playing at once
* are the two midranges fighting
* are two vocals talking over each other
* have the transients smeared into mush
* do the two grids actually agree

Those numbers are what goes to the model. **No audio, in any form -- samples,
spectrograms, encoded waveforms -- ever leaves this module toward the LLM.**
llama3.1:8b is text-only; the measurements are the text.

What this costs, measured on this crate: the block loop runs a 24-bar window in
~1.3 s at the live blocksize, the measurements add ~0.3 s. Decoding (~400 ms)
and time-stretching (~110 s for a whole track) are NOT paid here -- the live
decks have already done both by the time a transition is in its pre-roll, and
this module takes the loaded tracks rather than the analyses.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import bilinear, lfilter

from djai import config, render as render_mod
from djai.analysis import TrackAnalysis
from djai.deck import SAMPLE_RATE, LoadedTrack

log = logging.getLogger(__name__)

BEATS_PER_BAR = 4

#: Band edges, in Hz. "Sub" is the bass a swap is supposed to hand over; "mid"
#: is where two tracks turn to mud when both are present.
SUB_HZ = 200.0
MID_HZ = (200.0, 2000.0)

#: BS.1770's momentary block and its hop. Used for the gated integrated
#: loudness, where the standard fixes both.
MOMENTARY_S = 0.4
MOMENTARY_HOP_S = 0.1

#: BS.1770's SHORT-TERM window, which is what the dip is measured over. Three
#: seconds rather than the momentary 400 ms on purpose: a crossfade that sags is
#: a sustained fault lasting a bar or more, and at 400 ms the maximum drop is
#: set by whichever single block happened to fall between two kicks.
SHORT_TERM_S = 3.0
SHORT_TERM_HOP_S = 0.5

#: A deck counts as audible for vocal overlap above this fader x mid-band gain
#: (about -12 dB). Below it, a singer on that deck is not competing.
VOCAL_AUDIBLE = 0.25

#: Absolute and relative gates from BS.1770, in LUFS and LU.
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = 10.0


@dataclass
class DeckAState:
    """What the preview needs to know about the deck that is playing.

    Deliberately a plain snapshot rather than a reference to the live deck: the
    preview runs on a worker thread while that deck keeps playing, and reading a
    moving playhead halfway through a measurement would make the result depend
    on when the thread happened to be scheduled.
    """

    analysis: TrackAnalysis
    loaded: LoadedTrack
    #: Frame of track A where the transition begins. Computed by
    #: :func:`djai.phrase.plan_transition`, never by the model.
    transition_at_frame: float


@dataclass
class PreviewError(Exception):
    """A preview that could not be taken. Always means: commit what you have."""

    stage: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.stage}: {self.detail}"


# --- loudness, BS.1770 ----------------------------------------------------------


def _k_weighting(sr: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """The two BS.1770 pre-filter stages, as (b, a) pairs.

    Stage 1 is the head-shadow high shelf, stage 2 the RLB high-pass. Built from
    the specification's analogue prototypes rather than taken from another
    library's internals, so the numbers here are stable across versions of
    everything else. Cross-checked against pyloudnorm in the test suite.
    """
    # High shelf: +4 dB above ~1.5 kHz.
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = math.pow(10.0, gain_db / 20.0)
    vb = math.pow(vh, 0.4996667741545416)
    denom = 1.0 + k / q + k * k
    shelf_b = np.array([
        (vh + vb * k / q + k * k) / denom,
        2.0 * (k * k - vh) / denom,
        (vh - vb * k / q + k * k) / denom,
    ])
    shelf_a = np.array([
        1.0,
        2.0 * (k * k - 1.0) / denom,
        (1.0 - k / q + k * k) / denom,
    ])

    # RLB high-pass at ~38 Hz.
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / sr)
    hp_b = np.array([1.0, -2.0, 1.0])
    hp_a = np.array([
        1.0,
        2.0 * (k * k - 1.0) / (1.0 + k / q + k * k),
        (1.0 - k / q + k * k) / (1.0 + k / q + k * k),
    ])
    return (shelf_b, shelf_a), (hp_b, hp_a)


def _k_weighted(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply K-weighting to (frames, channels) float audio."""
    out = np.asarray(audio, dtype=np.float64)
    if out.ndim == 1:
        out = out[:, None]
    for b, a in _k_weighting(sr):
        out = lfilter(b, a, out, axis=0)
    return out


def _mean_squares(weighted: np.ndarray, sr: int, block_s: float, hop_s: float):
    """Per-block channel-summed mean square of K-weighted audio.

    Returns ``(mean_squares, block_starts)``. Channel weights are 1.0 for left
    and right, which is all this engine ever produces.
    """
    n_block = max(1, int(round(block_s * sr)))
    n_hop = max(1, int(round(hop_s * sr)))
    frames = weighted.shape[0]
    if frames < n_block:
        return np.empty(0), np.empty(0, dtype=int)
    starts = np.arange(0, frames - n_block + 1, n_hop)
    sq = np.square(weighted)
    # Cumulative sums turn "mean square of every overlapping block" into two
    # index operations instead of a Python loop over thousands of blocks.
    csum = np.concatenate([np.zeros((1, sq.shape[1])), np.cumsum(sq, axis=0)])
    sums = csum[starts + n_block] - csum[starts]
    return (sums.sum(axis=1) / n_block), starts


def _to_lkfs(mean_squares: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return -0.691 + 10.0 * np.log10(np.maximum(mean_squares, 1e-20))


def block_loudness(audio: np.ndarray, sr: int, block_s: float, hop_s: float):
    """Loudness of each overlapping block, in LKFS, with its start frame."""
    ms, starts = _mean_squares(_k_weighted(audio, sr), sr, block_s, hop_s)
    if ms.size == 0:
        return np.empty(0), starts
    return _to_lkfs(ms), starts


def integrated_from_mean_squares(ms: np.ndarray) -> float:
    """BS.1770 integrated loudness from momentary mean squares already taken.

    Split out so a caller that needs both the integrated number and the
    short-term curve of the same audio pays for the K-weighting filter once.
    """
    if ms.size == 0:
        return float("-inf")
    with np.errstate(divide="ignore"):
        loud = -0.691 + 10.0 * np.log10(np.maximum(ms, 1e-20))
    keep = loud > ABSOLUTE_GATE_LUFS
    if not keep.any():
        return float("-inf")
    relative = -0.691 + 10.0 * math.log10(max(float(ms[keep].mean()), 1e-20))
    keep &= loud > (relative - RELATIVE_GATE_LU)
    if not keep.any():
        return float("-inf")
    return float(-0.691 + 10.0 * math.log10(max(float(ms[keep].mean()), 1e-20)))


def integrated_lufs(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """BS.1770 integrated loudness, with both gates. -inf for silence."""
    ms, _ = _mean_squares(
        _k_weighted(audio, sr), sr, MOMENTARY_S, MOMENTARY_HOP_S
    )
    return integrated_from_mean_squares(ms)


def peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return 20.0 * math.log10(peak) if peak > 1e-9 else -120.0


# --- spectra --------------------------------------------------------------------


def _frame_matrix(mono: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
    """``(len(starts), n)`` of overlapping slices, without a Python loop.

    A preview is on a clock. Framing a 45-second window one slice at a time in
    Python costs seconds; one fancy-indexed gather costs milliseconds.
    """
    return mono[starts[:, None] + np.arange(n)[None, :]]


def _deck_spectra(
    loaded: LoadedTrack, frames: np.ndarray, block: int, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Power spectrum of one block of a deck's own audio at each playhead.

    ``frames`` is where the deck's playhead was at the end of each rendered
    block, straight out of the render envelope, so this measures what the deck
    was actually playing rather than where it was asked to be.
    """
    audio = loaded.audio
    total = audio.shape[0]
    out = np.zeros((frames.size, block // 2 + 1), dtype=np.float64)
    starts = frames.astype(np.int64) - block
    ok = (starts >= 0) & (starts + block <= total)
    if not ok.any():
        return out
    lo = int(starts[ok].min())
    hi = int(starts[ok].max()) + block
    seg = audio[lo:hi]
    mono = np.asarray(seg.mean(axis=1) if seg.ndim > 1 else seg, dtype=np.float64)
    window = np.hanning(block)
    matrix = _frame_matrix(mono, starts[ok] - lo, block) * window
    out[ok] = np.square(np.abs(np.fft.rfft(matrix, axis=1)))
    return out


def _band_slices(block: int, sr: int = SAMPLE_RATE):
    freqs = np.fft.rfftfreq(block, 1.0 / sr)
    sub = freqs < SUB_HZ
    mid = (freqs >= MID_HZ[0]) & (freqs <= MID_HZ[1])
    return sub, mid


# --- transients -----------------------------------------------------------------


def _onset_count(mono: np.ndarray, sr: int = SAMPLE_RATE) -> int:
    """Transients, by spectral flux with a simple adaptive threshold.

    The same function counts the mix and each source, so the ratio between them
    is meaningful even though the absolute count is only as good as the peak
    picking.
    """
    n, hop = 1024, 512
    mono = np.asarray(mono, dtype=np.float64)
    if mono.size < n * 4:
        return 0
    window = np.hanning(n)
    frames = 1 + (mono.size - n) // hop
    starts = np.arange(frames) * hop
    mags = np.abs(np.fft.rfft(_frame_matrix(mono, starts, n) * window, axis=1))
    flux = np.diff(mags, axis=0)
    flux = np.maximum(flux, 0.0).sum(axis=1)
    if flux.size < 5 or not np.isfinite(flux).all():
        return 0
    # Adaptive threshold: the local median plus a share of the local spread.
    thresh = np.median(flux) + 0.6 * (np.percentile(flux, 90) - np.median(flux))
    peaks = (flux[1:-1] > thresh) & (flux[1:-1] >= flux[:-2]) & (flux[1:-1] > flux[2:])
    return int(np.count_nonzero(peaks))


# --- the measurement ------------------------------------------------------------


def measure(
    result: "render_mod.PreviewRender",
    state: DeckAState,
    track_b: TrackAnalysis,
    loaded_b: LoadedTrack,
    params,
) -> dict[str, Any]:
    """Reduce a rendered transition to the numbers a revision can act on.

    Every value is a plain float or int: this dict is what gets written to the
    log, shown in the UI, and -- in compact text form -- handed to the model.
    """
    block = result.blocksize
    env = result.envelope
    t0 = result.transition_start_frame
    t1 = t0 + result.transition_frames
    mix = result.mix
    window = mix[t0:t1] if t1 <= mix.shape[0] else mix[t0:]

    out: dict[str, Any] = {
        "bars": float(result.bars),
        "blocks": len(env),
        "limiter_engaged": bool(result.limiter_engaged),
    }

    # --- level ---
    # K-weight once and take both numbers off it: the filter is the expensive
    # part, and integrated loudness and the short-term curve differ only in how
    # the filtered signal is blocked up afterwards.
    out["peak_dbfs"] = round(peak_dbfs(window), 2)
    mix_k = _k_weighted(window, SAMPLE_RATE)
    mix_momentary, _ = _mean_squares(
        mix_k, SAMPLE_RATE, MOMENTARY_S, MOMENTARY_HOP_S
    )
    lufs = integrated_from_mean_squares(mix_momentary)
    out["integrated_lufs"] = round(lufs, 2) if math.isfinite(lufs) else None

    # --- the dip ---
    # Compared instant by instant, not against a single number for the whole
    # track. A crossfade that sags is quieter than the material going into it AT
    # THAT MOMENT; measuring the mix's quietest moment against the sources'
    # average loudness overall would score every breakdown as a fault. Both
    # sources are read at the position the deck was actually playing, and the
    # two are averaged in the power domain -- which is what two signals at those
    # levels average to, rather than an artefact of averaging decibels.
    a_frames = np.array([r["a_frame"] for r in env], dtype=np.float64)
    b_frames = np.array([r["b_frame"] for r in env], dtype=np.float64)
    during_mask = np.array([bool(r["in_transition"]) for r in env])
    src_lufs: list[float] = []
    src_curves: list[np.ndarray] = []
    for loaded, frames in ((state.loaded, a_frames), (loaded_b, b_frames)):
        span = _source_span(loaded, frames[during_mask])
        if span is None:
            continue
        span_k = _k_weighted(span, SAMPLE_RATE)
        momentary, _ = _mean_squares(
            span_k, SAMPLE_RATE, MOMENTARY_S, MOMENTARY_HOP_S
        )
        value = integrated_from_mean_squares(momentary)
        if math.isfinite(value):
            src_lufs.append(value)
        short, _ = _mean_squares(
            span_k, SAMPLE_RATE, SHORT_TERM_S, SHORT_TERM_HOP_S
        )
        if short.size:
            src_curves.append(short)

    mix_short, _ = _mean_squares(
        mix_k, SAMPLE_RATE, SHORT_TERM_S, SHORT_TERM_HOP_S
    )
    out["source_lufs"] = [round(v, 2) for v in src_lufs]
    out["loudness_dip_db"] = _loudness_dip(mix_short, src_curves)

    # --- per-deck bands, as each deck's own audio through the applied gains ---
    during = [i for i, r in enumerate(env) if r["in_transition"]]
    if not during:
        out.update(
            low_end_overlap_bars=0.0, spectral_clash=0.0,
            transient_density_ratio=1.0, phase_coherence=1.0,
            vocal_overlap_bars=0.0,
        )
        return out

    idx = np.array(during, dtype=int)
    spec_a = _deck_spectra(state.loaded, a_frames[idx], block)
    spec_b = _deck_spectra(loaded_b, b_frames[idx], block)
    sub, mid = _band_slices(block)

    g_a = np.array([env[i]["a_gain"] for i in idx])
    g_b = np.array([env[i]["b_gain"] for i in idx])
    low_a = np.array([env[i]["a_low"] for i in idx])
    low_b = np.array([env[i]["b_low"] for i in idx])
    mid_a = np.array([env[i]["a_mid"] for i in idx])
    mid_b = np.array([env[i]["b_mid"] for i in idx])

    # Power scales with the square of a gain.
    sub_a = spec_a[:, sub].sum(axis=1) * np.square(g_a * low_a)
    sub_b = spec_b[:, sub].sum(axis=1) * np.square(g_b * low_b)
    mid_spec_a = spec_a[:, mid] * np.square(g_a * mid_a)[:, None]
    mid_spec_b = spec_b[:, mid] * np.square(g_b * mid_b)[:, None]

    bars_of = np.array([env[i]["bar"] for i in idx])
    out["low_end_overlap_bars"] = _low_overlap_bars(sub_a, sub_b, bars_of)
    out["spectral_clash"] = _spectral_clash(mid_spec_a, mid_spec_b)

    # --- transients: has the blend smeared them together? ---
    mono_mix = window.mean(axis=1) if window.ndim > 1 else window
    mix_onsets = _onset_count(np.asarray(mono_mix, dtype=np.float64))
    src_onsets = []
    for loaded, frames in ((state.loaded, a_frames[idx]), (loaded_b, b_frames[idx])):
        span = _source_span(loaded, frames)
        if span is not None:
            src_onsets.append(_onset_count(span.mean(axis=1)))
    # Beatmatched decks put their transients on the same grid, so the mix should
    # show at least as many as the busier deck. Materially fewer means they have
    # been smeared into each other.
    expected = max(src_onsets) if src_onsets else 0
    out["transient_density_ratio"] = (
        round(mix_onsets / expected, 3) if expected else 1.0
    )
    out["transients"] = {"mix": mix_onsets, "sources": src_onsets}

    # --- do the two grids agree? ---
    out["phase_coherence"] = _phase_coherence(result)

    # --- vocals, from the cached ranges: no audio needed ---
    out["vocal_overlap_bars"] = _vocal_overlap_bars(
        state, track_b, result, params
    )
    return out


def _source_span(loaded: LoadedTrack, frames: np.ndarray) -> np.ndarray | None:
    """The stretch of a deck's own audio that the render consumed."""
    valid = frames[frames > 0]
    if valid.size < 2:
        return None
    lo = int(max(0, valid.min()))
    hi = int(min(loaded.audio.shape[0], valid.max()))
    if hi - lo < SAMPLE_RATE // 2:
        return None
    return np.asarray(loaded.audio[lo:hi], dtype=np.float64)


def _resample_curve(curve: np.ndarray, n: int) -> np.ndarray:
    """Stretch a short-term curve onto ``n`` points by linear interpolation.

    Deck B plays at a different rate from its own source, so its source-time
    curve and the mix's output-time curve have different lengths for the same
    music. Both cover the same stretch of the transition, so mapping one onto
    the other's length lines them up.
    """
    if curve.size == n or curve.size == 0 or n <= 0:
        return curve
    return np.interp(
        np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, curve.size), curve
    )


def _loudness_dip(mix_ms: np.ndarray, src_curves: list[np.ndarray]) -> float:
    """Largest moment-by-moment drop below the sources' average, in dB."""
    if mix_ms.size == 0 or not src_curves:
        return 0.0
    aligned = [_resample_curve(c, mix_ms.size) for c in src_curves]
    reference_ms = np.mean(np.vstack(aligned), axis=0)
    with np.errstate(divide="ignore"):
        ref_db = -0.691 + 10.0 * np.log10(np.maximum(reference_ms, 1e-20))
        mix_db = -0.691 + 10.0 * np.log10(np.maximum(mix_ms, 1e-20))
    # Only where there was something to be quieter than: a passage that is
    # silent in both sources cannot be dipped by a crossfade.
    live = ref_db > ABSOLUTE_GATE_LUFS
    if not live.any():
        return 0.0
    return round(float(max(0.0, np.max(ref_db[live] - mix_db[live]))), 2)


def _per_bar(values: np.ndarray, bars_of: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of a per-block quantity within each whole bar."""
    whole = np.floor(bars_of).astype(int)
    bars = np.unique(whole)
    means = np.array([float(values[whole == bar].mean()) for bar in bars])
    return bars, means


def _low_overlap_bars(sub_a, sub_b, bars_of) -> float:
    """Bars in which both decks put significant energy below 200 Hz.

    Two decisions matter here, and both were wrong in the obvious version.

    Aggregated by BAR, not by block: a kick drum makes one block enormous and
    the next nearly empty, so a per-block test answers "was there a kick in this
    2048 samples" rather than "is this deck's bass playing".

    Measured against each deck's own 90th-percentile bar rather than its single
    loudest block: with the peak as the reference, one bass hit at full fader
    sets a bar so high that a deck sitting at half fader through the whole blend
    never clears it, and a blend with no bass swap at all measures as zero
    overlap. That is exactly the case this number exists to catch.
    """
    frac = config.PREVIEW_BAND_PRESENT_FRAC
    if sub_a.size == 0 or sub_b.size == 0:
        return 0.0
    bars, mean_a = _per_bar(sub_a, bars_of)
    _bars, mean_b = _per_bar(sub_b, bars_of)
    ref_a = float(np.percentile(mean_a, 90)) if mean_a.size else 0.0
    ref_b = float(np.percentile(mean_b, 90)) if mean_b.size else 0.0
    if ref_a <= 0.0 or ref_b <= 0.0:
        return 0.0
    both = (mean_a > frac * ref_a) & (mean_b > frac * ref_b)
    return float(int(np.count_nonzero(both)))


def _spectral_clash(mid_a: np.ndarray, mid_b: np.ndarray) -> float:
    """How much the two midranges are the same shape, and both present.

    Two terms, multiplied. The first is the cosine similarity of the two decks'
    midrange spectra, which says they occupy the same frequencies. The second is
    how balanced they are -- the quieter over the louder -- which is what stops a
    deck that has already faded out from counting as a clash simply because its
    spectrum still looks like the other one's.
    """
    if mid_a.size == 0 or mid_b.size == 0:
        return 0.0
    norm_a = np.linalg.norm(mid_a, axis=1)
    norm_b = np.linalg.norm(mid_b, axis=1)
    live = (norm_a > 0) & (norm_b > 0)
    if not live.any():
        return 0.0
    shape = np.sum(mid_a[live] * mid_b[live], axis=1) / (norm_a[live] * norm_b[live])
    balance = np.minimum(norm_a[live], norm_b[live]) / np.maximum(
        norm_a[live], norm_b[live]
    )
    # The 90th percentile, not the mean. "Is the midrange muddy" is a question
    # about the worst of the overlap; averaging it over the whole blend divides
    # that by the long stretches at each end where only one deck is playing and
    # nothing can clash.
    return round(float(np.percentile(shape * balance, 90)), 4)


def _phase_coherence(result: "render_mod.PreviewRender") -> float:
    """How well the two decks' beats line up across the overlap.

    Each of deck B's beats is placed against deck A's grid as a phase, and the
    length of the mean of those phases as unit vectors is the coherence: 1 is
    locked, 0 is beats scattered anywhere in the bar. Low means the two grids
    disagree -- which is a grid problem, not a transition problem.
    """
    t0 = result.transition_start_frame / SAMPLE_RATE
    t1 = (result.transition_start_frame + result.transition_frames) / SAMPLE_RATE
    a = np.array([
        r["time_s"] for r in result.grid
        if r["deck"] == "a" and r["kind"] == "beat" and t0 <= r["time_s"] <= t1
    ])
    b = np.array([
        r["time_s"] for r in result.grid
        if r["deck"] == "b" and r["kind"] == "beat" and t0 <= r["time_s"] <= t1
    ])
    if a.size < 2 or b.size < 1:
        return 1.0
    period = float(np.median(np.diff(a)))
    if period <= 0:
        return 1.0
    nearest = np.abs(b[:, None] - a[None, :]).min(axis=1)
    phase = (nearest / period) * 2.0 * np.pi
    return round(float(np.abs(np.mean(np.exp(1j * phase)))), 4)


def _vocal_overlap_bars(
    state: DeckAState, track_b: TrackAnalysis,
    result: "render_mod.PreviewRender", params,
) -> float:
    """Bars where both tracks are singing AND both can be heard.

    Which bars carry vocals comes from the cached vocal ranges, so no audio is
    touched for that. Whether a deck is audible comes from the render envelope:
    a vocal on a deck whose fader or mid band is down is not clashing with
    anything, and counting it would make every long blend look like a clash
    for bars in which one singer is already gone.
    """
    if state.analysis.bpm <= 0 or track_b.bpm <= 0:
        return 0.0
    from djai import phrase

    rows_by_bar: dict[int, list[dict]] = {}
    for row in result.envelope:
        if row["in_transition"]:
            rows_by_bar.setdefault(int(row["bar"]), []).append(row)

    count = 0
    for rows in rows_by_bar.values():
        first = rows[0]
        heard_a = float(np.mean([r["a_gain"] * r["a_mid"] for r in rows]))
        heard_b = float(np.mean([r["b_gain"] * r["b_mid"] for r in rows]))
        if heard_a < VOCAL_AUDIBLE or heard_b < VOCAL_AUDIBLE:
            continue
        bar_a = phrase.bar_at_frame(state.analysis, first["a_frame"])
        bar_b = phrase.bar_at_frame(track_b, first["b_frame"])
        if (state.analysis.has_vocals_between(bar_a, bar_a + 1.0)
                and track_b.has_vocals_between(bar_b, bar_b + 1.0)):
            count += 1
    return float(count)


def resolve_entry_frame(track_b: TrackAnalysis, params, default_frame: float) -> float:
    """Where deck B enters, for the entry point these parameters name.

    A lookup, not a computation: ``hot_cue_N`` is a marker the analysis already
    stores for the track, and ``mix_in`` is the frame placement already chose.
    The model names a cue; it never supplies a position.
    """
    index = params.hot_cue_index()
    if index is None:
        return float(default_frame)
    from djai.analysis import cue_seconds

    for cue in track_b.hot_cues or []:
        if int(cue.get("index", -1)) == index:
            return cue_seconds(cue) * SAMPLE_RATE
    return float(default_frame)


# --- the entry point ------------------------------------------------------------


def preview_transition(
    deck_a_state: DeckAState,
    track_b: TrackAnalysis,
    params,
    loaded_b: LoadedTrack,
    entry_frame_b: float,
    context_bars: float | None = None,
    blocksize: int | None = None,
) -> dict[str, Any]:
    """Render the planned transition and measure it. Worker thread only.

    Returns the measurement dict, with ``render_ms`` and ``measure_ms`` added.
    Raises :class:`PreviewError` if the render or the measurement could not be
    completed -- which the caller must treat as "commit the parameters you
    already have", never as a reason to delay the transition.
    """
    if context_bars is None:
        context_bars = config.PREVIEW_CONTEXT_BARS

    t_start = time.perf_counter()
    try:
        result = render_mod.render_preview(
            deck_a_state.analysis,
            track_b,
            deck_a_state.loaded,
            loaded_b,
            params,
            transition_at_frame_a=deck_a_state.transition_at_frame,
            entry_frame_b=resolve_entry_frame(track_b, params, entry_frame_b),
            context_bars=context_bars,
            blocksize=blocksize,
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "commit"
        raise PreviewError("render", f"{type(exc).__name__}: {exc}") from exc
    render_ms = (time.perf_counter() - t_start) * 1000.0

    t_measure = time.perf_counter()
    try:
        out = measure(result, deck_a_state, track_b, loaded_b, params)
    except Exception as exc:  # noqa: BLE001
        raise PreviewError("measure", f"{type(exc).__name__}: {exc}") from exc

    out["render_ms"] = round(render_ms, 1)
    out["measure_ms"] = round((time.perf_counter() - t_measure) * 1000.0, 1)
    return out
