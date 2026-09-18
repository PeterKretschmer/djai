"""Colour-coded low/mid/high waveforms at several zoom levels, cached on disk.

THREADING CONTEXT: the UI server's request threads, when a waveform is first
asked for. Filters a whole decoded track; never the audio thread.

A waveform here is three peak envelopes -- low, mid and high, split at the
deck EQ's own crossovers, so what the page shows is what the EQ knobs move --
at three levels of detail (about 25, 100 and 400 points per second) plus a
fixed 1,600-point overview. Each is stored as uint8 per band.

Time is in ORIGINAL track seconds: the envelopes come from the unstretched
audio, so a stretched deck's waveform lines up with the beat grid, which is
also in original seconds.

The cache (``<cache>/waveforms/<track_id>.npz``) carries
:data:`WAVEFORM_VERSION`, and nothing is read from it before that is checked.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

log = logging.getLogger(__name__)

WAVEFORM_VERSION: int = 1

#: Band edges, matching djai.deck's EQ crossovers.
LOW_XOVER_HZ: float = 250.0
HIGH_XOVER_HZ: float = 2500.0

#: Finest level's bucket, in frames. At 44.1 kHz that is ~400.9 points per
#: second: one bar at 128 BPM is ~750 points, enough for a 1-bar zoom.
FINE_BUCKET: int = 110
#: Each coarser level merges this many points of the one below it.
LEVEL_FACTORS: tuple[int, ...] = (1, 4, 16)
OVERVIEW_POINTS: int = 1600

#: Bands are filtered in chunks this long, with the filter state carried.
_CHUNK: int = 1 << 20


@dataclass
class Waveform:
    """Three bands at each level. ``levels[i]`` is ``(points_per_second, (3, n) uint8)``."""

    sample_rate: int
    duration_s: float
    levels: list[tuple[float, np.ndarray]]
    overview: np.ndarray  # (3, OVERVIEW_POINTS) uint8

    def meta(self) -> dict:
        return {
            "version": WAVEFORM_VERSION,
            "bands": ["low", "mid", "high"],
            "duration_s": round(self.duration_s, 4),
            "overview_points": int(self.overview.shape[1]),
            "levels": [
                {"level": i, "points_per_second": round(pps, 6), "points": int(arr.shape[1])}
                for i, (pps, arr) in enumerate(self.levels)
            ],
        }


def _lr4(cutoff: float, btype: str, sr: int) -> np.ndarray:
    stage = butter(2, cutoff, btype=btype, fs=sr, output="sos")
    return np.vstack([stage, stage])


def compute(audio: np.ndarray, sample_rate: int) -> Waveform:
    """Filter a track into bands and take peaks at every level. Seconds of CPU."""
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    mono = np.asarray(mono, dtype=np.float64)
    n = mono.shape[0]
    filters = {
        "low": _lr4(LOW_XOVER_HZ, "low", sample_rate),
        "rest": _lr4(LOW_XOVER_HZ, "high", sample_rate),
        "mid": _lr4(HIGH_XOVER_HZ, "low", sample_rate),
        "high": _lr4(HIGH_XOVER_HZ, "high", sample_rate),
    }
    state = {k: np.zeros((v.shape[0], 2)) for k, v in filters.items()}
    fine_n = n // FINE_BUCKET
    peaks = np.zeros((3, fine_n + 1), dtype=np.float64)
    carry = np.zeros((3, 0))
    written = 0
    for start in range(0, n, _CHUNK):
        chunk = mono[start:start + _CHUNK]
        low, state["low"] = sosfilt(filters["low"], chunk, zi=state["low"])
        rest, state["rest"] = sosfilt(filters["rest"], chunk, zi=state["rest"])
        mid, state["mid"] = sosfilt(filters["mid"], rest, zi=state["mid"])
        high, state["high"] = sosfilt(filters["high"], rest, zi=state["high"])
        bands = np.abs(np.vstack([low, mid, high]))
        if carry.shape[1]:
            bands = np.hstack([carry, bands])
        whole = bands.shape[1] // FINE_BUCKET
        if whole:
            blocks = bands[:, : whole * FINE_BUCKET].reshape(3, whole, FINE_BUCKET).max(axis=2)
            peaks[:, written:written + whole] = blocks
            written += whole
        carry = bands[:, whole * FINE_BUCKET:]
    if carry.shape[1]:
        peaks[:, written] = carry.max(axis=1)
        written += 1
    peaks = peaks[:, :written]
    scale = float(peaks.max()) or 1.0

    def quantise(x: np.ndarray) -> np.ndarray:
        return np.clip(np.round(x / scale * 255.0), 0, 255).astype(np.uint8)

    fine_pps = sample_rate / FINE_BUCKET
    levels = []
    for factor in LEVEL_FACTORS:
        m = int(np.ceil(peaks.shape[1] / factor))
        padded = np.zeros((3, m * factor))
        padded[:, : peaks.shape[1]] = peaks
        levels.append((fine_pps / factor, quantise(padded.reshape(3, m, factor).max(axis=2))))
    group = int(np.ceil(peaks.shape[1] / OVERVIEW_POINTS)) or 1
    padded = np.zeros((3, OVERVIEW_POINTS * group))
    take = min(peaks.shape[1], padded.shape[1])
    padded[:, :take] = peaks[:, :take]
    overview = quantise(padded.reshape(3, OVERVIEW_POINTS, group).max(axis=2))
    return Waveform(sample_rate, n / sample_rate, levels, overview)


def cache_file(cache_dir: Path, track_id: str) -> Path:
    return Path(cache_dir) / "waveforms" / f"{track_id}.npz"


def save(waveform: Waveform, cache_dir: Path, track_id: str) -> Path:
    path = cache_file(cache_dir, track_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"level{i}": arr for i, (_pps, arr) in enumerate(waveform.levels)}
    with path.open("wb") as fh:
        np.savez_compressed(
            fh,
            version=np.array(WAVEFORM_VERSION, dtype=np.int32),
            sample_rate=np.array(waveform.sample_rate, dtype=np.int32),
            duration_s=np.array(waveform.duration_s, dtype=np.float64),
            pps=np.array([pps for pps, _ in waveform.levels], dtype=np.float64),
            overview=waveform.overview,
            **arrays,
        )
    return path


def load(cache_dir: Path, track_id: str) -> Waveform | None:
    """The cached waveform, or None: missing, unreadable, or another version.

    The version is read and compared before any band data is touched.
    """
    path = cache_file(cache_dir, track_id)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != WAVEFORM_VERSION:
                return None
            pps = [float(v) for v in data["pps"]]
            levels = [(p, np.array(data[f"level{i}"])) for i, p in enumerate(pps)]
            return Waveform(
                int(data["sample_rate"]), float(data["duration_s"]), levels,
                np.array(data["overview"]),
            )
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        log.warning("waveform cache %s unreadable: %s", path, exc)
        return None


def for_track(track, cache_dir: Path | None, sample_rate: int) -> Waveform:
    """The waveform for a loaded track: from the cache, else computed and cached."""
    track_id = track.analysis.track_id
    if cache_dir is not None:
        cached = load(cache_dir, track_id)
        if cached is not None:
            return cached
    source = track.source if getattr(track, "source", None) is not None else track
    audio = source.audio[:-2] if source.audio.shape[0] > 2 else source.audio
    waveform = compute(audio, sample_rate)
    if cache_dir is not None:
        try:
            save(waveform, cache_dir, track_id)
        except OSError as exc:
            log.warning("could not cache waveform for %s: %s", track_id, exc)
    return waveform
