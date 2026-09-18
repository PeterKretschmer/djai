"""Offline track analysis and its on-disk cache.

THREADING CONTEXT: main thread only (the ``analyze`` subcommand), or a worker
thread during ``play`` when a cache entry is being *read*. Nothing in this
module may ever be called from the audio thread -- it decodes files, runs
librosa, and touches the filesystem.

For each audio file we compute BPM, a beat grid, estimated downbeats, musical
key (Camelot + name), per-beat RMS energy, and a ``grid_confidence`` in 0..1
derived from beat-interval variance. The result is written as a two-file cache
entry and is never recomputed while one exists for that content hash:
``./cache/<hash>.json`` holds the scalars, mix points and hot cues, and
``./cache/<hash>.npz`` holds the beat grid, downbeats and per-beat RMS as
float32 arrays. Both carry the schema version, and both are checked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import librosa
import numpy as np

from djai import beat_tracker, config

log = logging.getLogger(__name__)

# --- constants ---------------------------------------------------------------

#: Bumped whenever the analysis output format or algorithm changes; a sidecar
#: written by an older version is treated as a cache miss.
#: 3 added mix_in / mix_out. A sidecar written by an older version is a cache
#: miss, never a silent stale read -- see :func:`load_crate`, which counts them
#: and says so rather than quietly dropping the track from the crate.
#: 4 added hot_cues. Existing v3 sidecars are upgraded in place rather than
#: re-analysed: every input a hot cue needs (beat grid, downbeats, per-beat RMS,
#: mix points) is already in the file, so the upgrade costs no audio decoding.
#: 5 changed how tempo and grid_confidence are derived. There is deliberately
#: NO upgrade path to it: both values come from the audio, so a v4 sidecar
#: cannot be brought forward without re-reading the track. Anything older is a
#: cache miss and `load_crate` says so loudly.
#: 6 changed only storage: the beat grid, downbeats and per-beat RMS moved out
#: of the JSON into a float32 ``.npz`` beside it. Measured on the 96-track
#: library, those three lists were 78% of every sidecar as JSON number text.
#: Nothing was re-measured, so v5 sidecars ARE upgraded in place.
#:
#: float32 is exact to about 60 microseconds on a ten-minute beat timestamp,
#: under three samples at 44.1 kHz and far inside the 10 ms phase budget.
#: 7 added loudness (integrated LUFS, true peak, the gain that normalises it),
#: onset-agreement scores at the detected tempo, half and double with an
#: ambiguity flag, and `grid_manually_corrected`. v5 and v6 upgrade in place
#: with those empty; `analyze` then measures what is missing without
#: re-detecting the tempo.
#: 8 added musical structure: section bar ranges (intro, build, drop,
#: breakdown, outro), vocal bar ranges, a loudness-normalised intensity score
#: and an artist parsed from the file name. v5-v7 upgrade in place with those
#: empty; `analyze` measures them without touching the tempo or grid.
ANALYSIS_VERSION = 8

#: Hot cue positions are stored in **deck frames** -- seconds * this rate --
#: because that is the domain a deck's playhead lives in, and a cue that has to
#: be converted before it can be used is a cue that will eventually be converted
#: wrongly. Must equal :data:`djai.deck.SAMPLE_RATE`; importing it here would be
#: circular, so :mod:`djai.deck` asserts the two agree.
HOT_CUE_SR: int = 44100

#: The brief's ceiling: 8 cues per track, of which the first two are always the
#: mix points, leaving up to 6 detected or hand-set.
MAX_HOT_CUES: int = 8

#: A structural boundary is a change in per-beat RMS of at least this fraction
#: across the window either side of a downbeat.
HOT_CUE_RMS_CHANGE: float = 0.30

#: Half-window for that comparison, in bars.
HOT_CUE_WINDOW_BARS: int = 8

#: Detected cues closer together than this are the same musical event heard
#: twice; keep the stronger one.
HOT_CUE_MIN_SPACING_BARS: int = 8

#: Sample rate librosa analyses at. Lower than playback SR purely for speed;
#: every timestamp we emit is in *seconds*, so playback is unaffected.
ANALYSIS_SR = 22050

AUDIO_EXTENSIONS = (".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3", ".m4a")

DEFAULT_CACHE_DIR = Path("cache")

#: Bytes taken from the head and tail of a file for the content hash. Hashing
#: the whole file is needlessly slow for 50 MB tracks; head+tail+size is a
#: perfectly good content identity for a music library and means renaming or
#: moving a file still hits the cache.
_HASH_CHUNK = 1 << 20

BEATS_PER_BAR = 4

#: grid_confidence = exp(-_CONF_K * coefficient_of_variation_of_beat_intervals),
#: clipped to 0..1. cv=0.02 -> 0.85, cv=0.05 -> 0.67, cv=0.20 -> 0.20.
_CONF_K = 8.0

#: The confidence of a grid imported from Rekordbox or Serato. That grid is
#: ground truth: a person set it in software built for exactly that, so it is
#: not re-measured against onsets a detector found. Only an import sets it, so a
#: manually corrected grid at this value is known to be an imported one -- which
#: is how a forced re-analysis keeps it without the cache needing a new field.
#: Grids corrected by hand in this program keep their measured confidence.
IMPORTED_GRID_CONFIDENCE: float = 1.0

#: Which beat tracker analysis uses. "auto": Beat This! when it is installed
#: (``pip install .[beats]``), librosa otherwise. "beat_this": Beat This! or an
#: error. "librosa": the original detector and confidence, always. Set from
#: ``analyze --tracker``; the test suite pins "librosa".
BEAT_TRACKER: str = "auto"
BEAT_TRACKERS: tuple[str, ...] = ("auto", "beat_this", "librosa")

#: The BPM window a DJ actually counts in. Detected tempos are folded into it by
#: octaves before the comb refinement below runs.
BPM_MIN, BPM_MAX = 82.0, 164.0

#: Beat trackers routinely land on a half, double or dotted multiple of the real
#: tempo. These are the multipliers we re-score against the onset envelope.
_TEMPO_MULTIPLIERS = (1 / 3, 1 / 2, 2 / 3, 1.0, 3 / 2, 2.0, 3.0)

#: Log-normal prior over tempo, centred on 128 BPM. Breaks ties between comb
#: candidates that fit the onsets about equally well (a 2x comb hits every real
#: onset too, so onset fit alone is not decisive).
_TEMPO_PRIOR_CENTER = 128.0
_TEMPO_PRIOR_WIDTH = 0.55  # in octaves

#: Chroma is computed from the harmonic component starting at C3. Kick drums
#: have strong pitched tails in the bottom octaves (a 45 Hz tail reads as F#1)
#: which otherwise dominate the key estimate.
_CHROMA_FMIN_NOTE = "C3"
_CHROMA_OCTAVES = 4

_HOP = 512

#: Fine tempo search window around the coarse estimate, and its resolution.
_FINE_TEMPO_SPAN = 0.06
_FINE_TEMPO_STEPS = 2401
_FINE_TEMPO_CHUNK = 128

#: A beat counts as "playing" when its RMS is at least this fraction of the
#: track's median beat RMS. Ambient intros and outros sit below it.
MIX_POINT_RMS_FRACTION: float = 0.40

#: How long the energy must stay up before a downbeat counts as the mix-in.
MIX_IN_SUSTAIN_BARS: int = 8

#: Fallbacks when the energy trace gives no clear answer.
MIX_IN_FALLBACK_BARS: float = 16.0
MIX_OUT_FALLBACK_BARS: float = 32.0

_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Krumhansl-Schmuckler key profiles.
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Camelot wheel. Index is the pitch class of the tonic.
_CAMELOT_MAJOR = {
    "C": "8B", "C#": "3B", "D": "10B", "D#": "5B", "E": "12B", "F": "7B",
    "F#": "2B", "G": "9B", "G#": "4B", "A": "11B", "A#": "6B", "B": "1B",
}
_CAMELOT_MINOR = {
    "C": "5A", "C#": "12A", "D": "7A", "D#": "2A", "E": "9A", "F": "4A",
    "F#": "11A", "G": "6A", "G#": "1A", "A": "8A", "A#": "3A", "B": "10A",
}


# --- data model --------------------------------------------------------------


@dataclass
class TrackAnalysis:
    """Everything the mixing layers know about one track. Immutable in practice."""

    track_id: str
    path: str
    title: str
    duration_s: float
    bpm: float
    beats: list[float]
    downbeats: list[float]
    key_name: str
    camelot: str
    beat_rms: list[float]
    energy: float
    grid_confidence: float

    #: Where the track is worth mixing into and out of, in seconds. ``mix_out``
    #: is the placement anchor: a transition should *finish* there, not start
    #: wherever the next phrase boundary happens to fall.
    mix_in: float = 0.0
    mix_out: float = 0.0
    #: The same two points as bar indices counted from ``first_downbeat``.
    mix_in_bar: float = 0.0
    mix_out_bar: float = 0.0
    #: True when the energy trace gave no clear answer and the fallbacks were
    #: used, so callers can tell a measured point from a guessed one.
    mix_points_estimated: bool = True

    #: Up to :data:`MAX_HOT_CUES` markers playback can jump to, each a plain
    #: dict of ``index`` (1-based), ``sample_position`` (deck frames) and
    #: ``label``. Kept as dicts rather than a nested dataclass so the sidecar
    #: round-trips through :meth:`from_dict` without a custom decoder.
    hot_cues: list[dict] = field(default_factory=list)

    #: Integrated loudness (BS.1770) and 4x-oversampled true peak. None until
    #: measured: a sidecar upgraded from v5 or v6 has neither until `analyze`.
    lufs: float | None = None
    true_peak_dbtp: float | None = None
    #: The true peak of the track as a deck actually plays it: through the
    #: three-band EQ at unity. Measured separately because that EQ is flat in
    #: level but rotates phase, and on a limited master phase rotation
    #: re-exposes peaks the mastering limiter had pinned -- measured at +2.0 to
    #: +3.1 dB over the file's own true peak on three library tracks.
    played_peak_dbtp: float | None = None
    #: Gain applied at deck load to reach LOUDNESS_TARGET_LUFS, capped so the
    #: true peak stays under LOUDNESS_MAX_TRUE_PEAK_DBTP. 0 until measured.
    track_gain_db: float = 0.0

    #: Onset agreement of the grid at the detected tempo, half and double:
    #: ``{"base": .., "half": .., "double": ..}``. Empty until measured.
    tempo_scores: dict = field(default_factory=dict)
    #: Twice the tempo agrees nearly as well: the reading may be half-time.
    tempo_ambiguous: bool = False
    #: A person fixed this grid. Analysis never overwrites it.
    grid_manually_corrected: bool = False

    #: Structure, as ``{"label", "start_bar", "end_bar"}`` dicts in bars from
    #: ``first_downbeat``, end exclusive, in order. Labels are those in
    #: :data:`SECTION_LABELS`. Empty until measured.
    sections: list[dict] = field(default_factory=list)
    #: Bars with a voice in them, as ``[start_bar, end_bar]`` pairs, end
    #: exclusive. Empty is "none found" once ``vocal_fraction`` is set.
    vocal_bars: list[list[int]] = field(default_factory=list)
    #: Share of the track's bars that carry vocals. None until measured.
    vocal_fraction: float | None = None
    #: Composite 0..1 intensity, measured on loudness-normalised audio so a
    #: quiet banger outranks a loud ambient track. None until measured.
    intensity: float | None = None
    #: The three measurements behind it, for the log and for tuning.
    intensity_features: dict = field(default_factory=dict)
    #: From an "Artist - Title" file name; empty when the name has no artist.
    artist: str = ""

    analysis_version: int = ANALYSIS_VERSION

    # Cached numpy views, built lazily. Not serialised.
    _beats_np: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def beats_np(self) -> np.ndarray:
        """Beat timestamps in seconds as a float64 array (built once, then reused)."""
        if self._beats_np is None:
            self._beats_np = np.asarray(self.beats, dtype=np.float64)
        return self._beats_np

    @property
    def first_downbeat(self) -> float:
        """Time in seconds of bar 0. Falls back to the first beat, then to 0.0."""
        if self.downbeats:
            return self.downbeats[0]
        if self.beats:
            return self.beats[0]
        return 0.0

    @property
    def beat_period(self) -> float:
        """Nominal seconds per beat, from the reported BPM."""
        return 60.0 / self.bpm if self.bpm > 0 else 0.5

    @property
    def review_needed(self) -> bool:
        """Should a person check this grid before it is trusted in a set?

        Ambiguous tempo, or a grid too weak to hold a blend. A grid someone
        has already corrected is, by definition, reviewed.
        """
        if self.grid_manually_corrected:
            return False
        return self.tempo_ambiguous or (
            self.grid_confidence < config.TRANSITION_MIN_GRID_CONFIDENCE
        )

    @property
    def vocal_led(self) -> bool:
        """Is a voice a large part of this track? False until measured."""
        return (self.vocal_fraction or 0.0) >= VOCAL_LED_FRACTION

    def sections_labelled(self, label: str) -> list[dict]:
        """Every section with this label, in order."""
        return [s for s in self.sections if s.get("label") == label]

    def has_vocals_between(self, start_bar: float, end_bar: float) -> list[tuple[float, float]]:
        """The parts of ``[start_bar, end_bar)`` that carry vocals."""
        out = []
        for a, b in self.vocal_bars:
            lo, hi = max(float(a), start_bar), min(float(b), end_bar)
            if hi > lo:
                out.append((lo, hi))
        return out

    def to_json(self) -> str:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        return json.dumps(d, indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "TrackAnalysis":
        known = {f for f in cls.__dataclass_fields__ if not f.startswith("_")}
        return cls(**{k: v for k, v in d.items() if k in known})


# --- hot cues ----------------------------------------------------------------


def make_hot_cue(index: int, seconds: float, label: str) -> dict:
    """One cue, in the shape the sidecar and the wire both use."""
    return {
        "index": int(index),
        "sample_position": int(round(max(0.0, seconds) * HOT_CUE_SR)),
        "label": str(label),
    }


def cue_seconds(cue: dict) -> float:
    """A cue's position back in seconds."""
    return float(cue["sample_position"]) / HOT_CUE_SR


def _beat_index_at(beats: np.ndarray, t: float) -> int:
    """Index of the beat nearest ``t``."""
    if beats.size == 0:
        return 0
    return int(np.argmin(np.abs(beats - t)))


def auto_hot_cues(ta: "TrackAnalysis") -> list[dict]:
    """Cues 1 and 2 at the mix points, then up to four structural boundaries.

    A boundary is a downbeat where the mean per-beat RMS over the eight bars
    after it differs from the eight bars before it by at least
    :data:`HOT_CUE_RMS_CHANGE`. Landing on a downbeat is a requirement, not a
    rounding step: a cue that fires half a bar out is worse than no cue.

    Pure over already-cached fields -- no audio is read -- which is what lets a
    v3 sidecar be upgraded without re-analysing the track.
    """
    cues = [
        make_hot_cue(1, ta.mix_in, "mix in"),
        make_hot_cue(2, ta.mix_out, "mix out"),
    ]

    beats = ta.beats_np
    rms = np.asarray(ta.beat_rms, dtype=np.float64)
    if beats.size == 0 or rms.size == 0 or not ta.downbeats:
        return cues

    window = HOT_CUE_WINDOW_BARS * BEATS_PER_BAR
    bar_s = BEATS_PER_BAR * 60.0 / ta.bpm if ta.bpm > 0 else 2.0
    spacing_s = HOT_CUE_MIN_SPACING_BARS * bar_s

    candidates: list[tuple[float, float, str]] = []  # (strength, seconds, label)
    for t in ta.downbeats:
        i = _beat_index_at(beats, t)
        if i < window or i + window > rms.size:
            continue
        before = float(np.mean(rms[i - window:i]))
        after = float(np.mean(rms[i:i + window]))
        if before <= 1e-9:
            continue
        change = (after - before) / before
        if abs(change) < HOT_CUE_RMS_CHANGE:
            continue
        candidates.append((abs(change), float(t), "drop" if change > 0 else "break"))

    candidates.sort(key=lambda c: c[0], reverse=True)
    taken = [ta.mix_in, ta.mix_out]
    for _strength, t, label in candidates:
        if len(cues) >= 6:  # 2 mix points + 4 detected
            break
        if any(abs(t - other) < spacing_s for other in taken):
            continue
        taken.append(t)
        cues.append(make_hot_cue(len(cues) + 1, t, label))

    return cues


def with_hot_cues(ta: "TrackAnalysis") -> "TrackAnalysis":
    """Fill in ``hot_cues`` if the track has none. Returns the same object."""
    if not ta.hot_cues:
        ta.hot_cues = auto_hot_cues(ta)
    return ta


# --- hashing / cache ---------------------------------------------------------


def track_hash(path: Path) -> str:
    """Content hash used as the cache key. See ``_HASH_CHUNK`` for the rationale."""
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(_HASH_CHUNK))
        if size > 2 * _HASH_CHUNK:
            fh.seek(-_HASH_CHUNK, 2)
            h.update(fh.read(_HASH_CHUNK))
    return h.hexdigest()[:16]


def cache_path(track_id: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """The JSON half of a cache entry: scalars, mix points and hot cues."""
    return Path(cache_dir) / f"{track_id}.json"


def arrays_path(track_id: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """The array half of a cache entry: beat grid, downbeats, per-beat RMS."""
    return Path(cache_dir) / f"{track_id}.npz"


#: The per-beat and per-bar lists, stored as float32 arrays rather than JSON.
ARRAY_FIELDS: tuple[str, ...] = ("beats", "downbeats", "beat_rms")

#: Schema versions that can be brought forward in place, without re-reading
#: audio. Anything older is a genuine cache miss.
#:
#: v5 and v6 are upgradable: v6 changed only where three lists are stored, and
#: v7 only added fields that start empty. v4 and older still are not: v5
#: changed the tempo itself, and no arithmetic on a stale sidecar recovers a
#: number that came out of the audio. v7 is upgradable for the same reason as
#: v6: v8 only added fields that start empty.
UPGRADABLE_FROM: tuple[int, ...] = (5, 6, 7)


def write_sidecar(ta: "TrackAnalysis", cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Write both halves of a track's cache entry. Returns the JSON path.

    Arrays first, JSON last: a crash between the two leaves an array file with
    no JSON, which reads as a miss, rather than a JSON file whose arrays are
    missing or from an older write.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    d = {k: v for k, v in asdict(ta).items() if not k.startswith("_")}
    arrays = {name: np.asarray(d.pop(name), dtype=np.float32) for name in ARRAY_FIELDS}
    with arrays_path(ta.track_id, cache_dir).open("wb") as fh:
        np.savez_compressed(
            fh,
            analysis_version=np.array(ta.analysis_version, dtype=np.int32),
            **arrays,
        )
    path = cache_path(ta.track_id, cache_dir)
    path.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
    return path


def read_sidecar(path: Path) -> tuple[dict | None, str]:
    """One cache entry as a complete dict, and how it was obtained.

    The status is ``"ok"``, ``"upgraded"`` (a v5 entry rewritten as v6 with no
    audio read), ``"stale"`` (too old to bring forward, or its halves disagree),
    or ``"corrupt"``.

    Nothing is trusted before the version is checked. The JSON's version decides
    whether the array file is opened at all, and the array file carries its own
    version, which must agree before a single beat is read from it.
    """
    path = Path(path)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        log.warning("cache read failed for %s: %s", path, exc)
        return None, "corrupt"
    if not isinstance(d, dict):
        return None, "corrupt"

    version = d.get("analysis_version")
    if version != ANALYSIS_VERSION and version not in UPGRADABLE_FROM:
        return None, "stale"

    if any(name not in d for name in ARRAY_FIELDS):
        # From v6 the lists live in the array file, which must be at the same
        # schema version as this JSON before a single beat is read from it.
        npz = path.with_suffix(".npz")
        try:
            with np.load(npz, allow_pickle=False) as data:
                if int(data["analysis_version"]) != version:
                    log.warning("array file %s is not at schema v%s", npz, version)
                    return None, "stale"
                for name in ARRAY_FIELDS:
                    d[name] = data[name].astype(np.float64).tolist()
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
            log.warning("array file for %s is missing or unreadable: %s", path, exc)
            return None, "corrupt"

    if version != ANALYSIS_VERSION:
        upgraded = upgrade_sidecar(d, path)
        return (upgraded, "upgraded") if upgraded is not None else (None, "stale")
    return d, "ok"


def upgrade_sidecar(d: dict, path: Path) -> dict | None:
    """Bring a v5 or v6 entry to the current schema, or None if it cannot be.

    Nothing is re-measured, so no audio is read. v5 held everything inline; v6
    split the three long lists into a float32 array file; v7 added fields that
    start empty -- loudness, tempo scores -- which `analyze` fills in later
    without touching the tempo. The entry is rewritten once, and what is
    returned is read back from disk, so this session sees what the next will.
    """
    if d.get("analysis_version") not in UPGRADABLE_FROM:
        return None
    if any(name not in d for name in ARRAY_FIELDS):
        return None
    d = dict(d)
    d["analysis_version"] = ANALYSIS_VERSION
    try:
        ta = TrackAnalysis.from_dict(d)
    except TypeError:
        return None
    in_memory = {k: v for k, v in asdict(ta).items() if not k.startswith("_")}
    if Path(path).stem != ta.track_id:
        # Only entries named by their content hash are rewritten. Anything else
        # would leave the old file beside the new one and load the track twice.
        return in_memory
    try:
        written = write_sidecar(ta, Path(path).parent)
    except OSError as exc:
        # Still usable in memory this session; it just re-upgrades next time.
        log.warning("could not rewrite upgraded sidecar %s: %s", path, exc)
        return in_memory
    reread, status = read_sidecar(written)
    return reread if status == "ok" else None


def load_cached(track_id: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> TrackAnalysis | None:
    """Return a cached analysis, or None on a miss / stale version / corrupt file."""
    p = cache_path(track_id, cache_dir)
    if not p.exists():
        return None
    d, _status = read_sidecar(p)
    if d is None:
        return None
    try:
        return TrackAnalysis.from_dict(d)
    except TypeError as exc:
        log.warning("cache entry %s has an unexpected shape: %s", p, exc)
        return None


def load_crate(cache_dir: Path = DEFAULT_CACHE_DIR) -> list[TrackAnalysis]:
    """Load every valid analysis in the cache whose audio file still exists.

    A sidecar from an older schema is never read. It is also never dropped
    quietly: a schema bump can invalidate an entire library at once, and a
    crate that silently shrinks to nothing looks like lost work rather than a
    re-analysis prompt.
    """
    out: list[TrackAnalysis] = []
    stale = 0
    upgraded = 0
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return out
    for p in sorted(cache_dir.glob("*.json")):
        d, status = read_sidecar(p)
        if status == "stale":
            stale += 1
        if d is None:
            continue
        if status == "upgraded":
            upgraded += 1
        try:
            ta = TrackAnalysis.from_dict(d)
        except TypeError:
            continue
        if Path(ta.path).exists():
            out.append(ta)
        else:
            log.warning("cached track %s no longer on disk: %s", ta.track_id, ta.path)

    if upgraded:
        note = (
            f"{upgraded} sidecar(s) upgraded in place to schema "
            f"v{ANALYSIS_VERSION} (no audio re-decoded)."
        )
        log.info(note)
        print(f"[analysis] {note}")

    if stale:
        message = (
            f"{stale} cached sidecar(s) predate analysis schema v{ANALYSIS_VERSION} "
            f"and were NOT loaded. Re-run `python -m djai analyze <folder>` to "
            f"rebuild them (cached audio is not re-decoded for tracks already at "
            f"the current version)."
        )
        log.warning(message)
        print(f"[analysis] {message}")
    return out


# --- analysis primitives -----------------------------------------------------


def _grid_confidence(beats: np.ndarray, sr: int) -> float:
    """0..1 from the coefficient of variation of beat intervals.

    librosa reports beats on a frame grid, so every interval carries +-half a
    hop of quantisation noise that has nothing to do with the track's timing.
    That noise alone would cap confidence around 0.83, so we subtract its known
    variance (uniform over one quantum, and two quantised times per interval)
    before taking the CV. What is left is real tempo instability.
    """
    if beats.size < 4:
        return 0.0
    iv = np.diff(beats)
    mean = float(np.mean(iv))
    if mean <= 0:
        return 0.0
    quantum = _HOP / sr
    var_quantisation = 2.0 * quantum * quantum / 12.0
    var_real = max(0.0, float(np.var(iv)) - var_quantisation)
    cv = math.sqrt(var_real) / mean
    return float(np.clip(np.exp(-_CONF_K * cv), 0.0, 1.0))


def _fine_tempo_and_phase(
    oenv: np.ndarray, sr: int, coarse_bpm: float
) -> tuple[float, float]:
    """Localise tempo and beat phase by a single-bin DFT of the onset envelope.

    ``_refine_tempo`` can only return librosa's discrete tempogram values times
    a small ratio, so it lands ~1% off (120.94 for a 122.0 BPM track). Here we
    evaluate ``|sum(w * exp(-2j*pi*f*t))|`` on a fine frequency grid around that
    coarse estimate: the peak is the beat frequency and its phase angle is the
    beat offset, both at far better than frame resolution. The +-6% window keeps
    the search away from the half/double harmonics already resolved upstream.

    Returns ``(bpm, first_beat_time)``.
    """
    t = np.arange(oenv.size, dtype=np.float64) * (_HOP / sr)
    w = oenv - float(np.mean(oenv))
    if w.size < 16 or not np.any(w):
        return coarse_bpm, 0.0

    bpms = coarse_bpm * np.linspace(
        1.0 - _FINE_TEMPO_SPAN, 1.0 + _FINE_TEMPO_SPAN, _FINE_TEMPO_STEPS
    )
    mags = np.empty(bpms.size)
    phases = np.empty(bpms.size)
    # Chunked so the candidates x frames matrix stays small.
    for lo in range(0, bpms.size, _FINE_TEMPO_CHUNK):
        hi = min(lo + _FINE_TEMPO_CHUNK, bpms.size)
        freqs = (bpms[lo:hi] / 60.0)[:, None]
        z = np.exp(-2j * np.pi * freqs * t[None, :]) @ w
        mags[lo:hi] = np.abs(z)
        phases[lo:hi] = np.angle(z)

    peak = int(np.argmax(mags))
    bpm = float(bpms[peak])
    # Parabolic interpolation on the magnitude peak for sub-step precision.
    if 0 < peak < bpms.size - 1:
        y0, y1, y2 = mags[peak - 1], mags[peak], mags[peak + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            shift = 0.5 * (y0 - y2) / denom
            if abs(shift) <= 1.0:
                bpm = float(bpms[peak] + shift * (bpms[1] - bpms[0]))

    period = 60.0 / bpm
    # exp(-i*angle) is maximal where 2*pi*f*t == angle, i.e. beats sit at
    # t = angle/(2*pi*f) + k*period. Fold into the first period.
    t0 = (phases[peak] / (2.0 * np.pi)) * period
    t0 = t0 % period
    return bpm, float(t0)


def _fold_to_range(bpm: float, lo: float = BPM_MIN, hi: float = BPM_MAX) -> float:
    """Double or halve ``bpm`` until it lands in [lo, hi)."""
    if bpm <= 0 or not np.isfinite(bpm):
        return _TEMPO_PRIOR_CENTER
    for _ in range(8):
        if bpm < lo:
            bpm *= 2.0
        elif bpm >= hi:
            bpm /= 2.0
        else:
            break
    return bpm


def _tempo_prior(bpm: float) -> float:
    octaves = np.log2(bpm / _TEMPO_PRIOR_CENTER)
    return float(np.exp(-0.5 * (octaves / _TEMPO_PRIOR_WIDTH) ** 2))


def _comb_score(oenv: np.ndarray, sr: int, bpm: float) -> float:
    """Mean onset strength on the best-aligned beat comb at ``bpm``.

    A comb at twice the true tempo also lands on every real onset, but half its
    teeth fall between them, so the *mean* (not the sum) is what discriminates.
    """
    frames_per_beat = (60.0 / bpm) * sr / _HOP
    if frames_per_beat < 2 or oenv.size < frames_per_beat * 4:
        return 0.0
    n_beats = int((oenv.size - 1) / frames_per_beat)
    if n_beats < 4:
        return 0.0
    base = np.arange(n_beats) * frames_per_beat
    best = 0.0
    for phase_step in range(12):
        phase = phase_step * frames_per_beat / 12.0
        idx = np.rint(base + phase).astype(np.int64)
        idx = idx[idx < oenv.size]
        if idx.size:
            best = max(best, float(np.mean(oenv[idx])))
    return best


#: Tempo search range. The brief's 70-180: wide enough for a 98 BPM track and
#: a 160 BPM one, both of which the old 82-164 fold could not represent.
TEMPO_SEARCH_MIN: float = 70.0
TEMPO_SEARCH_MAX: float = 180.0


def _onset_peaks(oenv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Onset peak frames and their strengths. Computed once per track."""
    peaks = np.asarray(
        librosa.util.peak_pick(
            oenv, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.0, wait=2
        ),
        dtype=np.int64,
    )
    if peaks.size == 0:
        return peaks, peaks.astype(np.float64)
    return peaks, oenv[peaks].astype(np.float64)


def _agreement(
    peaks: np.ndarray, weights: np.ndarray, n_frames: int, frames_per_beat: float
) -> float:
    """Onset-to-grid agreement for one tempo, searching over phase."""
    if frames_per_beat < 2 or peaks.size < 8:
        return 0.0
    n_beats = int((n_frames - 1) / frames_per_beat)
    if n_beats < 8:
        return 0.0
    total = float(weights.sum())
    if total <= 0:
        return 0.0

    best = 0.0
    for phase_step in range(16):
        phase = phase_step * frames_per_beat / 16.0
        rel = (peaks - phase) / frames_per_beat
        off = np.abs((rel + 0.5) % 1.0 - 0.5)
        hit = np.clip(1.0 - off / 0.25, 0.0, 1.0)
        agreement = float(np.dot(weights, hit) / total)
        if agreement <= best:
            continue
        occupied = np.unique(np.rint(rel[hit > 0.5]).astype(np.int64))
        coverage = min(1.0, float(occupied.size) / float(n_beats))
        best = max(best, agreement * coverage)
    return float(np.clip(best, 0.0, 1.0))


def grid_agreement(oenv: np.ndarray, sr: int, bpm: float) -> float:
    """How well a beat grid at ``bpm`` agrees with the actual onsets, 0..1.

    Two halves, multiplied, because either alone is fooled by a harmonic:

    * **agreement** -- of the onset energy in the track, how much of it sits on
      a beat. A grid at half the true tempo scores badly here: half the onsets
      land squarely between its beats.
    * **coverage** -- of the grid's beats, how many actually carry an onset. A
      grid at twice the true tempo scores badly here: every other beat is
      empty.

    Mean onset strength on the comb, which is what this replaces, has neither
    property. It rises as the grid gets sparser, so it preferred half-tempo on
    13 of 20 real tracks and had to be propped up by a tempo prior centred on
    128 -- which is how a wrong answer near the prior became a stable one.
    """
    peaks, weights = _onset_peaks(oenv)
    return _agreement(peaks, weights, oenv.size, (60.0 / bpm) * sr / _HOP)


def _refine_tempo(oenv: np.ndarray, sr: int, raw_bpm: float) -> float:
    """The tempo alone. Kept scalar because `reference.py` imports this."""
    return _refine_tempo_scored(oenv, sr, raw_bpm)[0]


def _refine_tempo_scored(
    oenv: np.ndarray, sr: int, raw_bpm: float
) -> tuple[float, float]:
    """Pick the tempo whose grid best agrees with the onsets. ``(bpm, score)``.

    A dense sweep of the whole 70-180 range rather than a handful of multiples
    of whatever the beat tracker guessed. That matters because the errors seen
    on real tracks were not octaves: 160 BPM read as 119 (x0.744) and 105 read
    as 131 (x1.25). A multiplier list of octaves and triplets cannot reach
    either, so the wrong answer survived refinement every time.

    No tempo prior is applied anywhere. librosa 1.0 removed the ``tempo()``
    helper the brief names, and its predecessor applied a log-normal prior
    around ``start_bpm=120`` by default -- which is the behaviour we are trying
    to get away from, so scoring the range directly is both available and
    closer to the intent.
    """
    peaks, weights = _onset_peaks(oenv)
    n = oenv.size
    if peaks.size < 8:
        return _fold_to_range(raw_bpm), 0.0

    def score(bpm: float) -> float:
        return _agreement(peaks, weights, n, (60.0 / bpm) * sr / _HOP)

    # Coarse sweep at 1 BPM, then a fine pass around the best few.
    coarse = np.arange(TEMPO_SEARCH_MIN, TEMPO_SEARCH_MAX + 0.001, 1.0)
    scored = sorted(((score(float(b)), float(b)) for b in coarse), reverse=True)

    best_score, best_bpm = scored[0]
    for _s, around in scored[:3]:
        for b in np.arange(around - 1.0, around + 1.0, 0.05):
            if not TEMPO_SEARCH_MIN <= b <= TEMPO_SEARCH_MAX:
                continue
            s = score(float(b))
            if s > best_score:
                best_score, best_bpm = s, float(b)

    # KNOWN LIMITATION, stated rather than papered over. Onset agreement
    # cannot separate a grid from the same grid at double speed: on material
    # with eighth-note content every beat of the doubled grid also lands on an
    # onset, so it scores at least as well. Measured on a synthetic 75 BPM
    # track: 150 scored 0.889 against 75's 0.483, and the doubled reading is
    # not obviously wrong for that material.
    #
    # Two things bound the damage. The search range is 70-180, so a doubling
    # can only happen to tracks under 90. And the decks beat-match by ratio,
    # so a track read at double time still mixes in phase -- it is the LLM's
    # context and the bar counts that suffer, not the sync. Tie-breaking on
    # beat-strength alternation was tried and misfired in both directions; it
    # is not in here because it did not work, not because it was not tried.
    #
    # Explicit octave correction, as the brief asks: below 85 test double,
    # above 175 test half, and keep whichever agrees better.
    for low, high, factor in ((0.0, 85.0, 2.0), (175.0, 1e9, 0.5)):
        if low <= best_bpm < high:
            other = best_bpm * factor
            if TEMPO_SEARCH_MIN <= other <= TEMPO_SEARCH_MAX:
                other_score = score(other)
                if other_score > best_score:
                    best_score, best_bpm = other_score, other
    return best_bpm, best_score




def _estimate_key(chroma: np.ndarray) -> tuple[str, str]:
    """Krumhansl-Schmuckler key estimate. Returns (key_name, camelot)."""
    profile = chroma.mean(axis=1)
    if not np.any(profile):
        return "unknown", "?"
    profile = profile - profile.mean()

    best_score = -np.inf
    best: tuple[str, str] = ("unknown", "?")
    for mode, ks in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
        ref = ks - ks.mean()
        for tonic in range(12):
            rotated = np.roll(profile, -tonic)
            denom = np.linalg.norm(rotated) * np.linalg.norm(ref)
            score = float(np.dot(rotated, ref) / denom) if denom else 0.0
            if score > best_score:
                best_score = score
                name = _PITCH_NAMES[tonic]
                table = _CAMELOT_MAJOR if mode == "major" else _CAMELOT_MINOR
                best = (f"{name} {mode}", table[name])
    return best


def _estimate_downbeats(
    beats: np.ndarray, beat_chroma: np.ndarray, beat_rms: np.ndarray
) -> list[float]:
    """Pick the 4/4 bar phase, then return every beat on that phase.

    Scores each of the 4 candidate phases by how much harmonic *change* and
    energy accumulate on its beats: bar lines are where the chroma turns over
    and the kick lands. Beat-synchronous chroma is the signal, per spec.
    """
    n = beats.size
    if n < BEATS_PER_BAR * 2:
        return [float(b) for b in beats[:1]]

    # Harmonic novelty per beat: distance from the previous beat's chroma.
    c = beat_chroma
    norm = np.linalg.norm(c, axis=0)
    norm[norm == 0] = 1.0
    cn = c / norm
    novelty = np.zeros(n)
    novelty[1:] = np.linalg.norm(np.diff(cn, axis=1), axis=0)

    def z(x: np.ndarray) -> np.ndarray:
        s = float(np.std(x))
        return (x - float(np.mean(x))) / s if s > 0 else np.zeros_like(x)

    score_signal = z(novelty) + z(beat_rms[:n])

    best_phase, best_score = 0, -np.inf
    for phase in range(BEATS_PER_BAR):
        s = float(np.mean(score_signal[phase::BEATS_PER_BAR]))
        if s > best_score:
            best_score, best_phase = s, phase
    return [float(b) for b in beats[best_phase::BEATS_PER_BAR]]


def _mix_points(
    beats: np.ndarray, beat_rms: np.ndarray, downbeat_phase: int, bpm: float
) -> tuple[float, float, bool]:
    """Locate where the track is worth mixing into and out of.

    ``mix_in`` is the first downbeat whose energy has come up and *stays* up for
    :data:`MIX_IN_SUSTAIN_BARS`; requiring the sustain is what stops a single
    loud stab in an ambient intro from being mistaken for the drop.

    ``mix_out`` is the last downbeat before the energy drops away and never
    returns -- the start of the outro.

    Returns ``(mix_in_s, mix_out_s, estimated)``.
    """
    n = int(min(beats.size, beat_rms.size))
    bar_seconds = BEATS_PER_BAR * 60.0 / bpm if bpm > 0 else 2.0

    def fallback() -> tuple[float, float, bool]:
        if beats.size == 0:
            return 0.0, 0.0, True
        first_db = float(beats[downbeat_phase]) if beats.size > downbeat_phase else float(beats[0])
        last_db_idx = downbeat_phase + ((beats.size - 1 - downbeat_phase) // BEATS_PER_BAR) * BEATS_PER_BAR
        last_db = float(beats[max(last_db_idx, 0)])
        mi = first_db + MIX_IN_FALLBACK_BARS * bar_seconds
        mo = last_db - MIX_OUT_FALLBACK_BARS * bar_seconds
        if mo <= mi:
            mi, mo = first_db, last_db
        return mi, mo, True

    if n < BEATS_PER_BAR * 4 or not np.any(beat_rms[:n]):
        return fallback()

    median = float(np.median(beat_rms[:n]))
    if median <= 0:
        return fallback()
    threshold = MIX_POINT_RMS_FRACTION * median
    sustain = MIX_IN_SUSTAIN_BARS * BEATS_PER_BAR

    # --- mix in: first downbeat with `sustain` beats of unbroken energy ---
    mix_in_idx: int | None = None
    for k in range(downbeat_phase, n, BEATS_PER_BAR):
        window = beat_rms[k : min(k + sustain, n)]
        if window.size < min(sustain, n - k):
            break
        if window.size and bool(np.all(window >= threshold)):
            mix_in_idx = k
            break

    # --- mix out: start of the final run that never comes back up ---
    j = n
    while j > 0 and beat_rms[j - 1] < threshold:
        j -= 1
    # j is the first index of the trailing quiet run (n when there is none).
    mix_out_idx: int | None = None
    for k in range(downbeat_phase, n, BEATS_PER_BAR):
        if k <= j:
            mix_out_idx = k
        else:
            break

    if mix_in_idx is None or mix_out_idx is None:
        return fallback()
    mix_in_s = float(beats[mix_in_idx])
    mix_out_s = float(beats[mix_out_idx])
    if mix_out_s <= mix_in_s:
        return fallback()
    return mix_in_s, mix_out_s, False


# --- tempo ambiguity -----------------------------------------------------------


def tempo_candidates(oenv: np.ndarray, sr: int, bpm: float) -> dict[str, float]:
    """Onset agreement of the grid at ``bpm``, at half of it, and at double."""
    peaks, weights = _onset_peaks(oenv)
    n = oenv.size

    def score(candidate: float) -> float:
        if candidate <= 0:
            return 0.0
        return _agreement(peaks, weights, n, (60.0 / candidate) * sr / _HOP)

    return {
        "base": round(score(bpm), 4),
        "half": round(score(bpm / 2.0), 4),
        "double": round(score(bpm * 2.0), 4),
    }


def tempo_ambiguity(bpm: float, scores: dict) -> bool:
    """Might this reading be half-time? Flag it rather than guess.

    The rule: the grid at twice the tempo agrees within
    :data:`djai.config.TEMPO_AMBIGUITY_RATIO` of the detected one -- applied
    only where twice the tempo is itself one the detector searches.

    Measured on the 96-track library, the rule applied to every track flagged
    59 of them, including 8 of the 10 whose tempo is known and was read right
    or within 2%. Eighth-note hats make any grid's double agree well, so a
    correctly read 128 BPM track has a strong "double" at 256 -- a tempo no one
    mixes at, which the detector never considered. The double only says
    something about a reading when it is a tempo the reading could have been.
    """
    base = float(scores.get("base", 0.0) or 0.0)
    double = float(scores.get("double", 0.0) or 0.0)
    if base <= 0.0 or bpm * 2.0 > TEMPO_SEARCH_MAX:
        return False
    return double >= config.TEMPO_AMBIGUITY_RATIO * base


# --- loudness -----------------------------------------------------------------


def _true_peak_db(data: np.ndarray, sr: int, through_eq: bool = False) -> float:
    """4x-oversampled peak in dB -- the BS.1770 true-peak method -- optionally
    of the signal as a deck plays it: through the three-band EQ at unity.

    The EQ path is the deck's own network: the same Linkwitz-Riley crossovers,
    split and summed the same way, run once offline. It is flat in magnitude
    but rotates phase, which is enough to raise the peaks of a limited master.

    Worked in ten-second chunks with the filter state carried across and a
    short overlap for the oversampler, so a ten-minute track costs a few
    hundred megabytes rather than several gigabytes.
    """
    from scipy.signal import butter, resample_poly, sosfilt

    from djai import deck as deck_module

    n, channels = data.shape
    if n == 0:
        return -180.0
    chunk = max(sr * 10, 4096)
    pad = 256

    filters: list[np.ndarray] = []
    states: list[np.ndarray] = []
    if through_eq:
        def lr4(cutoff: float, btype: str) -> np.ndarray:
            stage = butter(deck_module.EQ_ORDER, cutoff, btype=btype, fs=sr, output="sos")
            return np.vstack([stage, stage])

        filters = [
            lr4(deck_module.EQ_LOW_XOVER, "low"),
            lr4(deck_module.EQ_LOW_XOVER, "high"),
            lr4(deck_module.EQ_HIGH_XOVER, "low"),
            lr4(deck_module.EQ_HIGH_XOVER, "high"),
        ]
        states = [np.zeros((f.shape[0], 2, channels)) for f in filters]

    peak = 0.0
    tail = np.zeros((0, channels))
    for start in range(0, n, chunk):
        x = data[start:start + chunk].astype(np.float64)
        if through_eq:
            low, states[0] = sosfilt(filters[0], x, axis=0, zi=states[0])
            rest, states[1] = sosfilt(filters[1], x, axis=0, zi=states[1])
            mid, states[2] = sosfilt(filters[2], rest, axis=0, zi=states[2])
            high, states[3] = sosfilt(filters[3], rest, axis=0, zi=states[3])
            y = low + mid + high
        else:
            y = x
        # The oversampler sees `tail` as context. Its first `pad` samples were
        # already counted; the rest were held back from the previous chunk's
        # end, where the oversampler had no future to look at.
        segment = np.vstack([tail, y])
        over = resample_poly(segment, 4, 1, axis=0)
        first = pad * 4 if tail.shape[0] else 0
        final = start + chunk >= n
        last = over.shape[0] if final else max(first, over.shape[0] - pad * 4)
        if last > first:
            peak = max(peak, float(np.max(np.abs(over[first:last]))))
        tail = y[-2 * pad:]
    return 20.0 * math.log10(max(peak, 1e-9))


def measure_loudness(path: Path) -> tuple[float, float, float]:
    """Integrated loudness (LUFS), the file's true peak, and its true peak as
    a deck plays it (both dBTP).

    Decoded at the file's own rate and channel layout, because loudness is
    defined over what a listener hears, not over the mono analysis signal.
    """
    import pyloudnorm
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    lufs = float(pyloudnorm.Meter(sr).integrated_loudness(data))
    return lufs, _true_peak_db(data, sr), _true_peak_db(data, sr, through_eq=True)


def loudness_gain_db(
    lufs: float | None,
    true_peak_dbtp: float | None,
    played_peak_dbtp: float | None = None,
) -> float:
    """The gain that brings a track to the target, never past the peak cap.

    Capped on the higher of the file's true peak and its true peak as played,
    so what leaves a deck -- not what sits in the file -- stays under the cap.
    """
    if lufs is None or not math.isfinite(lufs):
        return 0.0
    gain = config.LOUDNESS_TARGET_LUFS - lufs
    peaks = [
        p for p in (true_peak_dbtp, played_peak_dbtp)
        if p is not None and math.isfinite(p)
    ]
    if peaks:
        gain = min(gain, config.LOUDNESS_MAX_TRUE_PEAK_DBTP - max(peaks))
    return round(gain, 2)


def complete_measurements(ta: "TrackAnalysis", path: Path) -> None:
    """Fill in what an upgraded entry lacks, without re-detecting the tempo.

    Loudness from the file, and tempo scores at the *stored* tempo -- so a
    manually corrected grid is scored as corrected and is never replaced.
    """
    if ta.lufs is None or ta.played_peak_dbtp is None:
        try:
            lufs, true_peak, played_peak = measure_loudness(path)
            ta.lufs = round(lufs, 2)
            ta.true_peak_dbtp = round(true_peak, 2)
            ta.played_peak_dbtp = round(played_peak, 2)
        except Exception as exc:  # a file loudness cannot be read from still plays
            log.warning("loudness measurement failed for %s: %s", path, exc)
            # NaN, not None: measured and failed, so the next run does not retry
            # forever and `analyze` stays a no-op on an unchanged library.
            ta.lufs, ta.true_peak_dbtp, ta.played_peak_dbtp = float("nan"), None, float("nan")
        ta.track_gain_db = loudness_gain_db(
            ta.lufs, ta.true_peak_dbtp, ta.played_peak_dbtp
        )
    if not ta.tempo_scores:
        y, sr = librosa.load(str(path), sr=ANALYSIS_SR, mono=True)
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)
        ta.tempo_scores = tempo_candidates(oenv, sr, ta.bpm)
        ta.tempo_ambiguous = (
            False if ta.grid_manually_corrected
            else tempo_ambiguity(ta.bpm, ta.tempo_scores)
        )
    if ta.intensity is None:
        try:
            y, sr = librosa.load(str(path), sr=ANALYSIS_SR, mono=True)
            measure_structure(ta, y, sr)
        except Exception as exc:  # an entry without structure still plays
            log.warning("structure measurement failed for %s: %s", path, exc)
            ta.intensity = float("nan")
    if not ta.artist:
        ta.artist = artist_from_title(ta.title)


# --- musical structure: sections, vocals, intensity ---------------------------

#: The section labels, in the order a typical track visits them.
SECTION_LABELS: tuple[str, ...] = ("intro", "build", "drop", "breakdown", "outro")

#: Half-width of the checkerboard novelty kernel, in bars.
_SECTION_KERNEL_BARS: int = 4
#: No section is shorter than this, in bars.
_SECTION_MIN_BARS: int = 8
#: A segment within this many dB of the loudest segment is a drop.
_DROP_LEVEL_DB: float = -3.0
#: A build rises by at least this many dB per bar. Measured on the synthetic
#: set: builds rise 0.31-0.54 dB/bar (0.23 where one merged with a breakdown),
#: breakdowns 0.00-0.19.
_BUILD_SLOPE_DB_PER_BAR: float = 0.2

#: Vocal decision thresholds, per bar, tuned on synthetic sung lines against
#: pads and drums: the normalised flux of the harmonic 200 Hz-4 kHz band (a
#: voice moves; a pad holds), and that band's share of all energy.
VOCAL_FLUX_MIN: float = 0.11
VOCAL_SHARE_MIN: float = 0.5
VOCAL_BAND_HZ: tuple[float, float] = (200.0, 4000.0)
#: A track is vocal-led when at least this share of its bars carry vocals.
VOCAL_LED_FRACTION: float = 0.25

#: Intensity weights and the fixed ranges each measurement is scaled over.
#: Absolute, not relative to a crate, so one track can be compared with
#: another anywhere. Ranges set from a 96-track pop/dance library, where the
#: measurements run: percussive share 0.15-0.33, flux 2.3-2.8, 2.3-4.3
#: onsets per second.
_INTENSITY_WEIGHTS = {"percussive": 0.45, "flux": 0.35, "density": 0.20}
_INTENSITY_PERCUSSIVE_FULL: float = 0.5
_INTENSITY_FLUX_RANGE: tuple[float, float] = (1.5, 5.0)
_INTENSITY_DENSITY_FULL: float = 8.0


def artist_from_title(title: str) -> str:
    """The artist in an ``"Artist - Title"`` name, or empty."""
    parts = str(title or "").split(" - ", 1)
    return parts[0].strip() if len(parts) == 2 and parts[0].strip() else ""


def _bar_starts(ta: "TrackAnalysis") -> tuple[np.ndarray, int]:
    """Start times of every whole bar in the file, and the bar number of the first.

    Bar 0 is ``first_downbeat``; bars before it are negative.
    """
    bar_s = BEATS_PER_BAR * 60.0 / ta.bpm if ta.bpm > 0 else 2.0
    first = ta.first_downbeat
    n_pre = int(math.floor(first / bar_s + 1e-6))
    t0 = first - n_pre * bar_s
    n = max(1, int((ta.duration_s - t0) / bar_s))
    return t0 + np.arange(n + 1) * bar_s, -n_pre


def _zrows(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)


def detect_sections(
    mfcc_bars: np.ndarray, chroma_bars: np.ndarray, rms_db_bars: np.ndarray, bar0: int
) -> list[dict]:
    """Label bar ranges from beat-synchronous timbre, harmony and level.

    Boundaries: a checkerboard novelty over the bar self-similarity matrix of
    MFCCs and chroma, added to the level change across each bar line, and
    weighted toward 8- and 4-bar phrase lines. Peaks at least
    :data:`_SECTION_MIN_BARS` apart become boundaries.

    Labels, per segment: the first and last of three or more are intro and
    outro; otherwise within 3 dB of the loudest segment is a drop; a segment
    rising into a drop is a build; anything else is a breakdown. Adjacent
    segments with the same label merge.

    Measured on 10 synthetic tracks with known structure: boundaries 44/45
    within 2 bars, no false boundaries (see tests/test_phase3_structure.py).
    """
    n = int(rms_db_bars.size)
    if n < 2 * _SECTION_MIN_BARS:
        return []
    half = _SECTION_KERNEL_BARS
    feats = np.vstack([_zrows(mfcc_bars), 0.5 * _zrows(chroma_bars)]).T
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
    ssm = feats @ feats.T
    taper = np.exp(-0.5 * (np.linspace(-1, 1, 2 * half) / 0.5) ** 2)
    sign = np.concatenate([-np.ones(half), np.ones(half)]) * taper
    kernel = np.outer(sign, sign)
    padded = np.pad(ssm, half, mode="edge")
    novelty = np.array(
        [float(np.sum(kernel * padded[i:i + 2 * half, i:i + 2 * half])) for i in range(n)]
    )
    novelty = np.maximum(novelty, 0.0)
    level = np.zeros(n)
    for i in range(n):
        before, after = rms_db_bars[max(0, i - half):i], rms_db_bars[i:i + half]
        if before.size and after.size:
            level[i] = abs(float(after.mean() - before.mean()))
    score = novelty / (novelty.max() + 1e-9) + level / (level.max() + 1e-9)
    weight = np.array([
        1.0 if (i + bar0) % 8 == 0 else 0.6 if (i + bar0) % 4 == 0 else 0.25
        for i in range(n)
    ])
    score *= weight
    threshold = float(score.mean() + 0.5 * score.std())
    bounds: list[int] = []
    for i in np.argsort(score)[::-1]:
        if score[i] < threshold:
            break
        if i < _SECTION_MIN_BARS // 2 or i > n - _SECTION_MIN_BARS // 2:
            continue
        if all(abs(int(i) - j) >= _SECTION_MIN_BARS for j in bounds):
            bounds.append(int(i))
    edges = [0] + sorted(bounds) + [n]
    segments = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    levels = [float(rms_db_bars[a:b].mean()) for a, b in segments]
    top = max(levels)
    many = len(segments) >= 3

    out: list[dict] = []
    for k, (a, b) in enumerate(segments):
        rel = levels[k] - top
        slope = float(np.polyfit(np.arange(b - a), rms_db_bars[a:b], 1)[0]) if b - a > 1 else 0.0
        if many and k == 0:
            label = "intro"
        elif many and k == len(segments) - 1:
            label = "outro"
        elif rel > _DROP_LEVEL_DB:
            label = "drop"
        elif (
            k + 1 < len(segments)
            and levels[k + 1] - top > _DROP_LEVEL_DB
            and slope > _BUILD_SLOPE_DB_PER_BAR
        ):
            label = "build"
        else:
            label = "breakdown"
        if out and out[-1]["label"] == label:
            out[-1]["end_bar"] = b + bar0
        else:
            out.append({"label": label, "start_bar": a + bar0, "end_bar": b + bar0})
    return out


def _runs(flags: np.ndarray, bar0: int, min_bars: int = 2, bridge: int = 1) -> list[list[int]]:
    """True bars as ``[start, end)`` ranges: short gaps bridged, blips dropped."""
    flags = flags.astype(bool).copy()
    n = flags.size
    i = 0
    while i < n:
        if not flags[i]:
            j = i
            while j < n and not flags[j]:
                j += 1
            if 0 < i and j < n and j - i <= bridge:
                flags[i:j] = True
            i = j
        else:
            i += 1
    out: list[list[int]] = []
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            if j - i >= min_bars:
                out.append([i + bar0, j + bar0])
            i = j
        else:
            i += 1
    return out


def measure_structure(ta: "TrackAnalysis", y: np.ndarray, sr: int) -> None:
    """Sections, vocal bars and intensity for one track, from mono audio.

    Offline, and slow (roughly 10-25 s a track, most of it HPSS). Fills the
    v8 fields in place and never touches the tempo, grid or hot cues.
    """
    hop = _HOP
    starts, bar0 = _bar_starts(ta)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    n_frames = S.shape[1]
    frames = np.clip(
        librosa.time_to_frames(starts, sr=sr, hop_length=hop), 0, max(0, n_frames - 1)
    )

    def per_bar(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        cols = [
            x[:, a:b].mean(axis=1) if b > a else x[:, min(a, x.shape[1] - 1)]
            for a, b in zip(frames[:-1], frames[1:])
        ]
        return np.stack(cols, axis=1)

    mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=64)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=14)[1:]
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    rms = librosa.feature.rms(S=S)[0]
    rms_db = 20.0 * np.log10(per_bar(rms)[0] + 1e-6)
    ta.sections = detect_sections(per_bar(mfcc), per_bar(chroma), rms_db, bar0)

    # Vocals: the harmonic part of the voice band, which a voice both fills and
    # keeps moving.
    H, P = librosa.decompose.hpss(S, margin=2.0)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= VOCAL_BAND_HZ[0]) & (freqs <= VOCAL_BAND_HZ[1])
    Hb = H[band]
    eps = 1e-9
    share = Hb.sum(axis=0) / (H.sum(axis=0) + P.sum(axis=0) + eps)
    delta = np.diff(Hb, axis=1, prepend=Hb[:, :1])
    flux = np.sqrt((np.maximum(delta, 0.0) ** 2).sum(axis=0)) / (
        np.sqrt((Hb ** 2).sum(axis=0)) + eps
    )
    flags = (per_bar(flux)[0] > VOCAL_FLUX_MIN) & (per_bar(share)[0] > VOCAL_SHARE_MIN)
    ta.vocal_bars = _runs(flags, bar0)
    ta.vocal_fraction = round(
        sum(b - a for a, b in ta.vocal_bars) / max(1, flags.size), 4
    )

    # Intensity, on loudness-normalised audio so mastering level does not count.
    lufs = ta.lufs if ta.lufs is not None and np.isfinite(ta.lufs) else None
    gain = 10.0 ** ((config.LOUDNESS_TARGET_LUFS - lufs) / 20.0) if lufs is not None else 1.0
    Sn = S * gain
    Hn, Pn = librosa.decompose.hpss(Sn, margin=1.0)
    percussive = float((Pn ** 2).sum() / ((Pn ** 2).sum() + (Hn ** 2).sum() + 1e-12))
    oenv = librosa.onset.onset_strength(S=librosa.amplitude_to_db(Sn), sr=sr, hop_length=hop)
    onset_flux = float(np.mean(oenv)) if oenv.size else 0.0
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop)
    density = float(len(onsets)) / max(1e-6, len(y) / sr)
    ta.intensity_features = {
        "percussive": round(percussive, 4),
        "flux": round(onset_flux, 4),
        "density": round(density, 3),
    }
    ta.intensity = intensity_score(percussive, onset_flux, density)


def intensity_score(percussive: float, onset_flux: float, density: float) -> float:
    """Combine the three intensity measurements into 0..1."""
    lo, hi = _INTENSITY_FLUX_RANGE
    parts = {
        "percussive": min(1.0, max(0.0, percussive / _INTENSITY_PERCUSSIVE_FULL)),
        "flux": min(1.0, max(0.0, (onset_flux - lo) / (hi - lo))),
        "density": min(1.0, max(0.0, density / _INTENSITY_DENSITY_FULL)),
    }
    return round(sum(_INTENSITY_WEIGHTS[k] * v for k, v in parts.items()), 4)


# --- manual grid correction ---------------------------------------------------

#: Tap tempo sets nothing until it has this many taps.
MIN_TAPS: int = 8
#: And they must be steady. Past this spread in the tap intervals (as a
#: coefficient of variation) it is not a tempo yet, it is someone finding one.
MAX_TAP_CV: float = 0.08


def bpm_from_taps(times: Sequence[float]) -> float | None:
    """Tempo from tap timestamps in seconds, or None if too few or too uneven."""
    if len(times) < MIN_TAPS:
        return None
    intervals = np.diff(np.asarray(times, dtype=np.float64))
    if intervals.size == 0 or np.any(intervals <= 0):
        return None
    if float(np.std(intervals)) / float(np.mean(intervals)) > MAX_TAP_CV:
        return None
    return 60.0 / float(np.median(intervals))


def regrid(ta: "TrackAnalysis", bpm: float, first_downbeat: float) -> None:
    """Rebuild a track's grid at ``bpm`` with a bar line at ``first_downbeat``.

    In place and without the audio. Beats are laid across the track at the new
    period, downbeats every fourth beat on the given bar line's phase, per-beat
    RMS re-sampled onto the new beats by time, and mix points snapped to the
    nearest new downbeat. Hot cues are sample positions and do not move. Marks
    the grid manually corrected, which analysis never overwrites.
    """
    if not (math.isfinite(bpm) and bpm > 0):
        raise ValueError(f"bpm must be positive, got {bpm!r}")
    period = 60.0 / bpm
    anchor = float(first_downbeat)
    t0 = anchor - math.floor(anchor / period) * period
    n = max(1, int((max(float(ta.duration_s), t0) - t0) / period) + 1)
    beats = t0 + np.arange(n) * period
    phase = int(round((anchor - t0) / period)) % BEATS_PER_BAR
    downbeats = beats[phase::BEATS_PER_BAR]

    old_beats = np.asarray(ta.beats, dtype=np.float64)
    old_rms = np.asarray(ta.beat_rms, dtype=np.float64)
    if old_beats.size >= 2 and old_rms.size == old_beats.size:
        rms = np.interp(beats, old_beats, old_rms)
    else:
        rms = np.zeros(n)

    def snap(t: float) -> float:
        if downbeats.size == 0:
            return float(t)
        return float(downbeats[int(np.argmin(np.abs(downbeats - t)))])

    bar_s = BEATS_PER_BAR * period
    ta.bpm = round(float(bpm), 4)
    ta.beats = [round(float(b), 5) for b in beats]
    ta.downbeats = [round(float(b), 5) for b in downbeats]
    ta.beat_rms = [round(float(r), 6) for r in rms]
    ta.mix_in = round(snap(ta.mix_in), 5)
    ta.mix_out = round(snap(ta.mix_out), 5)
    first = float(downbeats[0]) if downbeats.size else t0
    ta.mix_in_bar = round((ta.mix_in - first) / bar_s, 4)
    ta.mix_out_bar = round((ta.mix_out - first) / bar_s, 4)
    ta.grid_manually_corrected = True
    ta.tempo_ambiguous = False
    ta._beats_np = None


def _detect_beats(y: np.ndarray, sr: int, path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Beat This! ``(beats, downbeats)`` under :data:`BEAT_TRACKER`, else None.

    In "auto" a missing install or a failure on one file falls back to librosa
    for that file, with a warning; in "beat_this" either is an error.
    """
    if BEAT_TRACKER not in BEAT_TRACKERS:
        raise ValueError(f"unknown beat tracker {BEAT_TRACKER!r}; one of {BEAT_TRACKERS}")
    if BEAT_TRACKER == "librosa":
        return None
    if not beat_tracker.available():
        if BEAT_TRACKER == "beat_this":
            raise RuntimeError(
                "Beat This! is not installed: pip install .[beats] "
                "(PyTorch with CUDA: see the note in pyproject.toml)"
            )
        return None
    try:
        return beat_tracker.detect(y, sr)
    except Exception as exc:
        if BEAT_TRACKER == "beat_this":
            raise
        log.warning("Beat This! failed on %s, using librosa: %s", path, exc)
        return None


def analyze_file(
    path: Path, grid: tuple[float, float] | None = None
) -> TrackAnalysis:
    """Decode and analyse one audio file. Slow (seconds per track) and blocking.

    ``grid`` -- ``(bpm, first_downbeat_seconds)`` -- skips tempo detection and
    lays the grid there instead. It is how a manually corrected grid survives
    a forced re-analysis: everything else is measured afresh against it.
    """
    path = Path(path)
    y, sr = librosa.load(str(path), sr=ANALYSIS_SR, mono=True)
    duration = float(len(y) / sr) if sr else 0.0

    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)

    # Beat This! detections, when that tracker is in use. They build the grid
    # of a track nobody has gridded, and on every track they are one of the two
    # witnesses the grid's confidence is measured against.
    detected = _detect_beats(y, sr, path)
    fitted = None
    if grid is None and detected is not None:
        fitted = beat_tracker.fit_constant_grid(*detected)

    if grid is not None:
        # A person's grid: no detection, and the bar line they chose.
        bpm = float(grid[0])
        grid_period = 60.0 / bpm
        t0 = float(grid[1]) - math.floor(float(grid[1]) / grid_period) * grid_period
    elif fitted is not None:
        # Beat This!'s grid: its tempo, phase and bar 1, held constant.
        bpm = float(fitted[0])
        grid_period = 60.0 / bpm
        t0 = float(fitted[1]) - math.floor(float(fitted[1]) / grid_period) * grid_period
    else:
        # Pass 1: take the whole tempo distribution and pick the candidate
        # whose grid agrees best with the onsets. No `start_bpm` is passed
        # anywhere in this function: librosa defaults that argument to 120,
        # and a prior is exactly what turns a hard track into a confidently
        # wrong one.
        raw_tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=_HOP, trim=False)
        bpm, _tempo_score = _refine_tempo_scored(
            oenv, sr, float(np.atleast_1d(raw_tempo)[0])
        )

        # Pass 2: localise tempo and phase precisely, then lay down a
        # constant-tempo grid. The decks change tempo by resampling at a
        # constant rate, so a constant-tempo grid is the only grid they can
        # hold phase against.
        bpm, t0 = _fine_tempo_and_phase(oenv, sr, bpm)
        if not np.isfinite(bpm) or bpm <= 0:
            bpm = _TEMPO_PRIOR_CENTER
            t0 = 0.0
    period = 60.0 / bpm
    n_beats = max(1, int((duration - t0) / period) + 1)
    beats = t0 + np.arange(n_beats) * period

    # Confidence is how well the fitted grid agrees with the actual onsets --
    # not how uniform the detected beat intervals are.
    #
    # Interval variance answers "is this grid steady", and a grid derived from
    # a wrong tempo is perfectly steady. So a track whose detection failed used
    # to score high, which is the opposite of useful: the number exists to say
    # whether the grid can be trusted. Onset agreement says that.
    #
    # With Beat This! detections this is replaced below, once downbeats exist:
    # onset agreement turned out not to see drift or a wrong bar 1 either.
    if detected is None:
        grid_confidence = grid_agreement(oenv, sr, bpm)
        # Uniformity still matters -- a grid that agrees on average but drifts
        # is not one a 24-bar blend can hold phase against -- so it caps it.
        _, beat_frames = librosa.beat.beat_track(
            onset_envelope=oenv, sr=sr, hop_length=_HOP, bpm=bpm, trim=False
        )
        beats_raw = np.asarray(
            librosa.frames_to_time(beat_frames, sr=sr, hop_length=_HOP), dtype=np.float64
        )
        grid_confidence = min(grid_confidence, _grid_confidence(beats_raw, sr))

    rms = librosa.feature.rms(y=y, hop_length=_HOP)[0]
    # Key comes from the harmonic component above C3: percussion and sub-bass
    # otherwise swamp the chroma (see _CHROMA_FMIN_NOTE).
    chroma = librosa.feature.chroma_cqt(
        y=librosa.effects.harmonic(y, margin=3.0),
        sr=sr,
        hop_length=_HOP,
        fmin=librosa.note_to_hz(_CHROMA_FMIN_NOTE),
        n_octaves=_CHROMA_OCTAVES,
    )

    # Sync per-beat features to the *fitted* grid, not the raw detected beats:
    # `beats` is what every downstream layer indexes against.
    grid_frames = np.asarray(
        librosa.time_to_frames(beats, sr=sr, hop_length=_HOP), dtype=np.int64
    )
    grid_frames = np.clip(grid_frames, 0, max(rms.size - 1, 0))
    grid_frames = np.unique(grid_frames)
    if grid_frames.size:
        beat_rms = librosa.util.sync(rms[np.newaxis, :], grid_frames, aggregate=np.mean)[0]
        beat_chroma = librosa.util.sync(chroma, grid_frames, aggregate=np.mean)
    else:
        beat_rms = np.zeros(0)
        beat_chroma = np.zeros((12, 0))
    # librosa.util.sync can return one extra column (the tail after the last
    # beat); trim so every per-beat array lines up with `beats`.
    beat_rms = np.asarray(beat_rms, dtype=np.float64)[: beats.size]
    beat_chroma = np.asarray(beat_chroma, dtype=np.float64)[:, : beats.size]
    if beat_rms.size < beats.size:
        beat_rms = np.pad(beat_rms, (0, beats.size - beat_rms.size))
    if beat_chroma.shape[1] < beats.size:
        beat_chroma = np.pad(beat_chroma, ((0, 0), (0, beats.size - beat_chroma.shape[1])))

    key_name, camelot = _estimate_key(chroma)
    downbeats = _estimate_downbeats(beats, beat_chroma, beat_rms)
    bar_line = grid[1] if grid is not None else fitted[1] if fitted is not None else None
    if bar_line is not None:
        bar_phase = int(round((float(bar_line) - t0) / period)) % BEATS_PER_BAR
        downbeats = [float(b) for b in beats[bar_phase::BEATS_PER_BAR]]

    if detected is not None:
        grid_confidence = beat_tracker.grid_confidence(
            beats, np.asarray(downbeats, dtype=np.float64), *detected,
            beat_tracker.onset_drift_ms(oenv, sr, _HOP, beats, duration),
        )

    scores = tempo_candidates(oenv, sr, bpm)
    try:
        lufs, true_peak, played_peak = (round(v, 2) for v in measure_loudness(path))
    except Exception as exc:  # a file loudness cannot be read from still plays
        log.warning("loudness measurement failed for %s: %s", path, exc)
        lufs, true_peak, played_peak = float("nan"), None, float("nan")

    # Bar 0 is the first downbeat, so mix points are expressed relative to it.
    downbeat_phase = 0
    if downbeats and beats.size:
        period = 60.0 / bpm if bpm > 0 else 0.5
        downbeat_phase = int(max(0, round((downbeats[0] - float(beats[0])) / period)))
    mix_in_s, mix_out_s, estimated = _mix_points(beats, beat_rms, downbeat_phase, bpm)
    first_downbeat_s = downbeats[0] if downbeats else (float(beats[0]) if beats.size else 0.0)
    bar_seconds = BEATS_PER_BAR * 60.0 / bpm if bpm > 0 else 2.0

    ta = with_hot_cues(TrackAnalysis(
        track_id=track_hash(path),
        path=str(path.resolve()),
        title=path.stem,
        duration_s=duration,
        bpm=round(bpm, 4),
        beats=[round(float(b), 5) for b in beats],
        downbeats=[round(float(b), 5) for b in downbeats],
        key_name=key_name,
        camelot=camelot,
        beat_rms=[round(float(r), 6) for r in beat_rms],
        energy=float(np.mean(beat_rms)) if beat_rms.size else 0.0,
        grid_confidence=round(grid_confidence, 4),
        mix_in=round(mix_in_s, 5),
        mix_out=round(mix_out_s, 5),
        mix_in_bar=round((mix_in_s - first_downbeat_s) / bar_seconds, 4),
        mix_out_bar=round((mix_out_s - first_downbeat_s) / bar_seconds, 4),
        mix_points_estimated=bool(estimated),
        lufs=lufs,
        true_peak_dbtp=true_peak,
        played_peak_dbtp=played_peak,
        track_gain_db=loudness_gain_db(lufs, true_peak, played_peak),
        tempo_scores=scores,
        # The ambiguity flag catches librosa guessing the metrical level. Beat
        # This! chooses the level itself, and on a slow track its correct
        # answer (87 BPM, say) always has a strong double.
        tempo_ambiguous=(
            False if grid is not None or fitted is not None
            else tempo_ambiguity(bpm, scores)
        ),
        grid_manually_corrected=grid is not None,
        artist=artist_from_title(path.stem),
    ))
    try:
        measure_structure(ta, y, sr)
    except Exception as exc:  # a track without structure still plays
        log.warning("structure measurement failed for %s: %s", path, exc)
        ta.intensity = float("nan")
    return ta


# --- folder driver -----------------------------------------------------------


def iter_audio_files(folder: Path) -> Iterable[Path]:
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            yield p


def analyze_folder(
    folder: Path,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> tuple[list[TrackAnalysis], int, int]:
    """Analyse every audio file under ``folder``, skipping anything already cached.

    Returns ``(analyses, n_analyzed, n_skipped)``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: list[TrackAnalysis] = []
    analyzed = skipped = 0

    files: Sequence[Path] = list(iter_audio_files(folder))
    if not files:
        log.warning("no audio files found under %s", folder)

    for i, p in enumerate(files, 1):
        tid = track_hash(p)
        if not force:
            cached = load_cached(tid, cache_dir)
            if cached is not None:
                # Keep the stored path fresh if the file moved -- but only if it
                # actually moved. Two files with identical content share one
                # entry, and re-pointing it at whichever was seen last rewrote
                # the cache on every run. Found on the real library, where one
                # track is present twice under different names.
                resolved = str(p.resolve())
                if cached.path != resolved and not Path(cached.path).exists():
                    cached.path = resolved
                    write_sidecar(cached, cache_dir)
                if (
                    cached.lufs is None
                    or cached.played_peak_dbtp is None
                    or not cached.tempo_scores
                    or cached.intensity is None
                ):
                    # Upgraded from an older schema: measure what it lacks,
                    # keeping its tempo and grid exactly as they are.
                    try:
                        complete_measurements(cached, p)
                        write_sidecar(cached, cache_dir)
                        print(f"[{i}/{len(files)}] measured {p.name}")
                    except Exception as exc:  # the entry stays usable as it was
                        log.error("could not complete %s: %s", p, exc)
                results.append(cached)
                skipped += 1
                print(f"[{i}/{len(files)}] cached   {p.name}")
                continue
        try:
            previous = load_cached(tid, cache_dir) if force else None
            if previous is not None and previous.grid_manually_corrected:
                # A person fixed this grid; re-analysis measures everything
                # else against it and never replaces it.
                ta = analyze_file(p, grid=(previous.bpm, previous.first_downbeat))
                ta.hot_cues = previous.hot_cues or ta.hot_cues
                if previous.grid_confidence >= IMPORTED_GRID_CONFIDENCE:
                    # An imported grid stays ground truth.
                    ta.grid_confidence = IMPORTED_GRID_CONFIDENCE
            else:
                ta = analyze_file(p)
        except Exception as exc:  # a bad file must not abort the whole crate
            log.error("analysis failed for %s: %s", p, exc)
            print(f"[{i}/{len(files)}] FAILED   {p.name}: {exc}")
            continue
        write_sidecar(ta, cache_dir)
        results.append(ta)
        analyzed += 1
        print(
            f"[{i}/{len(files)}] analyzed {p.name}  "
            f"{ta.bpm:.1f} BPM  {ta.camelot} ({ta.key_name})  "
            f"conf={ta.grid_confidence:.2f}"
        )

    return results, analyzed, skipped
