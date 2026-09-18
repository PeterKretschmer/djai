"""Import beat grids, cues and playlists from Rekordbox and Serato.

THREADING CONTEXT: main thread (the ``import`` subcommand). Reads XML, crate
files and audio-file tags, and writes the analysis cache. Never runs during
playback.

Standard library only. Rekordbox's XML export is plain XML. Serato keeps
playlists in binary ``.crate`` files and each track's grid and cues inside its
own audio file, as ID3v2 GEOB frames ("Serato BeatGrid", "Serato Markers2");
both formats are small enough to read directly.

What an import does to the cache:

* An imported grid replaces the analysed one and is marked
  ``grid_manually_corrected`` -- a person made it, in another program -- so
  ``analyze --force`` lays everything else over it and never replaces it.
  It is ground truth: its ``grid_confidence`` is
  :data:`djai.analysis.IMPORTED_GRID_CONFIDENCE`, kept through re-analysis.
* Imported hot cues replace any cue at the same index; other cues are kept.
* Playlists are written to ``<cache>/library/playlists.json``, a subfolder so
  neither the crate loader nor ``clean`` mistakes it for a track entry.

Every file gets one summary row saying what happened to it.
"""

from __future__ import annotations

import base64
import json
import re
import struct
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from djai import analysis as an

#: Where imported playlists are kept, relative to the cache folder.
PLAYLISTS_FILE = Path("library") / "playlists.json"

#: Rekordbox POSITION_MARK types.
_RB_CUE, _RB_LOOP = 0, 4


@dataclass
class ImportedTrack:
    """One track as another program describes it."""

    path: Path
    title: str = ""
    artist: str = ""
    #: Constant grid: tempo and the time of a downbeat, in seconds. None when
    #: the source had no grid for this track.
    bpm: float | None = None
    first_downbeat_s: float | None = None
    #: The source's grid changed tempo; only its first tempo is used.
    variable_tempo: bool = False
    #: ``(index, seconds, label)``, index 1-based.
    hot_cues: list[tuple[int, float, str]] = field(default_factory=list)
    memory_cues_skipped: int = 0
    playlists: list[str] = field(default_factory=list)


@dataclass
class FileSummary:
    """What happened to one file."""

    path: str
    status: str  # imported | updated | missing | not analysed | failed
    bpm: float | None = None
    first_downbeat_s: float | None = None
    cues: int = 0
    memory_cues_skipped: int = 0
    playlists: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        grid = (
            f"{self.bpm:.2f} BPM, bar 1 at {self.first_downbeat_s:.3f}s"
            if self.bpm and self.first_downbeat_s is not None
            else "no grid"
        )
        extra = f" [{'; '.join(self.notes)}]" if self.notes else ""
        lists = f" in {', '.join(self.playlists)}" if self.playlists else ""
        return (
            f"{self.status:12s} {Path(self.path).name} | {grid} | {self.cues} cue(s)"
            f"{f', {self.memory_cues_skipped} memory cue(s) skipped' if self.memory_cues_skipped else ''}"
            f"{lists}{extra}"
        )


# --- paths ------------------------------------------------------------------


def relocate(path: Path, rules: list[tuple[str, str]]) -> Path:
    """Apply the first matching ``OLD=NEW`` prefix rewrite, for moved libraries."""
    text = str(path).replace("\\", "/")
    for old, new in rules:
        old_n = old.replace("\\", "/").rstrip("/")
        if text.lower().startswith(old_n.lower()):
            return Path(new.rstrip("/\\") + text[len(old_n):])
    return path


def rekordbox_location(location: str) -> Path:
    """``file://localhost/C:/Music/My%20Track.mp3`` -> ``C:/Music/My Track.mp3``."""
    parsed = urllib.parse.urlparse(location)
    path = urllib.parse.unquote(parsed.path)
    if parsed.netloc and parsed.netloc != "localhost":
        path = f"//{parsed.netloc}{path}"  # a network share
    if re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    return Path(path)


# --- Rekordbox ------------------------------------------------------------------


def parse_rekordbox_xml(xml_path: Path) -> list[ImportedTrack]:
    """Every track in a Rekordbox XML export, with its playlists attached."""
    root = ET.parse(str(xml_path)).getroot()
    by_id: dict[str, ImportedTrack] = {}
    by_location: dict[str, ImportedTrack] = {}
    collection = root.find("COLLECTION")
    for node in (collection if collection is not None else []):
        if node.tag != "TRACK":
            continue
        location = node.get("Location", "")
        if not location:
            continue
        track = ImportedTrack(
            path=rekordbox_location(location),
            title=node.get("Name", ""),
            artist=node.get("Artist", ""),
        )
        tempos = node.findall("TEMPO")
        if tempos:
            first = tempos[0]
            bpm = float(first.get("Bpm", "0") or 0)
            start = float(first.get("Inizio", "0") or 0)
            # Battito is the beat number of this marker within its bar; 1 is a
            # downbeat. The next downbeat is (5 - Battito) % 4 beats later.
            beat_in_bar = int(float(first.get("Battito", "1") or 1))
            if bpm > 0:
                track.bpm = bpm
                track.first_downbeat_s = start + ((5 - beat_in_bar) % 4) * 60.0 / bpm
            track.variable_tempo = any(
                abs(float(t.get("Bpm", "0") or 0) - bpm) > 1e-3 for t in tempos[1:]
            )
        for mark in node.findall("POSITION_MARK"):
            num = int(float(mark.get("Num", "-1") or -1))
            kind = int(float(mark.get("Type", "0") or 0))
            if num < 0:
                track.memory_cues_skipped += 1
                continue
            if kind not in (_RB_CUE, _RB_LOOP) or num >= an.MAX_HOT_CUES:
                track.memory_cues_skipped += 1
                continue
            name = mark.get("Name", "") or ("loop" if kind == _RB_LOOP else f"cue {num + 1}")
            track.hot_cues.append((num + 1, float(mark.get("Start", "0") or 0), _label(name)))
        tid = node.get("TrackID", "")
        if tid:
            by_id[tid] = track
        by_location[location] = track

    playlists = root.find("PLAYLISTS")
    if playlists is not None:
        _walk_rekordbox_playlists(playlists, [], by_id, by_location)
    return list(by_id.values()) or list(by_location.values())


def _walk_rekordbox_playlists(node, trail, by_id, by_location) -> None:
    for child in node:
        if child.tag != "NODE":
            continue
        name = child.get("Name", "")
        if child.get("Type") == "1":
            full = " / ".join(trail + [name])
            by_key = by_location if child.get("KeyType") == "1" else by_id
            for entry in child.findall("TRACK"):
                track = by_key.get(entry.get("Key", ""))
                if track is not None and full not in track.playlists:
                    track.playlists.append(full)
        else:
            nested = trail if name == "ROOT" else trail + [name]
            _walk_rekordbox_playlists(child, nested, by_id, by_location)


def _label(name: str) -> str:
    """Cue names as djai uses them: a cue called "Drop" is a drop."""
    stripped = name.strip()
    return "drop" if stripped.lower() == "drop" else stripped


# --- Serato -----------------------------------------------------------------------


def parse_serato_crate(crate_path: Path, root: Path) -> tuple[str, list[Path]]:
    """``(crate name, track paths)`` from a Serato ``.crate`` file.

    Serato stores each path relative to the root of the drive its _Serato_
    folder is on, so ``root`` is that drive (or wherever the library now is).
    """
    data = Path(crate_path).read_bytes()
    name = Path(crate_path).stem.replace("%%", " / ")
    paths: list[Path] = []
    for tag, payload in _tlv(data):
        if tag != "otrk":
            continue
        for inner, value in _tlv(payload):
            if inner == "ptrk":
                rel = value.decode("utf-16-be").strip("\x00")
                paths.append(Path(root) / rel)
    return name, paths


def _tlv(data: bytes):
    """Serato's container format: 4-byte tag, 4-byte big-endian length, payload."""
    i = 0
    while i + 8 <= len(data):
        tag = data[i:i + 4].decode("latin-1")
        (length,) = struct.unpack(">I", data[i + 4:i + 8])
        yield tag, data[i + 8:i + 8 + length]
        i += 8 + length


def read_id3_geob(audio_path: Path) -> dict[str, bytes]:
    """Every GEOB frame in a file's ID3v2 tag, by description. Empty if none."""
    with Path(audio_path).open("rb") as fh:
        header = fh.read(10)
        if len(header) < 10 or header[:3] != b"ID3":
            return {}
        major = header[3]
        size = _syncsafe(header[6:10])
        body = fh.read(size)
    frames: dict[str, bytes] = {}
    i = 0
    while i + 10 <= len(body):
        frame_id = body[i:i + 4]
        if frame_id == b"\x00\x00\x00\x00":
            break
        raw_size = body[i + 4:i + 8]
        frame_size = _syncsafe(raw_size) if major >= 4 else struct.unpack(">I", raw_size)[0]
        payload = body[i + 10:i + 10 + frame_size]
        i += 10 + frame_size
        if frame_id != b"GEOB" or not payload:
            continue
        encoding = payload[0]
        term = b"\x00\x00" if encoding in (1, 2) else b"\x00"
        rest = payload[1:]
        rest = rest[rest.index(b"\x00") + 1:]            # MIME type, always latin-1
        filename_end = _terminator(rest, term)
        rest = rest[filename_end + len(term):]
        desc_end = _terminator(rest, term)
        description = rest[:desc_end].decode(
            "utf-16" if encoding == 1 else "utf-16-be" if encoding == 2
            else "utf-8" if encoding == 3 else "latin-1", "replace",
        )
        frames[description] = rest[desc_end + len(term):]
    return frames


def _terminator(data: bytes, term: bytes) -> int:
    step = len(term)
    for j in range(0, len(data) - step + 1, step):
        if data[j:j + step] == term:
            return j
    return len(data)


def _syncsafe(four: bytes) -> int:
    return (four[0] << 21) | (four[1] << 14) | (four[2] << 7) | four[3]


def parse_serato_beatgrid(data: bytes) -> tuple[float | None, float | None, bool]:
    """``(bpm, first beat seconds, variable)`` from a "Serato BeatGrid" payload."""
    if len(data) < 6 or data[:2] != b"\x01\x00":
        return None, None, False
    (count,) = struct.unpack(">I", data[2:6])
    if count == 0:
        return None, None, False
    i = 6
    first = None
    for _ in range(count - 1):
        position, _beats = struct.unpack(">fI", data[i:i + 8])
        first = position if first is None else first
        i += 8
    position, bpm = struct.unpack(">ff", data[i:i + 8])
    first = position if first is None else first
    return float(bpm), float(first), count > 1


def parse_serato_markers2(data: bytes) -> list[tuple[int, float, str]]:
    """Hot cues ``(index, seconds, label)`` from a "Serato Markers2" payload."""
    if len(data) < 2 or data[:2] != b"\x01\x01":
        return []
    text = re.sub(rb"[^A-Za-z0-9+/]", b"", data[2:].split(b"\x00", 1)[0])
    try:
        decoded = base64.b64decode(text + b"=" * (-len(text) % 4))
    except (ValueError, TypeError):
        return []
    cues: list[tuple[int, float, str]] = []
    i = 2 if decoded[:2] == b"\x01\x01" else 0
    while i < len(decoded):
        end = decoded.find(b"\x00", i)
        if end <= i:
            break
        entry = decoded[i:end].decode("latin-1")
        if end + 5 > len(decoded):
            break
        (length,) = struct.unpack(">I", decoded[end + 1:end + 5])
        payload = decoded[end + 5:end + 5 + length]
        i = end + 5 + length
        if entry != "CUE" or len(payload) < 12:
            continue
        index = payload[1]
        (position_ms,) = struct.unpack(">I", payload[2:6])
        name = payload[12:].split(b"\x00", 1)[0].decode("utf-8", "replace")
        if index < an.MAX_HOT_CUES:
            cues.append((index + 1, position_ms / 1000.0, _label(name) or f"cue {index + 1}"))
    return cues


def parse_serato(serato_path: Path, root: Path | None = None) -> list[ImportedTrack]:
    """Tracks from a ``_Serato_`` folder (every crate in it) or one ``.crate``."""
    serato_path = Path(serato_path)
    crates = (
        [serato_path] if serato_path.suffix.lower() == ".crate"
        else sorted((serato_path / "Subcrates").glob("*.crate"))
    )
    if root is None:
        base = serato_path.parent if serato_path.suffix.lower() == ".crate" else serato_path
        root = Path(base.anchor or "/")
    tracks: dict[str, ImportedTrack] = {}
    for crate in crates:
        name, paths = parse_serato_crate(crate, root)
        for path in paths:
            key = str(path).lower()
            track = tracks.get(key)
            if track is None:
                track = tracks[key] = ImportedTrack(path=path, title=path.stem)
            if name not in track.playlists:
                track.playlists.append(name)
    for track in tracks.values():
        if not track.path.exists():
            continue
        frames = read_id3_geob(track.path)
        if "Serato BeatGrid" in frames:
            bpm, first, variable = parse_serato_beatgrid(frames["Serato BeatGrid"])
            # Serato's first grid marker sits on a downbeat.
            track.bpm, track.first_downbeat_s, track.variable_tempo = bpm, first, variable
        if "Serato Markers2" in frames:
            track.hot_cues = parse_serato_markers2(frames["Serato Markers2"])
    return list(tracks.values())


# --- applying an import -------------------------------------------------------------


def apply_import(
    tracks: list[ImportedTrack],
    cache_dir: Path,
    analyze_missing: bool = True,
    relocate_rules: list[tuple[str, str]] | None = None,
) -> list[FileSummary]:
    """Write imported grids, cues and playlists into the analysis cache."""
    cache_dir = Path(cache_dir)
    summaries: list[FileSummary] = []
    playlists: dict[str, list[str]] = {}
    for track in tracks:
        path = relocate(track.path, relocate_rules or [])
        summary = FileSummary(
            path=str(path), status="missing", bpm=track.bpm,
            first_downbeat_s=track.first_downbeat_s,
            memory_cues_skipped=track.memory_cues_skipped,
            playlists=list(track.playlists),
        )
        summaries.append(summary)
        if track.variable_tempo:
            summary.notes.append("variable tempo in source: its first tempo is used")
        if not path.exists():
            summary.notes.append("audio file not found")
            continue
        try:
            has_grid = bool(track.bpm) and track.first_downbeat_s is not None
            tid = an.track_hash(path)
            ta = an.load_cached(tid, cache_dir)
            if ta is None:
                if not analyze_missing:
                    summary.status = "not analysed"
                    summary.notes.append("not in the cache; run without --no-analyze")
                    continue
                ta = (
                    an.analyze_file(path, grid=(float(track.bpm), float(track.first_downbeat_s)))
                    if has_grid else an.analyze_file(path)
                )
                summary.status = "imported"
            else:
                if has_grid:
                    an.regrid(ta, float(track.bpm), float(track.first_downbeat_s))
                summary.status = "updated"
            if track.hot_cues:
                by_index = {c["index"]: c for c in ta.hot_cues}
                for index, seconds, label in track.hot_cues:
                    by_index[index] = an.make_hot_cue(index, seconds, label)
                ta.hot_cues = [by_index[i] for i in sorted(by_index)]
            summary.cues = len(track.hot_cues)
            if not has_grid:
                summary.notes.append("no grid in source: analysed grid kept")
            else:
                # Ground truth: it overrides the detected grid (above) and is
                # trusted as fully as a grid can be, whatever onset agreement a
                # detector would have measured for it.
                ta.grid_confidence = an.IMPORTED_GRID_CONFIDENCE
                summary.bpm, summary.first_downbeat_s = ta.bpm, ta.first_downbeat
            an.write_sidecar(ta, cache_dir)
            for name in track.playlists:
                playlists.setdefault(name, []).append(ta.track_id)
        except Exception as exc:  # one bad file must not stop the import
            summary.status = "failed"
            summary.notes.append(f"{type(exc).__name__}: {exc}")
    if playlists:
        write_playlists(playlists, cache_dir)
    return summaries


def write_playlists(playlists: dict[str, list[str]], cache_dir: Path) -> Path:
    """Merge into ``<cache>/library/playlists.json``: name -> track ids, in order."""
    path = Path(cache_dir) / PLAYLISTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_playlists(cache_dir)
    for name, ids in playlists.items():
        existing[name] = list(dict.fromkeys(ids))
    path.write_text(json.dumps(existing, indent=1), encoding="utf-8")
    return path


def load_playlists(cache_dir: Path) -> dict[str, list[str]]:
    path = Path(cache_dir) / PLAYLISTS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
