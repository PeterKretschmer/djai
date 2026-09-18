"""Disk housekeeping: recording retention and what `clean` may delete.

THREADING CONTEXT: main thread, or the control thread starting a recording.
Never the audio thread -- everything here lists or deletes files.

Only one thing here deletes on its own authority: the recording retention cap,
which counts nothing but the FLAC recordings this version writes. Everything
else is reported by :func:`find_reclaimable` and removed by :func:`delete` only
after the operator has confirmed, which is `python -m djai clean`'s job.

Deliberately never reported: audio files, reference sets, cache entries at a
usable schema version (even if their audio is on a drive that is not mounted
right now), and the current session's log.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

#: Session recordings the retention cap manages.
RECORDING_GLOB = "set_*.flac"
#: Recordings from before they were FLAC. Reported by `clean`, never auto-deleted.
LEGACY_RECORDING_GLOB = "set_*.wav"
#: Scratch folders `render` creates inside renders/. Must match
#: :data:`djai.cli.RENDER_TEMP_PREFIX`.
RENDER_TEMP_PREFIX = "_tmp_"
_ROTATED_LOG = re.compile(r"^session_\d{8}_\d{6}\.jsonl\.(\d+)\.gz$")


@dataclass(frozen=True)
class Reclaimable:
    """One file or folder `clean` would delete, and why."""

    path: Path
    size: int
    reason: str


def enforce_recording_retention(
    record_dir: Path, keep: int, protect: Path | None = None
) -> list[Path]:
    """Delete the oldest FLAC recordings beyond ``keep``. Returns what went.

    Oldest by name, which is the start timestamp, so touching a file does not
    move it in the queue. ``protect`` -- the recording that is just starting --
    is never deleted and counts toward ``keep``. WAV files are never touched.
    """
    record_dir = Path(record_dir)
    if keep < 1 or not record_dir.is_dir():
        return []
    found = sorted(record_dir.glob(RECORDING_GLOB), key=lambda p: p.name)
    protected = Path(protect).resolve() if protect is not None else None
    excess = len(found) - keep
    removed: list[Path] = []
    for p in found:
        if excess <= 0:
            break
        if protected is not None and p.resolve() == protected:
            continue
        try:
            p.unlink()
        except OSError:
            continue
        removed.append(p)
        excess -= 1
    return removed


def _size(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return path.stat().st_size
    except OSError:
        return 0


def _sidecar_version(path: Path) -> int | None:
    """Just the schema version of a JSON sidecar. Nothing else in it is read."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    version = data.get("analysis_version") if isinstance(data, dict) else None
    return version if isinstance(version, int) else None


def find_reclaimable(
    renders_dir: Path,
    record_dir: Path,
    log_dir: Path,
    cache_dir: Path,
    keep_recordings: int,
    keep_logs: int,
) -> list[Reclaimable]:
    """Everything `clean` would delete, largest categories first in the listing."""
    from djai import analysis as an

    items: list[Reclaimable] = []

    renders_dir = Path(renders_dir)
    if renders_dir.is_dir():
        for p in sorted(renders_dir.iterdir()):
            reason = (
                "scratch folder a render left behind"
                if p.is_dir() and p.name.startswith(RENDER_TEMP_PREFIX)
                else "offline render"
            )
            items.append(Reclaimable(p, _size(p), reason))

    record_dir = Path(record_dir)
    if record_dir.is_dir():
        flacs = sorted(record_dir.glob(RECORDING_GLOB), key=lambda p: p.name)
        for p in flacs[: max(0, len(flacs) - keep_recordings)]:
            items.append(
                Reclaimable(p, _size(p), f"recording older than the newest {keep_recordings}")
            )
        for p in sorted(record_dir.glob(LEGACY_RECORDING_GLOB), key=lambda p: p.name):
            items.append(
                Reclaimable(p, _size(p), "WAV recording from before recordings were FLAC")
            )

    log_dir = Path(log_dir)
    if log_dir.is_dir():
        for p in sorted(log_dir.iterdir()):
            match = _ROTATED_LOG.match(p.name)
            if match and int(match.group(1)) > keep_logs:
                items.append(
                    Reclaimable(p, _size(p), f"rotated log beyond the newest {keep_logs}")
                )

    cache_dir = Path(cache_dir)
    if cache_dir.is_dir():
        usable = {an.ANALYSIS_VERSION, *an.UPGRADABLE_FROM}
        for p in sorted(cache_dir.glob("*.json")):
            version = _sidecar_version(p)
            if version in usable:
                continue
            why = (
                "unreadable cache sidecar"
                if version is None
                else f"cache sidecar from schema v{version}, too old to use"
            )
            items.append(Reclaimable(p, _size(p), why))
            npz = p.with_suffix(".npz")
            if npz.exists():
                items.append(Reclaimable(npz, _size(npz), "array file of that sidecar"))
        for npz in sorted(cache_dir.glob("*.npz")):
            if not npz.with_suffix(".json").exists():
                items.append(Reclaimable(npz, _size(npz), "array file with no sidecar"))

        # Separated stems are the largest thing in the cache -- about 20 MB a
        # minute -- and a folder whose track is gone is dead weight. Stems of a
        # track still in the crate are never offered: re-separating one is
        # minutes of GPU time, not a download.
        stems_root = cache_dir / "stems"
        if stems_root.is_dir():
            for folder in sorted(stems_root.iterdir()):
                if not folder.is_dir() or (cache_dir / f"{folder.name}.json").exists():
                    continue
                items.append(
                    Reclaimable(
                        folder, _size(folder),
                        "separated stems for a track no longer in the cache",
                    )
                )

    return items


def delete(items: list[Reclaimable]) -> tuple[int, int, list[tuple[Path, str]]]:
    """Delete confirmed items. Returns ``(deleted, bytes_freed, failures)``."""
    deleted = 0
    freed = 0
    failed: list[tuple[Path, str]] = []
    for item in items:
        try:
            if item.path.is_dir():
                shutil.rmtree(item.path)
            elif item.path.exists():
                item.path.unlink()
            else:
                continue
        except OSError as exc:
            failed.append((item.path, str(exc)))
            continue
        deleted += 1
        freed += item.size
    return deleted, freed, failed
