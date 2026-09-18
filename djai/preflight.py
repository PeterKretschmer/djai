"""Pre-gig checks: find out in the dressing room, not in front of the room.

THREADING CONTEXT: main thread, offline. Nothing here runs while a set is
playing -- every check is allowed to be slow, and decoding the whole crate
deliberately is. Opens and immediately closes an audio stream, so it must not
be run against a device the engine already holds.

Every check reports a specific reason rather than a pass/fail bit, because the
useful output at 9pm is "track 41 of 120 will not decode: <name>", not "FAIL".
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from djai import config
from djai.analysis import TrackAnalysis, load_crate

#: Every check's name, in the order they run. Kept explicit so the CLI can
#: report "not reached" for checks after an aborted one.
CHECK_NAMES = ("devices", "crate", "grid", "decode", "ollama", "disk")


@dataclass
class Check:
    """One check's outcome. ``detail`` is the reason, and it is always useful."""

    name: str
    ok: bool
    detail: str
    #: Problems that do not fail the check but are worth saying out loud.
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        mark = "ok  " if self.ok else "FAIL"
        return f"[{mark}] {self.name:8s} {self.detail}"


@dataclass
class Report:
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def warnings(self) -> list[str]:
        return [w for c in self.checks for w in c.warnings]


# --- individual checks --------------------------------------------------------


def check_devices(blocksize: int = config.AUDIO_BLOCKSIZE) -> Check:
    """At least one output device that actually opens, not merely enumerates."""
    from djai.engine import _can_open, list_output_devices

    try:
        devices = list_output_devices()
    except Exception as exc:
        return Check("devices", False, f"could not enumerate audio devices: {exc}")

    if not devices:
        return Check(
            "devices", False,
            "no output device with 2+ channels; check that an audio interface "
            "is connected and not held exclusively by another program",
        )

    openable = []
    for index, name, api, channels in devices:
        if _can_open(index, blocksize):
            openable.append((index, name, api, channels))

    if not openable:
        listed = ", ".join(f"{i}:{n[:24]}" for i, n, _a, _c in devices[:4])
        return Check(
            "devices", False,
            f"{len(devices)} device(s) enumerate but none opens at blocksize "
            f"{blocksize} ({listed}) -- likely held exclusively by another program",
        )

    warnings = []
    multi = [d for d in openable if d[3] >= 4]
    if not multi:
        warnings.append(
            "no device has 4+ output channels, so single-device cue "
            "(--cue-channels) is unavailable; use --cue-device for a second card"
        )
    if len(openable) < 2 and not multi:
        warnings.append("only one usable output device: cue output will be off")

    names = ", ".join(f"{i}:{n[:28]} [{a}]" for i, n, a, _c in openable[:3])
    return Check(
        "devices", True,
        f"{len(openable)} of {len(devices)} device(s) open at blocksize "
        f"{blocksize} ({names})",
        warnings,
    )


def check_crate(cache_dir: Path) -> tuple[Check, list[TrackAnalysis]]:
    """The crate loads and is big enough to mix from."""
    try:
        crate = load_crate(Path(cache_dir))
    except Exception as exc:
        return Check("crate", False, f"could not read cache {cache_dir}: {exc}"), []

    if not crate:
        return (
            Check(
                "crate", False,
                f"no analysed tracks in {cache_dir}; run "
                f"`python -m djai analyze <folder>` first",
            ),
            [],
        )

    warnings = []
    if len(crate) < 8:
        warnings.append(
            f"only {len(crate)} tracks: the selector will repeat itself quickly"
        )
    missing = [t for t in crate if not Path(t.path).exists()]
    if missing:
        return (
            Check(
                "crate", False,
                f"{len(missing)} of {len(crate)} analysed tracks no longer exist "
                f"on disk, first: {missing[0].title}",
            ),
            crate,
        )
    return Check("crate", True, f"{len(crate)} analysed track(s)", warnings), crate


def check_grid(
    crate: list[TrackAnalysis], threshold: float | None = None
) -> Check:
    """Beat grids are trustworthy. Every transition is placed on these."""
    if threshold is None:
        threshold = config.PREFLIGHT_MIN_GRID_CONFIDENCE
    if not crate:
        return Check("grid", False, "no tracks to check")

    weak = sorted(
        (t for t in crate if t.grid_confidence < threshold),
        key=lambda t: t.grid_confidence,
    )
    if weak:
        worst = ", ".join(f"{t.title} ({t.grid_confidence:.2f})" for t in weak[:3])
        return Check(
            "grid", False,
            f"{len(weak)} of {len(crate)} track(s) below grid confidence "
            f"{threshold:.2f}: {worst}"
            + (f" and {len(weak) - 3} more" if len(weak) > 3 else ""),
        )

    lowest = min(crate, key=lambda t: t.grid_confidence)
    warnings = []
    estimated = [t for t in crate if getattr(t, "mix_points_estimated", False)]
    if estimated:
        warnings.append(
            f"{len(estimated)} track(s) have estimated rather than detected mix "
            f"points; their transitions are placed on a guess"
        )
    return Check(
        "grid", True,
        f"all {len(crate)} track(s) at or above {threshold:.2f} "
        f"(lowest {lowest.grid_confidence:.2f}, {lowest.title})",
        warnings,
    )


def check_decode(
    crate: list[TrackAnalysis],
    limit: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> Check:
    """Every track actually decodes. The slow check, and the one that saves you.

    A track that analysed months ago can be unplayable now: the file moved onto
    a disconnected drive, a codec changed, the file was re-encoded. Analysis
    reads a sidecar; only decoding proves the audio is there.
    """
    from djai.deck import load_track

    if not crate:
        return Check("decode", False, "no tracks to check")

    if limit is None:
        limit = config.PREFLIGHT_MAX_DECODE_CHECKS
    subject = crate if limit <= 0 else crate[:limit]

    failures: list[str] = []
    silent: list[str] = []
    for i, analysis in enumerate(subject, 1):
        if progress is not None:
            progress(i, len(subject), analysis.title)
        try:
            track = load_track(analysis)
        except Exception as exc:
            failures.append(f"{analysis.title}: {type(exc).__name__}: {exc}")
            continue
        if track.audio.shape[0] < 1:
            failures.append(f"{analysis.title}: decoded to zero frames")
        elif not float(abs(track.audio).max()) > 0.0:
            silent.append(analysis.title)

    if failures:
        shown = "; ".join(failures[:3])
        more = f" and {len(failures) - 3} more" if len(failures) > 3 else ""
        return Check(
            "decode", False,
            f"{len(failures)} of {len(subject)} track(s) will not decode: "
            f"{shown}{more}",
        )

    warnings = []
    if silent:
        warnings.append(f"{len(silent)} track(s) decode to silence: {silent[0]}")
    if limit > 0 and len(crate) > limit:
        warnings.append(
            f"only checked {limit} of {len(crate)} tracks "
            f"(PREFLIGHT_MAX_DECODE_CHECKS); set it to 0 to check them all"
        )
    return Check("decode", True, f"{len(subject)} track(s) decode", warnings)


def check_ollama(required: bool = True) -> Check:
    """The local model answers. Not fatal when the set will run --no-llm."""
    from djai.intent import IntentEngine

    engine = IntentEngine()
    try:
        ok, detail = engine.warmup()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        ok = False
    finally:
        try:
            engine.close()
        except Exception:
            pass

    if ok:
        return Check("ollama", True, detail)
    if not required:
        return Check(
            "ollama", True,
            f"not responding, and not required: {detail}",
            [f"the set will run rule-based only: {detail}"],
        )
    return Check(
        "ollama", False,
        f"{engine.model} at {engine.base_url} is not usable: {detail}. "
        f"Start Ollama, or run with --no-llm.",
    )


def check_disk(
    log_dir: Path,
    record_dir: Path | None = None,
    min_mb: int | None = None,
) -> Check:
    """Room for the log and, if recording, for the recording."""
    if min_mb is None:
        min_mb = config.PREFLIGHT_MIN_DISK_MB

    target = Path(log_dir)
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(str(probe))
    except Exception as exc:
        return Check("disk", False, f"could not check free space at {probe}: {exc}")

    free_mb = usage.free / (1024 * 1024)
    if free_mb < min_mb:
        return Check(
            "disk", False,
            f"{free_mb:,.0f} MB free at {probe}, below the {min_mb:,} MB minimum "
            f"(a recording is ~635 MB per hour)",
        )

    warnings = []
    try:
        target.mkdir(parents=True, exist_ok=True)
        stamp = target / ".djai_write_test"
        stamp.write_text("ok", encoding="utf-8")
        stamp.unlink()
    except Exception as exc:
        return Check("disk", False, f"cannot write to {target}: {exc}")

    if record_dir is not None:
        try:
            Path(record_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return Check("disk", False, f"cannot create {record_dir}: {exc}")
        hours = free_mb / 635.0
        if hours < 4:
            warnings.append(
                f"{free_mb:,.0f} MB free is about {hours:.1f} hours of recording"
            )

    return Check(
        "disk", True, f"{free_mb:,.0f} MB free at {probe} "
        f"(~{free_mb / 635.0:.0f}h of recording)", warnings
    )


# --- the whole run ------------------------------------------------------------


def run_preflight(
    cache_dir: Path,
    log_dir: Path,
    blocksize: int = config.AUDIO_BLOCKSIZE,
    require_ollama: bool = True,
    record_dir: Path | None = None,
    decode_limit: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> Report:
    """Run every check. Never raises: a crashed preflight is a failed preflight."""
    checks: list[Check] = [check_devices(blocksize)]

    crate_check, crate = check_crate(cache_dir)
    checks.append(crate_check)

    if crate:
        checks.append(check_grid(crate))
        checks.append(check_decode(crate, limit=decode_limit, progress=progress))
    else:
        checks.append(Check("grid", False, "not reached: the crate is empty"))
        checks.append(Check("decode", False, "not reached: the crate is empty"))

    checks.append(check_ollama(required=require_ollama))
    checks.append(check_disk(log_dir, record_dir=record_dir))
    return Report(checks)
