"""What the operator thought, and what the weights should be because of it.

THREADING CONTEXT: control threads only -- the UI worker that records a
verdict, and the session start that reads the logs. Never the audio thread, and
never in the cue path: learning reads files.

Three kinds of evidence, all already in the session logs:

* **verdicts** -- the operator pressing "that worked" or "that didn't" after a
  transition. The only signal that is actually about how it sounded.
* **overrides and skips** -- the operator loading something else, forcing a
  track, or dropping what was queued. Not a verdict on the sound, but a clear
  one on the *choice*: the selector picked something the operator would not.
* **previews** -- what the critic measured, and which measure was furthest
  outside the reference range at the time.

What is learned from them is deliberately small: per-measure weights for the
critic, and per-term weights for the selector's similarities. Nothing here
changes what a transition does, only what the search is trying to minimise and
what the ranking prefers -- so the worst a bad night of feedback can do is
re-rank candidates, never arm something unsafe.

Phase 4 (SPEC §5, §8) added four things on top:

* **labels** beyond the two verdicts -- "too early", "too late", "more like
  this", "less like this" -- matched from plain words by :func:`parse_phrase`,
  never by the model, and written with the same context as a verdict.
* **preferences** from suggestions: which of the tracks shown the operator
  cued, against the ones they passed over (:func:`choices`). A pairwise
  comparison of the ranking's own reason terms moves each term's multiplier.
* **stable and night layers.** The stable model is learned from the logs and
  kept as numbered versions under ``<log_dir>/models`` with a pointer to the
  current one, so a bad update rolls back exactly (:func:`train`,
  :func:`rollback`). The night layer is learned from tonight's log alone, in a
  tighter range, and never written into the stable model; the two are
  multiplied at use (:func:`effective`).
* **style conditioning**: every event carries the persona it happened under,
  and a model is learned per persona beside the global one.

The room microphone (djai.room) writes ``room_reading`` lines but is not
evidence here yet: with no mic in use there is nothing to learn from, and a
learner fitted to a signal nobody has measured would be a claim, not a model.

Weights move only with :data:`MIN_VERDICTS` behind them and are clamped to
:data:`WEIGHT_RANGE`, because the room is not a training set: an operator
pressing "that didn't" twice should tilt the next pick, not rewrite the
program's taste.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

#: The verdict values the UI can send.
WORKED = "worked"
DIDNT = "didnt"
#: Labels about timing and taste (Phase 4). Written as verdicts so every
#: reader of ``operator_feedback`` sees them, but only "worked" and "didnt"
#: count toward :data:`MIN_VERDICTS` and the critic.
TOO_EARLY = "too_early"
TOO_LATE = "too_late"
MORE_LIKE_THIS = "more_like_this"
LESS_LIKE_THIS = "less_like_this"
VERDICTS = (WORKED, DIDNT, TOO_EARLY, TOO_LATE, MORE_LIKE_THIS, LESS_LIKE_THIS)

#: Plain words for each label, matched against the whole line so a song
#: request that happens to contain "more like" is never taken for one.
_PHRASES: tuple[tuple[str, str], ...] = (
    (r"(that )?(worked|works|was (good|great|nice))", WORKED),
    (r"(that )?(didn'?t( work)?|did not( work)?|was (bad|rough|messy))", DIDNT),
    (r"(that was |that's |thats |it was |a bit |way )?too early", TOO_EARLY),
    (r"(that was |that's |thats |it was |a bit |way )?too late", TOO_LATE),
    (r"more (like|of) (this|that)", MORE_LIKE_THIS),
    (r"(less|fewer|not) like (this|that)|less of (this|that)", LESS_LIKE_THIS),
)

#: Session-log events this reads. Everything else is ignored, and every one of
#: these existed before this module did -- old logs still teach it.
EVENTS = (
    "operator_feedback", "preview", "override", "commands_skipped",
    "forced_next", "transition_complete", "manual_control",
    # Phase 4: what was suggested, and what the operator cued.
    "suggestions_shown", "cue_added",
)

#: How many verdicts must exist before any weight moves at all.
MIN_VERDICTS: int = 5

#: How far a weight may travel from 1.0, as a multiplier either way.
WEIGHT_RANGE: tuple[float, float] = (0.25, 4.0)

#: How much one verdict moves the measure it blames, before clamping.
STEP: float = 0.25

#: Logs older than this many files back are not read: a crate and a room change,
#: and last month's taste is not evidence about tonight.
MAX_LOG_FILES: int = 20


def record(
    session_log: Any,
    verdict: str,
    *,
    tracks: tuple[str | None, str | None] = (None, None),
    style: str = "",
    critic_score: float | None = None,
    worst_measure: str = "",
    similarity: dict | None = None,
    note: str = "",
    persona: str = "default",
    context: dict | None = None,
) -> dict:
    """Write one operator verdict to the session log. Returns the record.

    Everything the learner will need is written *with* the verdict: which
    transition, which style, what the critic measured and how alike the two
    tracks were. A verdict that has to be joined back to another line later is
    a verdict that gets lost.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    fields = {
        "verdict": verdict,
        "track_pair": [tracks[0], tracks[1]],
        "transition_style": style,
        "critic_score": critic_score,
        "worst_measure": worst_measure,
        "similarity": dict(similarity or {}),
        "trigger": f"operator: {verdict}",
        "action": note or {WORKED: "kept", DIDNT: "marked as not working"}.get(
            verdict, f"noted: {verdict.replace('_', ' ')}"),
        "persona": persona,
        **(context or {}),
    }
    if session_log is not None:
        session_log.write("operator_feedback", **fields)
    return fields


def read_events(log_dir: Path, max_files: int = MAX_LOG_FILES) -> list[dict]:
    """Every event this module cares about, oldest first, across recent logs."""
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []
    files = sorted(log_dir.glob("session_*.jsonl"))[-max_files:]
    out: list[dict] = []
    for path in files:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("event") in EVENTS:
                    out.append(row)
        except OSError as exc:
            log.warning("could not read %s: %s", path, exc)
    return out


def summarise(events: Iterable[dict]) -> dict[str, Any]:
    """Count what the logs say, without interpreting it yet."""
    events = list(events)
    labels = [e for e in events if e.get("event") == "operator_feedback"]
    verdicts = [e for e in labels if e.get("verdict") in (WORKED, DIDNT)]
    worked = [e for e in verdicts if e.get("verdict") == WORKED]
    didnt = [e for e in verdicts if e.get("verdict") == DIDNT]
    overrides = [
        e for e in events
        if e.get("event") in ("override", "forced_next", "commands_skipped")
    ]
    previews = [e for e in events if e.get("event") == "preview"]
    blamed: dict[str, int] = defaultdict(int)
    for e in didnt:
        measure = e.get("worst_measure") or ""
        if measure:
            blamed[measure] += 1
    label_counts: dict[str, int] = defaultdict(int)
    for e in labels:
        if e.get("verdict") not in (WORKED, DIDNT):
            label_counts[str(e.get("verdict"))] += 1
    picks = choices(events)
    return {
        "verdicts": len(verdicts),
        "worked": len(worked),
        "didnt": len(didnt),
        "labels": dict(label_counts),
        "suggestion_choices": len(picks),
        "suggestions_ignored": ignored(events),
        "overrides": len(overrides),
        "previews": len(previews),
        "blamed": dict(blamed),
        "worked_similarity": _mean_similarity(worked),
        "didnt_similarity": _mean_similarity(didnt),
    }


def _mean_similarity(events: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for e in events:
        for key, value in (e.get("similarity") or {}).items():
            if isinstance(value, (int, float)):
                totals[key] += float(value)
                counts[key] += 1
    return {k: round(totals[k] / counts[k], 4) for k in totals if counts[k]}


def _clamp(value: float, bounds: tuple[float, float] | None = None) -> float:
    lo, hi = bounds or WEIGHT_RANGE
    return round(min(max(value, lo), hi), 3)


#: Suggestion choices needed before any ranking term moves, and how hard the
#: pairwise difference pulls: reason terms run 0..1, so a chosen track that is
#: on average 0.25 better on a term than the ones passed over lifts its weight
#: by half.
MIN_CHOICES: int = 5
PREF_GAIN: float = 2.0

#: The ranking's reason terms a preference can move. The first seven are
#: djai.selector's; plan fit is djai.songs'.
PREFERENCE_TERMS: tuple[str, ...] = (
    "bpm", "key", "energy", "confidence", "timbre", "groove", "neighbourhood",
    "plan_fit",
)
#: The similarity terms "more like this" and "less like this" move.
SIMILARITY_TERMS: tuple[str, ...] = ("timbre", "groove", "neighbourhood")
LIKE_STEP: float = 0.25

#: Timing labels needed before the blend length moves, the bars one net label
#: is worth, and how far it may go either way. A "too early" shortens the
#: blend -- which, planned backwards from mix-out, starts it later.
MIN_TIMING: int = 2
TIMING_STEP_BARS: int = 4
TIMING_MAX_BARS: int = 8


def choices(events: Iterable[dict]) -> list[tuple[dict, list[dict]]]:
    """Each time the operator cued one of the tracks shown to them: the chosen
    row's reason terms, and the other rows'. Oldest first.

    A list of suggestions stands until the next one replaces it; a cue of a
    track on it is a choice among its rows, and consumes it.
    """
    out: list[tuple[dict, list[dict]]] = []
    shown: dict[str, dict] | None = None
    for e in events:
        kind = e.get("event")
        if kind == "suggestions_shown":
            shown = {r.get("track_id"): (r.get("terms") or {}) for r in e.get("rows") or []
                     if r.get("track_id")}
        elif kind == "cue_added" and shown:
            tid = e.get("track_id")
            if tid in shown and len(shown) > 1:
                out.append((shown[tid], [t for k, t in shown.items() if k != tid]))
                shown = None
    return out


def ignored(events: Iterable[dict]) -> int:
    """Lists of suggestions replaced, or outlived by a hand-over, with nothing
    on them cued. Counted, not learned from: passing over all five is not a
    judgement on any one of them."""
    count, open_list = 0, False
    for e in events:
        kind = e.get("event")
        if kind in ("suggestions_shown", "transition_complete"):
            count += open_list
            open_list = kind == "suggestions_shown"
        elif kind == "cue_added":
            open_list = False
    return count


def preference_weights(
    picks: list[tuple[dict, list[dict]]], bounds: tuple[float, float] | None = None,
    min_choices: int = MIN_CHOICES,
) -> dict[str, float]:
    """Per-term multipliers from pairwise choices: chosen minus the mean of the
    rest, averaged over every choice, scaled by :data:`PREF_GAIN`."""
    if len(picks) < min_choices:
        return {}
    diffs: dict[str, list[float]] = defaultdict(list)
    for chosen, others in picks:
        for key in PREFERENCE_TERMS:
            vals = [float(o[key]) for o in others if isinstance(o.get(key), (int, float))]
            if isinstance(chosen.get(key), (int, float)) and vals:
                diffs[key].append(float(chosen[key]) - sum(vals) / len(vals))
    return {k: _clamp(1.0 + PREF_GAIN * sum(v) / len(v), bounds)
            for k, v in diffs.items() if v}


def learn(
    events: Iterable[dict],
    *,
    min_verdicts: int = MIN_VERDICTS,
    min_choices: int = MIN_CHOICES,
    min_timing: int = MIN_TIMING,
    bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Weights for the critic and the selector, from what the logs hold.

    Critic: a transition the operator rejected blames whichever measure was
    furthest outside the reference range at the time, and that measure's weight
    rises -- the search will work harder on it next time. A transition they
    kept does the opposite, gently.

    Selector: compares how alike the pairs were across kept and rejected
    transitions. If the ones that worked were more alike in timbre than the
    ones that did not, timbre matters here and its weight rises. Overrides and
    skips count as rejections of the *choice*, which is the same question. On
    top: the pairwise preference from suggestion choices, and "more/less like
    this" on the similarity terms. The factors multiply, then are clamped.

    Timing: net "too late" minus "too early", in bars of blend length.

    Every weight is a multiplier on the default. Returns ``{"critic",
    "selector", "timing_bias_bars", "summary"}``; each part stays empty (or 0)
    until its own evidence threshold is met.
    """
    events = list(events)
    bounds = bounds or WEIGHT_RANGE
    summary = summarise(events)
    out: dict[str, Any] = {"critic": {}, "selector": {}, "timing_bias_bars": 0,
                           "summary": summary}
    notes: list[str] = []

    selector_weights: dict[str, float] = {}
    if summary["verdicts"] < min_verdicts:
        notes.append(f"{summary['verdicts']} verdict(s); {min_verdicts} needed before "
                     "any weight moves")
    else:
        critic_weights: dict[str, float] = {}
        for measure, blamed_count in summary["blamed"].items():
            critic_weights[measure] = _clamp(1.0 + STEP * blamed_count, bounds)
        # A measure nobody blamed across a decent number of kept transitions is
        # evidently not what goes wrong in this room.
        kept = summary["worked"]
        if kept >= min_verdicts:
            for event in events:
                if event.get("event") != "preview":
                    continue
                for entry in event.get("critic") or []:
                    measure = entry.get("worst")
                    if measure and measure not in summary["blamed"]:
                        critic_weights.setdefault(measure, _clamp(1.0 - STEP / 2, bounds))
        out["critic"] = critic_weights

        worked_sim = summary["worked_similarity"]
        didnt_sim = summary["didnt_similarity"]
        for key in set(worked_sim) | set(didnt_sim):
            if key not in worked_sim or key not in didnt_sim:
                continue
            gap = worked_sim[key] - didnt_sim[key]
            # A term that separates the two is a term worth weighting. The scale
            # is deliberately gentle: similarity runs 0..1, so a 0.2 gap is large.
            selector_weights[key] = _clamp(1.0 + 2.0 * gap, bounds)
        notes.append(
            f"{summary['verdicts']} verdict(s) ({summary['worked']} worked, "
            f"{summary['didnt']} did not), {summary['overrides']} override(s)"
        )

    prefs = preference_weights(choices(events), bounds, min_choices)
    for key, value in prefs.items():
        selector_weights[key] = _clamp(selector_weights.get(key, 1.0) * value, bounds)
    if summary["suggestion_choices"]:
        notes.append(f"{summary['suggestion_choices']} suggestion choice(s)"
                     + ("" if prefs else f", {min_choices} needed"))

    labels = summary["labels"]
    like = labels.get(MORE_LIKE_THIS, 0) - labels.get(LESS_LIKE_THIS, 0)
    if like:
        factor = 1.0 + LIKE_STEP * like
        for key in SIMILARITY_TERMS:
            selector_weights[key] = _clamp(selector_weights.get(key, 1.0) * factor, bounds)
        notes.append(f"like this {like:+d}")
    out["selector"] = selector_weights

    early, late = labels.get(TOO_EARLY, 0), labels.get(TOO_LATE, 0)
    if early + late >= min_timing:
        bias = TIMING_STEP_BARS * (late - early)
        out["timing_bias_bars"] = int(max(-TIMING_MAX_BARS, min(TIMING_MAX_BARS, bias)))
        notes.append(f"blend length {out['timing_bias_bars']:+d} bars "
                     f"({early} too early, {late} too late)")
    summary["learning"] = "; ".join(notes)
    return out


def weights_for(log_dir: Path) -> dict[str, dict[str, float]]:
    """Read the logs and learn from them. Safe to call at session start."""
    try:
        return learn(read_events(log_dir))
    except Exception as exc:  # noqa: BLE001 - never let learning stop a set
        log.warning("could not learn from %s: %s", log_dir, exc)
        return {"critic": {}, "selector": {}, "summary": {"error": str(exc)}}


def parse_phrase(text: str) -> str | None:
    """The label a plain line of chat means, or None. Deterministic.

    The whole line has to match -- "more like this" is feedback, "play more
    like this one by Calvin Harris" is not -- so a song request is never
    swallowed as a verdict.
    """
    t = re.sub(r"[^a-z' ]+", " ", (text or "").lower().replace("’", "'"))
    t = re.sub(r"\s+", " ", t).strip()
    for pattern, label in _PHRASES:
        if re.fullmatch(pattern, t):
            return label
    return None


def read_log(path: Path, offset: int = 0) -> list[dict]:
    """The events this module reads from one session log, from ``offset``."""
    out: list[dict] = []
    try:
        with Path(path).open("rb") as f:
            f.seek(offset)
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return out
    for line in data.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event") in EVENTS:
            out.append(row)
    return out


# --- versioned stable models ------------------------------------------------------
#
# A model is a JSON file: the weights learned from the logs, globally and per
# persona, plus what it was learned from. Versions are never rewritten; the
# pointer file says which is in use. Rolling back moves the pointer, so the
# weights in use afterwards are byte-for-byte the ones that were in use before.

#: Tonight's layer moves weights only this far, whatever the logs of the
#: evening say: a rough hour must not undo what months of logs settled.
NIGHT_RANGE: tuple[float, float] = (0.5, 2.0)


def model_dir(log_dir: Path) -> Path:
    return Path(log_dir) / "models"


def versions(mdir: Path) -> list[int]:
    out = []
    for p in Path(mdir).glob("v*.json"):
        try:
            out.append(int(p.stem[1:]))
        except ValueError:
            continue
    return sorted(out)


def current_version(mdir: Path) -> int | None:
    try:
        return int(json.loads((Path(mdir) / "current.json").read_text("utf-8"))["version"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load(mdir: Path, version: int | None = None) -> dict | None:
    """A stored model: ``version``, or the current one. None if there is none."""
    version = current_version(mdir) if version is None else version
    if version is None:
        return None
    try:
        return json.loads((Path(mdir) / f"v{version:04d}.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None


def _point_at(mdir: Path, version: int) -> None:
    tmp = Path(mdir) / "current.json.tmp"
    tmp.write_text(json.dumps({"version": version}), encoding="utf-8")
    tmp.replace(Path(mdir) / "current.json")


def build(events: list[dict]) -> dict:
    """The stable model's content: global weights, and one set per persona."""
    personas = sorted({str(e.get("persona")) for e in events if e.get("persona")})
    return {
        "global": learn(events),
        "personas": {p: learn([e for e in events if e.get("persona") == p])
                     for p in personas},
    }


def train(log_dir: Path, note: str = "", max_files: int = MAX_LOG_FILES,
          exclude: Path | None = None) -> dict:
    """Learn a new stable model from the logs, store it as the next version,
    and make it current. Returns the model. ``exclude`` leaves one log out --
    tonight's, when the model is bootstrapped at the start of a set."""
    log_dir = Path(log_dir)
    mdir = model_dir(log_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(log_dir.glob("session_*.jsonl"))
             if exclude is None or p.resolve() != Path(exclude).resolve()][-max_files:]
    events = [e for p in files for e in read_log(p)]
    version = (versions(mdir) or [0])[-1] + 1
    model = {
        "version": version,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
        "trained_on": [{"file": p.name, "bytes": p.stat().st_size} for p in files],
        "events": len(events),
        **build(events),
    }
    tmp = mdir / f"v{version:04d}.json.tmp"
    tmp.write_text(json.dumps(model, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(mdir / f"v{version:04d}.json")
    _point_at(mdir, version)
    return model


def rollback(mdir: Path, version: int | None = None) -> int:
    """Make ``version`` current, or the one before the current. Returns it."""
    have = versions(mdir)
    if not have:
        raise ValueError("no stored model to roll back to")
    if version is None:
        now = current_version(mdir)
        older = [v for v in have if now is None or v < now]
        if not older:
            raise ValueError(f"v{now} is the oldest model; nothing earlier")
        version = older[-1]
    if version not in have:
        raise ValueError(f"no model v{version}; have {', '.join(f'v{v}' for v in have)}")
    _point_at(mdir, version)
    return version


def night_layer(events: list[dict]) -> dict[str, Any]:
    """What tonight alone says, in the tighter :data:`NIGHT_RANGE`.

    Low thresholds on purpose -- a single "more like this" should tilt the
    next pick tonight -- which is exactly why this layer is kept apart from
    the stable one and never written into it.
    """
    return learn(events, min_verdicts=1, min_choices=2, min_timing=1, bounds=NIGHT_RANGE)


def effective(stable: dict | None, night: dict | None, persona: str = "default") -> dict:
    """The weights to use now: the stable model's (for this persona where it
    has learned something, globally otherwise) times tonight's, clamped."""
    base = (stable or {}).get("global") or {}
    mine = ((stable or {}).get("personas") or {}).get(persona) or {}
    night = night or {}
    out: dict[str, Any] = {}
    for part in ("critic", "selector"):
        chosen = dict(mine.get(part) or base.get(part) or {})
        for key, value in (night.get(part) or {}).items():
            chosen[key] = _clamp(chosen.get(key, 1.0) * float(value))
        out[part] = chosen
    timing = int(mine.get("timing_bias_bars") or base.get("timing_bias_bars") or 0)
    timing += int(night.get("timing_bias_bars") or 0)
    out["timing_bias_bars"] = max(-TIMING_MAX_BARS, min(TIMING_MAX_BARS, timing))
    return out
