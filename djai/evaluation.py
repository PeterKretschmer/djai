"""Reading a session back, and saying whether it went well (SPEC §7).

THREADING CONTEXT: offline only. Nothing here runs while a set is playing, and
nothing here opens an audio device.

Three things live here:

* **replay** -- turn a session log back into the sequence of decisions it
  recorded, with the state each was taken on. What this can and cannot
  reproduce is stated at :func:`replay`, because the honest answer is not
  "everything".
* **metrics** -- numbers over a replayed session. Every metric declares
  whether it is a *proxy* for what we actually care about, and the label
  travels with the number rather than living in a docstring.
* **baseline diff** -- the same metrics for two versions, and what moved.

Deliberately not here: any judgement about whether a number is good. That is
the critic's job (:mod:`djai.critic`) and the operator's.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from djai.supervisor import LOG_SCHEMA_VERSION

#: Schema versions this module knows how to read. A log from the future is
#: refused rather than misread: silently interpreting an unknown shape is how
#: an eval harness starts reporting confident nonsense.
READABLE_VERSIONS: frozenset[int] = frozenset({1, 2})


class LogVersionError(RuntimeError):
    """A log whose schema this build cannot read."""


@dataclass(frozen=True)
class Decision:
    """One decision, the state it was taken on, and how it turned out."""

    decision_id: str
    kind: str
    action: str
    ts: str
    snapshot: dict | None
    fields: dict
    #: The outcome record, when the window over this decision was closed.
    outcome: dict | None = None

    @property
    def resolved(self) -> bool:
        return self.outcome is not None

    def snapshot_value(self, *path: str) -> Any:
        """Dig a value out of the triggering snapshot, or None."""
        node: Any = self.snapshot
        for key in path:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        return node


@dataclass
class Session:
    """A whole session log, read back."""

    path: Path
    schema_version: int
    records: list[dict] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)

    @property
    def events(self) -> list[str]:
        return [r.get("event", "") for r in self.records]

    def of_kind(self, kind: str) -> list[Decision]:
        return [d for d in self.decisions if d.kind == kind]


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" \
        else path.open("r", encoding="utf-8")


def read_records(path: Path) -> Iterator[dict]:
    """Every JSON record in a log, in order. Tolerates a torn last line.

    A session killed mid-write leaves a partial final line. That is a normal
    way for a log to end -- the process was killed, which is exactly the case
    the log exists for -- so it is skipped rather than raised on.
    """
    with _open(Path(path)) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail of a killed session


def replay(path: Path) -> Session:
    """Read a session log back into its decision sequence.

    **What this reproduces, and what it does not.** The decision sequence, the
    state each decision was taken on, and each outcome are read back exactly as
    they were written -- that is what "replayable" means here and it is what
    the comparison in :func:`diff` rests on.

    It does **not** re-run the model. A decision that consulted
    ``llama3.1:8b`` replays from the response recorded beside it, not from a
    fresh call, because a sampled model is not reproducible and pretending
    otherwise would make every replay a new experiment. Deterministic
    decisions -- the selector, the supervisor, placement -- are reproducible in
    the strong sense: given the same snapshot they will decide the same thing
    again, and :func:`compare_sequences` is how that is checked.
    """
    path = Path(path)
    records = list(read_records(path))
    version = 1
    for record in records:
        if record.get("event") == "log_opened":
            version = int(record.get("schema_version", 1))
            break
    if version not in READABLE_VERSIONS:
        raise LogVersionError(
            f"{path.name} is schema v{version}; this build reads "
            f"{sorted(READABLE_VERSIONS)} (it writes v{LOG_SCHEMA_VERSION})"
        )

    outcomes: dict[str, dict] = {
        r["decision_id"]: r for r in records
        if r.get("event") == "outcome" and r.get("decision_id")
    }
    decisions = []
    for r in records:
        # A decision is any record carrying a decision id. The event keeps its
        # own name (`track_cued`, `transition_armed`, ...), which is also its
        # kind -- see SessionLog.decision for why that is additive rather than
        # a record type of its own.
        if r.get("event") == "outcome" or not r.get("decision_id"):
            continue
        did = r.get("decision_id", "")
        known = {"event", "ts", "decision_id", "kind", "action", "snapshot"}
        decisions.append(Decision(
            decision_id=did,
            kind=r.get("kind") or r.get("event", ""),
            action=r.get("action", ""),
            ts=r.get("ts", ""),
            snapshot=r.get("snapshot"),
            fields={k: v for k, v in r.items() if k not in known},
            outcome=outcomes.get(did),
        ))
    return Session(path=path, schema_version=version,
                   records=records, decisions=decisions)


def compare_sequences(a: Session, b: Session) -> list[str]:
    """Where two sessions' decision sequences diverge. Empty means identical.

    Compared on (kind, action) rather than on ids or timestamps: a replay that
    decided the same things in the same order has reproduced the session, even
    though it ran at a different wall-clock time.
    """
    left = [(d.kind, d.action) for d in a.decisions]
    right = [(d.kind, d.action) for d in b.decisions]
    if left == right:
        return []
    out = []
    for i in range(max(len(left), len(right))):
        want = left[i] if i < len(left) else None
        got = right[i] if i < len(right) else None
        if want != got:
            out.append(f"[{i}] {want!r} != {got!r}")
    return out


# --- metrics ------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    """A number, and whether it measures the thing or merely stands in for it.

    ``proxy=True`` means: this correlates with what we care about and is what
    we can measure without a room full of people. It is not the thing. The flag
    is on the value so it survives into any report built from it -- a proxy
    described as a proxy only in prose becomes a fact the moment someone
    copies the number into a table.
    """

    name: str
    value: float
    proxy: bool
    unit: str = ""
    note: str = ""

    def __str__(self) -> str:
        tag = " (proxy)" if self.proxy else ""
        return f"{self.name}={self.value:g}{self.unit}{tag}"


def metrics(session: Session) -> dict[str, Metric]:
    """Numbers over one replayed session.

    Counts and timings are direct measurements. Anything claiming to describe
    how the *set* felt is a proxy and says so.
    """
    decisions = session.decisions
    resolved = [d for d in decisions if d.resolved]
    events = session.events

    def master_bpms() -> list[float]:
        out = []
        for d in decisions:
            v = d.snapshot_value("master_bpm")
            if isinstance(v, (int, float)) and v > 0:
                out.append(float(v))
        return out

    def peaks() -> list[float]:
        out = []
        for r in session.records:
            snap = r.get("snapshot")
            if isinstance(snap, dict):
                v = snap.get("master_peak_in")
                if isinstance(v, (int, float)):
                    out.append(float(v))
        return out

    bpms = master_bpms()
    peak_values = peaks()
    ms: list[Metric] = [
        Metric("decisions", float(len(decisions)), proxy=False),
        Metric("decisions_resolved", float(len(resolved)), proxy=False,
               note="decisions whose outcome window was closed"),
        Metric("rejections", float(events.count("rejected")), proxy=False),
        Metric("underrun_events", float(events.count("underrun")), proxy=False),
    ]
    if bpms:
        ms.append(Metric("master_bpm_range", max(bpms) - min(bpms), proxy=False,
                         unit=" BPM", note="how far the set's tempo travelled"))
    if peak_values:
        ms.append(Metric("peak_pre_limiter_max", max(peak_values), proxy=False))
    if len(bpms) > 2:
        # Stand-in for "did the set have shape". It is a spread of tempo, not
        # a measure of whether anyone danced.
        mean = sum(bpms) / len(bpms)
        var = sum((b - mean) ** 2 for b in bpms) / len(bpms)
        ms.append(Metric("tempo_variance", var, proxy=True, unit=" BPM^2",
                         note="proxy for set shape; measures tempo spread only"))
    return {m.name: m for m in ms}


@dataclass(frozen=True)
class MetricDelta:
    name: str
    baseline: float | None
    current: float | None
    proxy: bool

    @property
    def change(self) -> float | None:
        if self.baseline is None or self.current is None:
            return None
        return self.current - self.baseline

    def __str__(self) -> str:
        tag = " (proxy)" if self.proxy else ""
        if self.baseline is None:
            return f"{self.name}: new = {self.current:g}{tag}"
        if self.current is None:
            return f"{self.name}: gone (was {self.baseline:g}){tag}"
        return (f"{self.name}: {self.baseline:g} -> {self.current:g} "
                f"({self.change:+g}){tag}")


def diff(baseline: dict[str, Metric], current: dict[str, Metric]) -> list[MetricDelta]:
    """What moved between two versions. Proxy labels are carried through."""
    out = []
    for name in sorted(set(baseline) | set(current)):
        b, c = baseline.get(name), current.get(name)
        out.append(MetricDelta(
            name=name,
            baseline=b.value if b else None,
            current=c.value if c else None,
            proxy=(c or b).proxy,
        ))
    return out


# --- the scenario suite --------------------------------------------------------
#
# The five the brief names. Each is a *description* -- what to build and what
# to assert -- kept here so the harness and the tests agree on what a scenario
# is, while the slow runs themselves live in scratch/ and open no device.


@dataclass(frozen=True)
class Scenario:
    """One reproducible situation to put the system in."""

    name: str
    description: str
    #: Native BPMs of the crate the scenario runs against.
    bpms: tuple[float, ...]
    #: Minutes of set to drive.
    minutes: float
    #: What must hold afterwards, in words. Each is asserted by the runner.
    expectations: tuple[str, ...]
    #: What the scenario deliberately breaks, if anything.
    fault: str = ""


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="clean_two_track",
        description="Two tracks a few BPM apart, one transition, nothing else.",
        bpms=(124.0, 126.0),
        minutes=8.0,
        expectations=(
            "exactly one transition completes",
            "no sample reaches full scale before the limiter",
            "master tempo ends on the second track's native BPM",
        ),
    ),
    Scenario(
        name="multi_track_differing_bpm",
        description="Four tracks with gaps past 10%, which is what pinned the "
                    "master clock before Phase 0B.",
        bpms=(120.0, 132.0, 100.0, 128.0),
        minutes=25.0,
        expectations=(
            "master tempo converges to each incoming track's native BPM",
            "no logged stretch ratio leaves the cap",
            "stretch is applied exactly once per path",
        ),
    ),
    Scenario(
        name="forced_energy_drop",
        description="Energy is pulled down mid-set; the system must notice and "
                    "correct rather than ride it down.",
        bpms=(128.0, 128.0, 126.0),
        minutes=15.0,
        fault="energy of the live deck forced low",
        expectations=(
            "a correcting decision is logged within one phrase",
            "audio is continuous across the correction",
        ),
    ),
    Scenario(
        name="forced_analysis_failure",
        description="A track whose analysis is missing or corrupt is offered to "
                    "the selector.",
        bpms=(124.0, 124.0),
        minutes=10.0,
        fault="one track's sidecar corrupted",
        expectations=(
            "the bad track is refused, by name, with a reason",
            "the set continues without a gap",
        ),
    ),
    Scenario(
        name="rapid_cueing",
        description="Cue after cue with no settling time -- the operator "
                    "changing their mind repeatedly.",
        bpms=(126.0, 128.0, 124.0, 130.0, 127.0),
        minutes=10.0,
        expectations=(
            "no dropout or underrun",
            "every cue is honoured or refused with a reason, never dropped",
            "no decision is left with an unclosed outcome window",
        ),
    ),
)


def scenario(name: str) -> Scenario:
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(f"unknown scenario {name!r}; have {[s.name for s in SCENARIOS]}")
