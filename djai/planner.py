"""Set planning and narrative (SPEC §6, Phase 3.3).

THREADING CONTEXT: control thread only. Pure functions over the crate, the
history and the cue queue; nothing here touches the engine. A plan for a
90-minute set on the 105-track crate takes well under a second.

**One planner, three horizons.** :func:`plan_set` plans every slot from now to
the end of the set with a beam search over the selector's own ranking. The
horizons are views of that one plan: *near* is the next two tracks, *mid* the
next 30 minutes, *full* the whole set. The plan is soft: it is re-made after
every cue, whenever the operator changes the queue, and whenever the set goes
off it, and only its first slot is ever acted on.

**Hard constraints.** Operator cues sit in fixed slots (next, after N) and the
plan routes around them: the slot before a cue is chosen so the cue is
reachable inside the stretch cap. Every planned pair is checked against the
cap; a pair that cannot meet inside it is either a cue the operator bridged
("tempo_bridge") or it is not planned at all.

**Narrative.** A persona gives the energy arc (tension and release), how much
contrast it wants, how often it calls back to something played earlier, and
how much risk it will take. All of it is hand-set heuristics, labelled as
such: no model has been trained on sets. :func:`critique` scores a plan
without a model; an LLM note, when one is available, is advisory only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from djai import config, selector
from djai.analysis import TrackAnalysis

#: Default set length, in minutes.
SET_MINUTES: float = 90.0
#: The mid horizon, in minutes.
MID_MINUTES: float = 30.0
#: Beam width and branch per slot.
BEAM: int = 5
BRANCH: int = 5
#: A callback is a track that sounds like one played at least this long ago.
CALLBACK_MIN_MINUTES: float = 20.0
CALLBACK_SIMILARITY: float = 0.9
#: A pick is a risk when it sounds this unlike the track before it, or clashes.
RISK_SIMILARITY: float = 0.5


@dataclass(frozen=True)
class Persona:
    name: str
    #: Energy arc: (fraction of the set, target intensity 0..1), piecewise linear.
    arc: tuple[tuple[float, float], ...]
    #: "invisible", "showy" or "auto" for generated transitions (SPEC §4).
    mode: str = "auto"
    #: Risky picks allowed per 30 minutes.
    risk_per_30: float = 1.0
    #: Weight on following the arc, on contrast, and on callbacks.
    w_arc: float = 3.0
    w_contrast: float = 0.6
    w_callback: float = 0.8


#: Hand-set. The two reference DJs from the SPEC's quality bar are described
#: in the words of their sets as the operator characterised them, not modelled.
PERSONAS: dict[str, Persona] = {
    # Build, peak, a breather, a second higher peak, a short landing.
    "default": Persona("default", ((0.0, 0.35), (0.3, 0.6), (0.5, 0.75), (0.6, 0.55),
                                   (0.85, 0.9), (1.0, 0.6))),
    # Big-room: a quick climb, repeated peaks with short releases, showy.
    "guetta": Persona("guetta", ((0.0, 0.5), (0.15, 0.8), (0.3, 0.6), (0.45, 0.9),
                                 (0.6, 0.65), (0.8, 0.95), (1.0, 0.8)),
                      mode="showy", risk_per_30=1.5, w_contrast=0.9),
    # Relentless and fast-moving: high throughout, more risk, more callbacks.
    "hype": Persona("hype", ((0.0, 0.6), (0.2, 0.85), (0.5, 0.75), (0.7, 0.95),
                             (1.0, 0.9)),
                    mode="showy", risk_per_30=2.0, w_callback=1.2),
    # Warm-up: low and slow, invisible, no risk.
    "warmup": Persona("warmup", ((0.0, 0.2), (0.7, 0.4), (1.0, 0.5)),
                      mode="invisible", risk_per_30=0.0, w_contrast=0.3),
}


def target_at(persona: Persona, fraction: float) -> float:
    """The arc's intensity at ``fraction`` of the set."""
    pts = persona.arc
    f = min(1.0, max(0.0, fraction))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if f <= x1:
            return y0 + (y1 - y0) * ((f - x0) / (x1 - x0) if x1 > x0 else 0.0)
    return pts[-1][1]


def intensity_of(track: TrackAnalysis, crate: list[TrackAnalysis]) -> float:
    """Measured intensity, or crate-relative energy where none was measured."""
    value = selector._intensity(track)
    return value if value is not None else selector.normalised_energy(track, crate)


def minutes_of(track: TrackAnalysis) -> float:
    """How long a track plays in a mix: mix-in to mix-out."""
    span = float(track.mix_out) - float(track.mix_in)
    if span <= 30.0:
        span = float(getattr(track, "duration_s", 0.0) or 210.0)
    return span / 60.0


@dataclass(frozen=True)
class Slot:
    track: TrackAnalysis
    start_min: float
    target: float
    intensity: float
    #: How the tempo gets here from the slot before: a selector technique, or
    #: "tempo_bridge" for an operator cue bridged by an echo out.
    tempo: str
    stretch_pct: float
    #: How far the outgoing deck is ridden toward this one, in %.
    ride_pct: float = 0.0
    cued: bool = False
    callback: bool = False
    risk: bool = False

    def as_dict(self) -> dict:
        return {
            "track_id": self.track.track_id, "title": self.track.title,
            "bpm": round(float(self.track.bpm), 1), "camelot": self.track.camelot,
            "start_min": round(self.start_min, 1), "target": round(self.target, 3),
            "intensity": round(self.intensity, 3), "tempo": self.tempo,
            "stretch_pct": round(self.stretch_pct, 2),
            "ride_pct": round(self.ride_pct, 2), "cued": self.cued,
            "callback": self.callback, "risk": self.risk,
        }


@dataclass
class Plan:
    persona: str
    minutes: float
    slots: list[Slot] = field(default_factory=list)
    score: float = 0.0
    critique: dict = field(default_factory=dict)
    reason: str = ""

    @property
    def near(self) -> list[Slot]:
        return self.slots[:2]

    @property
    def mid(self) -> list[Slot]:
        if not self.slots:
            return []
        t0 = self.slots[0].start_min
        return [s for s in self.slots if s.start_min - t0 < MID_MINUTES]

    def as_dict(self) -> dict:
        return {
            "persona": self.persona, "minutes": self.minutes, "reason": self.reason,
            "score": round(self.score, 3), "critique": self.critique,
            "near": [s.track.track_id for s in self.near],
            "mid": [s.track.track_id for s in self.mid],
            "slots": [s.as_dict() for s in self.slots],
        }

    def summary(self) -> str:
        c = self.critique
        head = (f"Plan ({self.persona}, {self.minutes:.0f} min, {len(self.slots)} tracks) - "
                f"critic {c.get('score', 0):.1f}/10 [{c.get('source', 'rules')}]")
        rows = []
        for i, s in enumerate(self.mid):
            flags = "".join((" CUED" if s.cued else "", " callback" if s.callback else "",
                             " risk" if s.risk else "",
                             " TEMPO BRIDGE" if s.tempo == "tempo_bridge" else ""))
            rows.append(f"  {s.start_min:5.1f}m  {s.track.title} ({s.track.bpm:.0f}, "
                        f"{s.tempo}, int {s.intensity:.2f} -> {s.target:.2f}){flags}")
        notes = c.get("notes") or []
        return "\n".join([head] + rows + [f"  note: {n}" for n in notes])


def _cap_path(from_bpm: float, track: TrackAnalysis):
    return selector.tempo_path(from_bpm, track.bpm, travel=True)


def plan_set(
    crate: list[TrackAnalysis],
    current: TrackAnalysis | None,
    history: list[TrackAnalysis],
    cues: list[dict],
    persona: Persona,
    elapsed_min: float = 0.0,
    minutes: float = SET_MINUTES,
    playing_bpm: float | None = None,
    excluded: set[str] | None = None,
    weights: dict[str, float] | None = None,
    cache_dir=None,
) -> Plan:
    """Plan from now to the end of the set. Deterministic.

    ``cues`` is the operator's queue (djai.setstate entries), in order: each
    is a fixed slot. ``history`` is the tracks played so far, oldest first,
    with their start minute found by summing their lengths.
    """
    excluded = set(excluded or set())
    played_ids = {t.track_id for t in history} | excluded
    if current is not None:
        played_ids.add(current.track_id)
    # Where each played track sat in the set, for callbacks.
    played_at: list[tuple[float, TrackAnalysis]] = []
    t = 0.0
    for track in history:
        played_at.append((t, track))
        t += minutes_of(track)

    by_id = {tr.track_id: tr for tr in crate}
    fixed: dict[int, dict] = {}
    slot_i = 0
    waiting = []
    for entry in cues:
        track = by_id.get(entry.get("track_id"))
        if track is None:
            continue
        if entry.get("mode") == "after" and entry.get("after", 0) > 0:
            waiting.append((entry["after"], entry, track))
            continue
        while slot_i in fixed:
            slot_i += 1
        fixed[slot_i] = dict(entry, _track=track)
        slot_i += 1
    for after, entry, track in waiting:
        i = after
        while i in fixed:
            i += 1
        fixed[i] = dict(entry, _track=track)
    cued_ids = {e["_track"].track_id for e in fixed.values()}

    start_bpm = playing_bpm or (current.bpm if current else 0.0)
    risk_budget = persona.risk_per_30 * max(minutes - elapsed_min, 0.0) / 30.0
    now_min = elapsed_min + (minutes_of(current) / 2.0 if current else 0.0)

    last_fixed = max(fixed) if fixed else -1

    def finished(minute: float, n_slots: int) -> bool:
        return minute >= minutes and n_slots > last_fixed

    # Beam state: (score, slots, tip track, tip bpm, minute, used ids, risks,
    # minute of the last callback)
    beam = [(0.0, [], current, start_bpm, now_min, set(played_ids) | cued_ids, 0, -1e9)]
    for _ in range(200):
        extended, grew = [], False
        for state in beam:
            score, slots, tip, tip_bpm, minute, used, risks, last_cb = state
            if finished(minute, len(slots)):
                extended.append(state)
                continue
            k = len(slots)
            target = target_at(persona, minute / minutes if minutes > 0 else 1.0)
            options = []
            if k in fixed:
                track = fixed[k]["_track"]
                path = _cap_path(tip_bpm, track) if tip_bpm else selector.TempoPath("direct")
                # A cue the cap cannot reach is still played: the operator is
                # asked, and it comes in on a tempo bridge (djai.cli).
                tempo = path.technique if path.blendable else "tempo_bridge"
                options.append((track, path, tempo, 0.0, True, False, False))
            else:
                ranked = selector.rank_candidates(
                    crate, tip, used, history=list(history) + [s.track for s in slots],
                    cache_dir=cache_dir, weights=weights, playing_bpm=tip_bpm or None,
                    travel=True,
                )
                nxt = fixed.get(k + 1)
                fallback = None
                for c in ranked:
                    played_bpm = c.track.bpm * (c.tempo_path.rate_b or 1.0)
                    if (nxt is not None
                            and not _cap_path(played_bpm, nxt["_track"]).blendable
                            and _cap_path(tip_bpm or played_bpm, nxt["_track"]).blendable):
                        continue      # would strand the cue that follows it
                    sim = c.similarity or {}
                    simv = sum(sim.values()) / len(sim) if sim else 1.0
                    risky = simv < RISK_SIMILARITY or c.key_relation == "clash"
                    if risky and risks + 1 > risk_budget:
                        # Over budget: kept only as a last resort, so the plan
                        # never ends early for want of a safe pick.
                        if fallback is None:
                            fallback = (c.track, c.tempo_path, c.tempo_path.technique,
                                        c.score, False, False, True)
                        continue
                    cb = False
                    if minute - last_cb >= 30.0:
                        for at, old in played_at + [(x.start_min, x.track) for x in slots]:
                            if minute - at < CALLBACK_MIN_MINUTES:
                                continue
                            sim_old = selector.similarity(old, c.track, cache_dir)
                            if sim_old and (sum(sim_old.values()) / len(sim_old)
                                            >= CALLBACK_SIMILARITY):
                                cb = True
                                break
                    options.append((c.track, c.tempo_path, c.tempo_path.technique,
                                    c.score, False, cb, risky))
                    if len(options) >= BRANCH:
                        break
                if not options and fallback is not None:
                    options.append(fallback)
            if not options:
                extended.append(state)       # the crate ran out on this branch
                continue
            grew = True
            for track, path, tempo, sel_score, cued, cb, risky in options:
                inten = intensity_of(track, crate)
                prev = [x.intensity for x in slots[-2:]]
                flat = len(prev) == 2 and max(prev + [inten]) - min(prev + [inten]) < 0.05
                s_score = (
                    sel_score
                    + persona.w_arc * (1.0 - abs(inten - target))
                    - (persona.w_contrast if flat else 0.0)
                    + (persona.w_callback if cb else 0.0)
                )
                played_bpm = track.bpm * (path.rate_b or 1.0) if path.blendable else track.bpm
                slot = Slot(
                    track=track, start_min=minute, target=target, intensity=inten,
                    tempo=tempo,
                    stretch_pct=((path.rate_b or 1.0) - 1.0) * 100.0 if path.blendable else 0.0,
                    ride_pct=path.ride_percent * 100.0 if path.blendable else 0.0,
                    cued=cued, callback=cb, risk=risky,
                )
                extended.append((
                    score + s_score, slots + [slot], track, played_bpm,
                    minute + minutes_of(track), used | {track.track_id},
                    risks + (1 if risky else 0), minute if cb else last_cb,
                ))
        # Ranked by score per slot, so a longer plan is not preferred for
        # being longer.
        extended.sort(key=lambda b: (-b[0] / max(1, len(b[1])),
                                     [x.track.track_id for x in b[1]]))
        beam = extended[:BEAM]
        if not grew:
            break
    best = beam[0]
    plan = Plan(persona=persona.name, minutes=minutes, slots=best[1])
    plan.critique = critique(plan, persona, history)
    plan.score = plan.critique["score"]
    return plan


def critique(plan: Plan, persona: Persona, history: list[TrackAnalysis] = ()) -> dict:
    """Score a plan's narrative, 0..10, without a model. Heuristic, labelled.

    Arc fit, contrast (no three flat tracks in a row), key clashes, stretch
    cap (any unbridged pair beyond it is a violation, and zeroes the score),
    artist repeats, callbacks and risk against the persona's budget.
    """
    slots = plan.slots
    if not slots:
        return {"score": 0.0, "source": "rules", "notes": ["empty plan"], "cap_violations": 0}
    arc_fit = sum(1.0 - abs(s.intensity - s.target) for s in slots) / len(slots)
    flats = sum(
        1 for a, b, c in zip(slots, slots[1:], slots[2:])
        if max(a.intensity, b.intensity, c.intensity)
        - min(a.intensity, b.intensity, c.intensity) < 0.05
    )
    monotony = flats / max(1, len(slots) - 2)
    clashes = 0
    for a, b in zip(slots, slots[1:]):
        if selector.key_relation(a.track.camelot, b.track.camelot) == "clash":
            clashes += 1
    clash_rate = clashes / max(1, len(slots) - 1)
    cap = config.MAX_STRETCH_RATIO * 100.0 + 1e-6
    violations = [s.track.title for s in slots
                  if s.tempo == "none" or (s.tempo != "tempo_bridge" and (
                      abs(s.stretch_pct) > cap or abs(s.ride_pct) > cap))]
    bridges = [s.track.title for s in slots if s.tempo == "tempo_bridge"]
    artists = [getattr(s.track, "artist", "") for s in slots]
    repeats = sum(1 for i, a in enumerate(artists)
                  if a and a in artists[max(0, i - selector.ARTIST_WINDOW):i])
    callbacks = sum(1 for s in slots if s.callback)
    risks = sum(1 for s in slots if s.risk)
    expected_cb = max(1.0, (slots[-1].start_min - slots[0].start_min) / 30.0)
    score = 10.0 * (
        0.5 * arc_fit + 0.2 * (1.0 - monotony) + 0.2 * (1.0 - clash_rate)
        + 0.1 * min(1.0, callbacks / expected_cb)
    ) - 0.5 * repeats
    notes = []
    if violations:
        score = 0.0
        notes.append(f"stretch cap exceeded without a bridge: {', '.join(violations)}")
    if bridges:
        notes.append(f"tempo bridges (operator cues): {', '.join(bridges)}")
    if monotony > 0.2:
        notes.append(f"{flats} flat stretch(es) of three tracks")
    if clash_rate > 0.2:
        notes.append(f"{clashes} key clash(es)")
    if repeats:
        notes.append(f"{repeats} artist repeat(s) inside {selector.ARTIST_WINDOW}")
    return {
        "score": round(max(0.0, score), 2), "source": "rules",
        "arc_fit": round(arc_fit, 3), "monotony": round(monotony, 3),
        "clash_rate": round(clash_rate, 3), "cap_violations": len(violations),
        "tempo_bridges": len(bridges), "callbacks": callbacks, "risks": risks,
        "artist_repeats": repeats, "notes": notes,
    }


def plan_fit(plan: Plan | None, track: TrackAnalysis, crate: list[TrackAnalysis]) -> float:
    """How well ``track`` fits the plan, 0..1, for ranking suggestions.

    The planned next track is 1.0; one planned in the next half hour 0.6;
    otherwise how close its intensity is to what the arc wants next, at most 0.5.
    """
    if plan is None or not plan.slots:
        return 0.0
    first = next((s for s in plan.slots if not s.cued), plan.slots[0])
    if track.track_id == first.track.track_id:
        return 1.0
    if any(s.track.track_id == track.track_id for s in plan.mid):
        return 0.6
    return round(0.5 * max(0.0, 1.0 - 2.0 * abs(intensity_of(track, crate) - first.target)), 3)
