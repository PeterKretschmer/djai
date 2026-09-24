"""Choosing the next track. A pure function over the cache -- no LLM, no state.

THREADING CONTEXT: main / monitor thread. Pure and side-effect free, so it is
safe to call from anywhere except the audio thread (it iterates the crate).

The LLM never picks a track. It produces at most an *energy direction*, and
this module turns that into an actual choice using tempo, key and measured
energy. That separation is deliberate: track selection stays deterministic and
explainable, and a hallucinated title can never reach the decks.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from djai import config
from djai.analysis import TrackAnalysis

#: Tempo window, as a fraction of the current track's BPM. +-6% is about the
#: limit before resampling-based tempo matching becomes audible as pitch shift.
BPM_TOLERANCE: float = 0.06

#: Weights for the ranking score. Tempo proximity matters most, then key, then
#: how well the track matches the requested energy direction.
W_BPM: float = 1.0
W_KEY: float = 0.8
W_ENERGY: float = 1.2
W_CONFIDENCE: float = 0.4

#: Energy step implied by "more energy" / "less energy", in normalised units.
ENERGY_STEP: float = 0.18

# --- how two tempos can meet ------------------------------------------------------
#
# The rule this replaces was a wall: more than 8% apart and the mix was a cut,
# whatever the music was doing. That is not what a DJ does with a 100 BPM track
# and a 128 BPM one. They ride the tempo up over several tracks, or they mix the
# slow one half-time under the fast one, and a cut is what happens when neither
# is available -- not the first answer.
#
# Four ways two tempos meet, in the order they are preferred:
#
# * direct     -- already within the stretch range; nothing to do.
# * ride       -- close enough that the outgoing deck can be walked toward the
#                 incoming one while it plays. A pitch fader moved 3% over eight
#                 bars is inaudible as pitch and doubles the reach.
# * half/double -- an 87 BPM track under a 174 BPM one is not a tempo problem at
#                 all: every other beat lines up, and neither track changes speed.
# * none       -- no path. This, and only this, is what a cut is for.

#: How far the playing deck may be walked, as a fraction of its tempo. Six per
#: cent is a long way for a pitch fader, which is why a bigger ride is given
#: more bars to happen in rather than being done faster.
RIDE_PERCENT: float = 0.06

#: Bars a ride is spread over: about three per cent of tempo per eight bars,
#: never fewer than eight bars and never more than thirty-two. Eight bars at
#: 128 BPM is 15 seconds, so a 6% ride takes about half a minute -- which is
#: what riding a set from 107 to 119 BPM should feel like.
#: Bars kept clear between the end of a ride and the mix-out it was planned
#: for. The transition itself sits in that window -- the longest blend plus a
#: phrase -- because a pitch fader still moving when the blend starts would
#: walk the two decks apart while they are both audible.
RIDE_MARGIN_BARS: float = 36.0

RIDE_BARS_MIN: int = 8
RIDE_BARS_MAX: int = 32
RIDE_BARS_PER_PERCENT: float = 3.0


def ride_bars(ride: float) -> int:
    """How many bars a ride of this size is spread over."""
    bars = int(round(abs(ride) * 100.0 * RIDE_BARS_PER_PERCENT))
    return max(RIDE_BARS_MIN, min(RIDE_BARS_MAX, bars))

#: Metric relationships that need no stretching at all: the same tempo, or one
#: track counted half or double against the other.
METRIC_RATIOS: tuple[float, ...] = (1.0, 2.0, 0.5)

#: A technique costs something in the ranking: a ride is a gesture, and a
#: half-time mix is a bigger one. Direct is free.
TECHNIQUE_COST: dict[str, float] = {
    "direct": 0.0,
    "ride": 0.25,
    "double": 0.6,
    "half": 0.6,
}


@dataclass(frozen=True)
class TempoPath:
    """How the incoming track can be brought to the playing one, if it can.

    ``rate_b`` is the rate the incoming deck plays at, ``ride_percent`` how far
    the outgoing deck is walked before the blend, and ``ratio`` the metric
    relationship (1 for a straight mix, 2 when the incoming track is counted
    double, 0.5 when it is counted half).
    """

    technique: str
    ratio: float = 1.0
    rate_b: float = 1.0
    ride_percent: float = 0.0
    gap_percent: float = 0.0
    #: Bars the ride is spread over; 0 when there is no ride.
    bars: int = 0

    @property
    def blendable(self) -> bool:
        return self.technique != "none"

    def describe(self) -> str:
        if self.technique == "direct":
            return f"direct ({self.gap_percent:+.1f}%)"
        if self.technique == "ride":
            return (
                f"ride {self.ride_percent * 100:+.1f}% over {self.bars} bars "
                f"({self.gap_percent:+.1f}% apart)"
            )
        if self.technique in ("half", "double"):
            ridden = (
                f" after a {self.ride_percent * 100:+.1f}% ride over {self.bars} bars"
                if self.ride_percent else ""
            )
            return f"{self.technique}-time (x{self.ratio:g}){ridden}"
        return f"no blend path ({self.gap_percent:+.1f}% apart)"


def _meet(playing_bpm: float, counted_bpm: float, limit: float, ride_limit: float):
    """``(rate, ride)`` that brings ``counted_bpm`` to ``playing_bpm``, or None.

    Everything is decided on the **rate the deck is actually asked for**, not on
    the difference between two printed tempos. Those are not the same: a track
    8% slower than the mix needs a rate of 1.087, which is 8.7% of stretch and
    over the limit. Testing the tempo gap instead let exactly that through.
    """
    if playing_bpm <= 0 or counted_bpm <= 0:
        return None

    def inside(rate: float) -> float:
        """Strictly within the limit, never sitting on it.

        A deck exactly at the edge of the range gives ``playing / counted ==
        1 + limit`` exactly, and ``abs(1.08 - 1.0)`` is 0.08000000000000007.
        The supervisor compares with no epsilon, so that rate is refused at
        the load and the autopilot simply fails to cue -- measured in a
        20-track soak, where the same track was rejected on every attempt and
        only 4 of 20 tracks ever reached a deck. The producer owes its
        consumer a value the consumer accepts.
        """
        edge = limit * (1.0 - 1e-9)
        return float(min(1.0 + edge, max(1.0 - edge, rate)))

    lo, hi = counted_bpm * (1.0 - limit), counted_bpm * (1.0 + limit)
    if lo <= playing_bpm <= hi:
        return inside(playing_bpm / counted_bpm), 0.0
    # The playing deck rides just far enough to bring the rate inside the range.
    ridden = lo if playing_bpm < lo else hi
    ride = ridden / playing_bpm - 1.0
    if abs(ride) > ride_limit + 1e-9:
        return None
    return inside(ridden / counted_bpm), ride


def tempo_path(
    playing_bpm: float,
    incoming_bpm: float,
    stretch_limit: float | None = None,
    ride_limit: float = RIDE_PERCENT,
    travel: bool = False,
) -> TempoPath:
    """How these two tempos can meet, or that they cannot.

    Pure arithmetic over the four techniques above, preferring the smallest
    gesture that works: straight, then a ride, then counting the incoming
    track half or double, then admitting there is no path.
    """
    limit = config.MAX_STRETCH_RATIO if stretch_limit is None else float(stretch_limit)
    if playing_bpm <= 0 or incoming_bpm <= 0:
        return TempoPath("none")
    gap = (incoming_bpm - playing_bpm) / playing_bpm * 100.0

    if travel:
        # The set is going somewhere. Bringing the incoming track back to the
        # tempo that is playing would undo the journey one mix at a time, so
        # the playing deck is ridden toward the incoming track instead and the
        # track keeps as much of its own tempo as the ride can buy.
        ride = min(ride_limit, abs(incoming_bpm / playing_bpm - 1.0))
        ride = math.copysign(ride, incoming_bpm - playing_bpm)
        ridden = playing_bpm * (1.0 + ride)
        rate = ridden / incoming_bpm
        if abs(rate - 1.0) <= limit + 1e-9:
            technique = "direct" if not ride else "ride"
            return TempoPath(
                technique, 1.0, rate, ride, gap, ride_bars(ride) if ride else 0,
            )

    straight = _meet(playing_bpm, incoming_bpm, limit, ride_limit)
    if straight is not None:
        rate, ride = straight
        if not ride:
            return TempoPath("direct", 1.0, rate, 0.0, gap)
        return TempoPath("ride", 1.0, rate, ride, gap, ride_bars(ride))

    for ratio, name in ((2.0, "double"), (0.5, "half")):
        metric = _meet(playing_bpm, incoming_bpm * ratio, limit, ride_limit)
        if metric is not None:
            rate, ride = metric
            return TempoPath(name, ratio, rate, ride, gap, ride_bars(ride) if ride else 0)

    return TempoPath("none", 1.0, 1.0, 0.0, gap)


# --- what a track sounds like ------------------------------------------------
#
# BPM, key and energy say how two tracks fit; none of them says what either one
# sounds like. Two 128 BPM tracks in 8A can be a tech-house roller and a pop
# vocal, and following one with the other is a change of room, not a mix.
#
# Three descriptions, all from measurements already on disk:
#
# * timbre  -- how the track's energy splits across drums, bass, other and
#   vocals, from the separated stems (Phase 3), plus how percussive it is.
#   A track that is mostly drums and bass sits a long way from one that is
#   mostly voice and pads, whatever their tempo.
# * groove  -- the shape of a bar: per-beat loudness from the beat grid,
#   normalised, plus onset density. A four-to-the-floor bar and a shuffled,
#   syncopated bar measure differently here.
# * neighbourhood -- the two above taken together. It is NOT a genre label:
#   nothing in this crate carries one, and inventing names for clusters would
#   be dressing up a distance as knowledge. It answers "does this sound like
#   the last one", which is what the selector needs.

#: Weights of the three similarities. Deliberately below tempo and key: a
#: similar-sounding track at the wrong tempo is still unmixable, and feedback
#: (see :mod:`djai.feedback`) is what moves these from their starting values.
W_TIMBRE: float = 0.6
W_GROOVE: float = 0.5
W_NEIGHBOURHOOD: float = 0.4

#: Similarity is wanted, but sameness is not: a set of near-identical tracks
#: scores perfectly on all three. Past this similarity the score stops rising.
SIMILARITY_PLATEAU: float = 0.9

# --- set arc -----------------------------------------------------------------

#: Where a set is. Each asks for a different intensity, measured per track as
#: an absolute 0..1 score (analysis.intensity), not relative to the crate.
SET_PHASES: tuple[str, ...] = ("warmup", "build", "peak", "cooldown")

#: Intensity bands for the two phases that hold a level.
PHASE_BANDS: dict[str, tuple[float, float]] = {
    "warmup": (0.0, 0.45),
    "peak": (0.6, 1.0),
}
#: How far a build raises, and a cooldown lowers, intensity per track.
PHASE_STEP: float = 0.08
#: Weight of the set phase against tempo, key, energy and grid confidence.
#: Measured on the 96-track library, 10 picks from the median track:
#:
#:   weight  warmup mean / in band   peak mean / in band   build end  cooldown end
#:   1.2     0.467 / 60%             0.539 / 10%           0.488      0.345
#:   3.0     0.414 / 90%             0.605 / 40%           0.619      0.363
#:   4.0     0.421 / 70%             0.607 / 40%           0.629      0.340
#:
#: 3.0 is where warmup and peak separate cleanly. Peak's band is limited by
#: how few tracks in this crate are that intense within the tempo window, not
#: by the weight. Heavier weights do no better.
W_PHASE: float = 3.0

#: The same artist within this many tracks is penalised.
ARTIST_WINDOW: int = 5
ARTIST_PENALTY: float = 1.5
#: The same key within this many tracks is penalised: harmonic safety is not
#: the same as playing one key all night.
KEY_WINDOW: int = 3
KEY_REPEAT_PENALTY: float = 0.6
#: Two vocal-led tracks back to back is penalised.
VOCAL_BACK_TO_BACK_PENALTY: float = 0.8

_CAMELOT_RE = re.compile(r"^(\d{1,2})([AB])$")


@dataclass(frozen=True)
class Candidate:
    """One scored option, with the reasoning kept for the session log."""

    track: TrackAnalysis
    score: float
    bpm_delta_pct: float
    key_relation: str
    energy_delta: float
    #: The set phase this was scored for, or "" for none.
    set_phase: str = ""
    #: Named penalties applied, for the log.
    penalties: tuple[str, ...] = ()
    #: How alike this track and the one playing sound: timbre, groove and the
    #: two together. Empty when there was nothing to compare against.
    similarity: dict = field(default_factory=dict)
    #: How the two tempos meet: direct, a ride, half or double time.
    tempo_path: "TempoPath | None" = None
    #: Each reason term before its weight, 0..1: bpm, key, energy, confidence,
    #: and the similarity terms. What the ranking added up, and what a
    #: preference learned from the operator's choices moves (djai.feedback).
    terms: dict = field(default_factory=dict)

    def reason(self) -> str:
        text = (
            f"{self.track.title} | {self.track.bpm:.1f} BPM "
            f"({self.bpm_delta_pct:+.1f}%) | {self.track.camelot} "
            f"({self.key_relation}) | energy {self.energy_delta:+.2f} "
            f"| score {self.score:.3f}"
        )
        if self.set_phase:
            intensity = _intensity(self.track)
            shown = "?" if intensity is None else f"{intensity:.2f}"
            text += f" | {self.set_phase} intensity {shown}"
        if self.tempo_path is not None and self.tempo_path.technique != "direct":
            text += f" | {self.tempo_path.describe()}"
        if self.similarity:
            text += (
                f" | timbre {self.similarity.get('timbre', 0):.2f}"
                f" groove {self.similarity.get('groove', 0):.2f}"
            )
        if self.penalties:
            text += " | penalised: " + ", ".join(self.penalties)
        return text


def _intensity(track: TrackAnalysis | None) -> float | None:
    """A track's measured intensity, or None if it has none (or it failed)."""
    value = getattr(track, "intensity", None) if track is not None else None
    if value is None or not isinstance(value, (int, float)) or value != value:
        return None
    return float(value)


def phase_fit(track: TrackAnalysis, current: TrackAnalysis | None, set_phase: str) -> float:
    """How well a track's intensity suits the set phase, 0..1.

    warmup and peak hold a band; build asks for a step up from what is playing,
    cooldown a step down. A track with no intensity measured scores a neutral
    0.5, so an unmeasured crate still plays.
    """
    intensity = _intensity(track)
    if intensity is None or set_phase not in SET_PHASES:
        return 0.5
    if set_phase in PHASE_BANDS:
        lo, hi = PHASE_BANDS[set_phase]
        err = lo - intensity if intensity < lo else intensity - hi if intensity > hi else 0.0
    else:
        now = _intensity(current)
        if now is None:
            now = 0.5
        target = now + PHASE_STEP if set_phase == "build" else now - PHASE_STEP
        err = abs(intensity - target)
    return max(0.0, 1.0 - 2.0 * err)


# --- timbre, groove and the neighbourhood ----------------------------------------

#: Per-track descriptions, keyed by track id: building one reads a JSON
#: manifest and walks the beat grid, and a journey scores the same track many
#: times over. Cleared by :func:`forget_descriptions` when a crate is reloaded.
_DESCRIPTIONS: dict[str, dict[str, tuple[float, ...]]] = {}


def forget_descriptions() -> None:
    """Drop the cached descriptions. For tests, and for a re-analysed crate."""
    _DESCRIPTIONS.clear()


def timbre_vector(track: TrackAnalysis, cache_dir=None) -> tuple[float, ...]:
    """What the track is made of: stem balance, plus how percussive it is.

    Four stem shares (drums, bass, other, vocals) when the stems have been
    separated, and the analysis' own texture numbers either way. Without stems
    it still describes something -- just less.
    """
    from djai import stems as stems_mod

    features = getattr(track, "intensity_features", None) or {}
    texture = (
        float(features.get("percussive", 0.0)),
        min(float(features.get("flux", 0.0)) / 5.0, 1.0),
        float(getattr(track, "vocal_fraction", 0.0) or 0.0),
    )
    balance = None
    if cache_dir is not None:
        try:
            balance = stems_mod.balance(track.track_id, cache_dir)
        except Exception:  # noqa: BLE001 - a description is never worth a crash
            balance = None
    if balance:
        return tuple(balance.get(name, 0.0) for name in stems_mod.STEM_NAMES) + texture
    return (0.0, 0.0, 0.0, 0.0) + texture


def groove_vector(track: TrackAnalysis) -> tuple[float, ...]:
    """The shape of a bar: where the loudness sits across four beats.

    Per-beat RMS, folded onto the bar from the track's own downbeat phase and
    normalised so it describes the pattern rather than the level. A straight
    four-to-the-floor bar and a shuffled one with the weight on the offbeat are
    different here even at the same tempo and energy.
    """
    rms = list(getattr(track, "beat_rms", []) or [])
    if len(rms) < 8:
        return (0.25, 0.25, 0.25, 0.25, 0.0)
    beats = list(getattr(track, "beats", []) or [])
    downbeats = list(getattr(track, "downbeats", []) or [])
    phase = 0
    if beats and downbeats:
        period = float(getattr(track, "beat_period", 0.5)) or 0.5
        phase = int(round((downbeats[0] - beats[0]) / period)) % 4
    sums = [0.0, 0.0, 0.0, 0.0]
    counts = [0, 0, 0, 0]
    for i, value in enumerate(rms):
        slot = (i - phase) % 4
        sums[slot] += float(value)
        counts[slot] += 1
    profile = [s / c if c else 0.0 for s, c in zip(sums, counts)]
    total = sum(profile)
    if total <= 0:
        return (0.25, 0.25, 0.25, 0.25, 0.0)
    features = getattr(track, "intensity_features", None) or {}
    density = min(float(features.get("density", 0.0)) / 8.0, 1.0)
    return tuple(round(v / total, 4) for v in profile) + (round(density, 4),)


def _description(track: TrackAnalysis, cache_dir=None) -> dict[str, tuple[float, ...]]:
    got = _DESCRIPTIONS.get(track.track_id)
    if got is None:
        got = {"timbre": timbre_vector(track, cache_dir), "groove": groove_vector(track)}
        _DESCRIPTIONS[track.track_id] = got
    return got


def _closeness(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """1 when two descriptions match, 0 when they are as far apart as possible."""
    if not a or not b or len(a) != len(b):
        return 0.5
    diff = sum(abs(x - y) for x, y in zip(a, b)) / len(a)
    return max(0.0, 1.0 - diff * 2.0)


def similarity(
    a: TrackAnalysis, b: TrackAnalysis, cache_dir=None
) -> dict[str, float]:
    """How alike two tracks sound: timbre, groove, and the two together.

    Each is 0..1. The neighbourhood is the pair's overall closeness -- what a
    genre label would be for, without pretending to name one.
    """
    da, db = _description(a, cache_dir), _description(b, cache_dir)
    timbre = _closeness(da["timbre"], db["timbre"])
    groove = _closeness(da["groove"], db["groove"])
    return {
        "timbre": round(timbre, 4),
        "groove": round(groove, 4),
        "neighbourhood": round((timbre + groove) / 2.0, 4),
    }


def _similarity_terms(sim: dict[str, float]) -> dict[str, float]:
    """The similarity terms as the score sees them.

    Plateaued: past :data:`SIMILARITY_PLATEAU` a closer match adds nothing, so
    the selector prefers tracks that sit together without walking into a corner
    where every pick is the same record.
    """
    return {key: min(float(sim.get(key, 0.5)), SIMILARITY_PLATEAU)
            for key in ("timbre", "groove", "neighbourhood")}


def _similarity_score(sim: dict[str, float], weights: dict[str, float] | None) -> float:
    """The similarity terms' contribution to a candidate's score.

    ``weights`` are learned *multipliers* on the defaults (djai.feedback says
    so and clamps them around 1.0). Before Phase 4 they were read here as the
    weights themselves, so a learned "no opinion" of 1.0 silently raised
    timbre from 0.6 to 1.0.
    """
    terms = _similarity_terms(sim)
    return sum(default * _mult(weights, key) * terms[key]
               for key, default in (("timbre", W_TIMBRE), ("groove", W_GROOVE),
                                    ("neighbourhood", W_NEIGHBOURHOOD)))


def _mult(weights: dict[str, float] | None, key: str) -> float:
    """A learned multiplier for one term; 1.0 when nothing was learned."""
    return float((weights or {}).get(key, 1.0))


def parse_camelot(code: str) -> tuple[int, str] | None:
    m = _CAMELOT_RE.match((code or "").strip().upper())
    if not m:
        return None
    number, letter = int(m.group(1)), m.group(2)
    if not 1 <= number <= 12:
        return None
    return number, letter


def key_relation(a: str, b: str) -> str:
    """Classify how two Camelot codes relate.

    ``same``, ``relative`` (major/minor of the same number), ``adjacent``
    (+-1 around the wheel), or ``clash``.
    """
    pa, pb = parse_camelot(a), parse_camelot(b)
    if pa is None or pb is None:
        return "unknown"
    (na, la), (nb, lb) = pa, pb
    if na == nb and la == lb:
        return "same"
    if na == nb:
        return "relative"
    if la == lb and (na - nb) % 12 in (1, 11):
        return "adjacent"
    return "clash"


#: Score contribution per key relation. A clash is not disqualifying on its own
#: -- if it is the only track left in tempo, playing it beats stopping.
_KEY_SCORE = {
    "same": 1.0,
    "relative": 0.9,
    "adjacent": 0.85,
    "unknown": 0.3,
    "clash": 0.0,
}


def normalised_energy(track: TrackAnalysis, crate: list[TrackAnalysis]) -> float:
    """Map a track's raw RMS energy onto 0..1 across the crate.

    Absolute RMS is meaningless on its own (it tracks mastering loudness more
    than arrangement), so energy is only ever compared *within* the crate.
    """
    values = [t.energy for t in crate] or [track.energy]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return 0.5
    return (track.energy - lo) / (hi - lo)


#: Subtracted from a track whose grid needs review. Equal to the highest score
#: any track can earn, so a flagged track always ranks below every clean one --
#: deprioritised, not excluded, so a crate of flagged tracks still plays.
REVIEW_PENALTY: float = W_BPM + W_KEY + W_ENERGY + W_CONFIDENCE + W_PHASE


def rank_candidates(
    crate: list[TrackAnalysis],
    current: TrackAnalysis | None,
    played: set[str] | None = None,
    energy_direction: float = 0.0,
    bpm_tolerance: float = BPM_TOLERANCE,
    set_phase: str | None = None,
    history: list[TrackAnalysis] | None = None,
    cache_dir=None,
    weights: dict[str, float] | None = None,
    playing_bpm: float | None = None,
    travel: bool = False,
) -> list[Candidate]:
    """Score every eligible track, best first.

    ``playing_bpm`` is the tempo the mix is actually running at, which is not
    the playing track's printed tempo once a deck has been stretched or ridden;
    it defaults to the printed one. ``travel`` plans paths that move the set's
    tempo rather than hold it -- see :func:`tempo_path`.

    ``energy_direction`` is -1..+1: negative asks for calmer, positive for more
    driving, 0 keeps the current level.

    ``set_phase`` (one of :data:`SET_PHASES`) adds an intensity term.
    ``history`` is the tracks played so far, oldest first, and drives the
    repetition penalties: artist within :data:`ARTIST_WINDOW`, key within
    :data:`KEY_WINDOW`, and two vocal-led tracks back to back.
    """
    played = played or set()
    phase = set_phase if set_phase in SET_PHASES else ""
    recent = list(history or [])
    if current is not None and (not recent or recent[-1].track_id != current.track_id):
        recent.append(current)
    recent_artists = {
        t.artist.lower() for t in recent[-ARTIST_WINDOW:] if getattr(t, "artist", "")
    }
    recent_keys = {t.camelot for t in recent[-KEY_WINDOW:] if t.camelot}
    last_vocal_led = bool(recent and getattr(recent[-1], "vocal_led", False))
    eligible: list[Candidate] = []
    if not crate:
        return eligible

    current_bpm = playing_bpm if playing_bpm else (current.bpm if current else None)
    current_energy = normalised_energy(current, crate) if current else 0.5
    target_energy = min(
        1.0, max(0.0, current_energy + energy_direction * ENERGY_STEP * 2.0)
    )

    for track in crate:
        if track.track_id in played:
            continue
        if current is not None and track.track_id == current.track_id:
            continue
        # A quarantined grid is too weak to place a transition on, so the
        # system never reaches for the track itself. It stays fully playable
        # and cueable by hand -- this is the only gate, and it is not one the
        # operator passes through. See TrackAnalysis.quarantined.
        if getattr(track, "quarantined", False):
            continue

        path = TempoPath("direct")
        if current_bpm:
            delta = (track.bpm - current_bpm) / current_bpm
            # Not a wall any more: how these two tempos can meet. A track is
            # only refused when there is no way to meet it at all -- and that
            # is what a cut is for. See tempo_path.
            path = tempo_path(current_bpm, track.bpm, travel=travel)
            if not path.blendable:
                continue
            stretch = abs(path.rate_b - 1.0) / max(config.MAX_STRETCH_RATIO, 1e-6)
            bpm_score = max(0.0, 1.0 - stretch) - TECHNIQUE_COST.get(path.technique, 0.0)
        else:
            delta = 0.0
            bpm_score = 1.0

        relation = (
            key_relation(current.camelot, track.camelot) if current else "unknown"
        )
        energy = normalised_energy(track, crate)
        energy_err = abs(energy - target_energy)

        terms = {
            "bpm": bpm_score, "key": _KEY_SCORE.get(relation, 0.0),
            "energy": 1.0 - energy_err, "confidence": track.grid_confidence,
        }
        score = (
            W_BPM * _mult(weights, "bpm") * terms["bpm"]
            + W_KEY * _mult(weights, "key") * terms["key"]
            + W_ENERGY * _mult(weights, "energy") * terms["energy"]
            + W_CONFIDENCE * _mult(weights, "confidence") * terms["confidence"]
        )
        sim: dict[str, float] = {}
        if current is not None:
            sim = similarity(current, track, cache_dir)
            score += _similarity_score(sim, weights)
            terms.update(_similarity_terms(sim))
        if phase:
            score += W_PHASE * phase_fit(track, current, phase)
        if getattr(track, "review_needed", False):
            score -= REVIEW_PENALTY

        penalties: list[str] = []
        artist = getattr(track, "artist", "")
        if artist and artist.lower() in recent_artists:
            score -= ARTIST_PENALTY
            penalties.append(f"artist within {ARTIST_WINDOW}")
        if track.camelot and track.camelot in recent_keys:
            score -= KEY_REPEAT_PENALTY
            penalties.append(f"key within {KEY_WINDOW}")
        if last_vocal_led and getattr(track, "vocal_led", False):
            score -= VOCAL_BACK_TO_BACK_PENALTY
            penalties.append("vocal-led back to back")

        eligible.append(
            Candidate(
                track=track,
                score=score,
                bpm_delta_pct=delta * 100.0,
                key_relation=relation,
                energy_delta=energy - current_energy,
                set_phase=phase,
                penalties=tuple(penalties),
                similarity=sim,
                tempo_path=path,
                terms={k: round(float(v), 4) for k, v in terms.items()},
            )
        )

    eligible.sort(key=lambda c: (-c.score, c.track.title))
    return eligible


# --- planning a journey rather than a next track ----------------------------------

#: How many tracks ahead a plan looks, and how many partial plans are kept at
#: each step. Three deep is where a plan starts to mean something -- one step
#: is the old behaviour, two cannot show an arc -- and eight wide was enough on
#: this crate for a wider beam to stop changing the first pick.
JOURNEY_STEPS: int = 3
JOURNEY_BEAM: int = 8

#: Candidates considered at each step of the plan. The ranking is already
#: sorted, so this is the top of it.
JOURNEY_BRANCH: int = 6

#: A plan is worth no more than the sum of its steps, discounted: the track
#: after next is a guess about a room that has not happened yet, and it should
#: not outvote the track that plays in two minutes.
JOURNEY_DISCOUNT: float = 0.6

#: Weight of "is this going where the set is going" against everything else.
#: Enough to choose a slightly worse-matched track that moves the tempo, not
#: enough to take an unmixable one because it is the right speed.
W_TEMPO_JOURNEY: float = 2.0


@dataclass(frozen=True)
class Journey:
    """A plan: the next few tracks, in order, and what it is worth."""

    steps: tuple[Candidate, ...]
    score: float

    @property
    def first(self) -> Candidate | None:
        return self.steps[0] if self.steps else None

    def reason(self) -> str:
        if not self.steps:
            return "no journey"
        legs = " -> ".join(
            f"{c.track.title} ({c.track.bpm:.0f})" for c in self.steps
        )
        return f"journey {self.score:.2f}: {legs}"


def plan_journey(
    crate: list[TrackAnalysis],
    current: TrackAnalysis | None,
    played: set[str] | None = None,
    energy_direction: float = 0.0,
    set_phase: str | None = None,
    history: list[TrackAnalysis] | None = None,
    steps: int = JOURNEY_STEPS,
    cache_dir=None,
    weights: dict[str, float] | None = None,
    target_bpm: float | None = None,
    playing_bpm: float | None = None,
    travel: bool = False,
) -> Journey:
    """Plan the next few tracks together, and return the best plan.

    A beam search over the ranking: at each step the best few continuations of
    each surviving plan are kept. Picking the single best next track is greedy
    -- it will happily take the strongest match now and leave the set in a
    corner with nothing to follow it, which on this crate is what stranded the
    fastest and slowest tracks. Looking three ahead lets a slightly worse pick
    win when it opens a better path.

    Only the first step is ever played: the plan is re-made from what the room
    actually did. The rest is for the log, the UI, and Phase 5's tempo journey.
    """
    played = set(played or set())
    if current is not None:
        # The track on the deck is not a candidate three steps from now either:
        # without this the plan happily plays it again two tracks later.
        played = played | {current.track_id}
    # A plan step leaves the set at the tempo that step is *played* at, which
    # after a ride or a stretch is not the track's printed tempo. Carrying it
    # along the beam is what makes the depth-3 plan agree with what the room
    # will actually be doing when step two is cued.
    start_bpm = playing_bpm if playing_bpm else (current.bpm if current else 0.0)
    # A target is a request to travel: paths that hold the tempo would score
    # every step against a set that never actually moves, and the plan would
    # stall one mix in.
    travel = travel or target_bpm is not None
    beam: list[tuple[float, tuple[Candidate, ...], TrackAnalysis | None, set[str], float]] = [
        (0.0, (), current, played, start_bpm)
    ]
    for depth in range(max(1, steps)):
        extended: list[
            tuple[float, tuple[Candidate, ...], TrackAnalysis | None, set[str], float]
        ] = []
        for score, chain, tip, seen, tip_bpm in beam:
            ranked = rank_candidates(
                crate, tip, seen, energy_direction, set_phase=set_phase,
                history=list(history or []) + [c.track for c in chain],
                cache_dir=cache_dir, weights=weights,
                playing_bpm=tip_bpm or None, travel=travel,
            )
            if not ranked:
                extended.append((score, chain, tip, seen, tip_bpm))
                continue
            for candidate in ranked[:JOURNEY_BRANCH]:
                step_score = candidate.score
                played_bpm = candidate.track.bpm * (candidate.tempo_path.rate_b or 1.0)
                if target_bpm and tip_bpm > 0:
                    # Progress toward the tempo the set is going to, as a share
                    # of the distance left. This is what turns a wall into a
                    # journey: 100 -> 128 is not one mix, it is four, and each
                    # one only has to be blendable.
                    was = abs(target_bpm - tip_bpm)
                    now = abs(target_bpm - played_bpm)
                    if was > 1e-6:
                        step_score += W_TEMPO_JOURNEY * (was - now) / was
                extended.append((
                    score + (JOURNEY_DISCOUNT ** depth) * step_score,
                    chain + (candidate,),
                    candidate.track,
                    seen | {candidate.track.track_id},
                    played_bpm,
                ))
        if not extended:
            break
        extended.sort(key=lambda item: (-item[0], item[1][0].track.title if item[1] else ""))
        beam = extended[:JOURNEY_BEAM]
    best_score, best_chain, _tip, _seen, _tip_bpm = beam[0]
    return Journey(steps=best_chain, score=round(best_score, 4))


def nearest_tempo(
    crate: list[TrackAnalysis],
    current: TrackAnalysis | None,
    played: set[str] | None = None,
) -> Candidate | None:
    """The closest tempo in the crate, whatever the gap. None if nothing is left.

    The last resort when :func:`rank_candidates` offers nothing at all: a track
    whose tempo has no partner within the stretch range still has to be
    followed by something, and a hard cut needs no beat-matching. Scored 0 and
    labelled as what it is, so the log says why an unmatched track was chosen.
    Ties break on title, so the choice is deterministic.
    """
    played = played or set()
    options = [
        t for t in crate
        if t.track_id not in played
        and not (current is not None and t.track_id == current.track_id)
        # Still the system choosing, so quarantine still applies: a last resort
        # is not a reason to place a set on a grid that cannot hold one.
        and not getattr(t, "quarantined", False)
    ]
    if not options:
        return None
    if current is None or current.bpm <= 0:
        return Candidate(track=options[0], score=0.0, bpm_delta_pct=0.0,
                         key_relation="unknown", energy_delta=0.0,
                         penalties=("no tempo to match",))
    best = min(options, key=lambda t: (abs(t.bpm - current.bpm), t.title))
    delta = (best.bpm - current.bpm) / current.bpm
    relation = key_relation(current.camelot, best.camelot)
    energy = normalised_energy(best, crate) - normalised_energy(current, crate)
    return Candidate(
        track=best, score=0.0, bpm_delta_pct=delta * 100.0, key_relation=relation,
        energy_delta=energy,
        penalties=("nearest tempo: nothing within the stretch range",),
    )


def select_next(
    crate: list[TrackAnalysis],
    current: TrackAnalysis | None,
    played: set[str] | None = None,
    energy_direction: float = 0.0,
    bpm_tolerance: float = BPM_TOLERANCE,
    set_phase: str | None = None,
    history: list[TrackAnalysis] | None = None,
    cache_dir=None,
    weights: dict[str, float] | None = None,
) -> Candidate | None:
    """Best next track, or None if nothing in the crate fits.

    Pure: same inputs always give the same answer.
    """
    ranked = rank_candidates(
        crate, current, played, energy_direction, bpm_tolerance,
        set_phase=set_phase, history=history, cache_dir=cache_dir, weights=weights,
    )
    return ranked[0] if ranked else None
