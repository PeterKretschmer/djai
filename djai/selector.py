"""Choosing the next track. A pure function over the cache -- no LLM, no state.

THREADING CONTEXT: main / monitor thread. Pure and side-effect free, so it is
safe to call from anywhere except the audio thread (it iterates the crate).

The LLM never picks a track. It produces at most an *energy direction*, and
this module turns that into an actual choice using tempo, key and measured
energy. That separation is deliberate: track selection stays deterministic and
explainable, and a hallucinated title can never reach the decks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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
) -> list[Candidate]:
    """Score every eligible track, best first.

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

    current_bpm = current.bpm if current else None
    current_energy = normalised_energy(current, crate) if current else 0.5
    target_energy = min(
        1.0, max(0.0, current_energy + energy_direction * ENERGY_STEP * 2.0)
    )

    for track in crate:
        if track.track_id in played:
            continue
        if current is not None and track.track_id == current.track_id:
            continue

        if current_bpm:
            delta = (track.bpm - current_bpm) / current_bpm
            # Hard limit, independent of the preference window below: past this
            # the deck cannot match tempo without an audible artefact -- a
            # stretch this wide smears badly, and the resampling fallback would
            # shift pitch by more than a semitone. A track this far away must
            # never be offered, whatever `bpm_tolerance` is set to.
            if abs(delta) > config.MAX_STRETCH_RATIO:
                continue
            if abs(delta) > bpm_tolerance:
                continue
            bpm_score = 1.0 - abs(delta) / bpm_tolerance
        else:
            delta = 0.0
            bpm_score = 1.0

        relation = (
            key_relation(current.camelot, track.camelot) if current else "unknown"
        )
        energy = normalised_energy(track, crate)
        energy_err = abs(energy - target_energy)

        score = (
            W_BPM * bpm_score
            + W_KEY * _KEY_SCORE.get(relation, 0.0)
            + W_ENERGY * (1.0 - energy_err)
            + W_CONFIDENCE * track.grid_confidence
        )
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
            )
        )

    eligible.sort(key=lambda c: (-c.score, c.track.title))
    return eligible


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
) -> Candidate | None:
    """Best next track, or None if nothing in the crate fits.

    Pure: same inputs always give the same answer.
    """
    ranked = rank_candidates(
        crate, current, played, energy_direction, bpm_tolerance,
        set_phase=set_phase, history=history,
    )
    return ranked[0] if ranked else None
