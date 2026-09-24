"""Song lookup and suggestions for the operator (Phase 3.2).

THREADING CONTEXT: control thread only (REPL, UI server, chat). Pure functions
over the crate; nothing here touches the engine.

**Lookup** is deterministic fuzzy matching on title and artist. The language
model may extract the search text from a chat line; it never picks the track.
One clear match returns it; several close ones return all of them for the
operator to choose from; nothing close enough says so. It never guesses.

**Suggest** is the selector's own ranking (djai.selector.rank_candidates),
shown with every reason it used, plus mix-point quality and plan fit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from djai import selector
from djai.analysis import TrackAnalysis

#: A query token this similar to a track token counts as that token. Tolerates
#: one wrong, missing or swapped letter in a word of six or so.
TOKEN_MATCH: float = 0.8
#: A track scoring at least this is a match at all.
MATCH_SCORE: float = 0.8
#: Every match within this of the best is a candidate: the query did not
#: separate them, so the operator chooses.
AMBIGUITY_MARGIN: float = 0.05

#: Words that describe a version rather than name a song. Dropped from the
#: query so "animals remix" still finds "Animals"; kept in the track's own
#: tokens so "levels skrillex" can still prefer the Skrillex remix.
TAG_WORDS: frozenset[str] = frozenset({
    "feat", "ft", "featuring", "remix", "edit", "mix", "radio", "extended",
    "original", "vip", "bootleg", "rework", "version", "dub", "club", "the",
})

_BRACKETS = re.compile(r"[\(\[][^\)\]]*[\)\]]")
_FEAT = re.compile(r"\b(feat|ft|featuring)\b\.?.*$", re.IGNORECASE)
_NON_WORD = re.compile(r"[^\w\s]")


def _fold(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.lower().replace("'", "")).split())


def core_title(title: str) -> str:
    """The song's name: no brackets, no featured artists, no version suffix
    ("Heads Will Roll - A-Trak Remix Radio Edit"), no punctuation."""
    parts = (title or "").split(" - ")
    while len(parts) > 1 and set(_tokens(parts[-1])) & TAG_WORDS:
        parts.pop()
    return _fold(_FEAT.sub("", _BRACKETS.sub(" ", " - ".join(parts))))


def _tokens(text: str) -> list[str]:
    return _fold(text).split()


def _query_tokens(query: str) -> list[str]:
    return [t for t in _tokens(query) if t not in TAG_WORDS]


def match_score(query: str, track: TrackAnalysis) -> float:
    """How well ``query`` names ``track``, 0..1.

    The better of two readings: each query word against the track's words
    (partial queries, reordered words, typos), and the whole query against the
    song's core title (a typo spread across a short title).
    """
    words = _query_tokens(query)
    if not words:
        return 0.0
    artist = getattr(track, "artist", "") or ""
    own = _tokens(f"{track.title} {artist}")
    if not own:
        return 0.0
    per_word = []
    for w in words:
        best = max(SequenceMatcher(None, w, t).ratio() for t in own)
        per_word.append(best if best >= TOKEN_MATCH else 0.0)
    token_score = sum(per_word) / len(per_word)
    whole = SequenceMatcher(None, " ".join(words), core_title(track.title)).ratio()
    return max(token_score, whole)


@dataclass(frozen=True)
class Lookup:
    """``status`` is "match", "ambiguous" or "none"."""

    status: str
    tracks: tuple[TrackAnalysis, ...] = ()
    scores: tuple[float, ...] = ()

    @property
    def track(self) -> TrackAnalysis | None:
        return self.tracks[0] if self.status == "match" else None


def lookup(crate: list[TrackAnalysis], query: str, limit: int = 5) -> Lookup:
    """Find the track ``query`` names. Deterministic; never guesses."""
    query = (query or "").strip()
    if not query:
        return Lookup("none")
    for t in crate:
        if t.track_id == query:
            return Lookup("match", (t,), (1.0,))
    scored = sorted(
        ((match_score(query, t), t) for t in crate),
        key=lambda st: (-st[0], st[1].title),
    )
    scored = [(s, t) for s, t in scored if s >= MATCH_SCORE]
    if not scored:
        return Lookup("none")
    best = scored[0][0]
    close = [(s, t) for s, t in scored if best - s <= AMBIGUITY_MARGIN]
    if len(close) > 1:
        # Several read equally well. One whose core title IS the query wins;
        # "levels" names "Levels", not "Levels x Slide".
        # A typo in the whole title ("levles") still counts as naming it.
        wanted = " ".join(_query_tokens(query))
        exact = [(s, t) for s, t in close
                 if SequenceMatcher(None, wanted, core_title(t.title)).ratio() >= TOKEN_MATCH]
        if len(exact) == 1:
            return Lookup("match", (exact[0][1],), (exact[0][0],))
        close = close[:limit]
        return Lookup("ambiguous", tuple(t for _, t in close),
                      tuple(round(s, 3) for s, _ in close))
    return Lookup("match", (scored[0][1],), (round(best, 3),))


# --- suggestions --------------------------------------------------------------


def mix_quality(current: TrackAnalysis | None, track: TrackAnalysis) -> float | None:
    """Best mix-out quality of the playing track times the candidate's best
    mix-in quality (djai.understanding.mix_regions), or None if either is
    missing."""
    in_q = track.mix_in_regions[0]["quality"] if track.mix_in_regions else None
    if current is None:
        return in_q
    out_q = current.mix_out_regions[0]["quality"] if current.mix_out_regions else None
    if in_q is None or out_q is None:
        return None
    return round(float(in_q) * float(out_q), 4)


@dataclass(frozen=True)
class Suggestion:
    """One ranked candidate with every reason the ranking used."""

    track: TrackAnalysis
    rank: int
    score: float
    bpm_gap_pct: float
    #: How far the incoming deck is stretched from its native tempo, in %.
    stretch_pct: float
    #: direct, ride, half or double.
    tempo_match: str
    key_relation: str
    energy_delta: float
    similarity: float
    mix_quality: float | None
    plan_fit: float = 0.0
    penalties: tuple[str, ...] = field(default_factory=tuple)
    #: The ranking's reason terms before their weights (selector.Candidate.terms
    #: plus plan fit): logged when shown, so a later cue can be learned from.
    terms: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        t = self.track
        return {
            "rank": self.rank, "track_id": t.track_id, "title": t.title,
            "artist": getattr(t, "artist", ""), "bpm": round(float(t.bpm), 1),
            "camelot": t.camelot, "score": round(self.score, 3),
            "bpm_gap_pct": round(self.bpm_gap_pct, 1),
            "stretch_pct": round(self.stretch_pct, 2),
            "tempo_match": self.tempo_match, "key_relation": self.key_relation,
            "energy_delta": round(self.energy_delta, 3),
            "similarity": round(self.similarity, 3),
            "mix_quality": self.mix_quality, "plan_fit": round(self.plan_fit, 3),
            "penalties": list(self.penalties),
        }

    def line(self) -> str:
        mq = "?" if self.mix_quality is None else f"{self.mix_quality:.2f}"
        how = self.tempo_match
        if how in ("half", "double"):
            how = f"{how}-time match"
        return (
            f"{self.rank}. {self.track.title} | {self.track.bpm:.1f} BPM "
            f"({self.bpm_gap_pct:+.1f}%, stretch {self.stretch_pct:+.1f}%, {how}) | "
            f"{self.track.camelot} {self.key_relation} | energy {self.energy_delta:+.2f} | "
            f"similarity {self.similarity:.2f} | mix point {mq} | plan fit {self.plan_fit:.2f}"
        )


#: Weight of plan fit in the suggestion ranking, against the selector's score.
W_PLAN_FIT: float = 1.5


def suggest(
    crate: list[TrackAnalysis],
    current: TrackAnalysis | None,
    excluded: set[str],
    n: int = 5,
    plan_fit=None,
    **rank_kwargs,
) -> list[Suggestion]:
    """The top ``n`` next tracks, each with its reasons.

    ``excluded`` is played and unplayable ids; quarantined tracks are dropped
    by the ranking itself. ``plan_fit(track) -> 0..1`` adds the set plan's
    opinion (Phase 3.3); without it every fit is 0.
    """
    ranked = selector.rank_candidates(crate, current, excluded, **rank_kwargs)
    w_fit = W_PLAN_FIT * float((rank_kwargs.get("weights") or {}).get("plan_fit", 1.0))
    rows = []
    for c in ranked:
        fit = float(plan_fit(c.track)) if plan_fit else 0.0
        path = c.tempo_path
        sim = c.similarity or {}
        rows.append((c.score + w_fit * fit, c, fit, path, sim))
    rows.sort(key=lambda r: (-r[0], r[1].track.title))
    out = []
    for i, (score, c, fit, path, sim) in enumerate(rows[: max(1, n)], start=1):
        out.append(Suggestion(
            track=c.track, rank=i, score=score, bpm_gap_pct=c.bpm_delta_pct,
            stretch_pct=((path.rate_b if path and path.rate_b else 1.0) - 1.0) * 100.0,
            tempo_match=path.technique if path else "direct",
            key_relation=c.key_relation, energy_delta=c.energy_delta,
            similarity=(sum(sim.values()) / len(sim)) if sim else 0.0,
            mix_quality=mix_quality(current, c.track), plan_fit=fit,
            penalties=c.penalties, terms={**c.terms, "plan_fit": round(fit, 4)},
        ))
    return out
