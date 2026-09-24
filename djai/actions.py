"""The action language (SPEC §3): intent -> strategy -> execution.

THREADING CONTEXT: control thread (the REPL, the UI server's handlers). Builds
scheduled commands; never touches the audio thread except through the
scheduler, like everything else.

Three layers:

* **Intent** -- what the operator (or the model) wants, in words: "more
  tension", "cut to the next one".
* **Strategy** -- a named recipe that turns an intent into actions
  (:data:`STRATEGIES`). A small model picks one of a dozen names far more
  reliably than it writes a sequence of moves, so that is the default route.
* **Execution** -- actions, each a strict, flat JSON object, compiled into
  scheduled commands by tempo-relative trajectory generators.

The payload is ``{"intent": str, "strategy": name, "actions": [...]}``. Actions
are optional: a strategy with none is expanded; explicit actions are used as
given. Everything is checked against :data:`PLAN_SCHEMA` by a hand-written
validator (no dependency; the schema is also what Ollama constrains decoding
to), then :func:`repair` rejects or repairs dangerous combinations, then every
compiled command still has to pass the supervisor's own ``validate`` -- no
existing rule is bypassed. A payload that fails anywhere is reported, never
half-applied, and never raises into the scheduler.

Multi-deck coordination: every action in a plan shares one anchor (its
``start``) and places itself with ``at_bar``, and the whole plan runs under
one :class:`~djai.commands.CancelToken`, so a later plan supersedes it cleanly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from djai import config, glue, phrase
from djai.commands import (
    StartTransition,
    BeatJump,
    CancelToken,
    Command,
    ExitLoop,
    SetEQ,
    SetFilter,
    SetGain,
    SetLoop,
    SetPitch,
)
from djai.deck import SAMPLE_RATE

SCHEMA_VERSION: int = 1
MAX_ACTIONS: int = 12

DECKS: tuple[str, ...] = ("live", "incoming", "a", "b")
SHAPES: tuple[str, ...] = ("linear", "ease_in", "ease_out", "s_curve")
#: Where a plan's anchor falls. Every one is a grid line: nothing starts "now".
STARTS: tuple[str, ...] = ("next_beat", "next_bar", "next_4_bars", "next_8_bars")
_START_BEATS: dict[str, int] = {"next_beat": 1, "next_bar": 4, "next_4_bars": 16, "next_8_bars": 32}
FX: tuple[str, ...] = ("riser", "reverse", "filter_swell", "volume_dip")
TRANSITION_OPS: tuple[str, ...] = ("go", "cancel", "style")
TRANSITION_STYLES: tuple[str, ...] = (
    "auto", "bass_swap", "cut", "echo_out", "filter_sweep", "loop_roll_out",
    "drop_swap", "beat_repeat_in", "backspin", "double_drop", "reverb_out",
    "brake", "noise_riser", "filter_echo",
)


def _num(lo: float, hi: float) -> dict:
    return {"type": "number", "minimum": lo, "maximum": hi}


def _enum(values) -> dict:
    return {"enum": list(values)}


_COMMON = {"at_bar": _num(0, 32), "start": _enum(STARTS)}
_DECK = _enum(DECKS)
_SHAPE = _enum(SHAPES)


def _variant(kind: str, props: dict, required: tuple[str, ...]) -> dict:
    return {
        "type": "object",
        "properties": {"kind": {"const": kind}, **props, **_COMMON},
        "required": ["kind", *required],
        "additionalProperties": False,
    }


#: One schema per action kind; ``kind`` is the discriminator.
ACTION_SCHEMAS: dict[str, dict] = {
    "cue": _variant("cue", {"energy": _num(-1, 1), "mix_point": {
        "type": "integer", "minimum": 1, "maximum": 3}}, ()),
    "fader": _variant("fader", {"deck": _DECK, "to": _num(0, 1), "bars": _num(0, 32),
                                "shape": _SHAPE}, ("deck", "to")),
    "eq": _variant("eq", {"deck": _DECK, "band": _enum(("low", "mid", "high")),
                          "to": _num(0, 2), "bars": _num(0, 32), "shape": _SHAPE},
                   ("deck", "band", "to")),
    "filter": _variant("filter", {"deck": _DECK, "to": _num(-1, 1), "resonance": _num(0, 1),
                                  "bars": _num(0, 32), "shape": _SHAPE}, ("deck", "to")),
    "loop": _variant("loop", {"deck": _DECK, "beats": _enum((0.5, 1, 2, 4, 8, 16)),
                              "hold_bars": _num(0.25, 32)}, ("deck", "beats")),
    "roll": _variant("roll", {"deck": _DECK, "beats": _enum((0.125, 0.25, 0.5, 1)),
                              "bars": _num(0.25, 4)}, ("deck", "beats")),
    "beat_jump": _variant("beat_jump", {"deck": _DECK, "beats": {
        "type": "integer", "minimum": -64, "maximum": 64}}, ("deck", "beats")),
    "pitch": _variant("pitch", {"deck": _DECK, "percent": _num(-20, 20), "bars": _num(0, 32),
                                "shape": _SHAPE}, ("deck", "percent")),
    "fx": _variant("fx", {"effect": _enum(FX), "deck": _DECK, "beats": _num(0.25, 32),
                          "amount": _num(0, 1), "groove": _enum(glue.GROOVES)}, ("effect",)),
    "dynamics": _variant("dynamics", {"direction": _num(-1, 1)}, ("direction",)),
    "transition": _variant("transition", {"op": _enum(TRANSITION_OPS),
                                          "style": _enum(TRANSITION_STYLES)}, ("op",)),
}

#: The whole payload. Also passed to Ollama as ``format``.
PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "maxLength": 200},
        "strategy": None,  # filled in below, once STRATEGIES exists
        "actions": {"type": "array", "maxItems": MAX_ACTIONS,
                    "items": {"oneOf": list(ACTION_SCHEMAS.values())}},
    },
    "required": ["intent", "strategy"],
    "additionalProperties": False,
}


# --- the validator -----------------------------------------------------------------


def check(schema: dict, value: Any, path: str = "$") -> list[str]:
    """Every way ``value`` violates ``schema``. Empty means valid.

    The subset of JSON Schema this module uses, and nothing more: type, const,
    enum, minimum/maximum, maxLength, required, properties,
    additionalProperties=false, items, maxItems and oneOf. Numbers must be
    finite and booleans are never numbers.
    """
    errors: list[str] = []
    if "const" in schema and value != schema["const"]:
        return [f"{path}: must be {schema['const']!r}"]
    if "enum" in schema:
        # A bool is never an enum member: True == 1 must not pass as 1 beat.
        if isinstance(value, bool) or isinstance(value, (dict, list)) or value not in schema["enum"]:
            return [f"{path}: {value!r} is not one of {schema['enum']}"]
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected an object"]
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing {key!r}")
        for key, sub in value.items():
            if key not in props:
                if schema.get("additionalProperties") is False:
                    errors.append(f"{path}: unexpected {key!r}")
                continue
            errors += check(props[key], sub, f"{path}.{key}")
    elif t == "array":
        if not isinstance(value, list):
            return [f"{path}: expected an array"]
        if len(value) > schema.get("maxItems", len(value)):
            errors.append(f"{path}: more than {schema['maxItems']} items")
        items = schema.get("items")
        if items is not None:
            for i, item in enumerate(value[: schema.get("maxItems", len(value))]):
                errors += check(items, item, f"{path}[{i}]")
    elif t in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{path}: expected a number"]
        if isinstance(value, float) and not math.isfinite(value):
            return [f"{path}: must be finite"]  # (an int is always finite, however big)
        if t == "integer" and value != int(value):
            return [f"{path}: expected an integer"]
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            return [f"{path}: {value} outside {schema.get('minimum')}..{schema.get('maximum')}"]
    elif t == "string":
        if not isinstance(value, str):
            return [f"{path}: expected a string"]
        if len(value) > schema.get("maxLength", len(value)):
            errors.append(f"{path}: longer than {schema['maxLength']}")
    if "oneOf" in schema:
        # A discriminated union: pick the variant by `kind` so the error names
        # what was actually wrong, instead of eleven "must be" lines.
        kind = value.get("kind") if isinstance(value, dict) else None
        variant = ACTION_SCHEMAS.get(kind) if isinstance(kind, str) else None
        if variant is None:
            return [f"{path}: kind must be one of {sorted(ACTION_SCHEMAS)}"]
        errors += check(variant, value, path)
    return errors


# --- strategies (the middle layer) ------------------------------------------------------

#: name -> (what it is for, the actions it expands to). Short on purpose: the
#: descriptions are the model's whole menu.
STRATEGIES: dict[str, tuple[str, list[dict]]] = {
    "lift": ("more energy now, on this track",
             [{"kind": "dynamics", "direction": 0.8}]),
    "calm": ("less energy now, on this track",
             [{"kind": "dynamics", "direction": -0.7}]),
    "tension": ("build toward something",
                [{"kind": "fx", "effect": "riser", "beats": 16, "amount": 0.6},
                 {"kind": "fx", "effect": "filter_swell", "deck": "live", "beats": 16,
                  "amount": 0.5}]),
    "breathe": ("pull back for a moment, then return",
                [{"kind": "filter", "deck": "live", "to": -0.5, "bars": 2, "shape": "s_curve"},
                 {"kind": "filter", "deck": "live", "to": 0.0, "bars": 2, "shape": "s_curve",
                  "at_bar": 4}]),
    "stutter": ("a short roll on the playing track",
                [{"kind": "roll", "deck": "live", "beats": 0.5, "bars": 1}]),
    "rewind": ("a quick reverse on the playing track",
               [{"kind": "fx", "effect": "reverse", "deck": "live", "beats": 1}]),
    "next_now": ("move to the next track at the next phrase",
                 [{"kind": "transition", "op": "go"}]),
    "echo_out": ("next transition trails off in a delay",
                 [{"kind": "transition", "op": "style", "style": "echo_out"}]),
    "cut_next": ("next transition is a hard cut",
                 [{"kind": "transition", "op": "style", "style": "cut"}]),
    "cancel": ("abandon the transition in progress",
               [{"kind": "transition", "op": "cancel"}]),
    "harder_next": ("pick a harder next track",
                    [{"kind": "cue", "energy": 0.8}]),
    "none": ("nothing to do", []),
}
PLAN_SCHEMA["properties"]["strategy"] = _enum(STRATEGIES)


# --- trajectories -------------------------------------------------------------------------


def _shape(name: str, t: float) -> float:
    if name == "ease_in":
        return t * t
    if name == "ease_out":
        return 1.0 - (1.0 - t) ** 2
    if name == "s_curve":
        return t * t * (3.0 - 2.0 * t)
    return t


#: Nothing the language does is instant: a move takes at least a beat, in
#: sixteenth-note steps. A zero-length fader move is a hard cut, and a filter
#: flung from the detent in one step is a 3 ms fade -- both are what the
#: invariants forbid.
MIN_MOVE_BEATS: float = 1.0
STEPS_PER_BEAT: float = 4.0


def trajectory(v0: float, v1: float, beats: float, shape: str = "linear",
               steps_per_beat: float = STEPS_PER_BEAT) -> list[tuple[float, float]]:
    """``(beat offset, value)`` steps from ``v0`` to ``v1`` over ``beats``.

    Tempo-relative by construction: offsets are beats, turned into frames only
    when compiled against the master clock. The last step lands exactly on
    ``v1``. Zero beats is a single step (callers compiling a plan never ask
    for one: see :data:`MIN_MOVE_BEATS`). Each step is ramped over one block
    by the deck.
    """
    if beats <= 0:
        return [(0.0, v1)]
    n = max(1, min(64, int(math.ceil(beats * steps_per_beat))))
    return [(beats * i / n, v1 if i == n else v0 + (v1 - v0) * _shape(shape, i / n))
            for i in range(1, n + 1)]


# --- context and plan objects --------------------------------------------------------------


@dataclass
class DeckView:
    """What repair and compile need to know about one deck, read once."""

    name: str
    loaded: bool
    playing: bool
    in_transition: bool
    looping: bool
    reversing: bool
    gain: float
    eq: dict[str, float]
    filter_pos: float
    rate: float
    position: float
    beat_frames: float  # this deck's own frames per beat
    analysis: Any = None


@dataclass
class Context:
    live: str
    incoming: str
    frames: int
    master_beat: float
    master_bpm: float
    transition_active: bool
    decks: dict[str, DeckView]
    #: Engine frame an armed-but-not-started blend begins at, or None.
    blend_at: int | None = None

    @property
    def beat_frames(self) -> float:
        return SAMPLE_RATE * 60.0 / (self.master_bpm or 120.0)

    def anchor(self, start: str) -> int:
        """Engine frame of the next grid line of this kind, strictly ahead."""
        every = _START_BEATS.get(start, 1)
        beats_ahead = (math.floor(self.master_beat / every) + 1) * every - self.master_beat
        return int(round(self.frames + beats_ahead * self.beat_frames))


def context_of(session) -> Context:
    eng = session.engine
    frames, master_beat, pos_a, pos_b = eng.clock
    views = {}
    for name, pos in (("a", pos_a), ("b", pos_b)):
        d = eng.deck(name)
        t = d.track
        views[name] = DeckView(
            name=name, loaded=t is not None, playing=bool(d.playing),
            in_transition=eng.deck_in_transition(name), looping=bool(d.loop_active),
            reversing=bool(getattr(d, "reversing", False)), gain=float(d.gain.target),
            eq={"low": float(d.eq_low.target), "mid": float(d.eq_mid.target),
                "high": float(d.eq_high.target)},
            filter_pos=float(d.filter_pos.target), rate=float(d.rate), position=float(pos),
            beat_frames=(t.analysis.beat_period * SAMPLE_RATE) if t is not None else 0.0,
            analysis=t.analysis if t is not None else None,
        )
    return Context(
        live=session.live_deck, incoming=session.cued_deck(), frames=int(frames),
        master_beat=float(master_beat), master_bpm=float(eng.master_bpm or 120.0),
        transition_active=bool(eng.transition_active), decks=views,
        blend_at=min((c.execute_at for c in session.scheduler.pending()
                      if isinstance(c, StartTransition)), default=None),
    )


@dataclass
class Plan:
    """A payload that passed the schema, before repair."""

    intent: str
    strategy: str
    actions: list[dict]
    expanded: bool = False  # actions came from the strategy, not the payload


@dataclass
class Outcome:
    """What happened to one payload, end to end. Always returned, never raised."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    plan: Plan | None = None
    commands: list[Command] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    fallback: str | None = None

    def summary(self) -> str:
        if not self.ok:
            return "Refused: " + "; ".join(self.errors[:3])
        parts = [f"{self.plan.strategy}: {len(self.commands)} command(s) scheduled"]
        if self.repairs:
            parts.append("repaired: " + "; ".join(self.repairs))
        if self.rejected:
            parts.append(f"{len(self.rejected)} rejected by the supervisor")
        if self.replies:
            parts.append(" ".join(r for r in self.replies if r))
        if self.fallback:
            parts.append(f"(fallback: {self.fallback})")
        return " | ".join(parts)


# --- parse and validate -----------------------------------------------------------------------


def parse(payload: Any) -> tuple[Plan | None, list[str]]:
    """Text or an already-decoded object -> a schema-valid :class:`Plan`, or errors."""
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError, RecursionError) as exc:
            return None, [f"not JSON: {type(exc).__name__}"]
    try:
        errors = check(PLAN_SCHEMA, payload)
    except RecursionError:
        return None, ["nested too deeply"]
    if errors:
        return None, errors
    actions = list(payload.get("actions") or [])
    expanded = False
    if not actions:
        actions = [dict(a) for a in STRATEGIES[payload["strategy"]][1]]
        expanded = True
    return Plan(payload["intent"], payload["strategy"], actions, expanded), []


# --- repair (the supervisor's half of the language) -----------------------------------------------

#: Which control an action takes over, so two actions cannot fight for one.
def _control(a: dict, deck: str | None) -> tuple:
    kind = a["kind"]
    if kind == "fader":
        return (deck, "gain")
    if kind == "eq":
        return (deck, "eq", a["band"])
    if kind == "filter":
        return (deck, "filter")
    if kind in ("loop", "roll", "beat_jump"):
        return (deck, "playhead")
    if kind == "pitch":
        return (deck, "rate")
    if kind == "fx":
        eff = a["effect"]
        if eff == "reverse":
            return (deck, "playhead")
        if eff == "filter_swell":
            return (deck, "filter")
        return ("master", eff)
    return ("session", kind)


def _span_beats(a: dict) -> float:
    kind = a["kind"]
    if kind in ("fader", "eq", "filter", "pitch"):
        return 4.0 * a.get("bars", 0.0)
    if kind == "loop":
        return 4.0 * a.get("hold_bars", 1.0)
    if kind == "roll":
        return 4.0 * a.get("bars", 1.0)
    if kind == "fx":
        return a.get("beats", 4.0)
    return 0.0


#: A deck the room is hearing is never faded below this unless something else
#: is playing: dead air is the one failure this program does not allow.
MIN_LIVE_LEVEL: float = 0.3
#: Longest reverse, in beats. Longer stops reading as a gesture.
MAX_REVERSE_BEATS: float = 4.0


def repair(plan: Plan, ctx: Context) -> tuple[list[dict], list[str]]:
    """Reject or repair the dangerous parts of a plan. Returns (actions, notes).

    Session-level actions (cue, dynamics, transition) pass through; the
    session methods they call carry their own guards.
    """
    out: list[dict] = []
    notes: list[str] = []
    taken: dict[tuple, tuple[float, float]] = {}
    for a in plan.actions:
        a = dict(a)
        kind = a["kind"]
        deck = None
        view = None
        if "deck" in a or kind in ("fader", "eq", "filter", "loop", "roll", "beat_jump", "pitch") \
                or (kind == "fx" and a["effect"] in ("reverse", "filter_swell")):
            alias = a.get("deck", "live")
            deck = ctx.live if alias == "live" else ctx.incoming if alias == "incoming" else alias
            a["deck"] = deck
            view = ctx.decks[deck]
            if not view.loaded:
                notes.append(f"{kind} on deck {deck}: empty deck, dropped")
                continue
            owned = kind in ("fader", "eq", "filter", "loop", "roll", "pitch", "beat_jump") or (
                kind == "fx" and a["effect"] == "filter_swell")
            if view.in_transition and owned:
                notes.append(f"{kind} on deck {deck}: the blend owns this deck, dropped")
                continue
            if owned and ctx.blend_at is not None:
                ends = ctx.anchor(a.get("start", "next_beat")) + (
                    4.0 * a.get("at_bar", 0.0) + _span_beats(a)) * ctx.beat_frames
                if ends >= ctx.blend_at:
                    notes.append(f"{kind} on deck {deck}: would still be moving when the armed blend starts, dropped")
                    continue
            if kind in ("loop", "roll", "beat_jump") and view.reversing:
                notes.append(f"{kind} on deck {deck}: deck is reversing, dropped")
                continue
            if kind == "fx" and a["effect"] == "reverse" and view.looping:
                notes.append(f"reverse on deck {deck}: deck is looping, dropped")
                continue
        if kind == "fx" and a["effect"] == "riser" and ctx.transition_active:
            notes.append("riser: the blend owns the riser, dropped")
            continue
        if kind == "fx" and a["effect"] == "reverse" and a.get("beats", 1.0) > MAX_REVERSE_BEATS:
            a["beats"] = MAX_REVERSE_BEATS
            notes.append(f"reverse shortened to {MAX_REVERSE_BEATS:g} beats")
        if kind == "pitch":
            cap = config.MAX_STRETCH_RATIO * 100.0
            view = ctx.decks[deck]
            # The cap is on the deck's absolute rate, not on this move.
            current = (view.rate - 1.0) * 100.0
            target = current + a["percent"]
            clamped = max(-cap, min(cap, target))
            if abs(clamped - target) > 1e-9:
                a["percent"] = clamped - current
                notes.append(f"pitch clamped to the {cap:.0f}% stretch cap")
        if kind == "fader" and deck == ctx.live:
            other = ctx.decks[ctx.incoming]
            if a["to"] < MIN_LIVE_LEVEL and not other.playing:
                a["to"] = MIN_LIVE_LEVEL
                notes.append(f"fader on the live deck held at {MIN_LIVE_LEVEL:g}: nothing else is playing")
        if kind == "eq" and a["band"] == "low" and view is not None and a["to"] > 0.5:
            other = ctx.decks["b" if deck == "a" else "a"]
            if other.playing and view.playing and other.eq["low"] > 0.5 and not ctx.transition_active:
                a["to"] = 0.0
                notes.append("low EQ held down: two basslines at once")
        # Conflicts: two actions on one control at overlapping times.
        ctl = _control(a, deck)
        if ctl[0] not in ("session",):
            lo = 4.0 * a.get("at_bar", 0.0)
            hi = lo + max(_span_beats(a), 0.25)
            prev = taken.get(ctl)
            if prev is not None and lo < prev[1] and prev[0] < hi:
                notes.append(f"{kind} on {'/'.join(str(c) for c in ctl if c)}: conflicts with an earlier action, dropped")
                continue
            taken[ctl] = (lo, hi)
        out.append(a)
    return out, notes


# --- compile (execution layer) -------------------------------------------------------------------------


def _loop_start(view: DeckView, at_frame: int, ctx: Context) -> float:
    """The deck's beat line at or before where its playhead will be at ``at_frame``.

    At or before, never after: a loop that starts ahead of the playhead wraps
    it on the first block, and that is a click.
    """
    ahead = max(0, at_frame - ctx.frames) * max(view.rate, 0.0)
    pos = view.position + ahead
    beat = math.floor(phrase.beat_at_frame(view.analysis, pos) + 1e-3)
    return max(0.0, phrase.frame_at_beat(view.analysis, beat))


def compile_actions(actions: list[dict], ctx: Context, token: CancelToken,
                    origin: str = "action") -> tuple[list[Command], list[dict]]:
    """Deck actions -> scheduled commands; session actions returned as they are."""
    cmds: list[Command] = []
    session_actions: list[dict] = []
    bf = ctx.beat_frames
    for a in actions:
        kind = a["kind"]
        if kind in ("cue", "dynamics", "transition"):
            session_actions.append(a)
            continue
        base = ctx.anchor(a.get("start", "next_beat")) + int(round(4.0 * a.get("at_bar", 0.0) * bf))
        deck = a.get("deck")
        view = ctx.decks.get(deck) if deck else None

        def at(beats: float) -> int:
            return int(round(base + beats * bf))

        common = {"origin": origin, "token": token}
        if kind == "fader":
            for off, v in trajectory(view.gain, a["to"], max(MIN_MOVE_BEATS, 4 * a.get("bars", 0.0)), a.get("shape", "linear")):
                cmds.append(SetGain(deck=deck, gain=round(v, 4), execute_at=at(off), **common))
        elif kind == "eq":
            band = a["band"]
            for off, v in trajectory(view.eq[band], a["to"], max(MIN_MOVE_BEATS, 4 * a.get("bars", 0.0)), a.get("shape", "linear")):
                cmds.append(SetEQ(deck=deck, **{band: round(v, 4)}, execute_at=at(off), **common))
        elif kind == "filter":
            res = a.get("resonance")
            for i, (off, v) in enumerate(trajectory(view.filter_pos, a["to"], max(MIN_MOVE_BEATS, 4 * a.get("bars", 0.0)),
                                                    a.get("shape", "linear"))):
                cmds.append(SetFilter(deck=deck, position=round(v, 4),
                                      resonance=res if i == 0 else None, execute_at=at(off), **common))
        elif kind in ("loop", "roll"):
            start = _loop_start(view, base, ctx)
            length = a["beats"] * view.beat_frames
            hold = 4.0 * (a.get("hold_bars", 1.0) if kind == "loop" else a.get("bars", 1.0))
            cmds.append(SetLoop(deck=deck, start_frame=start, length_frames=length,
                                execute_at=base, **common))
            cmds.append(ExitLoop(deck=deck, execute_at=at(hold), **common))
        elif kind == "beat_jump":
            cmds.append(BeatJump(deck=deck, frames=a["beats"] * view.beat_frames,
                                 execute_at=base, **common))
        elif kind == "pitch":
            target = view.rate * (1.0 + a["percent"] / 100.0)
            for off, v in trajectory(view.rate, target, max(MIN_MOVE_BEATS, 4 * a.get("bars", 0.0)), a.get("shape", "linear")):
                cmds.append(SetPitch(deck=deck, rate=round(v, 6), execute_at=at(off), **common))
        elif kind == "fx":
            eff, amount = a["effect"], a.get("amount", 0.5)
            beats, groove = a.get("beats", 4.0), a.get("groove", "straight")
            if eff == "riser":
                cmds += glue.noise_riser(base, bf, beats=beats, peak=0.4 * amount,
                                         groove=groove, token=token, origin=origin)
            elif eff == "volume_dip":
                cmds += glue.volume_dip(base, bf, beats=beats, depth=1.0 - 0.6 * amount,
                                        groove=groove, token=token, origin=origin)
            elif eff == "filter_swell":
                cmds += glue.filter_swell(deck, base, bf, beats=beats, peak=0.8 * amount,
                                          groove=groove, token=token, origin=origin)
            elif eff == "reverse":
                cmds += glue.short_reverse(deck, base, bf, beats=beats, token=token, origin=origin)
    return cmds, session_actions


# --- execution against a session -----------------------------------------------------------------------


def run_session_action(session, a: dict) -> str:
    kind = a["kind"]
    if kind == "dynamics":
        return session.energy_correction(a["direction"]) or "No energy move (in a blend, or none asked)."
    if kind == "cue":
        session.cue_mix_point = a.get("mix_point")
        ok = session.cue_next(a.get("energy", 0.0), origin="action")
        return "Cued the next track." if ok else "Could not cue a track."
    op = a["op"]
    if op == "go":
        return session.force_transition()
    if op == "cancel":
        return session.cancel_transition()
    return session.set_transition_style(a.get("style", "auto"))


def execute(session, payload: Any, origin: str = "action") -> Outcome:
    """Parse, validate, repair, compile and schedule one payload. Never raises.

    A new plan supersedes the previous one: its pending commands are
    withdrawn first, so two plans never interleave on the same controls.
    """
    try:
        plan, errors = parse(payload)
        if plan is None:
            return Outcome(ok=False, errors=errors)
        ctx = context_of(session)
        actions, notes = repair(plan, ctx)
        previous = getattr(session, "_action_token", None)
        if previous is not None:
            # Superseded: withdraw the old plan, but close what it opened.
            for cmd in glue.closing_commands(session.scheduler.cancel(previous)):
                session.scheduler.submit(cmd)
        token = CancelToken(f"plan:{plan.strategy}")
        session._action_token = token
        cmds, session_actions = compile_actions(actions, ctx, token, origin)
        scheduled, rejected = [], []
        for cmd in cmds:
            rejection = session.supervisor.validate(cmd)
            if rejection is not None:
                rejected.append(f"{cmd.describe()}: {rejection.reason}")
                continue
            session.scheduler.submit(cmd)
            scheduled.append(cmd)
        replies = [run_session_action(session, a) for a in session_actions]
        outcome = Outcome(ok=True, repairs=notes, plan=plan, commands=scheduled,
                          rejected=rejected, replies=replies)
        session.session_log.write(
            "action_plan",
            trigger=f"{origin}: {plan.intent[:80]!r}",
            action=outcome.summary(),
            strategy=plan.strategy,
            actions=len(actions),
            repairs=notes,
            rejected=rejected,
        )
        return outcome
    except Exception as exc:  # the language layer never takes the set down
        return Outcome(ok=False, errors=[f"internal: {type(exc).__name__}: {exc}"])


# --- the model (llama3.1:8b) and its deterministic fallback -------------------------------------------------

SYSTEM_PROMPT = (
    "You turn a DJ's request into one JSON plan.\n"
    'Reply {"intent": <the request, short>, "strategy": <one name>, "actions": []}.\n'
    "Strategies:\n"
    + "\n".join(f"{name} - {desc}" for name, (desc, _) in STRATEGIES.items())
    + "\nUse a strategy and leave actions empty unless the request names a precise "
    "move (a deck, a filter, a fader, a loop, a pitch change). Never name a track.\n"
    "Decks: live (playing now), incoming (next), a, b. Filter -1..1: negative is "
    "darker (low-pass), positive thinner (high-pass). Bass is eq band low.\n"
    "Examples:\n"
    'User: go harder now\n{"intent": "harder now", "strategy": "lift", "actions": []}\n'
    'User: kill the bass on deck b over 2 bars\n{"intent": "kill bass b", "strategy": '
    '"none", "actions": [{"kind": "eq", "deck": "b", "band": "low", "to": 0, "bars": 2}]}\n'
    'User: jump back 4 beats\n{"intent": "jump back", "strategy": "none", "actions": '
    '[{"kind": "beat_jump", "deck": "live", "beats": -4}]}\n'
    'User: riser over 8 beats\n{"intent": "riser", "strategy": "none", "actions": '
    '[{"kind": "fx", "effect": "riser", "beats": 8}]}'
)

#: Keyword -> strategy, for when the model is down or its answer fails the schema.
_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("cancel", "abort", "undo the mix", "back out"), "cancel"),
    (("echo",), "echo_out"),
    (("cut",), "cut_next"),
    (("next", "mix now", "change track"), "next_now"),
    (("tension", "build", "riser"), "tension"),
    (("breathe", "breather", "pull back"), "breathe"),
    (("stutter", "roll"), "stutter"),
    (("rewind", "reverse"), "rewind"),
    (("calm", "chill", "down", "softer", "less"), "calm"),
    (("harder", "more", "up", "energy", "lift", "bang"), "lift"),
)


def fallback_plan(text: str) -> dict:
    """The deterministic route: a keyword picks the strategy, or nothing does."""
    low = (text or "").lower()
    for words, strategy in _KEYWORDS:
        if any(w in low for w in words):
            return {"intent": low[:200], "strategy": strategy, "actions": []}
    return {"intent": low[:200], "strategy": "none", "actions": []}


def ask_model(engine, text: str, state: dict) -> tuple[Any, str | None, float]:
    """One model call. Returns (payload or None, failure reason, seconds)."""
    import time

    import httpx

    started = time.time()
    if engine is None or not getattr(engine, "available", False):
        return None, "model unavailable", 0.0
    body = {
        "model": engine.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"State: {json.dumps(state, default=str)}\nUser: {text}"},
        ],
        "stream": False,
        "format": PLAN_SCHEMA,
        "keep_alive": "30m",
        "options": {"num_thread": 6, "temperature": 0.1, "num_predict": 300},
    }
    try:
        r = engine._client.post(f"{engine.base_url}/api/chat", json=body)
        r.raise_for_status()
        content = (r.json().get("message") or {}).get("content") or ""
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        return None, f"model call failed: {type(exc).__name__}", time.time() - started
    return content, None, time.time() - started


def act(session, text: str, engine=None) -> Outcome:
    """The operator's entry point: text in, a plan executed or a fallback used.

    JSON typed directly is executed as given. Otherwise the model is asked,
    and if it is unavailable or its answer fails the schema, the keyword
    fallback decides. Either way the reason is on the outcome, so a parse
    failure is visible rather than silently swallowed.
    """
    stripped = (text or "").strip()
    if stripped.startswith("{"):
        return execute(session, stripped, origin="operator:json")
    state = {"bpm": round(session.engine.master_bpm, 1), "transition": session.engine.transition_active}
    raw, why, _ = ask_model(engine, stripped, state)
    if raw is not None:
        outcome = execute(session, raw, origin="model")
        if outcome.ok:
            return outcome
        why = "model output failed the schema: " + "; ".join(outcome.errors[:2])
    outcome = execute(session, fallback_plan(stripped), origin="fallback")
    outcome.fallback = why
    return outcome
