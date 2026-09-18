"""The chat layer: natural language in, one structured action out.

THREADING CONTEXT: the **REPL thread** (or any worker). :meth:`IntentEngine.interpret`
makes a blocking HTTP call to a local Ollama server that can take seconds and
will spike CPU and GPU on this machine. It must never be reached from the audio
thread, the scheduler thread, or the monitor thread.

This is the only module that talks to an LLM, and it has no access to the decks.
It returns an :class:`Intent` -- a description of what the user seems to want --
and nothing more. Turning that into commands is :mod:`djai.cli`'s job, choosing
a track is :mod:`djai.selector`'s, and approving anything is
:mod:`djai.supervisor`'s. The model can therefore be wrong, slow, or malformed
without ever reaching the audio path.

Running the model locally changes the risk profile, not the architecture. An 8B
model follows instructions less reliably than a frontier API, so three things
guard the boundary:

1. Ollama constrains decoding to a JSON schema, with the six valid actions as a
   literal ``enum`` -- the model cannot emit an action that does not exist.
2. Anything that still gets through is parsed strictly here.
3. Whatever survives that is validated by :mod:`djai.supervisor` before it can
   reach the scheduler, exactly as before.

And because a local model fails more often than a hosted one, failure is not an
error path: it falls back to keyword matching so the program stays playable.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from djai import config

log = logging.getLogger(__name__)

#: The complete set of actions. Anything else is a parse failure, by design --
#: a hallucinated action name must never become a command.
ACTIONS: frozenset[str] = frozenset(
    {
        "next_track",
        "set_energy",
        "hold_blend",
        "skip_queued",
        "describe_state",
        # The model may ASK for a transition style by name. It never computes
        # timing, placement or curves -- those stay in transition.py and
        # phrase.py, and the supervisor rejects any name it does not know.
        "set_transition_style",
        # Where the set is heading. The selector turns the phase into an
        # intensity target; the model still never names a track.
        "set_phase",
        "none",
    }
)

#: The set phases the model may name. Mirrors djai.selector.SET_PHASES, which
#: intent.py does not import so it stays independent of the crate code.
SET_PHASES: tuple[str, ...] = ("warmup", "build", "peak", "cooldown")

#: Passed to Ollama as ``format`` so decoding is constrained to this shape. The
#: ``enum`` is the important part: it makes an invalid action unrepresentable
#: rather than merely discouraged.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(ACTIONS)},
        "params": {
            "type": "object",
            "properties": {
                "energy": {"type": "number"},
                "direction": {"type": "number"},
                "bars": {"type": "integer"},
                "phase": {"type": "string", "enum": ["warmup", "build", "peak", "cooldown"]},
            },
        },
        "reply": {"type": "string"},
    },
    "required": ["action", "params", "reply"],
}

#: Deliberately flat and short: no nested sections, no tags, no instructions to
#: reason or think. llama3.1:8b follows a list and three examples far better
#: than it follows prose about how to behave.
SYSTEM_PROMPT = """\
You turn a DJ's request into one JSON action.

Actions and their params:
next_track - play a different track next. params: {"energy": number -1 to 1}
set_energy - steer the whole set harder or calmer. params: {"direction": number -1 to 1}
hold_blend - keep both tracks blended longer. params: {"bars": integer 4 to 64}
skip_queued - cancel the queued commands. params: {}
describe_state - say what is playing. params: {}
set_transition_style - choose how the next blend is done. params: {"style": one of "bass_swap", "cut", "echo_out", "filter_sweep", "loop_roll_out", "drop_swap", "beat_repeat_in", "backspin", "double_drop", "reverb_out", "brake", "noise_riser", "filter_echo", "auto"}
set_phase - where the set is: params: {"phase": one of "warmup", "build", "peak", "cooldown"}
none - anything else. params: {}

Transition styles: bass_swap is the normal blend; cut is a hard switch;
echo_out fades the outgoing track into a delay; filter_sweep filters it away.
High-energy styles: loop_roll_out rolls a shrinking loop; drop_swap swaps on
the drop; beat_repeat_in stutters; backspin spins the record back; double_drop
plays both drops together.
Effects: reverb_out washes the track out in reverb; brake stops the record;
noise_riser builds noise into the next track; filter_echo filters and echoes.
auto lets the rules decide. You may name a style. You may never choose when a
transition happens or how long it lasts.

Negative numbers mean calmer, positive mean harder. reply is one short sentence
said back to the user. Never name a track; you cannot choose tracks.

Example 1
State: {"bpm": 124, "key": "8A", "bars_in": 48, "transition": false, "played": 3}
User: give me something harder
{"action": "next_track", "params": {"energy": 0.8}, "reply": "Cueing something harder for the next phrase."}

Example 2
State: {"bpm": 128, "key": "9A", "bars_in": 12, "transition": true, "played": 7}
User: what is playing right now
{"action": "describe_state", "params": {}, "reply": "128 BPM in 9A, twelve bars into the blend."}

Example 3
State: {"bpm": 122, "key": "5A", "bars_in": 80, "transition": false, "played": 2}
User: keep them blended a while longer
{"action": "hold_blend", "params": {"bars": 16}, "reply": "Holding the blend another 16 bars."}

Example 4
State: {"bpm": 126, "key": "7A", "bars_in": 64, "transition": false, "played": 5}
User: just slam straight into the next one
{"action": "set_transition_style", "params": {"style": "cut"}, "reply": "Next one comes in on a hard cut."}

Example 5
State: {"bpm": 124, "key": "4A", "bars_in": 90, "transition": false, "played": 9}
User: let this one trail off into a delay
{"action": "set_transition_style", "params": {"style": "echo_out"}, "reply": "Echoing this one out."}
"""

# --- designing a transition ---------------------------------------------------

#: Constrains decoding to the eleven fields. As with the command schema, the
#: enums are the load-bearing part: an invented curve name is unrepresentable
#: rather than merely discouraged.
TRANSITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "length_bars": {"type": "integer", "minimum": 4, "maximum": 32},
        "curve": {
            "type": "string",
            "enum": ["equal_power", "linear", "fast_in", "slow_in", "s_curve"],
        },
        "low_swap_bar": {"type": ["integer", "null"]},
        "low_swap_bars": {"type": "integer", "minimum": 1, "maximum": 8},
        "deck_a_high_rolloff": {"type": "boolean"},
        "deck_b_low_delay_bars": {"type": "integer", "minimum": 0, "maximum": 16},
        "echo_bars": {"type": "integer", "minimum": 0, "maximum": 4},
        "filter_sweep": {"type": "string", "enum": ["none", "hp_out", "lp_in"]},
        "filter_resonance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "entry_point": {"type": "string"},
        "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "loop_out_bars": {"type": "number", "minimum": 0.0, "maximum": 8.0},
        "loop_halving": {"type": "boolean"},
        "beat_repeat_division": {"type": "integer", "enum": [0, 8, 16]},
        "backspin_bars": {"type": "number", "minimum": 0.0, "maximum": 4.0},
        "double_drop_bars": {"type": "number", "minimum": 0.0, "maximum": 16.0},
        "align_mode": {"type": "string", "enum": ["phrase", "drop"]},
        "reverb_bars": {"type": "integer", "minimum": 0, "maximum": 4},
        "brake_beats": {"type": "integer", "enum": [0, 1, 2]},
        "riser_bars": {"type": "integer", "enum": [0, 4, 5, 6, 7, 8]},
        "vocal_aware": {"type": "boolean"},
    },
    "required": [
        "length_bars", "curve", "low_swap_bar", "low_swap_bars",
        "deck_a_high_rolloff", "deck_b_low_delay_bars", "echo_bars",
        "filter_sweep", "filter_resonance", "entry_point", "intensity",
        "loop_out_bars", "loop_halving", "beat_repeat_division",
        "backspin_bars", "double_drop_bars", "align_mode",
        "reverb_bars", "brake_beats", "riser_bars", "vocal_aware",
    ],
}

#: Deliberately short. llama3.1:8b degrades on long prompts, and everything
#: this omits is enforced downstream by the supervisor anyway.
TRANSITION_PROMPT = """You design DJ transitions. Return ONLY the JSON object.

You choose WHAT happens. You never choose WHEN: the engine places the
transition. Never output times, seconds, samples, or bar numbers in a track.
All bars are counted from the start of the transition itself.

Fields:
length_bars 4-32: how long the blend is.
curve: equal_power (even, safe), linear (flat), fast_in (incoming arrives
early and sits underneath), slow_in (incoming lands late, sudden), s_curve
(gentle ends, quick middle).
low_swap_bar: bar within the blend where the bass hands over, or null for no
managed swap.
low_swap_bars 1-8: how long that handover takes. 1 is a hard swap.
deck_a_high_rolloff: dull the outgoing track's top end as it leaves.
deck_b_low_delay_bars 0-16: hold the incoming bass out this many bars.
echo_bars 0-4: bars of delay on the outgoing track as it goes.
filter_sweep: none, hp_out (filter the outgoing track away), lp_in (incoming
opens up from dull).
filter_resonance 0.0-1.0: how much the sweep rings. 0 if none.
entry_point: "mix_in", or "hot_cue_N" using only a cue listed in the context
with "usable": true. A cue marked usable:false has too little track after it.
intensity 0.0-1.0: how far to take the effects.

High-energy tools, all off by default (0 or false):
loop_out_bars 0-8: loop the outgoing track's last bars. loop_halving halves
that loop as it runs, building tension.
beat_repeat_division 0, 8 or 16: stutter its last bar in eighth or sixteenth
slices.
backspin_bars 0-4: spin the outgoing record down and backwards.
double_drop_bars 0-16: play both drops together for this long.
align_mode: "phrase" lines up on a phrase boundary, "drop" lines the two
tracks' drops up. Use "drop" or double_drop_bars ONLY when both tracks list a
"drop" cue.

Effects, off by default: reverb_bars 0-4 sends the outgoing track through
echo into reverb; brake_beats 0, 1 or 2 stops its record; riser_bars 0 or 4-8
adds noise rising into the hand-over. vocal_aware true keeps vocals apart.

Design for THESE two tracks.

energy_direction positive means lift, negative means darker or calmer.
Positive: shorter blend, earlier swap, higher intensity, curve fast_in or
s_curve. Negative: longer blend, later swap, lower intensity, and consider
hp_out or echo_bars.

Then adjust for the numbers you were given:
- A large gap between deck_a energy and deck_b energy wants a shorter blend.
- Similar energies want a longer one.
- Keys that do not match want hp_out, or a longer deck_b_low_delay_bars.
- A wider bpm gap wants a shorter blend and a harder swap.
Use any value in range, not round numbers.

The examples show the shape of an answer; do not reuse their numbers.

"lift this"
{"length_bars":8,"curve":"fast_in","low_swap_bar":2,"low_swap_bars":1,"deck_a_high_rolloff":false,"deck_b_low_delay_bars":0,"echo_bars":0,"filter_sweep":"none","filter_resonance":0,"entry_point":"mix_in","intensity":0.85,"loop_out_bars":0,"loop_halving":false,"beat_repeat_division":0,"backspin_bars":0,"double_drop_bars":0,"align_mode":"phrase","reverb_bars":0,"brake_beats":0,"riser_bars":0,"vocal_aware":true}

"keep it dark"
{"length_bars":28,"curve":"slow_in","low_swap_bar":20,"low_swap_bars":6,"deck_a_high_rolloff":true,"deck_b_low_delay_bars":10,"echo_bars":3,"filter_sweep":"hp_out","filter_resonance":0.55,"entry_point":"mix_in","intensity":0.25,"loop_out_bars":0,"loop_halving":false,"beat_repeat_division":0,"backspin_bars":0,"double_drop_bars":0,"align_mode":"phrase","reverb_bars":0,"brake_beats":0,"riser_bars":0,"vocal_aware":true}

"slam it"
{"length_bars":4,"curve":"linear","low_swap_bar":0,"low_swap_bars":1,"deck_a_high_rolloff":false,"deck_b_low_delay_bars":0,"echo_bars":0,"filter_sweep":"none","filter_resonance":0,"entry_point":"hot_cue_2","intensity":1.0,"loop_out_bars":0,"loop_halving":false,"beat_repeat_division":0,"backspin_bars":0,"double_drop_bars":0,"align_mode":"phrase","reverb_bars":0,"brake_beats":0,"riser_bars":0,"vocal_aware":true}

"let it roll"
{"length_bars":16,"curve":"equal_power","low_swap_bar":null,"low_swap_bars":2,"deck_a_high_rolloff":false,"deck_b_low_delay_bars":4,"echo_bars":0,"filter_sweep":"none","filter_resonance":0,"entry_point":"mix_in","intensity":0.45,"loop_out_bars":0,"loop_halving":false,"beat_repeat_division":0,"backspin_bars":0,"double_drop_bars":0,"align_mode":"phrase","reverb_bars":0,"brake_beats":0,"riser_bars":0,"vocal_aware":true}

"bring it in on the drop"
{"length_bars":12,"curve":"s_curve","low_swap_bar":5,"low_swap_bars":2,"deck_a_high_rolloff":true,"deck_b_low_delay_bars":0,"echo_bars":1,"filter_sweep":"lp_in","filter_resonance":0.3,"entry_point":"hot_cue_3","intensity":0.7,"loop_out_bars":0,"loop_halving":false,"beat_repeat_division":0,"backspin_bars":0,"double_drop_bars":0,"align_mode":"phrase","reverb_bars":0,"brake_beats":0,"riser_bars":0,"vocal_aware":true}

"roll it out into the next one"
{"length_bars":8,"curve":"fast_in","low_swap_bar":0,"low_swap_bars":1,"deck_a_high_rolloff":false,"deck_b_low_delay_bars":0,"echo_bars":0,"filter_sweep":"none","filter_resonance":0,"entry_point":"mix_in","intensity":0.9,"loop_out_bars":4,"loop_halving":true,"beat_repeat_division":0,"backspin_bars":0,"double_drop_bars":0,"align_mode":"phrase","reverb_bars":0,"brake_beats":0,"riser_bars":0,"vocal_aware":true}

"stutter it and slam"
{"length_bars":4,"curve":"fast_in","low_swap_bar":0,"low_swap_bars":1,"deck_a_high_rolloff":false,"deck_b_low_delay_bars":0,"echo_bars":0,"filter_sweep":"none","filter_resonance":0,"entry_point":"mix_in","intensity":0.95,"loop_out_bars":0,"loop_halving":false,"beat_repeat_division":16,"backspin_bars":0,"double_drop_bars":0,"align_mode":"phrase","reverb_bars":0,"brake_beats":0,"riser_bars":0,"vocal_aware":true}

"double drop them" (both tracks list a hot cue labelled drop)
{"length_bars":12,"curve":"equal_power","low_swap_bar":0,"low_swap_bars":1,"deck_a_high_rolloff":false,"deck_b_low_delay_bars":0,"echo_bars":0,"filter_sweep":"none","filter_resonance":0,"entry_point":"mix_in","intensity":0.8,"loop_out_bars":0,"loop_halving":false,"beat_repeat_division":0,"backspin_bars":0,"double_drop_bars":12,"align_mode":"drop","reverb_bars":0,"brake_beats":0,"riser_bars":0,"vocal_aware":true}

"wash it out in reverb"
{"length_bars":8,"curve":"fast_in","low_swap_bar":4,"low_swap_bars":1,"deck_a_high_rolloff":true,"deck_b_low_delay_bars":0,"echo_bars":3,"filter_sweep":"none","filter_resonance":0,"entry_point":"mix_in","intensity":0.75,"loop_out_bars":0,"loop_halving":false,"beat_repeat_division":0,"backspin_bars":0,"double_drop_bars":0,"align_mode":"phrase","reverb_bars":3,"brake_beats":0,"riser_bars":0,"vocal_aware":true}

"stop the record and drop the next one in"
{"length_bars":4,"curve":"fast_in","low_swap_bar":3,"low_swap_bars":1,"deck_a_high_rolloff":false,"deck_b_low_delay_bars":0,"echo_bars":0,"filter_sweep":"none","filter_resonance":0,"entry_point":"mix_in","intensity":1.0,"loop_out_bars":0,"loop_halving":false,"beat_repeat_division":0,"backspin_bars":0,"double_drop_bars":0,"align_mode":"phrase","reverb_bars":0,"brake_beats":2,"riser_bars":0,"vocal_aware":true}

"build it with noise, filter and echo it out"
{"length_bars":16,"curve":"slow_in","low_swap_bar":14,"low_swap_bars":2,"deck_a_high_rolloff":false,"deck_b_low_delay_bars":8,"echo_bars":2,"filter_sweep":"hp_out","filter_resonance":0.45,"entry_point":"mix_in","intensity":0.8,"loop_out_bars":0,"loop_halving":false,"beat_repeat_division":0,"backspin_bars":0,"double_drop_bars":0,"align_mode":"phrase","reverb_bars":0,"brake_beats":0,"riser_bars":6,"vocal_aware":true}
"""

#: The revision pass. Deliberately shorter than the design prompt: by the time
#: this runs the deterministic rules have already made the safe corrections, and
#: the model is being asked for a second opinion on a narrow question.
REVISION_PROMPT = """You revise a DJ transition that has been rendered and
measured. Return ONLY the JSON object, with all the same fields.

You are given the parameters that were rendered, the measurements taken from
that render, and the corrections the rules already applied. Improve on them if
you can; return them unchanged if you cannot.

You choose WHAT happens. You never choose WHEN. Never output times, seconds,
samples, or bar numbers in a track. All bars are counted from the start of the
transition itself.

What the measurements mean:
loudness_dip_db: how far the mix sagged below the two tracks' own loudness.
Above 3 is a crossfade curve that is subtracting instead of blending;
equal_power is the safe shape, and a shorter length_bars sags for less time.
low_end_overlap_bars: bars with two basslines at once. Should be 0-2. Swap the
bass earlier (lower low_swap_bar) or faster (lower low_swap_bars).
spectral_clash: 0-1, how much the two midranges are the same shape and both
present. High is muddy: deck_a_high_rolloff true, or filter_sweep hp_out.
vocal_overlap_bars: bars with both singers. Should be 0. Hold the incoming deck
back with deck_b_low_delay_bars, or enter at a later hot cue.
peak_dbfs: master peak. Above -1.0 is too hot; lower intensity.
transient_density_ratio: below ~0.8 means the drums have smeared together.
phase_coherence: 0-1, how well the grids agree. Not yours to fix.

Change as little as possible. A revision that alters every field is worse than
one that alters the one field the numbers point at.
"""

# --- keyword fallback --------------------------------------------------------

#: Words that push the energy up or down, used to give a fallback action a
#: direction as well as a name.
_UP_WORDS = (
    "harder", "heavier", "bigger", "louder", "faster", "peak", "banger",
    "hype", "drive", "driving", "more energy", "pump", "rowdy", "intense",
    "up",
)
_DOWN_WORDS = (
    "calmer", "softer", "chill", "chiller", "mellow", "deeper", "slower",
    "quieter", "smooth", "relax", "wind down", "less energy", "down", "easy",
)

#: Words that name a transition style without naming it exactly. Checked
#: before the generic rules so "slam into the next one" is not read as
#: "next_track" with a shrug.
_STYLE_WORDS: tuple[tuple[str, str], ...] = (
    # The high-energy styles come first, being the more specific phrases:
    # "stutter it and slam" is a stutter, not a cut.
    (r"\b(loop roll|roll it out|roll out|rolling out|loop_roll_out)\b", "loop_roll_out"),
    (r"\b(drop swap|drop-swap|drop_swap|swap on the drop)\b", "drop_swap"),
    (r"\b(stutter|beat repeat|beat-repeat|beat_repeat_in)\b", "beat_repeat_in"),
    (r"\b(backspin|back spin|spinback|spin it back)\b", "backspin"),
    (r"\b(double drop|double-drop|double_drop)\b", "double_drop"),
    # Effects, before the plain echo and filter words they contain.
    (r"\b(filter and echo|filter echo|filter_echo|filter it and echo)\b", "filter_echo"),
    (r"\b(reverb|wash it out|wash out|reverb_out)\b", "reverb_out"),
    (r"\b(brake|power down|stop the record|turntable stop)\b", "brake"),
    (r"\b(riser|noise riser|white noise|noise_riser)\b", "noise_riser"),
    (r"\b(bass swap|bass-swap|bass_swap|normal blend|usual)\b", "bass_swap"),
    (r"\b(cut|slam|hard switch|straight in|straight into|jump straight)\b", "cut"),
    (r"\b(echo|delay|trail off|trail it|tail off)\b", "echo_out"),
    (r"\b(filter|sweep|high ?pass|filter out)\b", "filter_sweep"),
    (r"\b(auto|automatic|you (decide|choose)|whatever fits)\b", "auto"),
)


def _style_from_text(text: str) -> str | None:
    for pattern, style in _STYLE_WORDS:
        if re.search(pattern, text):
            return style
    return None


#: Ordered most specific first: the first pattern that matches wins.
_KEYWORD_RULES: tuple[tuple[str, str], ...] = (
    (
        r"\b(bass swap|bass-swap|bass_swap|cut|slam|hard switch|straight into|"
        r"echo|delay|trail off|filter|sweep|high ?pass|"
        r"loop roll|roll it out|roll out|drop swap|swap on the drop|stutter|"
        r"beat repeat|backspin|back spin|double drop|reverb|wash it out|brake|"
        r"power down|stop the record|riser|white noise)\b",
        "set_transition_style",
    ),
    (r"\b(warm ?up|warmup|peak time|peak hour|cool ?down|cooldown|"
     r"wind the set down|build the set|start building)\b", "set_phase"),
    (r"\b(skip|cancel|forget|nevermind|never mind|drop that|undo)\b", "skip_queued"),
    (r"\b(hold|blend|longer|stay|keep them|keep it going|extend)\b", "hold_blend"),
    (
        r"\b(what|which|status|state|playing|currently|now playing|tell me)\b",
        "describe_state",
    ),
    (r"\b(next|another|new track|change|switch|different|move on)\b", "next_track"),
    (r"\b(energy|vibe|mood|harder|heavier|calmer|softer|chill|mellow|deeper)\b",
     "set_energy"),
)


def _energy_from_text(text: str) -> float:
    """Read a direction out of the raw words. 0.0 when nothing points either way."""
    lowered = text.lower()
    score = 0.0
    for word in _UP_WORDS:
        if word in lowered:
            score += 1.0
    for word in _DOWN_WORDS:
        if word in lowered:
            score -= 1.0
    return max(-1.0, min(1.0, score))


def keyword_intent(text: str) -> "Intent | None":
    """Best-effort action from the raw text alone. None if nothing matches.

    This is the whole point of the fallback chain: a local model that times out
    or answers badly should cost the user a worse answer, not a dead prompt.
    """
    lowered = (text or "").lower()
    for pattern, action in _KEYWORD_RULES:
        if not re.search(pattern, lowered):
            continue
        params: dict[str, Any] = {}
        if action == "set_transition_style":
            style = _style_from_text(lowered)
            if style is None:
                continue  # matched a word but named no style; try the next rule
            params = {"style": style}
            reply = (
                "Letting the rules pick the transition."
                if style == "auto"
                else f"Next transition: {style.replace('_', ' ')}."
            )
        elif action == "set_phase":
            if re.search(r"\b(cool ?down|cooldown|wind the set down)\b", lowered):
                phase = "cooldown"
            elif re.search(r"\b(peak time|peak hour)\b", lowered):
                phase = "peak"
            elif re.search(r"\b(build the set|start building)\b", lowered):
                phase = "build"
            else:
                phase = "warmup"
            params = {"phase": phase}
            reply = f"Set phase: {phase}."
        elif action == "next_track":
            params = {"energy": _energy_from_text(lowered)}
            reply = "Cueing the next track."
        elif action == "set_energy":
            direction = _energy_from_text(lowered)
            params = {"direction": direction}
            reply = (
                "Taking the energy up." if direction > 0
                else "Bringing the energy down." if direction < 0
                else "Adjusting the energy."
            )
        elif action == "hold_blend":
            reply = "Holding the blend longer."
        elif action == "skip_queued":
            reply = "Dropping what was queued."
        else:
            reply = ""
        return Intent(action=action, params=params, reply=reply)
    return None


# --- the result --------------------------------------------------------------


@dataclass
class Intent:
    """One parsed instruction.

    ``fallback`` is set when the local model did not produce this and the
    keyword chain did; it carries the reason, and the REPL logs it.
    """

    action: str
    params: dict[str, Any] = field(default_factory=dict)
    reply: str = ""
    ok: bool = True
    error: str | None = None
    raw: str | None = None
    latency_s: float = 0.0
    fallback: str | None = None

    @property
    def energy_direction(self) -> float:
        """The energy value this intent implies, clamped to -1..1."""
        for key in ("energy", "direction", "level"):
            value = self.params.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(-1.0, min(1.0, float(value)))
        return 0.0

    @property
    def hold_bars(self) -> int:
        value = self.params.get("bars")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return int(max(4, min(64, value)))
        return 16


def parse_response(text: str) -> Intent:
    """Parse a model reply into an :class:`Intent`. Never raises.

    Strict about the *schema* -- an unknown action, a non-object ``params`` or a
    non-string ``reply`` is a failure, because those are what could turn into a
    bad command. The single formatting leniency is pulling the outermost JSON
    object out of surrounding prose, which costs nothing in safety and which a
    small model needs more often than a large one.
    """
    raw = (text or "").strip()
    if not raw:
        return Intent(action="none", ok=False, error="empty response", raw=text)

    data: Any
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return Intent(action="none", ok=False, error="not JSON", raw=text)
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            return Intent(
                action="none", ok=False, error=f"invalid JSON: {exc}", raw=text
            )

    if not isinstance(data, dict):
        return Intent(
            action="none", ok=False, error="top level is not an object", raw=text
        )

    action = data.get("action")
    if not isinstance(action, str) or action not in ACTIONS:
        return Intent(
            action="none", ok=False, error=f"unknown action {action!r}", raw=text
        )

    params = data.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return Intent(
            action="none", ok=False, error="params is not an object", raw=text
        )

    reply = data.get("reply", "")
    if not isinstance(reply, str):
        return Intent(action="none", ok=False, error="reply is not a string", raw=text)

    return Intent(action=action, params=params, reply=reply, ok=True)


# --- the engine --------------------------------------------------------------


class IntentEngine:
    """Wraps the local Ollama server. Blocking; REPL thread only."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout_s = (
            config.OLLAMA_TIMEOUT_S if timeout_s is None else float(timeout_s)
        )
        #: False once a warmup has failed: the REPL then goes straight to the
        #: keyword chain instead of paying a timeout on every command.
        self.available: bool = True
        self._client = httpx.Client(timeout=self.timeout_s)

    def close(self) -> None:
        self._client.close()

    # --- transport -----------------------------------------------------------

    def _body(self, user_content: str, num_predict: int = 200) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "format": RESPONSE_SCHEMA,
            # Keeps the model resident in VRAM between commands. Without it a
            # cold reload costs multiple seconds on the next thing typed.
            "keep_alive": "30m",
            "options": {
                # Six of sixteen hardware threads. The remaining headroom is
                # what the audio callback lives in; do not raise this.
                "num_thread": 6,
                "temperature": 0.1,
                "num_predict": num_predict,
            },
        }

    def warmup(self, timeout_s: float | None = None) -> tuple[bool, str]:
        """Load the model into VRAM before the audio stream opens.

        Returns ``(ok, detail)``. Never raises: an unreachable Ollama is a
        degraded mode, not a fatal error.
        """
        limit = config.OLLAMA_WARMUP_TIMEOUT_S if timeout_s is None else timeout_s
        started = time.time()
        try:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json=self._body("State: {}\nUser: hello", num_predict=1),
                timeout=limit,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            self.available = False
            detail = f"HTTP {exc.response.status_code} from {self.base_url}"
            if exc.response.status_code == 404:
                detail += f" (is {self.model!r} pulled? try: ollama pull {self.model})"
            return False, detail
        except httpx.TimeoutException:
            # Reachable but still loading. A cold read of an 8B model off disk
            # measured ~49 s on this machine and can exceed the warmup budget,
            # so this is NOT treated as unavailable: the server is there and
            # will be ready shortly. Only the first command pays for it.
            return False, (
                f"still loading after {limit:.0f}s - the first command may be "
                f"slow, then it stays resident"
            )
        except httpx.RequestError as exc:
            self.available = False
            return False, f"cannot reach Ollama at {self.base_url} ({exc})"

        self.available = True
        return True, f"{self.model} ready in {time.time() - started:.1f}s"

    def design_transition(
        self, context: dict[str, Any], timeout_s: float | None = None
    ) -> tuple[dict[str, Any] | None, str]:
        """Ask the model to design one transition. Never raises.

        Returns ``(params, "")`` on success or ``(None, reason)`` on any
        failure -- unreachable, slow, malformed, whatever. The caller's answer
        to every one of those is the same: use the rule-based preset. That is
        why this returns a reason rather than throwing: a transition is due
        either way, and the set does not stop for a model.

        The result is UNVALIDATED. It has been shape-constrained by Ollama's
        ``format``, which is not the same as being sane; `supervisor.py` is
        what decides whether it may run.
        """
        if not self.available:
            return None, "model unavailable"

        limit = self.timeout_s if timeout_s is None else float(timeout_s)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": TRANSITION_PROMPT},
                {"role": "user", "content": json.dumps(context, separators=(",", ":"))},
            ],
            "stream": False,
            "format": TRANSITION_SCHEMA,
            "keep_alive": "30m",
            "options": {
                "num_thread": 6,
                # Warmer than the command path on purpose: two transitions in a
                # row should not be the same shape just because the tracks were
                # similar. Measured -- at 0.7 an 8B model reproduced the
                # few-shot examples verbatim and returned two distinct designs
                # across twenty calls. The supervisor is what keeps the range
                # safe, so the temperature can afford to be high.
                "temperature": 1.0,
                "top_p": 0.95,
                "num_predict": 220,
            },
        }
        try:
            response = self._client.post(
                f"{self.base_url}/api/chat", json=body, timeout=limit
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except httpx.TimeoutException:
            return None, f"timed out after {limit:.0f}s"
        except httpx.HTTPStatusError as exc:
            return None, f"HTTP {exc.response.status_code}"
        except httpx.RequestError as exc:
            self.available = False
            return None, f"cannot reach Ollama ({exc})"
        except (KeyError, ValueError) as exc:
            return None, f"unreadable response ({exc})"

        raw = (content or "").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end <= start:
                return None, "not JSON"
            try:
                data = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None, "not JSON"
        if not isinstance(data, dict):
            return None, f"expected an object, got {type(data).__name__}"
        return data, ""

    def revise_transition(
        self, context: dict[str, Any], timeout_s: float | None = None
    ) -> tuple[dict[str, Any] | None, str]:
        """Ask the model to revise one transition, given what it measured.

        Same contract as :meth:`design_transition`: never raises, returns
        ``(params, "")`` or ``(None, reason)``, and the result is UNVALIDATED
        until `supervisor.py` has been through it.

        ``context`` carries the current parameters and the measurements as
        NUMBERS AND TEXT. No audio reaches this call in any form -- not samples,
        not a spectrogram, not an encoding of either. The model is text-only;
        measuring the render and describing it is the whole mechanism.

        This pass is optional and advisory. The deterministic rules in
        :mod:`djai.revise` have already run and are authoritative; whatever
        comes back here is either an improvement on their answer or discarded.
        """
        if not self.available:
            return None, "model unavailable"

        limit = self.timeout_s if timeout_s is None else float(timeout_s)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": REVISION_PROMPT},
                {"role": "user", "content": json.dumps(context, separators=(",", ":"))},
            ],
            "stream": False,
            "format": TRANSITION_SCHEMA,
            "keep_alive": "30m",
            "options": {
                "num_thread": 6,
                # Cold, unlike the design pass. This is a correction against
                # measurements, not a creative choice: the same numbers should
                # produce the same fix.
                "temperature": 0.1,
                "num_predict": 220,
            },
        }
        try:
            response = self._client.post(
                f"{self.base_url}/api/chat", json=body, timeout=limit
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except httpx.TimeoutException:
            return None, f"timed out after {limit:.1f}s"
        except httpx.HTTPStatusError as exc:
            return None, f"HTTP {exc.response.status_code}"
        except httpx.RequestError as exc:
            self.available = False
            return None, f"cannot reach Ollama ({exc})"
        except (KeyError, ValueError) as exc:
            return None, f"unreadable response ({exc})"

        raw = (content or "").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end <= start:
                return None, "not JSON"
            try:
                data = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None, "not JSON"
        if not isinstance(data, dict):
            return None, f"expected an object, got {type(data).__name__}"
        return data, ""

    def interpret(self, user_text: str, state: dict[str, Any]) -> Intent:
        """Resolve one line of user text. Never raises.

        Every failure mode -- unreachable, timeout, bad HTTP, malformed body,
        schema violation -- lands in the fallback chain rather than surfacing an
        error and stopping.
        """
        started = time.time()

        if not self.available:
            return self._fallback(
                user_text, "ollama unavailable (warmup failed)", started
            )

        content = (
            f"State: {json.dumps(state, separators=(',', ':'), default=str)}\n"
            f"User: {user_text}"
        )
        try:
            response = self._client.post(
                f"{self.base_url}/api/chat", json=self._body(content)
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException:
            return self._fallback(
                user_text, f"timeout after {self.timeout_s:.0f}s", started
            )
        except httpx.HTTPStatusError as exc:
            return self._fallback(
                user_text, f"HTTP {exc.response.status_code}", started
            )
        except httpx.RequestError as exc:
            return self._fallback(
                user_text, f"connection error: {type(exc).__name__}", started
            )
        except ValueError as exc:  # response body was not JSON
            return self._fallback(user_text, f"bad response body: {exc}", started)

        text = ""
        if isinstance(payload, dict):
            message = payload.get("message")
            if isinstance(message, dict):
                text = message.get("content") or ""

        intent = parse_response(text)
        if not intent.ok:
            log.warning("intent parse failed (%s): %r", intent.error, text)
            return self._fallback(user_text, intent.error or "parse failed", started, text)

        intent.latency_s = time.time() - started
        return intent

    # --- fallback chain ------------------------------------------------------

    def _fallback(
        self,
        user_text: str,
        reason: str,
        started: float,
        raw: str | None = None,
    ) -> Intent:
        """Keyword match first; an inert ``none`` carrying the raw text second."""
        intent = keyword_intent(user_text)
        if intent is None:
            intent = Intent(action="none", reply=raw if raw else user_text)
        intent.ok = True
        intent.fallback = reason
        intent.error = reason
        intent.raw = raw
        intent.latency_s = time.time() - started
        return intent
