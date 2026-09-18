r"""The one transition v1 implements: a bass-swap crossfade.

THREADING CONTEXT: :func:`gains_at` is called once per block from the **audio
thread**. It is pure float arithmetic over its argument -- no state, no
allocation beyond the small result tuple, no branching on anything but the
progress value. The engine owns the progress counter; this module owns the
shape.

The shape, in bars (default 16, see :data:`TRANSITION_BARS`)::

    bar    0      2      4      6      8     10     12     14     16
           |------|------|------|------|------|------|------|------|
    A gain 1.000                      0.707                    0.000
           `------------- cos, equal power ------------------------'
    B gain 0.000                      0.707                    1.000
           `------------- sin, equal power ------------------------'
    A low  1111111111111111111111111111|0000000000000000000000000000
    B low  0000000000000000000000000000|1111111111111111111111111111
                                       ^ bass hands over here

Two properties are load-bearing, and both were measured, not assumed:

**Constant power.** ``from_gain**2 + to_gain**2 == 1`` at every point, so two
uncorrelated tracks sum to the level of one. The previous shape held *both*
decks at unity gain from bar 8 to bar 20 -- 12 of 32 bars, ~23 s -- which
measured +5.1 to +8.8 dB over a single deck and drove the master limiter into
clipping on 26-45% of blocks. That was the "both tracks play over each other at
full level" fault.

**One bassline, always.** The low band hands over instantly at the midpoint
rather than crossfading. A crossfaded swap leaves both decks partially
attenuated for its duration; an instant handover means exactly one deck owns
the low end at every instant. The deck's own one-block parameter ramp
(:mod:`djai.deck`) declicks the step, so it is a ~46 ms transfer on a bar line,
which is what a DJ does with a bass kill.

Mid and high bands are never touched by the transition -- the engine leaves
them wherever the user or the LLM put them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import NamedTuple

from djai import config

# --- shape constants ---------------------------------------------------------

#: Total length of the transition, in bars. Config-driven; default 16.
TRANSITION_BARS: float = config.TRANSITION_BARS

#: Where the low band hands over, as a fraction of the transition. 0.5 puts it
#: at the midpoint, where both decks are at equal gain.
BASS_SWAP_AT: float = 0.5

_HALF_PI = math.pi / 2.0


class TransitionGains(NamedTuple):
    """Deck parameters at one instant of the transition."""

    from_gain: float
    from_low: float
    to_gain: float
    to_low: float


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def gains_at(
    progress_bars: float, transition_bars: float | None = None
) -> TransitionGains:
    """Deck gains and low-band levels ``progress_bars`` into the transition.

    Defined for any input: clamped before 0 and after the transition length, so
    a late or early call is still well behaved.
    """
    length = TRANSITION_BARS if transition_bars is None else transition_bars
    if length <= 0:
        return TransitionGains(from_gain=0.0, from_low=0.0, to_gain=1.0, to_low=1.0)

    progress = _clamp01(progress_bars / length)

    # Equal power: the squares sum to exactly 1 at every point.
    from_gain = math.cos(_HALF_PI * progress)
    to_gain = math.sin(_HALF_PI * progress)

    # Instant handover, so exactly one deck owns the low end at any instant.
    swapped = progress >= BASS_SWAP_AT
    from_low = 0.0 if swapped else 1.0
    to_low = 1.0 if swapped else 0.0

    return TransitionGains(
        from_gain=from_gain, from_low=from_low, to_gain=to_gain, to_low=to_low
    )


def transition_frames(
    bpm: float,
    sample_rate: int,
    beats_per_bar: int = 4,
    transition_bars: float | None = None,
) -> int:
    """How many output frames the transition lasts at ``bpm``."""
    length = TRANSITION_BARS if transition_bars is None else transition_bars
    seconds = length * beats_per_bar * 60.0 / bpm
    return int(round(seconds * sample_rate))


# =============================================================================
# The transition library
# =============================================================================
#
# Every style is expressed as a PRECOMPUTED ENVELOPE: a float32 array with one
# row per audio block, built on the control thread before the transition
# starts. The audio thread does one bounds-clamped row lookup per block and
# assigns the values. No curve maths, no branching on style, and no allocation
# ever happens in the callback -- which is what lets a filter sweep and an echo
# tail cost exactly what a bass swap costs.
#
# `bass_swap`'s shape is unchanged; it is now sampled from the same array as
# everything else rather than recomputed per block, which yields identical
# numbers by construction (see tests/test_transition_library.py).

#: The four curve styles. ``cue_jump_in`` is deliberately NOT here: it is an
#: entry method -- where deck B starts from -- and combines with any of these.
STYLES: tuple[str, ...] = (
    "bass_swap", "cut", "echo_out", "filter_sweep",
    # High-energy set. All five are shapes over the same envelope: the loop
    # ones move LOOP_BEATS, the backspin moves RATE_A, and the two drop
    # alignments are entry choices with a gain shape.
    "loop_roll_out", "drop_swap", "beat_repeat_in", "backspin", "double_drop",
    # Effects (Phase 3): a reverb tail fed by the echo, a turntable brake, a
    # noise riser into the incoming track, and a filter sweep with an echo on
    # its way out.
    "reverb_out", "brake", "noise_riser", "filter_echo",
)

#: Loop lengths `loop_roll_out` halves through, in bars.
LOOP_ROLL_BARS: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)
#: Bars spent at each length before halving.
LOOP_ROLL_STEP_BARS: float = 2.0
#: `beat_repeat_in` stutters the final bar at this division by default.
BEAT_REPEAT_DIVISION: int = 16
#: How long the stutter runs, in bars, at the end of the window.
BEAT_REPEAT_BARS: float = 1.0
#: A backspin runs over this many bars by default.
BACKSPIN_BARS: float = 1.0
#: How far the record is dragged backwards, as a rate multiplier at the end.
BACKSPIN_END_RATE: float = -1.4
#: Two drops together, before deck A leaves.
DOUBLE_DROP_BARS: float = 12.0
#: Extra headroom while two full-energy drops play together. Summed, two
#: mastered drops are ~6 dB hotter than one, and the limiter should be
#: catching transients rather than holding down the whole passage.
DOUBLE_DROP_HEADROOM: float = 0.62

#: Entry methods for the incoming deck.
ENTRIES: tuple[str, ...] = ("mix_in", "cue_jump_in")

#: Every name the intent layer and the supervisor will accept, plus "auto".
STYLE_CHOICES: tuple[str, ...] = STYLES + ("auto",)

# Envelope columns. Named indices rather than a record array: the audio thread
# reads these by integer, and an integer index cannot allocate.
FROM_GAIN, FROM_LOW, FROM_MID, FROM_HIGH = 0, 1, 2, 3
TO_GAIN, TO_LOW, TO_MID, TO_HIGH = 4, 5, 6, 7
#: How much of deck A is fed to the echo send. Zero for every style but
#: ``echo_out``, so the engine can skip the delay line entirely.
ECHO_SEND = 8
#: Deck A's loop length in BEATS for this block, or 0 for no loop. This is what
#: the shape functions write: a musical quantity, never a sample position. It is
#: not read by the audio thread -- :func:`_resolve_loops` turns it into the two
#: columns below before the envelope is ever handed over.
LOOP_BEATS = 9
#: The resolved region, in FRAMES, which is what the engine assigns. ``LOOP_LEN``
#: of 0 means no loop. ``LOOP_START`` is an offset from the outgoing deck's
#: position at the transition's first block, so the callback adds one float to
#: it and does nothing else: no division, no floor, no beat arithmetic.
LOOP_START = 10
LOOP_LEN = 11
#: Deck A's playback rate multiplier, 1.0 for normal. Negative plays backwards,
#: which is all `backspin` is.
RATE_A = 12
#: Each deck's filter knob, -1..1 (negative low-pass, positive high-pass, 0
#: open), and the resonance both decks use, 0..1. Knob positions, never
#: frequencies or coefficients: each deck's prebuilt table turns a position
#: into coefficients, so the callback assigns these like any other column.
FROM_FILTER = 13
TO_FILTER = 14
FILTER_RES = 15
#: How much of deck A, and of its echo tail, feeds the reverb. The reverb's
#: tail keeps ringing after this falls to zero; that is the point of it.
REVERB_SEND = 16
#: Level of the noise riser. The riser itself is rendered on the control
#: thread when the transition is armed; the callback only plays it back.
RISER_GAIN = 17
N_COLS = 18

#: Length of the two shorter styles, in bars.
ECHO_OUT_BARS: float = 2.0
FILTER_SWEEP_BARS: float = 8.0
#: How far `filter_sweep` turns deck A's knob toward high-pass. 0.9 of the way
#: is a cutoff near 6 kHz: the track is thinned to its hats, not muted.
FILTER_SWEEP_DEPTH: float = 0.9
#: And how much it rings: enough to hear the cutoff climb.
FILTER_SWEEP_RESONANCE: float = 0.4

# --- effects -------------------------------------------------------------------

#: `reverb_out`: deck A's last bars go into the echo, and the echo into a
#: reverb whose tail rings on over the incoming track.
REVERB_OUT_BARS: float = 4.0
#: How long the reverb keeps ringing after its send closes, in seconds.
REVERB_TAIL_SECONDS: float = 3.0
#: `brake`: the outgoing record stops over this many beats, like a turntable
#: with its motor switched off.
BRAKE_BEATS: int = 2
#: `noise_riser`: white noise swept upward over the bars before the hand-over.
NOISE_RISER_BARS: int = 8
#: The riser's level at its peak. Kept well under the music: it is tension on
#: top of a mix, not a second track.
RISER_LEVEL: float = 0.22
#: `filter_echo`: deck A filters out while an echo carries its last bars.
FILTER_ECHO_BARS: float = 8.0

#: Echo delay, in beats. A dotted eighth is the standard DJ delay: it fills the
#: bar without landing on the beat the incoming track is about to occupy.
ECHO_DELAY_BEATS: float = 0.75
#: How much of the delayed signal is fed back in. Below 1 so the tail dies.
ECHO_FEEDBACK: float = 0.45
#: Level the repeats come back into the mix at. Fixed, and deliberately not the
#: send: the send decides what goes INTO the delay line, and repeats already in
#: it must keep sounding and decay by themselves after the send closes. Scaling
#: the output by the send too was a bug -- closing the send silenced the tail.
ECHO_RETURN: float = 0.8
#: The echo keeps running until its repeats have decayed this far (-60 dB),
#: including past the hand-over into the incoming track.
ECHO_RING_OUT_DB: float = -60.0


# =============================================================================
# The transition parameter space
# =============================================================================
#
# The five named styles are presets in this space, not the only points in it.
# A model may design anywhere inside it; it may not step outside it, and it
# never sees a sample position, a bar boundary or an absolute time. Everything
# here is expressed relative to the transition's own length, and the length
# itself is an INPUT to `phrase.plan_transition`, which still decides where the
# transition sits by working backwards from mix-out.

#: Gain curve shapes. What each one does to the pair of faders.
CURVES: tuple[str, ...] = (
    "equal_power",   # cos/sin: two uncorrelated tracks sum to the level of one
    "linear",        # constant amplitude sum; dips in power through the middle
    "fast_in",       # the incoming deck arrives early and sits under the outgoing
    "slow_in",       # the incoming deck holds back, then lands late
    "s_curve",       # slow at both ends, quick through the middle
)

#: What, if anything, sweeps.
FILTER_SWEEPS: tuple[str, ...] = ("none", "hp_out", "lp_in")

#: Bounds. The supervisor enforces these; they live here because they describe
#: the space itself rather than any one caller's policy.
LENGTH_BARS_RANGE: tuple[float, float] = (4.0, 32.0)
LOW_SWAP_BARS_RANGE: tuple[int, int] = (1, 8)
DECK_B_LOW_DELAY_RANGE: tuple[int, int] = (0, 16)
ECHO_BARS_RANGE: tuple[int, int] = (0, 4)
INTENSITY_RANGE: tuple[float, float] = (0.0, 1.0)
FILTER_RESONANCE_RANGE: tuple[float, float] = (0.0, 1.0)
REVERB_BARS_RANGE: tuple[int, int] = (0, 4)
#: A brake is one or two beats, or none.
BRAKE_BEATS_CHOICES: tuple[int, ...] = (0, 1, 2)
#: A riser is 4 to 8 bars, or none.
RISER_BARS_CHOICES: tuple[int, ...] = (0, 4, 5, 6, 7, 8)
LOOP_OUT_BARS_RANGE: tuple[float, float] = (0.0, 8.0)
BACKSPIN_BARS_RANGE: tuple[float, float] = (0.0, 4.0)
DOUBLE_DROP_BARS_RANGE: tuple[float, float] = (0.0, 16.0)
#: Only these two divisions: a stutter coarser than an eighth is a loop, and
#: finer than a sixteenth is a buzz.
BEAT_REPEAT_DIVISIONS: tuple[int, ...] = (0, 8, 16)
ALIGN_MODES: tuple[str, ...] = ("phrase", "drop")

#: `drop_swap`, `beat_repeat_in` and `backspin` are short gestures: a bar or so
#: of something violent at the END of an otherwise ordinary window. The window
#: is the schema's minimum length rather than the gesture's own, for two
#: reasons. A deck needs a run-up -- the incoming track has to be somewhere
#: before it is slammed in -- and a preset that reported a shorter length than
#: the live path arms it with is exactly how :mod:`djai.render` and
#: :mod:`djai.cli` came to disagree about how long a transition was.
SHORT_GESTURE_BARS: float = LENGTH_BARS_RANGE[0]


@dataclass(frozen=True)
class TransitionParams:
    """One transition, as a shape rather than as a name.

    Every field is relative to the transition's own length. There is no field
    here that can carry a sample position, a frame, a bar number in the track,
    or a wall-clock time -- which is the structural reason a model working in
    this space cannot compute timing even if it tried to.
    """

    length_bars: float = TRANSITION_BARS
    curve: str = "equal_power"
    #: Bar within the transition where the low band hands over, or None for no
    #: managed swap (each deck's low band then follows its own fader).
    low_swap_bar: int | None = None
    #: How long that handover takes. 1 is effectively instant.
    low_swap_bars: int = 1
    deck_a_high_rolloff: bool = False
    #: Hold the incoming deck's low band out for this many bars.
    deck_b_low_delay_bars: int = 0
    #: Bars of beat-synced echo on the outgoing deck as it leaves.
    echo_bars: int = 0
    filter_sweep: str = "none"
    #: How much the swept filter rings at its cutoff, 0..1. Only heard when
    #: `filter_sweep` is not "none".
    filter_resonance: float = 0.0
    #: "mix_in", or "hot_cue_N" for a marker on the incoming track.
    entry_point: str = "mix_in"
    #: How far each effect is taken. Scales depth, never timing.
    intensity: float = 0.5

    # --- the high-energy set -------------------------------------------------
    #: Loop length deck A rolls out on, in bars. 0 for no loop.
    loop_out_bars: float = 0.0
    #: Halve that loop as it goes, building tension.
    loop_halving: bool = False
    #: Slice division for a stutter: eighths or sixteenths.
    beat_repeat_division: int = 0
    #: Bars a backspin runs over. 0 for none.
    backspin_bars: float = 0.0
    #: Bars both drops play together. 0 for none.
    double_drop_bars: float = 0.0
    #: "phrase" lines the transition up on a phrase boundary; "drop" lines the
    #: two tracks' drops up with each other.
    align_mode: str = "phrase"

    # --- effects -----------------------------------------------------------
    #: Bars of reverb, fed by deck A and its echo, over the end of the window.
    reverb_bars: int = 0
    #: Beats a brake stops deck A over, at the very end. 0 for none.
    brake_beats: int = 0
    #: Bars of noise riser ending at the hand-over. 0 for none.
    riser_bars: int = 0
    #: Ask placement to steer clear of two vocals at once. The supervisor
    #: refuses a clash either way; this decides whether placement tries to
    #: avoid one before it gets that far.
    vocal_aware: bool = True

    #: Which preset this came from, or "llm" when a model designed it. Not
    #: part of the schema a model returns -- it is provenance, for the log.
    name: str = "llm"

    def to_schema(self) -> dict:
        """The twenty-one fields, in the shape the model is asked to return."""
        return {
            "length_bars": self.length_bars,
            "curve": self.curve,
            "low_swap_bar": self.low_swap_bar,
            "low_swap_bars": self.low_swap_bars,
            "deck_a_high_rolloff": self.deck_a_high_rolloff,
            "deck_b_low_delay_bars": self.deck_b_low_delay_bars,
            "echo_bars": self.echo_bars,
            "filter_sweep": self.filter_sweep,
            "filter_resonance": self.filter_resonance,
            "entry_point": self.entry_point,
            "intensity": self.intensity,
            "loop_out_bars": self.loop_out_bars,
            "loop_halving": self.loop_halving,
            "beat_repeat_division": self.beat_repeat_division,
            "backspin_bars": self.backspin_bars,
            "double_drop_bars": self.double_drop_bars,
            "align_mode": self.align_mode,
            "reverb_bars": self.reverb_bars,
            "brake_beats": self.brake_beats,
            "riser_bars": self.riser_bars,
            "vocal_aware": self.vocal_aware,
        }

    def hot_cue_index(self) -> int | None:
        """The cue number in ``entry_point``, or None for a mix-in entry."""
        if not self.entry_point.startswith("hot_cue_"):
            return None
        try:
            return int(self.entry_point.split("_")[-1])
        except ValueError:
            return None


def preset_params(style: str, transition_bars: float | None = None) -> TransitionParams:
    """The named styles, expressed in the parameter space.

    These are what a model's design is compared against, what it falls back to
    when its answer is unusable, and what runs when there is no model at all.
    """
    length = TRANSITION_BARS if transition_bars is None else transition_bars
    if style == "cut":
        # A cut has no shape to speak of. It keeps the schema's floor for
        # length because the arm path measures it in blocks, not bars.
        return TransitionParams(
            length_bars=LENGTH_BARS_RANGE[0], curve="fast_in",
            low_swap_bar=0, low_swap_bars=1, intensity=1.0, name="cut",
        )
    if style == "echo_out":
        return TransitionParams(
            length_bars=ECHO_OUT_BARS, curve="fast_in",
            low_swap_bar=int(ECHO_OUT_BARS // 2), low_swap_bars=1,
            echo_bars=int(ECHO_OUT_BARS), intensity=0.8, name="echo_out",
        )
    if style == "filter_sweep":
        return TransitionParams(
            length_bars=FILTER_SWEEP_BARS, curve="s_curve",
            low_swap_bar=None, low_swap_bars=1,
            deck_a_high_rolloff=False, filter_sweep="hp_out",
            filter_resonance=FILTER_SWEEP_RESONANCE,
            intensity=FILTER_SWEEP_DEPTH, name="filter_sweep",
        )
    if style == "loop_roll_out":
        return TransitionParams(
            length_bars=style_bars(style), curve="fast_in",
            low_swap_bar=None, loop_out_bars=LOOP_ROLL_BARS[0],
            loop_halving=True, intensity=0.9, name="loop_roll_out",
        )
    if style == "drop_swap":
        # No entry_point here: the drop is found by `align_mode`, which works
        # off the cue's LABEL. A hot cue number would be a guess about a
        # particular track's cue list.
        return TransitionParams(
            length_bars=style_bars(style), curve="fast_in",
            low_swap_bar=None, align_mode="drop",
            intensity=1.0, name="drop_swap",
        )
    if style == "beat_repeat_in":
        return TransitionParams(
            length_bars=style_bars(style), curve="fast_in",
            low_swap_bar=None, beat_repeat_division=BEAT_REPEAT_DIVISION,
            intensity=0.95, name="beat_repeat_in",
        )
    if style == "backspin":
        return TransitionParams(
            length_bars=style_bars(style), curve="fast_in",
            low_swap_bar=None, backspin_bars=BACKSPIN_BARS,
            intensity=0.9, name="backspin",
        )
    if style == "double_drop":
        return TransitionParams(
            length_bars=style_bars(style), curve="equal_power",
            low_swap_bar=0, double_drop_bars=DOUBLE_DROP_BARS,
            align_mode="drop",
            intensity=0.8, name="double_drop",
        )
    if style == "reverb_out":
        return TransitionParams(
            length_bars=REVERB_OUT_BARS, curve="fast_in",
            low_swap_bar=int(REVERB_OUT_BARS // 2), low_swap_bars=1,
            echo_bars=2, reverb_bars=2, intensity=0.8, name="reverb_out",
        )
    if style == "brake":
        return TransitionParams(
            length_bars=SHORT_GESTURE_BARS, curve="fast_in",
            low_swap_bar=int(SHORT_GESTURE_BARS) - 1, low_swap_bars=1,
            brake_beats=BRAKE_BEATS, intensity=0.9, name="brake",
        )
    if style == "noise_riser":
        return TransitionParams(
            length_bars=float(NOISE_RISER_BARS), curve="slow_in",
            low_swap_bar=NOISE_RISER_BARS - 1, low_swap_bars=1,
            riser_bars=NOISE_RISER_BARS, intensity=0.9, name="noise_riser",
        )
    if style == "filter_echo":
        return TransitionParams(
            length_bars=FILTER_ECHO_BARS, curve="s_curve",
            low_swap_bar=None, filter_sweep="hp_out",
            filter_resonance=FILTER_SWEEP_RESONANCE, echo_bars=2,
            intensity=0.85, name="filter_echo",
        )
    return TransitionParams(
        length_bars=length, curve="equal_power",
        low_swap_bar=int(length * BASS_SWAP_AT), low_swap_bars=1,
        intensity=0.5, name="bass_swap",
    )


PRESETS: tuple[str, ...] = STYLES


def style_bars(style: str, transition_bars: float | None = None) -> float:
    """How long a style runs, in bars. ``cut`` is sub-bar and reports 0.

    This has to agree with the ``length_bars`` of the same style's preset: the
    live path arms from the preset and the renderer arms from here, and when
    they disagree the render is not the transition that ships.
    """
    if style == "cut":
        return 0.0
    if style == "echo_out":
        return ECHO_OUT_BARS
    if style == "filter_sweep":
        return FILTER_SWEEP_BARS
    if style == "loop_roll_out":
        return LOOP_ROLL_STEP_BARS * len(LOOP_ROLL_BARS)
    if style in ("drop_swap", "beat_repeat_in", "backspin"):
        return SHORT_GESTURE_BARS
    if style == "double_drop":
        return DOUBLE_DROP_BARS
    if style == "reverb_out":
        return REVERB_OUT_BARS
    if style == "brake":
        return SHORT_GESTURE_BARS
    if style == "noise_riser":
        return float(NOISE_RISER_BARS)
    if style == "filter_echo":
        return FILTER_ECHO_BARS
    return TRANSITION_BARS if transition_bars is None else transition_bars


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp01(t)


def _segment(t: float, start: float, end: float) -> float:
    """Progress within a sub-window of the transition, clamped to 0..1."""
    if end <= start:
        return 1.0 if t >= end else 0.0
    return _clamp01((t - start) / (end - start))


def _row_bass_swap(t: float, row) -> None:
    g = gains_at(t * TRANSITION_BARS, TRANSITION_BARS)
    row[FROM_GAIN] = g.from_gain
    row[FROM_LOW] = g.from_low
    row[TO_GAIN] = g.to_gain
    row[TO_LOW] = g.to_low


def _row_cut(t: float, row) -> None:
    """A hard switch: deck A is already gone on the first row."""
    row[FROM_GAIN] = 0.0
    row[TO_GAIN] = 1.0


def _row_echo_out(t: float, row) -> None:
    """Dry A falls, its echo send rises and then decays, B enters clean."""
    # Dry signal down over the first three quarters, so the tail is audible
    # against B rather than fighting A's own dry output.
    row[FROM_GAIN] = math.cos(_HALF_PI * _segment(t, 0.0, 0.75))
    # Send peaks early, then falls away so the repeats thin out.
    send_up = _segment(t, 0.0, 0.25)
    send_down = _segment(t, 0.5, 1.0)
    row[ECHO_SEND] = _clamp01(send_up * (1.0 - send_down))
    # B comes in clean and full-range underneath.
    row[TO_GAIN] = math.sin(_HALF_PI * _clamp01(t))
    row[TO_LOW] = 1.0 if t >= 0.5 else 0.0
    row[FROM_LOW] = 0.0 if t >= 0.5 else 1.0


def _row_filter_sweep(t: float, row) -> None:
    """Deck A's filter knob turns toward high-pass: the track thins from the bottom up.

    A real resonant high-pass on the deck, swept from 20 Hz to about 6 kHz --
    exponentially in frequency, because the knob is -- with a little resonance
    so the cutoff is heard climbing. It used to be imitated by dropping the
    three EQ bands in turn, which sounds like three steps rather than a sweep.
    """
    row[FROM_FILTER] = FILTER_SWEEP_DEPTH * _segment(t, 0.0, 0.9)
    row[FILTER_RES] = FILTER_SWEEP_RESONANCE
    # Level holds while the body thins, then closes out at the end.
    row[FROM_GAIN] = math.cos(_HALF_PI * _segment(t, 0.75, 1.0))
    # B enters full-range from the start -- it is what carries the bottom end
    # once A's low band is gone.
    row[TO_GAIN] = math.sin(_HALF_PI * _segment(t, 0.0, 0.6))
    row[TO_LOW] = 1.0


def _low_handover(row, x: float) -> None:
    """Hand the low band from A to B across ``x`` in 0..1.

    The one thing none of the high-energy styles may do is let two basslines
    play at once, and the one thing they must not do either is leave a gap
    where neither has one.
    """
    row[FROM_LOW] = 1.0 - x
    row[TO_LOW] = x


def _row_loop_roll_out(t: float, row) -> None:
    """Deck A loops its last region, halving the loop as tension builds.

    4 bars, then 2, then 1, then half a bar, two bars at each length. Deck B
    rises underneath across the whole roll, so that when the loop releases on
    the final downbeat there is a track already there to release into -- the
    roll is the tension, the hand-over is its resolution.
    """
    total = LOOP_ROLL_STEP_BARS * len(LOOP_ROLL_BARS)
    bar = t * total
    index = min(len(LOOP_ROLL_BARS) - 1, int(bar // LOOP_ROLL_STEP_BARS))
    row[LOOP_BEATS] = LOOP_ROLL_BARS[index] * 4.0
    # The roll stays up through the first half, then thins as B arrives. It is
    # never cut off: the last row hands over, and the deck's own gain ramp
    # carries the final step.
    row[FROM_GAIN] = math.cos(_HALF_PI * _segment(t, 0.5, 1.0))
    row[TO_GAIN] = math.sin(_HALF_PI * _clamp01(t))
    _low_handover(row, _segment(t, 0.5, 0.55))


def _row_drop_swap(t: float, row) -> None:
    """A's last phrase into B's drop: A thins and goes, B lands on the drop.

    ``align_mode="drop"`` is what makes this a drop swap -- it puts the end of
    the window on both tracks' drop cues. Everything here is the run-up to
    that moment. A loses its top end through the back half and its level over
    the final bar, so the arrival sounds like a switch rather than the end of
    a fade; B rises underneath and is full when the drop lands.
    """
    row[FROM_HIGH] = 1.0 - _segment(t, 0.5, 1.0)
    row[FROM_GAIN] = math.cos(_HALF_PI * _segment(t, 0.75, 1.0))
    row[TO_GAIN] = math.sin(_HALF_PI * _clamp01(t))
    _low_handover(row, _segment(t, 0.75, 1.0))


def _row_beat_repeat_in(t: float, row) -> None:
    """A's final bar stuttered in slices, then a hard hand-over to B.

    The stutter is a loop of one slice. A 1/16 division is a quarter-beat
    loop, which the deck handles like any other -- the slice length is in
    beats, so nothing here computes a sample position.
    """
    total = SHORT_GESTURE_BARS
    start = (total - BEAT_REPEAT_BARS) / total
    if t >= start:
        row[LOOP_BEATS] = 4.0 / BEAT_REPEAT_DIVISION
    x = _segment(t, start, 1.0)
    row[FROM_GAIN] = math.cos(_HALF_PI * x)
    row[TO_GAIN] = math.sin(_HALF_PI * _clamp01(t))
    _low_handover(row, x)


def _row_backspin(t: float, row) -> None:
    """A pitches down, reverses and decays over its final bar.

    Rate is an envelope column, not a branch in the callback: the engine
    assigns whatever the row says, and negative plays backwards.
    """
    total = SHORT_GESTURE_BARS
    start = (total - BACKSPIN_BARS) / total
    x = _segment(t, start, 1.0)
    row[RATE_A] = 1.0 + (BACKSPIN_END_RATE - 1.0) * x
    # The spin decays so it does not fight the incoming track.
    row[FROM_GAIN] = 1.0 - x ** 1.5
    row[TO_GAIN] = math.sin(_HALF_PI * _clamp01(t))
    _low_handover(row, x)


def _row_double_drop(t: float, row) -> None:
    """Both drops together, then A leaves.

    Held below unity throughout: two mastered drops summed are far hotter than
    one, and the master limiter should be catching transients here rather than
    riding the whole passage down.
    """
    head = DOUBLE_DROP_HEADROOM
    row[FROM_GAIN] = head * (1.0 - _segment(t, 0.75, 1.0))
    row[TO_GAIN] = head + (1.0 - head) * _segment(t, 0.75, 1.0)
    # Both basslines at once is the one thing that will not work, so A's low
    # goes immediately and B carries the bottom end.
    row[FROM_LOW] = 0.0
    row[TO_LOW] = 1.0


_ROW_BUILDERS = {
    "bass_swap": _row_bass_swap,
    "cut": _row_cut,
    "echo_out": _row_echo_out,
    "filter_sweep": _row_filter_sweep,
    "loop_roll_out": _row_loop_roll_out,
    "drop_swap": _row_drop_swap,
    "beat_repeat_in": _row_beat_repeat_in,
    "backspin": _row_backspin,
    "double_drop": _row_double_drop,
    # The effect styles are points in the parameter space and are built by the
    # same maths a designed transition uses, so a preset and a design asking
    # for the same effect cannot drift apart.
    "reverb_out": lambda t, row: _row_from_params(preset_params("reverb_out"), t, row),
    "brake": lambda t, row: _row_from_params(preset_params("brake"), t, row),
    "noise_riser": lambda t, row: _row_from_params(preset_params("noise_riser"), t, row),
    "filter_echo": lambda t, row: _row_from_params(preset_params("filter_echo"), t, row),
}


def _curve_pair(shape: str, t: float) -> tuple[float, float]:
    """Outgoing and incoming fader positions at ``t`` in 0..1."""
    t = _clamp01(t)
    if shape == "linear":
        return 1.0 - t, t
    if shape == "fast_in":
        # The incoming deck arrives early and waits underneath.
        return math.cos(_HALF_PI * t), math.sin(_HALF_PI * math.sqrt(t))
    if shape == "slow_in":
        # It holds back, then lands.
        return math.cos(_HALF_PI * (t * t)), math.sin(_HALF_PI * t * t)
    if shape == "s_curve":
        s = t * t * (3.0 - 2.0 * t)
        return math.cos(_HALF_PI * s), math.sin(_HALF_PI * s)
    # equal_power, and anything unrecognised, which the supervisor will have
    # rejected long before it reaches here.
    return math.cos(_HALF_PI * t), math.sin(_HALF_PI * t)


def _row_from_params(p: TransitionParams, t: float, row) -> None:
    """One envelope row for an arbitrary point in the parameter space.

    Control thread only, and only ever at build time. Everything here is
    ordinary float maths over ``t``; the audio thread reads the finished array.
    """
    length = max(1e-6, p.length_bars)
    bar = t * length
    depth = _clamp01(p.intensity)

    from_gain, to_gain = _curve_pair(p.curve, t)
    row[FROM_GAIN] = from_gain
    row[TO_GAIN] = to_gain

    # --- low band -----------------------------------------------------------
    # The incoming deck's low can be held out for a while regardless of the
    # swap: that is what stops two basslines arriving together.
    b_low_open = 1.0
    if p.deck_b_low_delay_bars > 0:
        b_low_open = 1.0 if bar >= p.deck_b_low_delay_bars else 0.0

    if p.low_swap_bar is None:
        # No managed swap: each deck's low band simply follows its fader.
        row[FROM_LOW] = 1.0
        row[TO_LOW] = b_low_open
    else:
        swap_at = float(p.low_swap_bar)
        over = max(1e-6, float(p.low_swap_bars))
        # Ramps across `low_swap_bars`, so 1 bar is effectively the instant
        # handover the original bass swap does.
        x = _clamp01((bar - swap_at) / over)
        row[FROM_LOW] = 1.0 - x
        row[TO_LOW] = x * b_low_open

    # --- deck A's high band --------------------------------------------------
    if p.deck_a_high_rolloff:
        # Rolls away over the back half, as far as intensity asks.
        row[FROM_HIGH] = 1.0 - depth * _segment(t, 0.4, 1.0)
    else:
        row[FROM_HIGH] = 1.0

    # --- sweeps ---------------------------------------------------------------
    if p.filter_sweep == "hp_out":
        # Deck A's filter turns toward high-pass, as far as intensity asks.
        row[FROM_FILTER] = depth * _segment(t, 0.0, 0.9)
        row[FILTER_RES] = p.filter_resonance
    elif p.filter_sweep == "lp_in":
        # Deck B arrives low-passed and opens up.
        row[TO_FILTER] = -depth * (1.0 - _segment(t, 0.1, 0.85))
        row[FILTER_RES] = p.filter_resonance

    # --- echo send ------------------------------------------------------------
    if p.echo_bars > 0:
        # Runs over the final `echo_bars`, rising as the dry signal leaves and
        # falling away before the end so the tail does not run into deck B.
        start = max(0.0, length - float(p.echo_bars))
        x = _clamp01((bar - start) / max(1e-6, length - start))
        row[ECHO_SEND] = depth * _clamp01(x * 2.0) * _clamp01((1.0 - x) * 2.5)

    # --- the high-energy set --------------------------------------------------
    # Every one of these is a gesture at the END of the window, sized in bars of
    # the window's own length. A designed value arrives already bounded by the
    # supervisor; it is clamped to the window again here because a 4-bar spin
    # inside a 4-bar transition and a 4-bar spin inside a 32-bar one are
    # different gestures, and only the second is what was asked for.
    if p.loop_out_bars > 0:
        # Deck A loops its last region. Halving spends LOOP_ROLL_STEP_BARS at
        # each length, so the roll's span is set by how many halvings there are;
        # without halving the one length simply runs twice.
        span = (
            LOOP_ROLL_STEP_BARS * len(LOOP_ROLL_BARS)
            if p.loop_halving
            else 2.0 * float(p.loop_out_bars)
        )
        span = min(span, length)
        start = length - span
        if bar >= start:
            loop = float(p.loop_out_bars)
            if p.loop_halving:
                loop /= 2.0 ** int((bar - start) // LOOP_ROLL_STEP_BARS)
            # A quarter beat is the floor. Below that a loop stops being a loop
            # and becomes a pitched buzz.
            row[LOOP_BEATS] = max(0.25, loop * 4.0)

    if p.beat_repeat_division > 0:
        # A stutter is a one-slice loop. It wins over a roll if both are asked
        # for, being the shorter and later gesture of the two.
        if bar >= length - BEAT_REPEAT_BARS:
            row[LOOP_BEATS] = 4.0 / float(p.beat_repeat_division)

    if p.backspin_bars > 0:
        spin = min(float(p.backspin_bars), length)
        x = _segment(bar, length - spin, length)
        row[RATE_A] = 1.0 + (BACKSPIN_END_RATE - 1.0) * x
        # `min` rather than assignment: whatever the curve was already doing to
        # deck A's level, a spin may take it down further but never back up.
        row[FROM_GAIN] = min(row[FROM_GAIN], 1.0 - x ** 1.5)

    if p.double_drop_bars > 0:
        # Two drops together, both held below unity, and then A leaves. This
        # overrides the curve on purpose: the whole point is that for these
        # bars neither deck is fading.
        head = DOUBLE_DROP_HEADROOM
        together = min(float(p.double_drop_bars), length)
        # A leaves across whatever is left of the window, or across its final
        # quarter when the two drops fill the window completely.
        out = _segment(bar, together if together < length else length * 0.75, length)
        row[FROM_GAIN] = head * (1.0 - out)
        row[TO_GAIN] = head + (1.0 - head) * out
        # Two full-range basslines at once is the one thing that will not work.
        row[FROM_LOW] = 0.0
        row[TO_LOW] = b_low_open

    # --- effects -----------------------------------------------------------
    # All three are gestures at the end of the window, sized in its own bars
    # or beats, and each moves continuously: nothing here steps.
    if p.reverb_bars > 0:
        span = min(float(p.reverb_bars), length)
        x = _segment(bar, length - span, length)
        # The send opens over the first half of its span and holds; the dry
        # signal leaves across the second half, so what is left of deck A at
        # the end is its reverb. The send itself closes over the last fifth,
        # ahead of the hand-over: deck A's raw signal still feeds it, and a
        # send slammed shut in one block would click inside the tail.
        row[REVERB_SEND] = depth * _clamp01(x * 2.0) * _clamp01((1.0 - x) * 5.0)
        row[FROM_GAIN] = min(row[FROM_GAIN], 1.0 - _clamp01((x - 0.5) * 2.0))

    if p.brake_beats > 0:
        beats = length * BEATS_PER_BAR_T
        span = min(float(p.brake_beats), beats)
        x = _segment(bar * BEATS_PER_BAR_T, beats - span, beats)
        # Linear deceleration to a stop; the level falls away at the very end
        # so the stopped record is not left droning.
        row[RATE_A] = min(row[RATE_A], 1.0 - x)
        row[FROM_GAIN] = min(row[FROM_GAIN], 1.0 - x ** 3)

    if p.riser_bars > 0:
        span = min(float(p.riser_bars), length)
        x = _segment(bar, length - span, length)
        # Squared, so most of the rise is in the last bars, where it builds.
        row[RISER_GAIN] = RISER_LEVEL * max(depth, 0.25) * x * x

    # --- headroom for the sends ------------------------------------------------
    # The echo and the reverb both take deck A from before its fader and its
    # filter, so they re-inject what the dry path is taking away. Measured: in
    # a 20-minute run of nothing but `filter_echo` the master reached 1.091
    # before the limiter, and `reverb_out` 1.026, while every other style
    # stayed under 0.89. Capping each send at the headroom the dry signal
    # leaves keeps deck A, with its sends, at unity or below.
    headroom = max(0.0, 1.0 - float(row[FROM_GAIN]))
    row[ECHO_SEND] = min(row[ECHO_SEND], headroom)
    row[REVERB_SEND] = min(row[REVERB_SEND], headroom)


#: Beats per bar, for the brake. transition.py counts in bars everywhere else.
BEATS_PER_BAR_T: float = 4.0


def _resolve_loops(env, blocksize: int, bpm: float, sample_rate: int) -> None:
    """Turn each row's loop length in beats into a region in frames.

    **Control thread, at build time.** This is why the audio thread does no
    loop arithmetic: it reads a start offset and a length and assigns them.

    The start is the beat line at or before the point where a loop first
    engages, and it is held for as long as the loop stays engaged. Halving a
    loop moves its end, never its beginning -- which is what a CDJ does, and
    what stops a roll walking forward through the track.

    Offsets are relative to the outgoing deck's position at the transition's
    first block. That lands them on its beat grid because every transition is
    placed on a downbeat of that deck.

    Precision: values are stored as float32. A 32-bar offset is exact to a
    quarter of a frame and a loop length to a thirtieth, and neither error
    accumulates -- the deck wraps against the stored numbers every lap.
    """
    import math

    env[:, LOOP_START] = 0.0
    env[:, LOOP_LEN] = 0.0
    if bpm <= 0.0 or sample_rate <= 0:
        # No tempo, no grid: a loop could not be placed on a beat, so none is.
        env[:, LOOP_BEATS] = 0.0
        return
    frames_per_beat = 60.0 / bpm * sample_rate
    start = 0.0
    engaged = False
    for i in range(env.shape[0]):
        beats = float(env[i, LOOP_BEATS])
        if beats <= 0.0:
            engaged = False
            continue
        if not engaged:
            # The engine advances its frame count before it looks a row up, so
            # row i is read for the block that starts (i - 1) blocks in.
            here = max(0, i - 1) * blocksize
            start = math.floor(here / frames_per_beat) * frames_per_beat
            engaged = True
        env[i, LOOP_START] = start
        env[i, LOOP_LEN] = beats * frames_per_beat


def build_envelope_from_params(
    params: TransitionParams,
    total_frames: int,
    blocksize: int,
    bpm: float = 0.0,
    sample_rate: int = 44100,
):
    """Precompute a transition from validated parameters. Control thread only.

    Same contract as :func:`build_envelope`: a float32 array of one row per
    audio block, which the engine indexes and assigns. Nothing about the
    callback changes because the parameter space exists.
    """
    import numpy as np

    blocksize = max(1, int(blocksize))
    total_frames = max(1, int(total_frames))
    rows = max(2, total_frames // blocksize + 2)

    env = np.zeros((rows, N_COLS), dtype=np.float32)
    env[:, (FROM_LOW, FROM_MID, FROM_HIGH, TO_MID, TO_HIGH)] = 1.0
    env[:, RATE_A] = 1.0        # normal speed unless a style says otherwise

    last = rows - 1
    for i in range(rows):
        t = _clamp01((i * blocksize) / total_frames)
        if i == last:
            t = 1.0
        _row_from_params(params, t, env[i])

    # However it ends, it ends handed over.
    env[last, FROM_GAIN] = 0.0
    env[last, ECHO_SEND] = 0.0
    env[last, TO_GAIN] = 1.0
    env[last, TO_LOW] = 1.0
    env[last, TO_MID] = 1.0
    env[last, TO_HIGH] = 1.0
    # Whatever a style did to deck A, it hands back a deck with no loop and
    # playing forwards. The next track to land on it inherits neither.
    env[last, LOOP_BEATS] = 0.0
    env[last, RATE_A] = 1.0
    # And with both filters open. The reverb's send and the riser close; the
    # reverb's tail rings on by itself.
    env[last, FROM_FILTER] = 0.0
    env[last, TO_FILTER] = 0.0
    env[last, REVERB_SEND] = 0.0
    env[last, RISER_GAIN] = 0.0
    _resolve_loops(env, blocksize, bpm, sample_rate)
    return env


def build_envelope(
    style: str,
    total_frames: int,
    blocksize: int,
    bpm: float = 0.0,
    sample_rate: int = 44100,
):
    """Precompute the whole transition, one row per audio block.

    **Control thread only.** Returns a C-contiguous float32 array of shape
    ``(rows, N_COLS)``; the engine indexes it and assigns, and does nothing
    else. An extra final row is appended holding the settled end state, so an
    overrun of a block or two reads a sane value instead of clamping onto a
    mid-curve one.
    """
    import numpy as np  # local: transition.py stays importable without numpy

    if style not in _ROW_BUILDERS:
        raise ValueError(f"unknown transition style {style!r}")
    blocksize = max(1, int(blocksize))
    total_frames = max(1, int(total_frames))
    rows = max(2, total_frames // blocksize + 2)

    env = np.zeros((rows, N_COLS), dtype=np.float32)
    # Bands default to unity; a style only writes the ones it moves.
    env[:, (FROM_LOW, FROM_MID, FROM_HIGH, TO_MID, TO_HIGH)] = 1.0
    env[:, RATE_A] = 1.0        # normal speed unless a style says otherwise
    build = _ROW_BUILDERS[style]

    last = rows - 1
    for i in range(rows):
        t = _clamp01((i * blocksize) / total_frames)
        if i == last:
            t = 1.0
        build(t, env[i])

    # The same send headroom a designed transition gets (see _row_from_params):
    # deck A plus its sends at unity or below. Applied here too because the
    # presets keep their own row builders. Measured before: `echo_out` forced
    # for 20 minutes reached 1.154 into the limiter with its send at full while
    # the dry signal was still at 0.73 -- a breach older than Phase 3, found
    # only when the effect styles were soaked one at a time.
    headroom = np.maximum(0.0, 1.0 - env[:, FROM_GAIN])
    np.minimum(env[:, ECHO_SEND], headroom, out=env[:, ECHO_SEND])
    np.minimum(env[:, REVERB_SEND], headroom, out=env[:, REVERB_SEND])

    # However a style ends, it ends handed over: B at unity and full range, A
    # silent. Anything else strands the mix on the last row.
    env[last, FROM_GAIN] = 0.0
    env[last, ECHO_SEND] = 0.0
    env[last, TO_GAIN] = 1.0
    env[last, TO_LOW] = 1.0
    env[last, TO_MID] = 1.0
    env[last, TO_HIGH] = 1.0
    # Whatever a style did to deck A, it hands back a deck with no loop and
    # playing forwards. The next track to land on it inherits neither.
    env[last, LOOP_BEATS] = 0.0
    env[last, RATE_A] = 1.0
    # And with both filters open. The reverb's send and the riser close; the
    # reverb's tail rings on by itself.
    env[last, FROM_FILTER] = 0.0
    env[last, TO_FILTER] = 0.0
    env[last, REVERB_SEND] = 0.0
    env[last, RISER_GAIN] = 0.0
    _resolve_loops(env, blocksize, bpm, sample_rate)
    return env


def echo_delay_frames(bpm: float, sample_rate: int) -> int:
    """Beat-synced delay time for ``echo_out``, derived from deck A's BPM."""
    if bpm <= 0:
        bpm = 128.0
    return int(round(ECHO_DELAY_BEATS * 60.0 / bpm * sample_rate))


# --- choosing a style ---------------------------------------------------------

#: Below this grid confidence on either track, a blend cannot be trusted to
#: stay in phase for its whole length, so the switch is made instant instead.
#:
#: No longer the preflight floor. Preflight asks whether a track is analysable
#: at all and fails the whole run; this asks whether one blend can hold phase.
#: They became different questions when v5 changed what the number measures.
MIN_GRID_CONFIDENCE: float = config.TRANSITION_MIN_GRID_CONFIDENCE

#: Tempo gap beyond which a blend is not attempted. Matches the stretch limit:
#: past this the incoming deck cannot be matched without audible artefacts.
MAX_BPM_DELTA: float = 0.08

#: "BPM close" for the filter-sweep rule.
CLOSE_BPM_DELTA: float = 0.03

#: How much of a rise counts as rising energy.
ENERGY_RISE: float = 1.05

#: How far into deck A's track counts as its outro, in bars before mix-out.
OUTRO_BARS: float = 32.0


class TransitionChoice(NamedTuple):
    """A style, how the incoming deck enters, and the rule that decided it."""

    style: str
    entry: str
    hot_cue_index: int | None
    rule: str


def camelot_compatible(x: str | None, y: str | None) -> bool:
    """Harmonic compatibility on the Camelot wheel.

    Compatible means the same key, its relative major/minor, or one step
    around the wheel in the same mode. Anything else clashes.
    """
    if not x or not y:
        return False
    import re

    mx = re.fullmatch(r"(\d{1,2})([AB])", x.strip().upper())
    my = re.fullmatch(r"(\d{1,2})([AB])", y.strip().upper())
    if not mx or not my:
        return False
    nx, lx = int(mx.group(1)), mx.group(2)
    ny, ly = int(my.group(1)), my.group(2)
    step = min((nx - ny) % 12, (ny - nx) % 12)
    if step == 0:
        return True           # same key, or its relative major/minor
    return step == 1 and lx == ly


def _energy_falling(analysis, position_s: float) -> bool:
    """Is deck A's per-beat energy lower than it was a phrase ago?"""
    beats = analysis.beats_np
    rms = analysis.beat_rms
    if beats.size == 0 or len(rms) < 32:
        return False
    import numpy as np

    i = int(np.searchsorted(beats, position_s))
    i = max(32, min(i, len(rms)))
    recent = rms[i - 16:i]
    earlier = rms[i - 32:i - 16]
    if not recent or not earlier:
        return False
    mean_recent = sum(recent) / len(recent)
    mean_earlier = sum(earlier) / len(earlier)
    return mean_earlier > 1e-9 and mean_recent < mean_earlier * 0.95


def _in_outro(analysis, position_s: float) -> bool:
    bar_s = 4 * 60.0 / analysis.bpm if analysis.bpm > 0 else 2.0
    return position_s >= analysis.mix_out - OUTRO_BARS * bar_s


#: A cue_jump_in entry must leave at least this many bars of track after it,
#: measured to mix-out. Learned from a soak: entering on a drop three quarters
#: of the way through a track meant the deck ran out almost as soon as it
#: became live, and the recovery path's gap was audible as dead air. A drop cue
#: is for skipping an intro, not for starting near the end.
MIN_ENTRY_RUNWAY_BARS: float = 48.0

#: And it has to be far enough in to be skipping anything at all.
MIN_ENTRY_SKIP_BARS: float = 8.0


def drop_cue(track) -> dict | None:
    """The hot cue marking this track's drop, if it has one.

    Drop-aligned transitions need one on BOTH tracks -- there is nothing to
    line up otherwise -- so this is what the supervisor asks before letting a
    `double_drop` or a `drop_swap` run.
    """
    for cue in getattr(track, "hot_cues", None) or []:
        if cue.get("label") == "drop":
            return cue
    return None


def entry_has_runway(track_b, seconds: float) -> bool:
    """Is there enough of the incoming track left after this entry point?

    One rule, used by both routes into a deck. The rule-based chooser asks it
    when it picks a drop cue, and the supervisor asks it of any cue a model
    names -- because a deck that enters near its own mix-out runs out shortly
    after the blend hands over, and the recovery that follows is audible as
    dead air. That happened twice: once through each route.
    """
    bar_s = 4 * 60.0 / track_b.bpm if track_b.bpm > 0 else 2.0
    return seconds <= track_b.mix_out - MIN_ENTRY_RUNWAY_BARS * bar_s


def _drop_cue(track_b) -> int | None:
    """A hot cue that starts deck B at a drop rather than at its intro.

    Returns the earliest qualifying cue, so the entry skips the intro without
    eating the track.
    """
    bar_s = 4 * 60.0 / track_b.bpm if track_b.bpm > 0 else 2.0
    earliest = MIN_ENTRY_SKIP_BARS * bar_s
    best = None
    for cue in getattr(track_b, "hot_cues", None) or []:
        if cue.get("label") != "drop":
            continue
        seconds = float(cue["sample_position"]) / 44100.0
        if seconds < earliest or not entry_has_runway(track_b, seconds):
            continue
        if best is None or seconds < best[1]:
            best = (int(cue["index"]), seconds)
    return best[0] if best else None


def choose_transition(deck_a, track_b, context=None) -> TransitionChoice:
    """Pick a transition style. Pure, rule-based, and never the LLM's job.

    The model may ask for a style by name (``context["style"]``); it may not
    compute one, and it never sees timing or curves. With no request, or with
    ``"auto"``, the rules below decide in order and the first match wins.
    """
    context = context or {}
    requested = context.get("style") or "auto"
    entry, cue_index, entry_rule = "mix_in", None, ""

    track_a = deck_a.track.analysis if deck_a.track is not None else None

    # Entry method is chosen independently of the curve, because it is a
    # question about deck B alone: is there a better place to start it than
    # its mix-in?
    drop = _drop_cue(track_b)
    if drop is not None:
        entry, cue_index = "cue_jump_in", drop
        entry_rule = f" + cue_jump_in (deck B hot cue {drop} lands a drop)"

    if requested in STYLES:
        return TransitionChoice(
            requested, entry, cue_index,
            f"style requested explicitly: {requested}{entry_rule}",
        )

    if track_a is None:
        return TransitionChoice("cut", entry, cue_index,
                                f"no track on deck A{entry_rule}")

    conf_a = float(track_a.grid_confidence)
    conf_b = float(track_b.grid_confidence)
    if conf_a < MIN_GRID_CONFIDENCE or conf_b < MIN_GRID_CONFIDENCE:
        return TransitionChoice(
            "cut", entry, cue_index,
            f"grid confidence below {MIN_GRID_CONFIDENCE:.2f} "
            f"(A {conf_a:.2f}, B {conf_b:.2f}){entry_rule}",
        )

    bpm_a = float(track_a.bpm) * float(getattr(deck_a, "rate", 1.0) or 1.0)
    delta = abs(float(track_b.bpm) - bpm_a) / bpm_a if bpm_a > 0 else 1.0
    if delta > MAX_BPM_DELTA:
        return TransitionChoice(
            "cut", entry, cue_index,
            f"BPM delta {delta:.1%} above {MAX_BPM_DELTA:.0%}{entry_rule}",
        )

    position_s = float(getattr(deck_a, "position", 0.0)) / 44100.0
    if _in_outro(track_a, position_s) and _energy_falling(track_a, position_s):
        return TransitionChoice(
            "echo_out", entry, cue_index,
            f"deck A in its outro with falling energy{entry_rule}",
        )

    compatible = camelot_compatible(track_a.camelot, track_b.camelot)
    if track_b.energy > track_a.energy * ENERGY_RISE and compatible:
        return TransitionChoice(
            "bass_swap", entry, cue_index,
            f"energy rising and keys compatible "
            f"({track_a.camelot} -> {track_b.camelot}){entry_rule}",
        )

    if not compatible and delta <= CLOSE_BPM_DELTA:
        return TransitionChoice(
            "filter_sweep", entry, cue_index,
            f"keys clash ({track_a.camelot} vs {track_b.camelot}) but BPM is "
            f"within {CLOSE_BPM_DELTA:.0%}{entry_rule}",
        )

    return TransitionChoice(
        "bass_swap", entry, cue_index, f"default{entry_rule}"
    )
