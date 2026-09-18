"""Musical-time conversions: sample position <-> beat <-> bar <-> phrase.

THREADING CONTEXT: main and monitor threads. These are pure functions over a
deck's analysis; none of them are called from the audio thread, because
scheduling decisions are made ahead of time and handed to the engine as
already-resolved sample positions.

Two clocks exist and must not be confused:

* **Deck time** -- a frame index into a deck's own audio buffer. Advances at
  ``deck.rate`` frames per output frame. All the ``*_of_*`` helpers here work in
  deck time.
* **Engine time** -- frames since the audio stream started, shared by both decks
  and monotonic. Commands carry ``execute_at`` in engine time.

:func:`deck_frame_to_engine_frame` is the bridge, and it is only valid while
the deck's rate stays constant -- which is the case between scheduling a
command and firing it.

Bar 0 begins at the track's first estimated downbeat, so bar and phrase numbers
are musically meaningful rather than counted from the file's first sample.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from djai.analysis import TrackAnalysis
from djai.deck import SAMPLE_RATE, Deck

log = logging.getLogger(__name__)

BEATS_PER_BAR: int = 4
BARS_PER_PHRASE: int = 32
BEATS_PER_PHRASE: int = BEATS_PER_BAR * BARS_PER_PHRASE

#: Automatic placement snaps to this grid. Finer than a 32-bar phrase on
#: purpose: snapping a 16-bar transition to 32 bars moves its start by up to a
#: full minute, which is what put hand-offs 39 s early or 3 s past the end.
PLACEMENT_GRID_BARS: int = 8

#: Shortest blend worth attempting. Below this a cut is more honest.
MIN_TRANSITION_BARS: float = 8.0


# --- deck-time conversions ---------------------------------------------------


def beat_at_frame(analysis: TrackAnalysis, frame: float) -> float:
    """Fractional beat index at a deck frame position. Beat 0 is the first beat
    of the grid (not necessarily a downbeat); may return a negative value for
    positions before the grid starts."""
    beats = analysis.beats_np
    if beats.size == 0:
        return 0.0
    # The grid is uniform by construction (see analysis._fine_tempo_and_phase),
    # so this is exact and needs no search. Deriving the step from `bpm` rather
    # than from the stored timestamps matters: both are rounded on the way to
    # JSON, and using two independently-rounded quantities here would make the
    # deck's beat position and the engine's beat clock disagree by ~1e-7 per
    # frame -- a slow phantom drift the supervisor would then chase.
    return (frame / SAMPLE_RATE - float(beats[0])) / analysis.beat_period


def frame_at_beat(analysis: TrackAnalysis, beat: float) -> float:
    """Inverse of :func:`beat_at_frame`."""
    beats = analysis.beats_np
    if beats.size == 0:
        return 0.0
    return (float(beats[0]) + beat * analysis.beat_period) * SAMPLE_RATE


def downbeat_offset_beats(analysis: TrackAnalysis) -> float:
    """How many beats past beat 0 the first downbeat sits (0 <= x < 4).

    Rounded before the modulo, and deliberately so: downbeats are grid points,
    so this is an integer by construction. Without the rounding, a first
    downbeat that coincides with beat 0 computes as a tiny *negative* float,
    and Python's ``%`` maps that to ~3.9999 rather than 0 -- which reports the
    cue point as bar -1 and shifts every phrase boundary by a bar.
    """
    if not analysis.downbeats or analysis.beats_np.size == 0:
        return 0.0
    off = beat_at_frame(analysis, analysis.first_downbeat * SAMPLE_RATE)
    return float(round(off) % BEATS_PER_BAR)


def bar_at_frame(analysis: TrackAnalysis, frame: float) -> float:
    """Fractional bar number, counted from the first downbeat (bar 0)."""
    beat = beat_at_frame(analysis, frame) - downbeat_offset_beats(analysis)
    return beat / BEATS_PER_BAR


def frame_at_bar(analysis: TrackAnalysis, bar: float) -> float:
    """Inverse of :func:`bar_at_frame`."""
    beat = bar * BEATS_PER_BAR + downbeat_offset_beats(analysis)
    return frame_at_beat(analysis, beat)


def phrase_at_frame(analysis: TrackAnalysis, frame: float) -> float:
    """Fractional 32-bar phrase number, counted from the first downbeat."""
    return bar_at_frame(analysis, frame) / BARS_PER_PHRASE


def next_phrase_boundary(deck: Deck, after_frame: float | None = None) -> int:
    """Deck-frame position of the next 32-bar phrase boundary.

    ``after_frame`` defaults to the deck's current playhead. Strictly *after*:
    sitting exactly on a boundary returns the following one, so a command can
    never be scheduled for a moment that has already passed.
    """
    if deck.track is None:
        return 0
    analysis = deck.track.analysis
    if after_frame is None:
        after_frame = deck.position
    bar = bar_at_frame(analysis, after_frame)
    phrase = bar / BARS_PER_PHRASE
    next_phrase = np.floor(phrase) + 1.0
    return int(round(frame_at_bar(analysis, next_phrase * BARS_PER_PHRASE)))


def next_downbeat(deck: Deck, after_frame: float | None = None) -> int:
    """Deck-frame position of the next bar line. Used for hard resyncs."""
    if deck.track is None:
        return 0
    analysis = deck.track.analysis
    if after_frame is None:
        after_frame = deck.position
    bar = bar_at_frame(analysis, after_frame)
    return int(round(frame_at_bar(analysis, np.floor(bar) + 1.0)))


# --- placement ---------------------------------------------------------------


@dataclass(frozen=True)
class TransitionPlan:
    """Where an automatic transition should happen, in deck-A frames."""

    #: Deck A frame at which the transition starts.
    start_frame: int
    #: Length actually planned. May be shorter than requested if the track was
    #: selected late.
    bars: float
    #: Deck B frame at which the incoming track starts playing: its mix-in.
    entry_frame_b: int
    #: True when there was not enough room left for a blend at all.
    is_cut: bool = False
    #: Why the plan differs from the ideal, for the session log. None if ideal.
    reason: str | None = None

    @property
    def end_frame(self) -> int:
        return self.start_frame


#: Outro-over-intro placement is only used when deck A's outro starts at least
#: this far into the track, the same bar the supervisor holds every automatic
#: transition to. An "outro" found early is a mislabel, not an ending.
OUTRO_MIN_START_FRACTION: float = 0.40


def _plan_outro_over_intro(
    deck_a: Deck, track_b: TrackAnalysis, transition_bars: float
) -> TransitionPlan | None:
    """Lay deck A's outro over deck B's intro, when both are labelled.

    The blend starts where A's outro starts and B enters at the start of its
    intro. Its length is the requested one, trimmed to what A has left, and
    never below :data:`MIN_TRANSITION_BARS`. None when either section is
    missing, A's outro has already begun, or it starts implausibly early --
    the caller then places back from mix-out as before.
    """
    analysis = deck_a.track.analysis
    outros = [s for s in getattr(analysis, "sections", None) or [] if s.get("label") == "outro"]
    intros = [s for s in getattr(track_b, "sections", None) or [] if s.get("label") == "intro"]
    if not outros or not intros:
        return None
    outro, intro = outros[-1], intros[0]
    start_bar = float(outro["start_bar"])
    now_bar = bar_at_frame(analysis, deck_a.position)
    if start_bar <= now_bar:
        return None
    start_frame = frame_at_bar(analysis, start_bar)
    total = analysis.duration_s * SAMPLE_RATE
    if total <= 0 or start_frame / total < OUTRO_MIN_START_FRACTION:
        return None
    last_bar = bar_at_frame(analysis, total) - 1.0
    bars = min(float(transition_bars), last_bar - start_bar)
    if bars < MIN_TRANSITION_BARS:
        return None
    entry_bar = max(0.0, float(intro["start_bar"]))
    entry_frame = int(round(max(0.0, frame_at_bar(track_b, entry_bar))))
    reason = (
        f"outro over intro: deck A's outro from bar {start_bar:.0f}, incoming "
        f"intro from bar {entry_bar:.0f}"
        + ("" if abs(bars - transition_bars) < 1e-6
           else f", {bars:.0f} of {transition_bars:.0f} bars fit")
    )
    return TransitionPlan(
        start_frame=int(round(start_frame)),
        bars=float(bars),
        entry_frame_b=entry_frame,
        reason=reason,
    )


def plan_transition(
    deck_a: Deck, track_b: TrackAnalysis, transition_bars: float
) -> TransitionPlan | None:
    """Place a transition so it *finishes* at deck A's mix-out point.

    This works backwards, and that is the whole point.
    :func:`next_phrase_boundary` answers "when is the next musically valid
    moment", which is an alignment question; it cannot answer "when should this
    track hand off", which is a placement question. Scheduling forwards from the
    moment a track happened to be selected is what put transitions in the middle
    of tracks.

    The steps are: take ``mix_out_bar`` as the target end, subtract the
    transition length, snap that start *down* to a
    :data:`PLACEMENT_GRID_BARS` boundary anchored at ``first_downbeat``.

    If that start has already passed -- the track was selected too late -- the
    next grid boundary is used instead and the transition is shortened to fit
    before mix-out, down to :data:`MIN_TRANSITION_BARS`. Below that floor there
    is no room to blend and a cut is planned instead.

    Returns None if deck A has no track loaded.
    """
    if deck_a.track is None:
        return None
    analysis = deck_a.track.analysis
    entry_b = int(round(track_b.mix_in * SAMPLE_RATE))

    outro_plan = _plan_outro_over_intro(deck_a, track_b, transition_bars)
    if outro_plan is not None:
        return outro_plan

    end_bar = analysis.mix_out_bar
    ideal_start_bar = end_bar - transition_bars
    start_bar = math.floor(ideal_start_bar / PLACEMENT_GRID_BARS) * PLACEMENT_GRID_BARS

    now_bar = bar_at_frame(analysis, deck_a.position)
    if start_bar > now_bar:
        # Snapping the start down leaves a remainder, so a fixed-length blend
        # would finish up to GRID-1 bars before mix-out. The length absorbs it
        # instead: the start stays on the 8-bar grid and the blend ends exactly
        # at mix-out, which is the point of planning backwards. The stretch is
        # bounded by the grid, so a 16-bar request becomes at most 23.
        bars = end_bar - start_bar
        return TransitionPlan(
            start_frame=int(round(frame_at_bar(analysis, start_bar))),
            bars=float(bars),
            entry_frame_b=entry_b,
            reason=(
                None
                if abs(bars - transition_bars) < 1e-6
                else f"stretched {transition_bars:.0f} -> {bars:.0f} bars so the "
                f"blend ends on mix-out at bar {end_bar:.1f}"
            ),
        )

    # Selected too late: take the next grid boundary and fit what remains.
    start_bar = math.ceil(now_bar / PLACEMENT_GRID_BARS) * PLACEMENT_GRID_BARS
    if start_bar <= now_bar:
        start_bar += PLACEMENT_GRID_BARS
    available = end_bar - start_bar

    if available >= MIN_TRANSITION_BARS:
        # Use the whole remainder so it still ends on mix-out.
        bars = available
        reason = (
            f"late selection: {transition_bars:.0f} bars would start at "
            f"bar {ideal_start_bar:.1f}, already past (now bar {now_bar:.1f}); "
            f"shortened to {bars:.0f} bars from bar {start_bar:.0f}"
        )
        log.info(reason)
        return TransitionPlan(
            start_frame=int(round(frame_at_bar(analysis, start_bar))),
            bars=float(bars),
            entry_frame_b=entry_b,
            reason=reason,
        )

    reason = (
        f"no room to blend: only {available:.1f} bars left before mix-out at "
        f"bar {end_bar:.1f} (now bar {now_bar:.1f}); cutting instead"
    )
    log.warning(reason)
    return TransitionPlan(
        start_frame=int(round(frame_at_bar(analysis, start_bar))),
        bars=0.0,
        entry_frame_b=entry_b,
        is_cut=True,
        reason=reason,
    )


# --- deck time <-> engine time ----------------------------------------------


def deck_frame_to_engine_frame(
    deck: Deck, engine_frame_now: float, target_deck_frame: float
) -> int:
    """Translate a deck-frame target into engine time at the deck's current rate.

    Valid only while the rate is unchanged. The supervisor's rate nudges are
    small (<=0.5%) and slow relative to a 32-bar phrase, so a command scheduled
    this way lands within a few milliseconds of its musical position -- well
    inside the tolerance of a 32-bar fade.
    """
    rate = deck.rate if deck.rate > 0 else 1.0
    delta_deck = target_deck_frame - deck.position
    return int(round(engine_frame_now + delta_deck / rate))


def bars_until(
    deck: Deck, engine_frame_now: float, execute_at_engine_frame: float
) -> float:
    """How many bars of *this deck's* music remain before an engine-time event.

    This is what the REPL prints when it tells the user when a queued command
    will fire.
    """
    if deck.track is None:
        return 0.0
    engine_frames = execute_at_engine_frame - engine_frame_now
    rate = deck.rate if deck.rate > 0 else 1.0
    target_deck_frame = deck.position + engine_frames * rate
    return bar_at_frame(deck.track.analysis, target_deck_frame) - bar_at_frame(
        deck.track.analysis, deck.position
    )
