"""Glue: short gestures that mask an interrupt, and the grooves that place them.

THREADING CONTEXT: control thread. Every function here returns scheduled
commands (and the riser renders its buffer here); nothing is ever called from
the audio thread, which only applies the commands.

The four primitives are the things a DJ reaches for to hide a change of plan:
a short reverse, a noise riser, a filter swell and a volume dip. Each is laid
out on the master beat grid in beats, so it is tempo-relative by construction,
and each command can carry a :class:`~djai.commands.CancelToken` so the whole
gesture can be withdrawn if the plan changes again.

**Micro-timing.** A groove template shifts each sixteenth of a beat by a few
milliseconds -- pushed ahead, laid back, or swung. Offsets are ±10-40 ms, the
range where a move reads as feel rather than as late; "straight" is the one
template with none. A reverse is never grooved: its exit resumes from the slip
position, so its timing does not move the deck's phase either way.
"""

from __future__ import annotations

import math

from djai.commands import (
    CancelToken,
    Command,
    DeckName,
    SetFilter,
    SetMasterGain,
    SetReverse,
    SetRiser,
)
from djai.deck import SAMPLE_RATE
from djai.engine import render_riser

#: Offsets in milliseconds for the four sixteenths of a beat. Positive is late.
GROOVES: dict[str, tuple[float, float, float, float]] = {
    "straight": (0.0, 0.0, 0.0, 0.0),
    "push": (-15.0, -10.0, -15.0, -10.0),
    "laid_back": (20.0, 12.0, 20.0, 12.0),
    "shuffle": (0.0, 30.0, 0.0, 30.0),
    "drag": (35.0, 25.0, 40.0, 25.0),
}
MIN_OFFSET_MS: float = 10.0
MAX_OFFSET_MS: float = 40.0

GLUE_KINDS: tuple[str, ...] = ("reverse", "riser", "filter_swell", "volume_dip")


def groove_offset_frames(groove: str, sixteenth: int) -> int:
    """How far a groove moves a hit on this sixteenth, in frames."""
    ms = GROOVES[groove][sixteenth % 4]
    return int(round(ms * SAMPLE_RATE / 1000.0))


def _at(start: int, beat_frames: float, beats: float, groove: str) -> int:
    """Engine frame of ``beats`` after ``start``, with the groove applied."""
    sixteenth = int(round(beats * 4))
    return int(round(start + beats * beat_frames)) + groove_offset_frames(groove, sixteenth)


def beat_grid(engine) -> tuple[int, float]:
    """The next master downbeat-of-a-beat, in engine frames, and a beat's length.

    Read from one consistent clock snapshot, so the two cannot disagree.
    """
    frames, master_beat, _, _ = engine.clock
    bpm = engine.master_bpm or 120.0
    beat_frames = SAMPLE_RATE * 60.0 / bpm
    to_next = (math.floor(master_beat) + 1 - master_beat) * beat_frames
    return int(round(frames + to_next)), beat_frames


def volume_dip(
    start: int, beat_frames: float, beats: float = 1.0, depth: float = 0.5,
    groove: str = "straight", token: CancelToken | None = None, origin: str = "glue",
) -> list[Command]:
    """Dip the master to ``depth`` for ``beats``, then back to unity."""
    depth = min(1.0, max(0.0, depth))
    return [
        SetMasterGain(gain=depth, execute_at=_at(start, beat_frames, 0, groove),
                      origin=origin, token=token),
        SetMasterGain(gain=1.0, execute_at=_at(start, beat_frames, beats, groove),
                      origin=origin, token=token),
    ]


def filter_swell(
    deck: DeckName, start: int, beat_frames: float, beats: float = 4.0,
    peak: float = 0.6, steps: int = 8, groove: str = "straight",
    token: CancelToken | None = None, origin: str = "glue",
) -> list[Command]:
    """Sweep a deck's high-pass up to ``peak`` and back to the detent.

    A half-sine over ``beats``: rises for the first half, falls for the
    second, and always lands on 0 so the deck is left unfiltered.
    """
    steps = max(2, int(steps))
    out: list[Command] = []
    for i in range(1, steps + 1):
        pos = 0.0 if i == steps else peak * math.sin(math.pi * i / steps)
        out.append(SetFilter(
            deck=deck, position=round(pos, 4),
            execute_at=_at(start, beat_frames, beats * i / steps, groove),
            origin=origin, token=token,
        ))
    return out


def noise_riser(
    start: int, beat_frames: float, beats: float = 8.0, peak: float = 0.25,
    steps: int = 8, groove: str = "straight", token: CancelToken | None = None,
    origin: str = "glue",
) -> list[Command]:
    """A riser that climbs to ``peak`` over ``beats`` and stops dead on the last.

    Renders its buffer here, on the control thread: seconds of band-passed
    noise, which the audio thread will only ever read.
    """
    steps = max(2, int(steps))
    # A quarter-second longer than the gesture: its last step may be grooved
    # late, and a buffer that runs out under a raised level stops dead.
    buf = render_riser(int(beats * beat_frames + 0.25 * SAMPLE_RATE))
    out: list[Command] = [SetRiser(
        gain=0.0, riser=buf, execute_at=_at(start, beat_frames, 0, groove),
        origin=origin, token=token,
    )]
    for i in range(1, steps):
        out.append(SetRiser(
            gain=round(peak * (i / (steps - 1)) ** 2, 4),
            execute_at=_at(start, beat_frames, beats * i / steps, groove),
            origin=origin, token=token,
        ))
    out.append(SetRiser(
        gain=0.0, execute_at=_at(start, beat_frames, beats, groove),
        origin=origin, token=token,
    ))
    return out


def short_reverse(
    deck: DeckName, start: int, beat_frames: float, beats: float = 1.0,
    token: CancelToken | None = None, origin: str = "glue",
) -> list[Command]:
    """Play ``deck`` backwards for ``beats``, then resume where it would be."""
    return [
        SetReverse(deck=deck, on=True, execute_at=int(round(start)),
                   origin=origin, token=token),
        SetReverse(deck=deck, on=False,
                   execute_at=int(round(start + beats * beat_frames)),
                   origin=origin, token=token),
    ]


def closing_commands(dropped: list[Command]) -> list[Command]:
    """What must still happen when a gesture is withdrawn part-way through.

    Cancelling a token withdraws the command that would have *ended* the
    gesture, and some gestures cannot be left open: a riser left sounding, a
    master left dipped, a deck left playing backwards or stuck in a loop.
    Returns those endings, due now, one per control -- the last one withdrawn
    wins. A filter or fader left mid-move is a level, not a fault, and the
    plan that superseded it moves on from there.
    """
    from dataclasses import replace

    from djai.commands import IMMEDIATE, ExitLoop

    ends: dict[tuple, Command] = {}
    for cmd in dropped:
        if isinstance(cmd, SetRiser) and cmd.gain == 0.0 and cmd.riser is None:
            ends[("riser",)] = cmd
        elif isinstance(cmd, SetMasterGain) and cmd.gain == 1.0:
            ends[("master",)] = cmd
        elif isinstance(cmd, SetReverse) and not cmd.on:
            ends[("reverse", cmd.deck)] = cmd
        elif isinstance(cmd, ExitLoop):
            ends[("loop", cmd.deck)] = cmd
    return [replace(c, execute_at=IMMEDIATE, token=None, origin=f"{c.origin}:closed")
            for c in ends.values()]
