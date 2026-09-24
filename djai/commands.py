"""Structured commands -- the entire vocabulary between the control layers and
the audio engine.

THREADING CONTEXT: constructed on the main / monitor threads, consumed on the
**audio thread**. Everything a command carries must therefore be inert: plain
floats, ints, strings, and already-decoded :class:`~djai.deck.LoadedTrack`
objects. Applying a command on the audio thread must never decode, allocate or
block -- which is why :class:`LoadTrack` carries finished audio rather than a
path.

Every command carries ``execute_at``, an **engine-time** frame position (see
:mod:`djai.phrase`), or the sentinel :data:`IMMEDIATE`. Anything quantized to
musical position resolves to a real frame number before it is queued; only
panic commands and supervisor interventions use ``IMMEDIATE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from djai.deck import LoadedTrack

#: ``execute_at`` sentinel meaning "apply at the top of the next audio callback".
IMMEDIATE: int = -1

DeckName = Literal["a", "b"]


class CancelToken:
    """A handle on a group of scheduled commands, so they can be withdrawn together.

    Read on the scheduler thread only; the audio thread never sees one. A
    single bool, set once: cancelling is idempotent and never undone.
    """

    __slots__ = ("cancelled", "name")

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def __repr__(self) -> str:
        return f"CancelToken({self.name!r}{', cancelled' if self.cancelled else ''})"


@dataclass(frozen=True)
class Command:
    """Base for every command. ``origin`` is for the session log only."""

    execute_at: int = IMMEDIATE
    origin: str = "system"
    #: The group this command belongs to (Phase 2.1). Cancelling the token
    #: withdraws every command still pending under it. Not part of equality.
    token: CancelToken | None = field(default=None, compare=False, repr=False)

    @property
    def is_immediate(self) -> bool:
        return self.execute_at == IMMEDIATE

    def describe(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class LoadTrack(Command):
    """Publish an already-decoded track to a deck and cue it to ``start_frame``.

    The decode happens on a worker thread; by the time this reaches the audio
    thread it is a reference assignment.
    """

    deck: DeckName = "a"
    track: LoadedTrack | None = None
    start_frame: int = 0
    rate: float = 1.0
    play: bool = True
    #: Make this deck drive the master beat clock once the load lands. Carried
    #: on the command rather than set separately, because the control thread
    #: cannot see the track on the deck until the audio thread has attached it.
    master: bool = False

    def describe(self) -> str:
        title = self.track.title if self.track else "<none>"
        return f"LoadTrack({self.deck} <- {title!r} @ rate {self.rate:.4f})"


@dataclass(frozen=True)
class StartTransition(Command):
    """Begin the bass-swap crossfade. The engine advances it per block; the
    shape lives in :mod:`djai.transition`."""

    from_deck: DeckName = "a"
    to_deck: DeckName = "b"
    total_frames: int = 0
    #: Placed by lining the two tracks' drops up. The supervisor's early-start
    #: guard is relaxed for these, and only these, down to its absolute floor.
    drop_aligned: bool = False

    def describe(self) -> str:
        return (
            f"StartTransition({self.from_deck} -> {self.to_deck}, "
            f"{self.total_frames} frames)"
        )


@dataclass(frozen=True)
class SetEQ(Command):
    """Set one deck's band gains. ``None`` leaves a band untouched."""

    deck: DeckName = "a"
    low: float | None = None
    mid: float | None = None
    high: float | None = None

    def describe(self) -> str:
        return f"SetEQ({self.deck} low={self.low} mid={self.mid} high={self.high})"


@dataclass(frozen=True)
class SetGain(Command):
    """Set one deck's channel level -- the mixer fader, in command form.

    Distinct from :class:`Cut`, which is a panic that slams a deck to silence
    and is not something an operator dials back up. This is an ordinary level
    move, smoothed by the deck like every other gain change.

    Like :class:`KillBass`, applying this to a deck in an active transition
    ends that transition: the crossfade drives the same gains every block, so
    leaving it running would simply overwrite the operator a moment later, and
    manual control has to outrank automation rather than argue with it.
    """

    deck: DeckName = "a"
    gain: float = 1.0

    def describe(self) -> str:
        return f"SetGain({self.deck}, {self.gain:.3f})"


@dataclass(frozen=True)
class SetFilter(Command):
    """Turn one deck's filter knob, and/or set its resonance.

    ``position`` is -1..1: negative sweeps a low-pass down, positive a
    high-pass up, 0 is the centre detent (no filter). ``resonance`` is 0..1.
    ``None`` leaves either untouched. A knob position, never a frequency: the
    deck's prebuilt table turns it into coefficients.
    """

    deck: DeckName = "a"
    position: float | None = None
    resonance: float | None = None

    def describe(self) -> str:
        return f"SetFilter({self.deck} position={self.position} resonance={self.resonance})"


@dataclass(frozen=True)
class SetKeyLock(Command):
    """Turn one deck's key lock on or off.

    Off puts the unstretched source back under the playhead (resampling). On
    records the choice; ``track``, when given, is a finished stretched copy of
    the deck's current track to swap in. The copy is always made on the
    stretch worker -- this carries the result, never the work.
    """

    deck: DeckName = "a"
    on: bool = True
    track: LoadedTrack | None = None

    def describe(self) -> str:
        extra = " +copy" if self.track is not None else ""
        return f"SetKeyLock({self.deck}, {'on' if self.on else 'off'}{extra})"


@dataclass(frozen=True)
class SwapStems(Command):
    """Put a stem mix of the deck's own track under the playhead, on a bar line.

    ``track`` is a finished mix built by :func:`djai.stems.load_mix` on a
    control thread -- this carries the audio, never the decoding or the
    separation. ``mix`` is its name, for the log. Applying it is the same two
    assignments key lock makes: the playhead does not move, and the deck
    crossfades out of the buffer it is leaving, so the change does not click.

    A stem move is a swap on a bar line rather than a per-block stem gain
    precisely so the audio callback keeps reading one buffer per deck.
    """

    deck: DeckName = "a"
    mix: str = "full"
    track: LoadedTrack | None = None

    def describe(self) -> str:
        return f"SwapStems({self.deck}, {self.mix})"


# --- performance controls (Phase 4) -------------------------------------------
# Every frame here is resolved from the beat grid on the control thread. The
# audio thread assigns; it never works out where a beat is.


@dataclass(frozen=True)
class SetLoop(Command):
    """Engage, or resize, a slip loop on one deck. Original track frames."""

    deck: DeckName = "a"
    start_frame: float = 0.0
    length_frames: float = 0.0

    def describe(self) -> str:
        return f"SetLoop({self.deck} @ {self.start_frame:.0f}, {self.length_frames:.0f} frames)"


@dataclass(frozen=True)
class ExitLoop(Command):
    """Leave a loop where straight playback would be, so phase is kept."""

    deck: DeckName = "a"

    def describe(self) -> str:
        return f"ExitLoop({self.deck})"


@dataclass(frozen=True)
class BeatJump(Command):
    """Move a deck's playhead by an exact grid distance, in original frames."""

    deck: DeckName = "a"
    frames: float = 0.0

    def describe(self) -> str:
        return f"BeatJump({self.deck}, {self.frames:+.0f} frames)"


@dataclass(frozen=True)
class SetPitch(Command):
    """The pitch fader: an operator's rate for one deck.

    Distinct from :class:`SetRate`, which is the supervisor's drift nudge. A
    deck moved by hand is no longer drift-corrected until it is synced again.
    """

    deck: DeckName = "a"
    rate: float = 1.0

    def describe(self) -> str:
        return f"SetPitch({self.deck}, {self.rate:.4f})"


@dataclass(frozen=True)
class SyncDeck(Command):
    """Match one deck's tempo to the master and shift it into phase."""

    deck: DeckName = "a"
    rate: float = 1.0
    shift_frames: float = 0.0

    def describe(self) -> str:
        return f"SyncDeck({self.deck}, rate {self.rate:.4f}, shift {self.shift_frames:+.0f})"


@dataclass(frozen=True)
class Cut(Command):
    """Panic: silence a deck (or the master) now. Bypasses the LLM entirely."""

    deck: DeckName | Literal["master"] = "master"

    def describe(self) -> str:
        return f"Cut({self.deck})"


@dataclass(frozen=True)
class KillBass(Command):
    """Panic: drop or restore a deck's low band now."""

    deck: DeckName | Literal["master"] = "master"
    killed: bool = True

    def describe(self) -> str:
        return f"KillBass({self.deck}, killed={self.killed})"


@dataclass(frozen=True)
class Stop(Command):
    """Panic: stop both decks and end the session."""

    def describe(self) -> str:
        return "Stop()"


# --- reactivity and glue (Phase 2.1) -------------------------------------------
# Built on the control thread by djai.glue and the session; the audio thread
# does assignments and per-block float maths with them, nothing more.


@dataclass(frozen=True)
class CancelTransition(Command):
    """Abandon the blend in flight and go back to the outgoing track.

    Not an abort: an abort leaves both decks playing at unity. This fades the
    incoming deck out over ``fade_frames`` and stops it, while the outgoing
    deck's level and EQ return to unity over the same span, so the room is
    left with exactly the track it had before the blend began.
    """

    fade_frames: int = 0

    def describe(self) -> str:
        return f"CancelTransition(fade {self.fade_frames} frames)"


@dataclass(frozen=True)
class SetReverse(Command):
    """Play one deck backwards, slip-style, or stop doing so.

    While reversed the deck's straight-through position keeps advancing, and
    turning reverse off resumes from it: the deck comes out where it would
    have been, in phase, whatever the reverse ran for.
    """

    deck: DeckName = "a"
    on: bool = True

    def describe(self) -> str:
        return f"SetReverse({self.deck}, {'on' if self.on else 'off'})"


@dataclass(frozen=True)
class SetRiser(Command):
    """Set the noise riser's level; with ``riser``, start that buffer from its top.

    The buffer is rendered on the control thread (see
    :func:`djai.engine.render_riser`); the audio thread only reads it. A
    transition's envelope drives the same level, so this is for use outside
    one -- the supervisor enforces that.
    """

    gain: float = 0.0
    riser: object = field(default=None, compare=False, repr=False)

    def describe(self) -> str:
        return f"SetRiser({self.gain:.3f}{', new buffer' if self.riser is not None else ''})"


@dataclass(frozen=True)
class SetMasterGain(Command):
    """The master level, ramped over one block. For volume dips, never a boost."""

    gain: float = 1.0

    def describe(self) -> str:
        return f"SetMasterGain({self.gain:.3f})"


# --- supervisor-only interventions -------------------------------------------
# These are not user-facing commands. They exist because supervisor.py is
# specified to nudge playback rate and to hard-resync a drifting deck, and both
# must reach the decks through the same queue as everything else rather than by
# mutating deck state from the monitor thread.


@dataclass(frozen=True)
class SetRate(Command):
    """Fractional playback-rate change. Drift correction only."""

    deck: DeckName = "a"
    rate: float = 1.0

    def describe(self) -> str:
        return f"SetRate({self.deck}, {self.rate:.6f})"


@dataclass(frozen=True)
class Resync(Command):
    """Hard-set a deck's playhead. Drift correction only."""

    deck: DeckName = "a"
    frame: int = 0

    def describe(self) -> str:
        return f"Resync({self.deck}, frame={self.frame})"


#: Commands the user can trigger without going through the LLM.
PANIC_TYPES: tuple[type[Command], ...] = (Cut, KillBass, Stop)

