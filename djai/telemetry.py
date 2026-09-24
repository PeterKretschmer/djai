"""The feature bus: what the room sounds like, as the callback saw it.

THREADING CONTEXT: exactly one writer -- the **audio thread**, through
:meth:`FeatureBus.publish` -- and any number of readers on any other thread.

The rule this module exists to keep is the first invariant: nothing blocks the
audio callback. So the writer never takes a lock, never allocates, never does
I/O, and -- the part that is easy to get wrong -- **never reads consumer
state**. There is no "is the reader ready" flag and no backpressure, because
either would let a stalled consumer reach into the audio thread. A reader that
falls behind loses old rows and nothing else; the drop is counted so the loss
is visible rather than silent.

Rows are a preallocated flat float64 array addressed by integer offsets. Flat
and integer-indexed deliberately: ``buf[slot, i]`` builds a tuple for the index
and ``buf[slot]`` builds a view object, and both allocate. ``buf[base + i]``
does not.

Publication is a single ``self._seq = n`` rebind *after* the row is filled. A
reader sees a row only once its sequence number has been published, and
re-checks the sequence after copying to know the row was not overwritten
underneath it -- the usual seqlock read, which needs no lock on either side.

Musical time (bar, bars to phrase end, time to downbeat) is **derived by the
reader**, not by the callback. Each row carries the deck's grid constants, so a
snapshot is self-describing: the derivation cannot race a track swap, and the
callback does not pay for arithmetic nobody may ask for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

#: 4/4 throughout, as everywhere else in this program.
BEATS_PER_BAR: int = 4
#: A phrase is 32 bars. Mirrors :data:`djai.phrase.BARS_PER_PHRASE`; not
#: imported so the bus stays independent of the crate code.
BARS_PER_PHRASE: int = 32

# --- row layout ---------------------------------------------------------------
#
# Fixed offsets rather than a dict or a dtype with names: the callback writes
# these by integer index, and an integer is the only index that does not
# allocate.

_SEQ = 0
_TIME = 1
_FRAMES = 2
_MASTER_BEAT = 3
_BEATS_PER_FRAME = 4
_TRANSITION = 5
_MASTER_PEAK_IN = 6
_MASTER_PEAK = 7
_LIMITER_GAIN = 8

#: Where deck A's block starts, and how long each deck's block is.
_DECK_A = 9
_DECK_STRIDE = 15
_DECK_B = _DECK_A + _DECK_STRIDE

# offsets within a deck block
_D_PLAYING = 0
_D_POSITION = 1
_D_RATE = 2
_D_STRETCH = 3
_D_NATIVE_BPM = 4
_D_BEAT0 = 5
_D_BEAT_PERIOD = 6
_D_DOWNBEAT_OFFSET = 7
_D_ENERGY = 8
_D_GAIN = 9
_D_EQ_LOW = 10
_D_EQ_MID = 11
_D_EQ_HIGH = 12
_D_FILTER = 13
_D_LOOP_FRAMES = 14

ROW_FIELDS: int = _DECK_B + _DECK_STRIDE

#: Sample rate the positions are counted in. Mirrors djai.deck.SAMPLE_RATE.
SAMPLE_RATE: int = 44100


@dataclass(frozen=True)
class DeckFeatures:
    """One deck, as of one callback. Musical time is derived here, not there."""

    name: str
    playing: bool
    position_frames: float
    rate: float
    stretch_ratio: float
    native_bpm: float
    energy: float
    gain: float
    eq: tuple[float, float, float]
    filter_pos: float
    loop_frames: float
    #: Grid constants the row carried, so the derivation below cannot race a
    #: track swap on the deck.
    beat0_s: float
    beat_period_s: float
    downbeat_offset_beats: float

    @property
    def tempo(self) -> float:
        """What this deck actually sounds like, in BPM."""
        return self.native_bpm * self.rate

    @property
    def beat(self) -> float:
        """Fractional beat index. Beat 0 is the first beat of the grid."""
        if self.beat_period_s <= 0.0:
            return 0.0
        return (self.position_frames / SAMPLE_RATE - self.beat0_s) / self.beat_period_s

    @property
    def bar(self) -> float:
        """Fractional bar, counted from the first downbeat."""
        return (self.beat - self.downbeat_offset_beats) / BEATS_PER_BAR

    @property
    def phase(self) -> float:
        """Where in the bar this deck is, 0..1."""
        return self.bar % 1.0

    @property
    def bars_to_phrase_end(self) -> float:
        """Bars remaining in the current 32-bar phrase."""
        return BARS_PER_PHRASE - (self.bar % BARS_PER_PHRASE)

    def as_dict(self) -> dict:
        """Compact, JSON-safe, and including what was derived.

        The derived values are written out rather than left to be recomputed:
        a log read back in six months must not depend on this class's
        arithmetic still being what it was.
        """
        return {
            "playing": self.playing,
            "position_frames": round(self.position_frames, 1),
            "rate": round(self.rate, 6),
            "stretch_ratio": round(self.stretch_ratio, 6),
            "native_bpm": round(self.native_bpm, 3),
            "tempo": round(self.tempo, 3),
            "bar": round(self.bar, 4),
            "phase": round(self.phase, 4),
            "bars_to_phrase_end": round(self.bars_to_phrase_end, 3),
            "seconds_to_downbeat": round(self.seconds_to_downbeat, 4),
            "energy": round(self.energy, 5),
            "gain": round(self.gain, 4),
            "eq": [round(v, 4) for v in self.eq],
            "filter_pos": round(self.filter_pos, 4),
            "loop_frames": round(self.loop_frames, 1),
        }

    @property
    def seconds_to_downbeat(self) -> float:
        """Wall-clock seconds until this deck's next bar line.

        At the rate the deck is actually running: a deck ridden 6% fast reaches
        its next downbeat 6% sooner, and a scheduler that used the printed
        tempo would fire late by exactly that.
        """
        if self.beat_period_s <= 0.0 or self.rate <= 0.0:
            return 0.0
        bars_left = 1.0 - (self.bar % 1.0)
        return bars_left * BEATS_PER_BAR * self.beat_period_s / self.rate


@dataclass(frozen=True)
class Snapshot:
    """The whole mix as of one callback, with the time it was taken."""

    seq: int
    #: ``time.perf_counter`` at the end of the callback that filled this row.
    monotonic: float
    frames_played: float
    master_beat: float
    master_bpm: float
    transition_active: bool
    master_peak_in: float
    master_peak: float
    limiter_gain: float
    a: DeckFeatures
    b: DeckFeatures

    def deck(self, name: str) -> DeckFeatures:
        return self.a if name == "a" else self.b

    def as_dict(self) -> dict:
        """What goes in the log beside a decision (SPEC §7)."""
        return {
            "seq": self.seq,
            "t": round(self.monotonic, 6),
            "frames_played": round(self.frames_played, 1),
            "master_beat": round(self.master_beat, 4),
            "master_bpm": round(self.master_bpm, 3),
            "transition_active": self.transition_active,
            "master_peak_in": round(self.master_peak_in, 5),
            "master_peak": round(self.master_peak, 5),
            "limiter_gain": round(self.limiter_gain, 5),
            "a": self.a.as_dict(),
            "b": self.b.as_dict(),
        }

    @property
    def age_seconds(self) -> float:
        """How stale this snapshot is now. The bus's latency, measured.

        Against :func:`time.perf_counter`, which is what stamped it. On
        Windows ``time.monotonic`` advances in ~15.6 ms steps, so a latency
        measured with it reads as either 0 ms or 16 ms and says nothing about
        a bus that answers in microseconds.
        """
        return time.perf_counter() - self.monotonic


class FeatureBus:
    """A wait-free ring of callback snapshots.

    ``capacity`` rows are preallocated at construction and never reallocated.
    At 512 rows and a 512-frame block that is about six seconds of history at
    44.1 kHz, which is more than any consumer here looks back.
    """

    def __init__(self, capacity: int = 512) -> None:
        if capacity < 2:
            raise ValueError("capacity must be at least 2")
        self._capacity = int(capacity)
        self._buf = np.zeros(self._capacity * ROW_FIELDS, dtype=np.float64)
        #: Rows published so far. The only thing the writer rebinds, and it is
        #: rebound last -- so a row is visible exactly when it is complete.
        self._seq = 0
        #: Rows a reader asked for and did not get because they had been
        #: overwritten. Counted, so falling behind is visible.
        self.dropped = 0
        self.published = 0

    # --- writer: AUDIO THREAD ---------------------------------------------
    #
    # No locks, no allocation, no I/O, and no read of any consumer's state.

    def publish(
        self,
        #: ``time.perf_counter`` at the end of the callback that filled this row.
    monotonic: float,
        frames_played: float,
        master_beat: float,
        beats_per_frame: float,
        transition_active: bool,
        master_peak_in: float,
        master_peak: float,
        limiter_gain: float,
        deck_a,
        deck_b,
    ) -> None:
        """Write one row and publish it. **Audio thread only.**

        ``deck_a``/``deck_b`` are :class:`djai.deck.Deck` objects; only plain
        float attributes are read off them, all of which the deck maintains as
        it plays. Nothing here searches, allocates or calls into analysis.
        """
        buf = self._buf
        seq = self._seq + 1
        base = (seq % self._capacity) * ROW_FIELDS

        buf[base + _SEQ] = seq
        buf[base + _TIME] = monotonic
        buf[base + _FRAMES] = frames_played
        buf[base + _MASTER_BEAT] = master_beat
        buf[base + _BEATS_PER_FRAME] = beats_per_frame
        buf[base + _TRANSITION] = 1.0 if transition_active else 0.0
        buf[base + _MASTER_PEAK_IN] = master_peak_in
        buf[base + _MASTER_PEAK] = master_peak
        buf[base + _LIMITER_GAIN] = limiter_gain

        self._write_deck(buf, base + _DECK_A, deck_a)
        self._write_deck(buf, base + _DECK_B, deck_b)

        # Last, and alone: everything above is visible to a reader only once
        # this lands, and a single attribute rebind is atomic under the GIL.
        self._seq = seq
        self.published += 1

    @staticmethod
    def _write_deck(buf: np.ndarray, at: int, deck) -> None:
        """AUDIO THREAD. Plain attribute reads and float stores, nothing else."""
        track = deck.track
        buf[at + _D_PLAYING] = 1.0 if deck.playing else 0.0
        buf[at + _D_POSITION] = deck.position
        buf[at + _D_RATE] = deck.rate
        buf[at + _D_STRETCH] = track.stretch_rate if track is not None else 1.0
        # Cached on the deck at attach time (see Deck.attach): reading them off
        # `analysis` here would be a property call per block for numbers that
        # only change when a record changes.
        buf[at + _D_NATIVE_BPM] = deck.grid_bpm
        buf[at + _D_BEAT0] = deck.grid_beat0_s
        buf[at + _D_BEAT_PERIOD] = deck.grid_beat_period_s
        buf[at + _D_DOWNBEAT_OFFSET] = deck.grid_downbeat_offset_beats
        buf[at + _D_ENERGY] = deck.current_beat_energy()
        buf[at + _D_GAIN] = deck.gain.cur
        buf[at + _D_EQ_LOW] = deck.eq_low.cur
        buf[at + _D_EQ_MID] = deck.eq_mid.cur
        buf[at + _D_EQ_HIGH] = deck.eq_high.cur
        buf[at + _D_FILTER] = deck.filter_pos.cur
        buf[at + _D_LOOP_FRAMES] = deck._loop[1]

    # --- readers: any other thread ----------------------------------------

    @property
    def sequence(self) -> int:
        """Rows published so far."""
        return self._seq

    def _read(self, seq: int) -> Snapshot | None:
        """Copy row ``seq`` out, or None if it has been overwritten.

        The sequence is checked after the copy as well as before: if the writer
        lapped this slot while it was being read, the row is discarded rather
        than returned half-new. That is the whole synchronisation.
        """
        if seq <= 0 or seq > self._seq:
            return None
        if self._seq - seq >= self._capacity:
            return None
        base = (seq % self._capacity) * ROW_FIELDS
        row = self._buf[base:base + ROW_FIELDS].copy()
        if int(row[_SEQ]) != seq or self._seq - seq >= self._capacity:
            return None  # lapped mid-read
        return _snapshot_from(row)

    def latest(self) -> Snapshot | None:
        """The most recent complete snapshot, or None before the first block."""
        return self._read(self._seq)

    def since(self, seq: int, limit: int = 0) -> list[Snapshot]:
        """Every snapshot published after ``seq``, oldest first.

        Rows that have been overwritten are counted in :attr:`dropped` and left
        out; the caller sees a gap in sequence numbers rather than a silent
        loss. ``limit`` caps how many are returned, newest kept.
        """
        newest = self._seq
        if newest <= seq:
            return []
        oldest = max(seq + 1, newest - self._capacity + 1)
        if oldest > seq + 1:
            self.dropped += oldest - (seq + 1)
        if limit > 0 and newest - oldest + 1 > limit:
            oldest = newest - limit + 1
        out = []
        for s in range(oldest, newest + 1):
            snap = self._read(s)
            if snap is not None:
                out.append(snap)
        return out


def _deck_from(row: np.ndarray, at: int, name: str) -> DeckFeatures:
    return DeckFeatures(
        name=name,
        playing=bool(row[at + _D_PLAYING]),
        position_frames=float(row[at + _D_POSITION]),
        rate=float(row[at + _D_RATE]),
        stretch_ratio=float(row[at + _D_STRETCH]),
        native_bpm=float(row[at + _D_NATIVE_BPM]),
        energy=float(row[at + _D_ENERGY]),
        gain=float(row[at + _D_GAIN]),
        eq=(float(row[at + _D_EQ_LOW]), float(row[at + _D_EQ_MID]),
            float(row[at + _D_EQ_HIGH])),
        filter_pos=float(row[at + _D_FILTER]),
        loop_frames=float(row[at + _D_LOOP_FRAMES]),
        beat0_s=float(row[at + _D_BEAT0]),
        beat_period_s=float(row[at + _D_BEAT_PERIOD]),
        downbeat_offset_beats=float(row[at + _D_DOWNBEAT_OFFSET]),
    )


def _snapshot_from(row: np.ndarray) -> Snapshot:
    beats_per_frame = float(row[_BEATS_PER_FRAME])
    return Snapshot(
        seq=int(row[_SEQ]),
        monotonic=float(row[_TIME]),
        frames_played=float(row[_FRAMES]),
        master_beat=float(row[_MASTER_BEAT]),
        master_bpm=beats_per_frame * 60.0 * SAMPLE_RATE,
        transition_active=bool(row[_TRANSITION]),
        master_peak_in=float(row[_MASTER_PEAK_IN]),
        master_peak=float(row[_MASTER_PEAK]),
        limiter_gain=float(row[_LIMITER_GAIN]),
        a=_deck_from(row, _DECK_A, "a"),
        b=_deck_from(row, _DECK_B, "b"),
    )
