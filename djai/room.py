"""What the room is hearing, and whether it is working (SPEC §5).

THREADING CONTEXT: control threads only -- the autopilot tick polls
:class:`RoomMonitor`. The audio thread's whole part in this is one ring write
per block (``Engine.output_tap``), the same pattern the recorder uses: nothing
here is ever called from the callback.

Three things live here:

* **proxies** -- numbers measured off the limited master output: loudness per
  bar, spectral flux, energy variance and loudness dynamics. They are
  *proxies*: they say what the system played, not what the crowd felt, and
  every reading carries ``proxy: True`` so the label travels with the number.
* **the room mic** -- optional and off by default. It measures the same
  proxies from a microphone and reports how well they agree with the master
  (``signal_quality``). The loop never acts on it, so turning it off changes
  nothing but the quality of what is observed. This module opens no input
  device: :meth:`RoomMic.feed` is where one would deliver its blocks.
* **the "is this working?" model** -- :class:`EnergyLoop`, a heuristic and
  labelled as one. It compares each bar's measured energy with the same
  measure taken on the playing track's own audio at that bar
  (:func:`source_energy`), so a breakdown the producer wrote is not a drop:
  what is left is what the mix did. Off by 2.5 dB for two bars is a drop or a
  spike, and the loop answers it with a bounded correction through the Short
  band -- first by putting back energy controls something else moved, then by
  stepping a correction level. :meth:`EnergyLoop.assess` is the seam a learned
  model would fill; nothing is trained here.

Measured (Phase 4): 25 real crate tracks played through the engine with the
loop on and nothing injected, 2,524 bars, no correction; the same tracks with
a filter, EQ or fader fault injected, 15/15 recovered in 9 bars, no reversal.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from djai import config

SAMPLE_RATE: int = 44100
#: Analysis hop and FFT size, in frames. ~23 ms hops, ~46 ms windows.
HOP: int = 1024
FFT: int = 2048
#: Bars of history the variance and dynamics proxies look back over.
SUMMARY_BARS: int = 8
#: Bars the mic-to-master agreement is measured over.
QUALITY_BARS: int = 16
#: Fewest hops a bar needs to be read at all: a bar the tap only partly saw
#: (the loop starting mid-bar, a gap) is dropped rather than misread.
MIN_HOPS_PER_BAR: int = 8


@dataclass(frozen=True)
class BarReading:
    """One bar of output, measured. Every value is a proxy."""

    bar: int
    #: K-weighted loudness, in dB (relative, not calibrated LUFS).
    loudness_db: float
    #: Level of the 2-16 kHz band, in dB: the brightness a crowd hears as
    #: intensity, and what the loop's EQ and filter moves mostly change.
    presence_db: float
    flux: float
    hops: int
    hop_db: tuple[float, ...] = field(default=(), repr=False)

    @property
    def energy_db(self) -> float:
        """The loop's energy proxy: loudness and presence, equally weighted.

        Loudness alone is owned by the kick and the bass -- on a mix the
        "harder" EQ lift moves it +0.35 dB while it is plainly audible -- and
        presence alone would miss a fader. Together each counts.
        """
        return 0.5 * (self.loudness_db + self.presence_db)

    def as_dict(self) -> dict:
        return {"bar": self.bar, "loudness_db": round(self.loudness_db, 2),
                "presence_db": round(self.presence_db, 2),
                "energy_db": round(self.energy_db, 2),
                "flux": round(self.flux, 4), "proxy": True}


def _k_weighting(n_fft: int) -> np.ndarray:
    """Power weights per rfft bin approximating ITU-R BS.1770 K-weighting: a
    2nd-order high-pass near 38 Hz and a +4 dB shelf above ~1.7 kHz.

    Loudness as the standard measures it, rather than raw RMS, which the kick
    and the bass own: on a real mix the EQ and filter moves the loop makes
    barely register in RMS while they are plainly audible.
    """
    f = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)
    hp = f ** 4 / (f ** 4 + 38.0 ** 4)
    x = (f / 1681.0) ** 2
    shelf = (1.0 + x * 10 ** (4.0 / 10.0)) / (1.0 + x)
    return hp * shelf


class OutputProxies:
    """Per-bar K-weighted loudness and flux from a stream of audio blocks.

    ``push`` takes audio with the engine frame its first sample was played at,
    and a function placing an engine frame on the playing track's bar grid.
    Bars come out once complete.
    """

    def __init__(self, keep_bars: int = 64) -> None:
        self._window = np.hanning(FFT)
        norm = np.sum(self._window ** 2) * FFT / 2.0
        self._kweight = _k_weighting(FFT) / norm
        f = np.fft.rfftfreq(FFT, 1.0 / SAMPLE_RATE)
        self._presence = ((f >= 2000.0) & (f < 16000.0)).astype(np.float64) / norm
        self._tail = np.zeros(FFT - HOP)
        self._pending = np.zeros(0)
        self._pending_start = 0
        self._next_frame: int | None = None
        self._prev_mag: np.ndarray | None = None
        self._bar: int | None = None
        self._bar_ms: list[float] = []
        self._bar_presence: list[float] = []
        self._bar_flux: list[float] = []
        self.readings: deque[BarReading] = deque(maxlen=keep_bars)

    def reset(self) -> None:
        """Forget the bar in progress (a jump, a new track, a gap)."""
        self._bar = None
        self._bar_ms, self._bar_presence, self._bar_flux = [], [], []

    def push(
        self, block: np.ndarray, start_frame: int,
        bar_of: Callable[[float], float | None],
    ) -> list[BarReading]:
        mono = block.mean(axis=1) if block.ndim == 2 else block
        mono = np.asarray(mono, dtype=np.float64)
        if self._next_frame is not None and start_frame != self._next_frame:
            # A gap or overlap in what the tap delivered: start clean.
            self._pending = np.zeros(0)
            self._tail = np.zeros(FFT - HOP)
            self._prev_mag = None
            self.reset()
        if self._pending.size == 0:
            self._pending_start = start_frame
        self._next_frame = start_frame + mono.size
        data = np.concatenate([self._pending, mono])
        out: list[BarReading] = []
        at = 0
        while data.size - at >= HOP:
            hop = data[at:at + HOP]
            frame = np.concatenate([self._tail, hop])
            self._tail = frame[HOP:]
            spectrum = np.abs(np.fft.rfft(frame * self._window))
            sq = spectrum * spectrum
            power = float(np.dot(sq, self._kweight))
            presence = float(np.dot(sq, self._presence))
            mag = np.log(spectrum + 1e-6)
            flux = (0.0 if self._prev_mag is None
                    else float(np.maximum(mag - self._prev_mag, 0.0).mean()))
            self._prev_mag = mag
            centre = self._pending_start + at + HOP / 2.0
            done = self._add_hop(power, presence, flux, bar_of(centre))
            if done is not None:
                out.append(done)
            at += HOP
        self._pending = data[at:]
        self._pending_start += at
        return out

    def _add_hop(self, ms: float, presence: float, flux: float,
                 bar: float | None) -> BarReading | None:
        if bar is None or not math.isfinite(bar):
            self.reset()
            return None
        index = int(math.floor(bar))
        done = None
        if self._bar is not None and index != self._bar:
            if index == self._bar + 1 and len(self._bar_ms) >= MIN_HOPS_PER_BAR:
                done = self._finish()
            self.reset()
        if self._bar is None:
            self._bar = index
        self._bar_ms.append(ms)
        self._bar_presence.append(presence)
        self._bar_flux.append(flux)
        return done

    def _finish(self) -> BarReading:
        ms = np.asarray(self._bar_ms)
        hop_db = tuple(float(10.0 * math.log10(v + 1e-12)) for v in ms)
        reading = BarReading(
            bar=int(self._bar), loudness_db=float(10.0 * math.log10(ms.mean() + 1e-12)),
            presence_db=float(10.0 * math.log10(np.mean(self._bar_presence) + 1e-12)),
            flux=float(np.mean(self._bar_flux)), hops=int(ms.size), hop_db=hop_db,
        )
        self.readings.append(reading)
        return reading

    def summary(self, bars: int = SUMMARY_BARS) -> dict:
        """The slower proxies, over the last ``bars`` bars. Labelled."""
        recent = list(self.readings)[-bars:]
        if not recent:
            return {"proxy": True, "bars": 0}
        levels = np.array([r.loudness_db for r in recent])
        hops = np.concatenate([np.asarray(r.hop_db) for r in recent])
        return {
            "proxy": True,
            "bars": len(recent),
            "loudness_db": round(float(levels[-1]), 2),
            "presence_db": round(float(recent[-1].presence_db), 2),
            "energy_variance": round(float(levels.var()), 3),
            "spectral_flux": round(float(np.mean([r.flux for r in recent])), 4),
            "loudness_dynamics_db": round(
                float(np.percentile(hops, 90) - np.percentile(hops, 10)), 2),
        }


class RoomMic:
    """An optional microphone's view of the room. Off by default.

    Blocks arrive through :meth:`feed` from whatever captures them; nothing in
    this program opens an input device yet. What it measures is reported and
    logged beside the master proxies, never acted on.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = config.ROOM_MIC_ENABLED if enabled is None else bool(enabled)
        self.proxies = OutputProxies()
        self._queue: deque[tuple[np.ndarray, int]] = deque(maxlen=256)

    def feed(self, block: np.ndarray, start_frame: int) -> None:
        """Hand over one captured block, stamped with the engine frame it was
        heard at. Any thread; cheap. Ignored while disabled."""
        if self.enabled:
            self._queue.append((np.array(block, copy=True), int(start_frame)))

    def drain(self, bar_of: Callable[[float], float | None]) -> list[BarReading]:
        out: list[BarReading] = []
        while self._queue:
            block, start = self._queue.popleft()
            out.extend(self.proxies.push(block, start, bar_of))
        return out

    def signal_quality(self, master: OutputProxies, bars: int = QUALITY_BARS) -> float | None:
        """How well the mic's bar energy follows the master's, -1..1.

        A mic drowned in crowd noise or badly placed reads low; one that hears
        the system clearly reads near 1. None until enough bars overlap.
        """
        mine = {r.bar: r.energy_db for r in list(self.proxies.readings)[-bars:]}
        pairs = [(r.energy_db, mine[r.bar]) for r in list(master.readings)[-bars:]
                 if r.bar in mine]
        if len(pairs) < 4:
            return None
        a, b = np.array(pairs).T
        if a.std() < 1e-9 or b.std() < 1e-9:
            return None
        return round(float(np.corrcoef(a, b)[0, 1]), 3)


@dataclass(frozen=True)
class Verdict:
    """What the loop made of one bar."""

    status: str  # settling, ok, watch, drop, spike, saturated
    bar: int
    deviation_db: float = 0.0
    #: The new correction level, when this bar issued one.
    direction: float | None = None
    reason: str = ""


class EnergyLoop:
    """Is this working? A heuristic, bounded, deliberately slow controller.

    The residual is measured bar loudness minus the track's own analysed bar
    energy; its level over the first :attr:`SETTLE_BARS` after a (re)baseline
    is what "working" means. A residual :attr:`TRIGGER_DB` off that for
    :attr:`CONFIRM_BARS` bars in a row is a drop or a spike. The answer is an
    integral step on the correction level (-1 calmer .. +1 harder), bounded per
    step and in total, and then a :attr:`REFRACTORY_BARS` wait so the loop sees
    what its last move did before it moves again. Between
    :attr:`RELEASE_DB` and the trigger it only watches: hysteresis, so a level
    sitting on the edge does not chatter.

    Why it does not oscillate: each step is sized below the inverse of the
    correction's authority in that direction, so one step does not overshoot
    the deviation it answers, and nothing moves until the last move has landed
    and been measured. The authority is lopsided -- measured through the engine
    on a synthetic mix, without the loop's level trim: +1 harder lifts the
    energy proxy ~1.9 dB, -0.25 calmer already takes ~2.6 dB off, because the
    low-pass bites at once -- so the first step's gain is too
    (:attr:`GAIN_UP_PER_DB` answering a drop, :attr:`GAIN_DOWN_PER_DB` a
    spike), and later steps in an episode are sized from what the last one
    actually did. A correction that controls drifted from where the loop put
    them is answered by putting them back, which cannot overshoot.
    """

    TRIGGER_DB: float = 2.5
    RELEASE_DB: float = 1.5
    #: An episode ends when the deviation, averaged since the last move
    #: landed, is inside this. Tighter than the release, so a correction that
    #: stops on the release edge -- where one bar reads 1.4 and the next 1.6 --
    #: is finished rather than left flickering.
    SETTLED_DB: float = 1.0
    CONFIRM_BARS: int = 2
    SETTLE_BARS: int = 4
    REFRACTORY_BARS: int = 10
    GAIN_UP_PER_DB: float = 0.25
    GAIN_DOWN_PER_DB: float = 0.08
    #: After a step has landed, the next one is sized from what that step
    #: actually did (dB moved per unit of correction), clamped to this range
    #: and damped by :attr:`SECANT_DAMPING`: the correction's authority varies
    #: tenfold between records, and a fixed gain either creeps or overshoots.
    AUTHORITY_DB: tuple[float, float] = (1.0, 15.0)
    SECANT_DAMPING: float = 0.8
    MAX_STEP: float = 0.5
    MAX_DIRECTION: float = 1.0

    def __init__(self) -> None:
        self.reset()

    def reset(self, direction: float = 0.0) -> None:
        """Re-learn what "working" is from the next bars; keep ``direction``."""
        self.direction = float(direction)
        self.baseline: float | None = None
        self._settle: list[float] = []
        self._off = 0
        self._sign = 0
        self._last_move: int | None = None
        #: Inside an episode -- a confirmed drop or spike not yet back inside
        #: :attr:`RELEASE_DB` -- the loop finishes the job without waiting for
        #: a fresh confirmation after each landed move.
        self._episode = False
        #: Deviations since the last move landed, for the episode's averages.
        self._devs: list[float] = []
        #: The last step: (deviation it answered, how far it moved the level).
        self._last_step: tuple[float, float] | None = None
        #: Bars assessed since the reset: the loop's own clock, because a
        #: track's bar numbers jump with a hot cue, a loop or a new track.
        self._count = 0

    def assess(self, residual_db: float, bar: int, drifted: bool = False) -> Verdict:
        """One bar's residual in, a verdict out. The learned-model seam.

        ``drifted`` says the deck's energy controls are no longer where the
        loop last put them (a stuck filter, an EQ something else moved). The
        first answer is then to put them back -- the smallest move there is,
        and one that cannot overshoot -- before stepping the level at all.
        """
        self._count += 1
        if self.baseline is None:
            self._settle.append(float(residual_db))
            if len(self._settle) >= self.SETTLE_BARS:
                self.baseline = float(np.median(self._settle))
            return Verdict("settling", bar)
        dev = float(residual_db) - self.baseline
        status = "ok" if abs(dev) <= self.RELEASE_DB else ("spike" if dev > 0 else "drop")
        waiting = (self._last_move is not None
                   and self._count - self._last_move < self.REFRACTORY_BARS)
        if self._episode and not waiting:
            # What the last move left, measured once it has landed.
            self._devs.append(dev)
            if len(self._devs) >= 2 and abs(float(np.mean(self._devs[-4:]))) <= self.SETTLED_DB:
                self._episode = False
                self._off, self._sign = 0, 0
                return Verdict(status, bar, dev, reason="settled")
        if self._episode:
            if waiting:
                return Verdict(status, bar, dev, reason="waiting on the last correction")
            if len(self._devs) < 2:
                return Verdict(status, bar, dev, reason="measuring the last correction")
            # Judged on its average, not on one bar's reading.
            dev = float(np.mean(self._devs[-4:]))
        else:
            if abs(dev) <= self.RELEASE_DB:
                self._off, self._sign = 0, 0
                return Verdict("ok", bar, dev)
            sign = 1 if dev > 0 else -1
            if abs(dev) >= self.TRIGGER_DB:
                self._off = self._off + 1 if sign == self._sign else 1
                self._sign = sign
            if self._off < self.CONFIRM_BARS:
                return Verdict("watch", bar, dev,
                               reason=f"{status}, bar {self._off}" if self._off else "")
            if waiting:
                return Verdict(status, bar, dev, reason="waiting on the last correction")
        how = f"{'spike' if dev > 0 else 'drop'} {dev:+.1f} dB"
        if drifted:
            new, how = self.direction, how + ", controls drifted: put back"
        else:
            gain = self.GAIN_UP_PER_DB if dev < 0 else self.GAIN_DOWN_PER_DB
            if self._episode and self._last_step is not None:
                before, moved = self._last_step
                authority = (dev - before) / moved
                lo, hi = self.AUTHORITY_DB
                if authority > 0:  # the step moved the level the way it meant to
                    gain = self.SECANT_DAMPING / min(max(authority, lo), hi)
            step = max(-self.MAX_STEP, min(self.MAX_STEP, -gain * dev))
            new = max(-self.MAX_DIRECTION, min(self.MAX_DIRECTION, self.direction + step))
            if abs(new - self.direction) < 0.02:
                return Verdict("saturated", bar, dev,
                               reason=f"{how} past the correction's reach")
        moved = round(new, 3) - self.direction
        self._last_step = (dev, moved) if abs(moved) > 1e-9 else None
        self.direction = round(new, 3)
        self._last_move = self._count
        self._episode = True
        self._devs = []
        self._off, self._sign = 0, 0
        return Verdict(status, bar, dev, direction=self.direction, reason=how)


def source_energy(audio: np.ndarray, beat0_s: float, beat_period_s: float,
                  downbeat_offset_beats: float) -> dict[int, float]:
    """Bar -> energy proxy (dB) of a track's own audio, on the given grid."""
    if beat_period_s <= 0.0 or audio is None or len(audio) == 0:
        return {}
    frames_per_bar = SAMPLE_RATE * 4.0 * beat_period_s
    first = (beat0_s + downbeat_offset_beats * beat_period_s) * SAMPLE_RATE
    proxies = OutputProxies(keep_bars=100000)
    chunk = 1 << 18
    for at in range(0, len(audio), chunk):
        proxies.push(audio[at:at + chunk], at, lambda f: (f - first) / frames_per_bar)
    return {r.bar: r.energy_db for r in proxies.readings}


class RoomMonitor:
    """Tap, proxies, mic and loop, polled from a control thread."""

    def __init__(self, engine, mic_enabled: bool | None = None) -> None:
        self.engine = engine
        self.tap, self._tap_start = engine.attach_output_tap()
        self._buf = np.zeros((self.tap._frames, 2), dtype=np.float32)
        self.master = OutputProxies()
        self.mic = RoomMic(mic_enabled)
        self.loop = EnergyLoop()
        self._expected: dict[int, dict[int, float]] = {}
        self._paused = False
        #: The operator touched the live deck's EQ or filter by hand: those
        #: knobs are theirs until the next hand-over, and the loop only watches.
        self.operator_owns = False
        #: Where the loop expects the live deck's energy controls to be, as
        #: (low, mid, high, filter); set by whoever last wrote them through
        #: the loop's own path. None: not known, never judged as drifted.
        self.controls: tuple[float, float, float, float] | None = (1.0, 1.0, 1.0, 0.0)

    def handover(self) -> None:
        """A new track is the mix and both decks are neutral: start again."""
        self.loop.reset(0.0)
        self.master.reset()
        self.operator_owns = False
        self.controls = (1.0, 1.0, 1.0, 0.0)

    def operator(self, direction: float) -> None:
        """The operator moved the energy: their level is the new "working"."""
        self.loop.reset(direction)

    def operator_touched(self) -> None:
        """A hand on the live deck's EQ or filter: theirs until the hand-over."""
        self.operator_owns = True
        self.controls = None
        self.loop.reset(self.loop.direction)

    def drifted(self, deck) -> bool:
        """Whether the deck's energy controls are off where the loop put them."""
        if self.controls is None or deck is None:
            return False
        now = (deck.eq_low.target, deck.eq_mid.target, deck.eq_high.target,
               deck.filter_pos.target)
        return any(abs(a - b) > 0.02 for a, b in zip(now, self.controls))

    def _expected_db(self, deck, bar: int) -> float | None:
        """The energy proxy of the playing track's own audio at ``bar``.

        Measured with the same code as the output, on the deck's own decoded
        buffer and the deck's own grid, so what is left after subtracting it
        is what the mix did -- EQ, filter, faders, the limiter -- and never the
        arrangement. Once per track, on this (control) thread: ~10k FFTs for a
        four-minute track.
        """
        track = deck.track
        if track is None:
            return None
        key = id(track)
        per_bar = self._expected.get(key)
        if per_bar is None:
            per_bar = source_energy(track.audio, deck.grid_beat0_s,
                                    deck.grid_beat_period_s, deck.grid_downbeat_offset_beats)
            self._expected = {key: per_bar}
        return per_bar.get(bar)

    def poll(self, live: str, paused: bool) -> list[tuple[BarReading, Verdict | None]]:
        """Read what the tap holds, measure it, and judge each finished bar."""
        took = self.tap.read_into(self._buf, self._buf.shape[0])
        snap = self.engine.features.latest()
        if took <= 0 or snap is None:
            return []
        start = self._tap_start + self.tap.read - took
        deck = snap.deck(live)
        track = self.engine.deck(live).track
        if (not deck.playing or deck.beat_period_s <= 0.0 or track is None):
            bar_of = lambda f: None  # noqa: E731
        else:
            frames_per_bar = SAMPLE_RATE * 4.0 * deck.beat_period_s / max(deck.rate, 1e-6)
            bar_of = lambda f: deck.bar + (f - snap.frames_played) / frames_per_bar  # noqa: E731
        readings = self.master.push(self._buf[:took], start, bar_of)
        if self.mic.enabled:
            self.mic.drain(bar_of)
        if paused:
            self._paused = True
            return [(r, None) for r in readings]
        if self._paused:
            # Back from a blend or a manual hold: whatever the level is now is
            # the decision someone made, not a fault. Learn it again.
            self._paused = False
            self.loop.reset(self.loop.direction)
        out = []
        drifted = self.drifted(self.engine.deck(live))
        for r in readings:
            expected = self._expected_db(self.engine.deck(live), r.bar)
            if expected is None:
                out.append((r, None))
                continue
            out.append((r, self.loop.assess(r.energy_db - expected, r.bar, drifted)))
        return out

    def state(self) -> dict:
        """For the log and the UI."""
        out = {"master": self.master.summary(), "direction": self.loop.direction,
               "baseline_db": None if self.loop.baseline is None
               else round(self.loop.baseline, 2),
               "heuristic": True, "mic": "off"}
        if self.mic.enabled:
            out["mic"] = "on"
            out["room"] = self.mic.proxies.summary()
            out["signal_quality"] = self.mic.signal_quality(self.master)
        return out
