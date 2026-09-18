"""Offline transition renderer: run a transition headless, faster than realtime.

THREADING CONTEXT: main thread only. This module opens no audio device and
starts no background threads. It drives :meth:`djai.engine.Engine.callback`
directly in a loop, which is the *same* code the audio thread runs -- the real
:mod:`djai.deck` resampler and EQ, the real :mod:`djai.engine` mixer, the real
:mod:`djai.transition` shape. Nothing here reimplements the signal path; if a
render sounds wrong, the live engine is wrong in the same way.

The point is diagnosis. A transition that misbehaves in a club is hard to
inspect; the same transition rendered to a WAV plus two CSVs can be read
directly. Outputs, for ``render_transition(a, b, "out")``:

* ``out.wav``              the mix
* ``out_envelope.csv.gz``  per audio block: what the automation actually applied
* ``out_grid.csv.gz``      both decks' beats and downbeats in output time

The two CSVs are gzipped: they compress several-fold and every CSV tool and
``pandas.read_csv`` reads ``.csv.gz`` directly.

Because there is no device, this runs as fast as numpy will go -- roughly
40-80x realtime for a 32-bar transition.
"""

from __future__ import annotations

import csv
import gzip
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from djai import config, phrase, transition
from djai.analysis import TrackAnalysis
from djai.commands import LoadTrack, StartTransition
from djai.deck import (
    CHANNELS,
    SAMPLE_RATE,
    Deck,
    LoadedTrack,
    StretchError,
    load_track,
    time_stretch,
)
from djai.engine import Engine
from djai.scheduler import Scheduler

log = logging.getLogger(__name__)

#: Bars of the outgoing deck rendered before the transition starts, so the
#: "before" state is visible in the envelope.
PRE_ROLL_BARS: float = 4.0

#: Bars rendered after the transition completes, to catch a bad hand-over.
POST_ROLL_BARS: float = 4.0

BEATS_PER_BAR: int = 4


@dataclass
class RenderResult:
    """Everything a diagnosis needs, in memory as well as on disk."""

    wav_path: Path
    envelope_path: Path
    grid_path: Path
    placement_path: Path
    #: The single placement row, keyed by :data:`PLACEMENT_COLUMNS`.
    placement: dict
    mix: np.ndarray
    #: One row per block: see :data:`ENVELOPE_COLUMNS`.
    envelope: list[dict]
    #: One row per beat of each deck, in output time.
    grid: list[dict]
    transition_start_frame: int
    transition_frames: int
    bars: float
    bpm_a: float
    bpm_b: float
    rate_b: float
    blocksize: int
    #: Whether deck B played a stretched copy (key lock on, rate outside the
    #: deadband, stretch succeeded) rather than resampling.
    b_stretched: bool = False

    @property
    def duration_s(self) -> float:
        return self.mix.shape[0] / SAMPLE_RATE


ENVELOPE_COLUMNS = [
    "block",
    "time_s",
    "in_transition",
    "bar",
    "a_gain",
    "a_low",
    "a_mid",
    "a_high",
    "b_gain",
    "b_low",
    "b_mid",
    "b_high",
    "a_bar_pos",
    "b_bar_pos",
    "peak",
    "rms",
    # Deck A's own motion, which the gain and EQ columns cannot show: a loop
    # and a backspin leave every gain flat while the record does something
    # violent. Appended rather than inserted, so a reader that indexes the
    # columns by position is unaffected.
    "a_rate",
    "a_loop_beats",
    "a_loop_start_bar",
    # Each deck's filter knob as applied, -1 (low-pass) .. 0 (open) .. 1
    # (high-pass). Appended for the same reason as the three above.
    "a_filter",
    "b_filter",
    # Effects (Phase 3): the reverb send and the noise riser's level, as the
    # engine applied them this block.
    "reverb_send",
    "riser_gain",
]

GRID_COLUMNS = ["deck", "kind", "beat_index", "bar_index", "time_s"]

#: One row per rendered transition. Everything here is in *track* time, not
#: render time: the question this answers is "where in the outgoing track does
#: the hand-off happen", which the render window would otherwise obscure.
PLACEMENT_COLUMNS = [
    "out_title",
    "out_duration_s",
    "start_s",
    "end_s",
    "start_frac",
    "end_frac",
    "start_bar",
    "end_bar",
    "transition_bars",
    "in_title",
    "in_duration_s",
    "in_entry_s",
    "in_entry_bar",
    "in_entry_frac",
    "bpm_a",
    "bpm_b",
    # Anchoring evidence: is bar 0 at sample 0, or at the first downbeat?
    "a_first_downbeat_s",
    "a_bar_at_sample0",
    "a_bar_at_first_downbeat",
    "b_first_downbeat_s",
    "b_bar_at_first_downbeat",
]


def plan_auto_placement(
    track_a: TrackAnalysis, track_b: TrackAnalysis, transition_bars: float
) -> phrase.TransitionPlan:
    """Placement as the automatic path computes it: backwards from mix-out.

    Wraps :func:`djai.phrase.plan_transition` with a probe deck positioned at
    the moment the autopilot would arm, so the renderer measures the shipping
    placement rule rather than a convenient window.
    """
    from djai.cli import AUTOPILOT_LEAD_PHRASES, AUTOPILOT_MARGIN_SECONDS

    seconds = transition_bars * BEATS_PER_BAR * 60.0 / track_a.bpm
    lead_s = seconds * AUTOPILOT_LEAD_PHRASES + AUTOPILOT_MARGIN_SECONDS
    armed_at_s = max(0.0, track_a.duration_s - lead_s)

    deck = Deck("probe")
    deck.track = LoadedTrack(analysis=track_a, audio=np.zeros((2, CHANNELS), np.float32))
    deck.position = armed_at_s * SAMPLE_RATE
    plan = phrase.plan_transition(deck, track_b, transition_bars)
    assert plan is not None
    return plan


def plan_live_placement(
    track_a: TrackAnalysis, transition_bars: float
) -> tuple[int, float]:
    """Where the *current production* path would start a transition in track A.

    Replicates what :mod:`djai.cli`'s autopilot does today: wait until the
    remaining time drops under its lead, then take the next 32-bar phrase
    boundary. Reproduced here so the placement CSV measures the shipping
    behaviour rather than a convenient render window.

    Returns ``(deck_a_frame, armed_at_seconds)``.
    """
    from djai.cli import AUTOPILOT_LEAD_PHRASES, AUTOPILOT_MARGIN_SECONDS

    seconds = transition_bars * BEATS_PER_BAR * 60.0 / track_a.bpm
    lead_s = seconds * AUTOPILOT_LEAD_PHRASES + AUTOPILOT_MARGIN_SECONDS
    armed_at_s = max(0.0, track_a.duration_s - lead_s)

    deck = Deck("probe")
    deck.track = LoadedTrack(analysis=track_a, audio=np.zeros((2, CHANNELS), np.float32))
    boundary = phrase.next_phrase_boundary(deck, after_frame=armed_at_s * SAMPLE_RATE)
    return int(boundary), armed_at_s


def _deck_snapshot(deck) -> tuple[float, float, float, float]:
    """The gains actually applied by the end of the block just rendered.

    ``.cur`` is what the per-block ramp reached, not what was requested, so this
    measures the automation rather than restating the intent.
    """
    return (deck.gain.cur, deck.eq_low.cur, deck.eq_mid.cur, deck.eq_high.cur)


def _beat_rows(
    deck_name: str,
    analysis: TrackAnalysis,
    start_deck_frame: float,
    start_output_frame: int,
    rate: float,
    n_output_frames: int,
    min_output_frame: int = 0,
) -> list[dict]:
    """Every beat of one deck, expressed in output time.

    Assumes the map from deck frames to output frames is linear, which holds
    while a deck plays straight at one rate. It does not hold for deck A once a
    loop or a backspin engages: the grid rows for A are then where its beats
    would have been, not where its looped or reversed audio actually is. The
    envelope's ``a_rate``, ``a_loop_beats`` and ``a_loop_start_bar`` columns
    show exactly when that is the case.
    """
    rows: list[dict] = []
    if rate <= 0:
        return rows
    first_beat = phrase.beat_at_frame(analysis, start_deck_frame)
    last_beat = phrase.beat_at_frame(
        analysis, start_deck_frame + n_output_frames * rate
    )
    downbeat_offset = phrase.downbeat_offset_beats(analysis)

    for k in range(int(np.floor(first_beat)) - 1, int(np.ceil(last_beat)) + 2):
        deck_frame = phrase.frame_at_beat(analysis, float(k))
        output_frame = start_output_frame + (deck_frame - start_deck_frame) / rate
        if output_frame < min_output_frame or output_frame > n_output_frames:
            continue
        is_downbeat = (k - downbeat_offset) % BEATS_PER_BAR == 0
        rows.append(
            {
                "deck": deck_name,
                "kind": "downbeat" if is_downbeat else "beat",
                "beat_index": k,
                "bar_index": round((k - downbeat_offset) / BEATS_PER_BAR, 4),
                "time_s": round(output_frame / SAMPLE_RATE, 6),
            }
        )
    return rows


@dataclass
class PreviewRender:
    """One headless render of a planned transition, in memory only.

    Same engine, same scheduler, same envelope as :func:`render_transition`;
    what differs is that nothing is written to disk, the window is only the
    transition plus a few bars either side, and the caller supplies tracks that
    are already decoded and stretched. In the pre-roll window both of those have
    already happened for the live decks, and they are the expensive part:
    measured on the crate here, decoding is ~400 ms and time-stretching a whole
    track is ~110 s, against ~1.3 s for the block loop over a 24-bar window.
    Re-doing either would put a preview two orders of magnitude over budget.
    """

    mix: np.ndarray
    envelope: list[dict]
    grid: list[dict]
    #: Output frame the transition starts at, and how long it runs.
    transition_start_frame: int
    transition_frames: int
    bars: float
    bpm_a: float
    bpm_b: float
    rate_b: float
    blocksize: int
    #: Deck A's frame at output frame 0, and deck B's at a known output frame,
    #: so a measurement can map any block back to what each deck was playing.
    a_start_deck_frame: float
    b_anchor_output_frame: int
    b_anchor_deck_frame: float
    #: Whether the master limiter pulled the mix down anywhere in the window.
    limiter_engaged: bool

    @property
    def duration_s(self) -> float:
        return self.mix.shape[0] / SAMPLE_RATE


def render_preview(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    loaded_a: LoadedTrack,
    loaded_b: LoadedTrack,
    params,
    transition_at_frame_a: float,
    entry_frame_b: float,
    context_bars: float = 4.0,
    blocksize: int | None = None,
) -> PreviewRender:
    """Render a *designed* transition headlessly, for measurement.

    ``params`` is a :class:`djai.transition.TransitionParams`, so this renders
    the point in the parameter space the model actually asked for rather than
    the nearest named preset. The envelope comes from
    :meth:`Engine.arm_transition_params`, the same call the live path makes.

    Raises nothing it can help: a caller in the pre-roll window must be able to
    treat a failure as "commit the original parameters", and an exception here
    is the signal for that.
    """
    engine = Engine(blocksize=blocksize) if blocksize else Engine()
    block = engine.blocksize

    bars = float(params.length_bars)
    seconds = bars * BEATS_PER_BAR * 60.0 / track_a.bpm
    total_frames = int(round(seconds * SAMPLE_RATE))

    context_frames = int(
        round(context_bars * BEATS_PER_BAR * 60.0 / track_a.bpm * SAMPLE_RATE)
    )

    swap_frame_a = int(round(transition_at_frame_a))
    start_a = max(
        int(round(track_a.first_downbeat * SAMPLE_RATE)), swap_frame_a - context_frames
    )
    engine.submit(
        LoadTrack(deck="a", track=loaded_a, start_frame=start_a, rate=1.0, master=True)
    )
    engine.deck_a.gain.jump(1.0)

    rate_b = track_a.bpm / track_b.bpm if track_b.bpm > 0 else 1.0
    start_b = int(round(entry_frame_b))
    n_total = (swap_frame_a - start_a) + total_frames + context_frames

    scheduler = Scheduler(engine)
    execute_at = swap_frame_a - start_a
    scheduler.submit(
        LoadTrack(
            deck="b", track=loaded_b, start_frame=start_b, rate=rate_b,
            play=True, execute_at=execute_at, origin="preview",
        )
    )
    engine.arm_transition_params(params, total_frames, track_a.bpm)
    scheduler.submit(
        StartTransition(
            from_deck="a", to_deck="b", total_frames=total_frames,
            execute_at=execute_at, origin="preview",
        )
    )
    engine.deck_b.gain.jump(0.0)

    buf = np.zeros((block, CHANNELS), dtype=np.float32)
    chunks: list[np.ndarray] = []
    envelope: list[dict] = []
    started = False
    start_block = 0
    b_anchor_output_frame = execute_at
    b_anchor_deck_frame = float(start_b)
    limiter_engaged = False

    n_blocks = int(np.ceil(n_total / block))
    for i in range(n_blocks):
        scheduler.tick(engine.frames_played)
        engine.callback(buf, block, None, None)
        if not started and engine.transition_active:
            started = True
            start_block = i
            b_anchor_output_frame = i * block + block
            b_anchor_deck_frame = engine.deck_b.position
        if engine.limiter_gain < 0.999:
            limiter_engaged = True
        chunks.append(buf.copy())

        a_gain, a_low, a_mid, a_high = _deck_snapshot(engine.deck_a)
        b_gain, b_low, b_mid, b_high = _deck_snapshot(engine.deck_b)
        bar = (i - start_block) * block / SAMPLE_RATE * track_a.bpm / 60.0 / BEATS_PER_BAR
        envelope.append(
            {
                "block": i,
                "time_s": round(i * block / SAMPLE_RATE, 6),
                "in_transition": 1 if (started and 0.0 <= bar <= bars) else 0,
                "bar": round(bar if started else -0.0, 5),
                "a_gain": a_gain, "a_low": a_low, "a_mid": a_mid, "a_high": a_high,
                "b_gain": b_gain, "b_low": b_low, "b_mid": b_mid, "b_high": b_high,
                # Where each deck's playhead actually is, in FRAMES of its own
                # track, which is what a per-deck measurement has to index by.
                "a_frame": float(engine.deck_a.position),
                "b_frame": (
                    float(engine.deck_b.position)
                    if engine.deck_b.track is not None else 0.0
                ),
                "a_filter": round(float(engine.deck_a.filter_pos.cur), 4),
                "b_filter": round(float(engine.deck_b.filter_pos.cur), 4),
            }
        )

    mix = np.concatenate(chunks)[:n_total]
    grid = _beat_rows("a", track_a, float(start_a), 0, 1.0, n_total)
    grid += _beat_rows(
        "b", track_b, b_anchor_deck_frame, b_anchor_output_frame, rate_b,
        n_total, min_output_frame=execute_at,
    )
    grid.sort(key=lambda r: (r["time_s"], r["deck"]))

    return PreviewRender(
        mix=mix,
        envelope=envelope,
        grid=grid,
        transition_start_frame=start_block * block,
        transition_frames=total_frames,
        bars=bars,
        bpm_a=track_a.bpm,
        bpm_b=track_b.bpm,
        rate_b=rate_b,
        blocksize=block,
        a_start_deck_frame=float(start_a),
        b_anchor_output_frame=b_anchor_output_frame,
        b_anchor_deck_frame=b_anchor_deck_frame,
        limiter_engaged=limiter_engaged,
    )


def render_transition(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    out_path: str | Path,
    blocksize: int | None = None,
    transition_bars: float | None = None,
    transition_at_frame_a: float | None = None,
    entry_frame_b: float | None = None,
    style: str = "bass_swap",
    key_lock: bool | None = None,
) -> RenderResult:
    """Render one A->B transition offline and write the WAV plus both CSVs.

    ``track_a`` and ``track_b`` are cached analyses; their audio is decoded
    here. Deck B is beatmatched to deck A and cued so its bar 0 lands on the
    downbeat where the transition begins -- the same alignment
    :func:`djai.cli.Session.arm_transition` performs live.

    ``key_lock`` is deck B's key lock; None takes the live default
    (``KEY_LOCK_DEFAULT``, when time-stretch is enabled). Offline there is no
    pre-roll for the stretch worker to use, so with key lock on the copy is
    made here before the first block. A failed stretch resamples, as live.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loaded_a: LoadedTrack = load_track(track_a)
    loaded_b: LoadedTrack = load_track(track_b)

    engine = Engine(blocksize=blocksize) if blocksize else Engine()
    block = engine.blocksize

    if style not in transition.STYLES:
        raise ValueError(
            f"unknown transition style {style!r}; "
            f"choose from {', '.join(transition.STYLES)}"
        )

    # Each style has its own natural length. An explicit `transition_bars`
    # still wins, so a render can be forced to any length for comparison.
    if transition_bars is not None:
        bars = transition_bars
    elif style == "bass_swap":
        bars = transition.TRANSITION_BARS
    elif style == "cut":
        # A cut has no length. One bar is used purely to place it and to give
        # the CSV a window either side of the switch; the transition itself is
        # still a single block.
        bars = 1.0
    else:
        bars = transition.style_bars(style)

    seconds = bars * BEATS_PER_BAR * 60.0 / track_a.bpm
    total_frames = int(round(seconds * SAMPLE_RATE))
    if style == "cut":
        total_frames = block

    pre_roll_frames = int(
        round(PRE_ROLL_BARS * BEATS_PER_BAR * 60.0 / track_a.bpm * SAMPLE_RATE)
    )

    # Where in track A the hand-off happens. Defaults to PRE_ROLL_BARS in,
    # which keeps short shape-inspection renders cheap; pass an explicit frame
    # to render the transition where it would really land in the track.
    if transition_at_frame_a is None:
        transition_at_frame_a = phrase.frame_at_bar(track_a, PRE_ROLL_BARS)
    swap_frame_a = int(round(transition_at_frame_a))

    # --- deck A: seek so the transition lands PRE_ROLL_BARS into the render ---
    start_a = max(
        int(round(track_a.first_downbeat * SAMPLE_RATE)), swap_frame_a - pre_roll_frames
    )
    engine.submit(
        LoadTrack(deck="a", track=loaded_a, start_frame=start_a, rate=1.0, master=True)
    )
    engine.deck_a.gain.jump(1.0)
    post_roll_frames = int(
        round(POST_ROLL_BARS * BEATS_PER_BAR * 60.0 / track_a.bpm * SAMPLE_RATE)
    )
    n_total = pre_roll_frames + total_frames + post_roll_frames

    rate_b = track_a.bpm / track_b.bpm if track_b.bpm > 0 else 1.0

    if key_lock is None:
        key_lock = config.KEY_LOCK_DEFAULT and config.TIME_STRETCH_ENABLED
    engine.deck_b.key_lock = bool(key_lock)
    if key_lock and (
        config.STRETCH_DEADBAND
        <= abs(rate_b - 1.0)
        <= config.MAX_STRETCH_RATIO + 1e-9
    ):
        try:
            loaded_b = time_stretch(loaded_b, rate_b)
        except StretchError as exc:
            log.warning("render: stretch failed (%s); deck B resamples", exc)

    if entry_frame_b is None:
        entry_frame_b = track_b.first_downbeat * SAMPLE_RATE
    start_b = int(round(entry_frame_b))

    # Schedule through the real Scheduler at the exact engine frame of deck A's
    # PRE_ROLL_BARS downbeat, rather than firing on whatever block boundary
    # happens to come next. This matters: it is the live path (the engine
    # compensates a late release at the cue point), and quantising the cue to a
    # 2048-frame block instead would plant a fixed 0-46 ms flam that looks
    # exactly like a beatmatching fault.
    scheduler = Scheduler(engine)
    execute_at = swap_frame_a - start_a
    scheduler.submit(
        LoadTrack(
            deck="b",
            track=loaded_b,
            start_frame=start_b,
            rate=rate_b,
            play=True,
            execute_at=execute_at,
            origin="render",
        )
    )
    # Build the envelope for this style before the command is queued, exactly
    # as the live session does. The render then exercises the same path.
    engine.arm_transition_plan(style, total_frames, track_a.bpm)
    scheduler.submit(
        StartTransition(
            from_deck="a",
            to_deck="b",
            total_frames=total_frames,
            execute_at=execute_at,
            origin="render",
        )
    )
    engine.deck_b.gain.jump(0.0)

    buf = np.zeros((block, CHANNELS), dtype=np.float32)
    chunks: list[np.ndarray] = []
    envelope: list[dict] = []

    started = False
    start_block = 0
    a_start_deck_frame = float(start_a)
    #: Deck B's real position at a known output frame, captured once it is
    #: actually running, so the grid reflects where it landed rather than where
    #: it was asked to land.
    b_anchor_output_frame = execute_at
    b_anchor_deck_frame = float(start_b)

    n_blocks = int(np.ceil(n_total / block))
    for i in range(n_blocks):
        frames_done = i * block

        # Exactly what the scheduler thread does, driven deterministically.
        scheduler.tick(engine.frames_played)
        engine.callback(buf, block, None, None)

        # `transition_active` is the usual signal, but a `cut` begins and ends
        # inside one callback and is already over by the time it is checked.
        #
        # The fallback is deliberately narrow. `frames_played` advances at the
        # END of a callback, so a bare "have we reached execute_at" test goes
        # true one block before the scheduler releases anything -- which shifts
        # every bar number in the CSV by a block and fails the phase 1 criteria
        # it is supposed to leave alone. Requiring the transition's whole
        # length to have elapsed makes it true exactly from the release block.
        cut_done = (
            total_frames <= block
            and engine.frames_played >= execute_at + total_frames
        )
        if not started and (engine.transition_active or cut_done):
            started = True
            start_block = i
            b_anchor_output_frame = frames_done + block
            b_anchor_deck_frame = engine.deck_b.position
        chunks.append(buf.copy())

        a_gain, a_low, a_mid, a_high = _deck_snapshot(engine.deck_a)
        b_gain, b_low, b_mid, b_high = _deck_snapshot(engine.deck_b)
        bar = (i - start_block) * block / SAMPLE_RATE * track_a.bpm / 60.0 / BEATS_PER_BAR
        # Stated explicitly rather than left to be inferred from `bar`: pre-roll
        # rows also sit at bar 0, and a reader filtering on `bar >= 0` silently
        # scoops them up along with the post-roll.
        in_transition = started and 0.0 <= bar <= bars
        # What the deck is actually doing at the end of this block, read back
        # from the deck rather than from the envelope that asked for it.
        loop_start, loop_len = engine.deck_a.loop_region
        frames_per_beat_a = 60.0 / track_a.bpm * SAMPLE_RATE if track_a.bpm > 0 else 0.0
        looping = loop_len > 0.0 and frames_per_beat_a > 0.0
        envelope.append(
            {
                "block": i,
                "time_s": round(frames_done / SAMPLE_RATE, 6),
                "in_transition": 1 if in_transition else 0,
                "bar": round(bar if started else -0.0, 5),
                "a_gain": round(a_gain, 6),
                "a_low": round(a_low, 6),
                "a_mid": round(a_mid, 6),
                "a_high": round(a_high, 6),
                "b_gain": round(b_gain, 6),
                "b_low": round(b_low, 6),
                "b_mid": round(b_mid, 6),
                "b_high": round(b_high, 6),
                "a_bar_pos": round(
                    phrase.bar_at_frame(track_a, engine.deck_a.position), 5
                ),
                "b_bar_pos": (
                    round(phrase.bar_at_frame(track_b, engine.deck_b.position), 5)
                    if engine.deck_b.track is not None
                    else 0.0
                ),
                "peak": round(float(np.max(np.abs(buf))), 6),
                "rms": round(float(np.sqrt(np.mean(np.square(buf, dtype=np.float64)))), 6),
                "a_rate": round(float(engine.deck_a.rate), 6),
                "a_loop_beats": (
                    round(loop_len / frames_per_beat_a, 4) if looping else 0.0
                ),
                # Empty rather than 0 when there is no loop: bar 0 is a real
                # position, and a blank reads as missing in any CSV tool.
                "a_loop_start_bar": (
                    round(phrase.bar_at_frame(track_a, loop_start), 4) if looping else ""
                ),
                "a_filter": round(float(engine.deck_a.filter_pos.cur), 4),
                "b_filter": round(float(engine.deck_b.filter_pos.cur), 4),
                "reverb_send": round(float(engine._rv_send), 4),
                "riser_gain": round(float(engine._riser_prev), 4),
            }
        )

    mix = np.concatenate(chunks)[:n_total]

    grid = _beat_rows("a", track_a, a_start_deck_frame, 0, 1.0, n_total)
    grid += _beat_rows(
        "b",
        track_b,
        b_anchor_deck_frame,
        b_anchor_output_frame,
        rate_b,
        n_total,
        min_output_frame=execute_at,
    )
    grid.sort(key=lambda r: (r["time_s"], r["deck"]))

    # --- placement, in track time ---
    start_s = swap_frame_a / SAMPLE_RATE
    end_s = (swap_frame_a + total_frames) / SAMPLE_RATE
    dur_a = track_a.duration_s or 1e-9
    dur_b = track_b.duration_s or 1e-9
    placement = {
        "out_title": track_a.title,
        "out_duration_s": round(track_a.duration_s, 3),
        "start_s": round(start_s, 3),
        "end_s": round(end_s, 3),
        "start_frac": round(start_s / dur_a, 4),
        "end_frac": round(end_s / dur_a, 4),
        "start_bar": round(phrase.bar_at_frame(track_a, swap_frame_a), 3),
        "end_bar": round(phrase.bar_at_frame(track_a, swap_frame_a + total_frames), 3),
        "transition_bars": bars,
        "in_title": track_b.title,
        "in_duration_s": round(track_b.duration_s, 3),
        "in_entry_s": round(start_b / SAMPLE_RATE, 3),
        "in_entry_bar": round(phrase.bar_at_frame(track_b, start_b), 3),
        "in_entry_frac": round(start_b / SAMPLE_RATE / dur_b, 4),
        "bpm_a": round(track_a.bpm, 3),
        "bpm_b": round(track_b.bpm, 3),
        "a_first_downbeat_s": round(track_a.first_downbeat, 3),
        "a_bar_at_sample0": round(phrase.bar_at_frame(track_a, 0.0), 3),
        "a_bar_at_first_downbeat": round(
            phrase.bar_at_frame(track_a, track_a.first_downbeat * SAMPLE_RATE), 3
        ),
        "b_first_downbeat_s": round(track_b.first_downbeat, 3),
        "b_bar_at_first_downbeat": round(
            phrase.bar_at_frame(track_b, track_b.first_downbeat * SAMPLE_RATE), 3
        ),
    }

    wav_path = out_path.with_suffix(".wav")
    envelope_path = out_path.parent / f"{out_path.stem}_envelope.csv.gz"
    grid_path = out_path.parent / f"{out_path.stem}_grid.csv.gz"
    placement_path = out_path.parent / f"{out_path.stem}_placement.csv"
    with placement_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PLACEMENT_COLUMNS)
        writer.writeheader()
        writer.writerow(placement)

    sf.write(str(wav_path), mix, SAMPLE_RATE, subtype="PCM_16")
    with gzip.open(envelope_path, "wt", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ENVELOPE_COLUMNS)
        writer.writeheader()
        writer.writerows(envelope)
    with gzip.open(grid_path, "wt", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=GRID_COLUMNS)
        writer.writeheader()
        writer.writerows(grid)

    return RenderResult(
        wav_path=wav_path,
        envelope_path=envelope_path,
        grid_path=grid_path,
        placement_path=placement_path,
        placement=placement,
        mix=mix,
        envelope=envelope,
        grid=grid,
        transition_start_frame=start_block * block,
        transition_frames=total_frames,
        bars=bars,
        bpm_a=track_a.bpm,
        bpm_b=track_b.bpm,
        rate_b=rate_b,
        blocksize=block,
        b_stretched=loaded_b.is_stretched,
    )
