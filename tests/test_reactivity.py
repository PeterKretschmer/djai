"""SPEC §1 (Phase 2.1): priority bands, cancellation, glue, grooves, re-lock.

THREADING CONTEXT: main thread (pytest). The scheduler is ticked by hand and
the engine's callback driven directly, so every timing here is exact.
"""

from __future__ import annotations

from djai.commands import CancelToken, SetEQ, SetPitch
from djai.deck import SAMPLE_RATE
from djai.scheduler import Scheduler
from tests.test_integration import session as _session_fixture

session = _session_fixture

BAR_120 = SAMPLE_RATE * 2  # one bar at 120 BPM


class _FakeEngine:
    def __init__(self) -> None:
        self.frames_played = 0
        self.master_bpm = 120.0
        self.got: list = []

    def submit(self, cmd) -> bool:
        self.got.append(cmd)
        return True


def _eq(at: int, token=None, origin="t") -> SetEQ:
    return SetEQ(deck="a", high=1.2, execute_at=at, origin=origin, token=token)


# --- bands ------------------------------------------------------------------------


def test_commands_are_banded_by_lead_time_in_bars():
    sched = Scheduler(_FakeEngine())
    assert sched.band_of(SetEQ(deck="a")) == "immediate"  # IMMEDIATE sentinel
    assert sched.band_of(_eq(1 * BAR_120)) == "immediate"
    assert sched.band_of(_eq(2 * BAR_120)) == "immediate"
    assert sched.band_of(_eq(4 * BAR_120)) == "short"
    assert sched.band_of(_eq(8 * BAR_120)) == "short"
    assert sched.band_of(_eq(16 * BAR_120)) == "medium"
    assert sched.band_of(_eq(32 * BAR_120)) == "medium"


def test_bands_follow_the_master_tempo():
    eng = _FakeEngine()
    sched = Scheduler(eng)
    at = 3 * BAR_120  # 3 bars at 120 BPM, 6 at 240
    assert sched.band_of(_eq(at)) == "short"
    eng.master_bpm = 60.0  # a bar is twice as long: 1.5 bars away
    assert sched.band_of(_eq(at)) == "immediate"


def test_same_frame_commands_release_most_urgent_band_first():
    eng = _FakeEngine()
    sched = Scheduler(eng)
    due = 20 * BAR_120
    plan = _eq(due, origin="plan")  # submitted 20 bars out: medium
    sched.submit(plan)
    eng.frames_played = due - BAR_120  # a bar before, a correction arrives
    fix = _eq(due, origin="fix")
    sched.submit(fix)
    bands = sched.pending_by_band()
    assert bands["medium"] == [plan] and bands["immediate"] == [fix]
    sched.tick(due)
    assert [c.origin for c in eng.got] == ["fix", "plan"]


# --- cancellation -------------------------------------------------------------------


def test_cancelling_a_token_withdraws_its_whole_group_and_nothing_else():
    eng = _FakeEngine()
    sched = Scheduler(eng)
    tok = CancelToken("ride")
    for i in range(1, 5):
        sched.submit(_eq(i * BAR_120, token=tok))
    keep = _eq(3 * BAR_120)
    sched.submit(keep)
    dropped = sched.cancel(tok)
    assert len(dropped) == 4
    assert sched.pending() == [keep]
    sched.tick(10 * BAR_120)
    assert eng.got == [keep]


def test_a_cancelled_token_stops_later_submissions_and_the_release_race():
    eng = _FakeEngine()
    sched = Scheduler(eng)
    tok = CancelToken()
    tok.cancel()
    sched.submit(_eq(BAR_120, token=tok))
    sched.submit(SetEQ(deck="a", token=tok))  # immediate, too
    assert len(sched) == 0 and eng.got == []

    # A command already queued whose token is cancelled without going through
    # the scheduler (the race the second check exists for) is still held back.
    tok2 = CancelToken()
    sched.submit(_eq(BAR_120, token=tok2))
    tok2.cancel()
    assert sched.tick(BAR_120) == []
    assert eng.got == []


def test_locking_the_clock_mid_glide_withdraws_every_glide_step(session):
    """Regression: the lock used to cancel only `tempo_ride` steps by origin,
    leaving a master glide to keep moving the clock the operator had locked."""
    deck = session.engine.deck(session.engine.master_deck or "a")
    deck.rate = 1.04
    session._schedule_master_glide(deck.name)
    glide = [c for c in session.scheduler.pending() if c.origin == "master_glide"]
    assert glide, "the glide should have queued pitch steps"
    assert {session.scheduler.band_of(c) for c in glide} >= {"medium"}
    session.set_master_tempo_lock(True)
    left = [c for c in session.scheduler.pending() if isinstance(c, SetPitch)]
    assert left == []


def test_a_new_glide_supersedes_one_in_flight(session):
    deck = session.engine.deck(session.engine.master_deck or "a")
    deck.rate = 1.04
    session._schedule_master_glide(deck.name)
    first = len(session._ride_steps)
    deck.rate = 1.02
    session._schedule_master_glide(deck.name)
    glide = [c for c in session.scheduler.pending() if c.origin == "master_glide"]
    assert len(glide) == len(session._ride_steps) <= first
    assert glide[0].rate < 1.02  # the new glide, starting from 1.02


# --- glue and cancel, on the engine itself -------------------------------------------

import numpy as np  # noqa: E402

from djai import glue  # noqa: E402
from djai.commands import (  # noqa: E402
    CancelTransition, LoadTrack, SetGain, SetReverse, StartTransition,
)
from djai.deck import CHANNELS, LoadedTrack  # noqa: E402
from djai.engine import Engine  # noqa: E402
from tests.test_deck import make_track  # noqa: E402

BLOCK = 512
BPM = 128.0
BEAT = SAMPLE_RATE * 60.0 / BPM


def _quiet(freqs, seconds=40.0, level=0.08) -> LoadedTrack:
    """Tones far under the limiter's threshold, so nulls are exact."""
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    sig = sum(level * np.sin(2 * np.pi * f * t) for f in freqs)
    audio = np.stack([sig, sig], axis=1).astype(np.float32)
    audio = np.vstack([audio, np.zeros((2, CHANNELS), dtype=np.float32)])
    base = make_track(n_frames=n, bpm=BPM)
    return LoadedTrack(analysis=base.analysis, audio=audio)


A_TONES, B_TONES = (60.0, 900.0, 5000.0), (110.0, 1500.0)


def _engine(with_b: bool) -> Engine:
    eng = Engine(blocksize=BLOCK, stretch=False)
    eng.submit(LoadTrack(deck="a", track=_quiet(A_TONES), master=True))
    eng.submit(SetGain(deck="a", gain=1.0))  # a fresh deck's fader is down
    if with_b:
        eng.submit(LoadTrack(deck="b", track=_quiet(B_TONES), play=False))
    return eng


def _run(eng: Engine, blocks: int, sched: Scheduler | None = None) -> np.ndarray:
    out = np.zeros((blocks * BLOCK, CHANNELS), dtype=np.float32)
    buf = np.zeros((BLOCK, CHANNELS), dtype=np.float32)
    for i in range(blocks):
        if sched is not None:
            sched.tick(eng.frames_played + BLOCK)
        eng.callback(buf, BLOCK, None, None)
        out[i * BLOCK:(i + 1) * BLOCK] = buf
    return out


def _db(x: np.ndarray, ref: np.ndarray) -> float:
    return 20 * np.log10((np.sqrt(np.mean(x ** 2)) + 1e-20) / np.sqrt(np.mean(ref ** 2)))


def test_cancelling_mid_transition_nulls_against_the_outgoing_track_alone():
    ref = _run(_engine(False), 2400)

    eng = _engine(True)
    head = _run(eng, 40)
    total = int(16 * 4 * BEAT)  # 16 bars, the bass swaps at the midpoint
    eng.arm_transition_plan("bass_swap", total, BPM)
    eng.submit(StartTransition(from_deck="a", to_deck="b", total_frames=total))
    blend = _run(eng, 1800)  # ~70% through: bass handed over, both audible
    assert eng.transition_active
    assert eng.deck_state("a").eq[0] < 1.0 and eng.deck_b.gain.cur > 0.5
    eng.submit(CancelTransition(fade_frames=int(BEAT)))
    tail = _run(eng, 560)
    test = np.concatenate([head, blend, tail])

    cancel_at = 1840 * BLOCK
    settled = cancel_at + int(BEAT) + 4 * BLOCK
    assert not eng.transition_active
    assert not eng.deck_b.playing, "the incoming deck stops once faded"
    # After the fade the room hears exactly the outgoing track, as if the
    # blend had never started.
    assert _db(test[settled:] - ref[settled:], ref[settled:]) < -90.0
    # And the fade itself has no click: the residual (what the cancel changed)
    # moves no faster than it did while the blend was running steadily.
    resid = test - ref
    steady = np.abs(np.diff(resid[cancel_at - 40 * BLOCK:cancel_at, 0])).max()
    during = np.abs(np.diff(resid[cancel_at:settled, 0])).max()
    assert during <= steady * 1.05


def test_cancel_with_nothing_in_flight_is_a_no_op():
    eng, ref = _engine(False), _run(_engine(False), 60)
    head = _run(eng, 20)
    eng.submit(CancelTransition(fade_frames=int(BEAT)))
    out = np.concatenate([head, _run(eng, 40)])
    assert np.array_equal(out, ref)


def test_a_slip_reverse_comes_out_where_straight_playback_would_be():
    ref = _run(_engine(False), 200)
    eng = _engine(False)
    head = _run(eng, 40)
    eng.submit(SetReverse(deck="a", on=True))
    mid = _run(eng, 20)
    assert eng.deck_a.reversing and eng.deck_a.position < 40 * BLOCK
    eng.submit(SetReverse(deck="a", on=False))
    tail = _run(eng, 140)
    test = np.concatenate([head, mid, tail])
    assert eng.deck_a.position == eng.frames_played  # rate 1: in phase
    after = 61 * BLOCK  # one block for the exit crossfade
    assert _db(test[after:] - ref[after:], ref[after:]) < -120.0
    # The turn-around and the exit are no sharper than the music itself.
    edge = np.abs(np.diff(ref[:, 0])).max()
    assert np.abs(np.diff(test[40 * BLOCK - 4:after, 0])).max() <= edge * 1.05


def test_a_reverse_is_refused_inside_a_loop():
    eng = _engine(False)
    _run(eng, 4)
    eng.deck_a.set_loop(0.0, 4 * BEAT)
    eng.deck_a.set_reverse(True)
    assert not eng.deck_a.reversing


def test_the_volume_dip_ramps_down_and_returns_exactly_to_unity():
    ref = _run(_engine(False), 120)
    eng = _engine(False)
    sched = Scheduler(eng)
    start = 20 * BLOCK
    for cmd in glue.volume_dip(start, BEAT, beats=2, depth=0.5):
        sched.submit(cmd)
    test = _run(eng, 120, sched)
    inside = slice(start + 2 * BLOCK, start + int(2 * BEAT) - BLOCK)
    assert np.allclose(test[inside], 0.5 * ref[inside], atol=1e-6)
    back = start + int(2 * BEAT) + 2 * BLOCK
    assert np.array_equal(test[back:], ref[back:])
    edge = np.abs(np.diff(ref[:, 0])).max()
    assert np.abs(np.diff(test[:, 0])).max() <= edge * 1.05


def test_the_riser_plays_outside_a_transition_and_stops_on_its_last_beat():
    ref = _run(_engine(False), 400)
    eng = _engine(False)
    sched = Scheduler(eng)
    start = 20 * BLOCK
    cmds = glue.noise_riser(start, BEAT, beats=8, peak=0.25)
    for cmd in cmds:
        sched.submit(cmd)
    test = _run(eng, 400, sched)
    end = cmds[-1].execute_at
    late = slice(end - int(2 * BEAT), end - BLOCK)
    assert _db(test[late] - ref[late], ref[late]) > -12.0, "the riser is audible"
    after = end + 2 * BLOCK
    assert np.array_equal(test[after:], ref[after:]), "and gone after its last beat"


def test_the_filter_swell_peaks_then_lands_on_the_detent():
    cmds = glue.filter_swell("a", 0, BEAT, beats=4, peak=0.6, steps=8)
    positions = [c.position for c in cmds]
    assert max(positions) == 0.6 and positions[-1] == 0.0
    assert all(b > a for a, b in zip(cmds, cmds[1:]) for a, b in [(a.execute_at, b.execute_at)])


def test_grooves_offset_each_hit_by_10_to_40_ms():
    for name, offsets in glue.GROOVES.items():
        if name == "straight":
            assert offsets == (0.0, 0.0, 0.0, 0.0)
            continue
        # A swing may leave the on-beats alone; every offset it does apply is
        # inside the micro-timing range.
        moved = [abs(ms) for ms in offsets if ms != 0.0]
        assert moved
        assert all(glue.MIN_OFFSET_MS <= ms <= glue.MAX_OFFSET_MS for ms in moved)
    straight = glue.volume_dip(0, BEAT, beats=0.25)
    swung = glue.volume_dip(0, BEAT, beats=0.25, groove="shuffle")
    # The dip's end lands on the second sixteenth, which the shuffle delays.
    delta_ms = (swung[1].execute_at - straight[1].execute_at) * 1000 / SAMPLE_RATE
    assert abs(delta_ms - 30.0) < 0.05


def test_cancelling_a_glue_token_withdraws_the_whole_gesture():
    eng = _engine(False)
    sched = Scheduler(eng)
    tok = CancelToken("glue")
    for cmd in glue.noise_riser(20 * BLOCK, BEAT, beats=4, token=tok):
        sched.submit(cmd)
    assert len(sched) > 0
    sched.cancel(tok)
    assert len(sched) == 0


# --- "go harder now": audible within 8 bars, over 50 randomised runs -----------------

from djai import cli  # noqa: E402
from djai.intent import Intent  # noqa: E402
from tests.test_integration import log_events, make_analysis  # noqa: E402

AUDIBLE_DB = 1.5  # a band moving this much, against its own level a moment ago


def _drum_bar_track(bpm: float, seconds: float, seed: int) -> np.ndarray:
    """A kick on every beat plus broadband noise, tiled bar by bar.

    Tiled so every bar is identical: any change the measurement sees is the
    correction, not the music.
    """
    rng = np.random.default_rng(seed)
    beat = int(round(SAMPLE_RATE * 60.0 / bpm))
    t = np.arange(beat) / SAMPLE_RATE
    kick = 0.30 * np.sin(2 * np.pi * 55.0 * t) * np.exp(-t * 18.0)
    bar = np.concatenate([kick] * 4) + 0.05 * rng.standard_normal(4 * beat)
    n = int(SAMPLE_RATE * seconds)
    sig = np.tile(bar, n // bar.size + 1)[:n]
    audio = np.stack([sig, sig], axis=1).astype(np.float32)
    return np.vstack([audio, np.zeros((2, CHANNELS), dtype=np.float32)])


def _bands_db(x: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.rfft(x[:, 0])) ** 2
    f = np.fft.rfftfreq(x.shape[0], 1.0 / SAMPLE_RATE)
    edges = ((20, 250), (250, 2500), (2500, 16000))
    return np.array([10 * np.log10(spec[(f >= lo) & (f < hi)].sum() + 1e-20) for lo, hi in edges])


def _harder_session(tmp_path, monkeypatch, bpm: float) -> cli.Session:
    crate = []
    for i, (tid, cam) in enumerate((("live", "8A"), ("next", "9A"), ("other", "8B"))):
        p = tmp_path / f"{tid}-{bpm:.0f}.wav"
        p.write_bytes(b"placeholder")
        crate.append(make_analysis(f"{tid}{bpm:.0f}", bpm + 0.5 * i, cam, 0.3 + 0.3 * i, p))

    def fake_load(analysis):
        return LoadedTrack(analysis=analysis, audio=_drum_bar_track(analysis.bpm, 120.0, 7))

    monkeypatch.setattr(cli, "load_track", fake_load)
    monkeypatch.setattr("djai.config.TIME_STRETCH_ENABLED", False)
    s = cli.Session(crate, log_dir=tmp_path / f"logs-{bpm:.0f}")
    s.start_first_track()
    return s


def _play(s: cli.Session, frames: int, block: int = 1024) -> np.ndarray:
    eng = s.engine
    out = np.zeros((frames, CHANNELS), dtype=np.float32)
    buf = np.zeros((block, CHANNELS), dtype=np.float32)
    done = 0
    while done < frames:
        n = min(block, frames - done)
        s.scheduler.tick(eng.frames_played + n)
        eng.callback(buf[:n], n, None, None)
        out[done:done + n] = buf[:n]
        done += n
    return out


def test_go_harder_now_is_audible_within_8_bars_over_50_randomised_runs(tmp_path, monkeypatch):
    rng = np.random.default_rng(20260923)
    sessions = {bpm: _harder_session(tmp_path, monkeypatch, bpm) for bpm in (118.0, 124.0, 131.0)}
    latencies = []
    try:
        for bpm, s in sessions.items():
            _play(s, int(SAMPLE_RATE * 2))  # past the load ramp
        for run in range(50):
            bpm = float(rng.choice(list(sessions)))
            s = sessions[bpm]
            deck = s.engine.deck(s.live_deck)
            # Back to neutral between runs, as the hand-over would leave it.
            s.scheduler.cancel_all()
            deck.eq_mid.jump(1.0)
            deck.eq_high.jump(1.0)
            deck.filter_pos.jump(0.0)
            bar = int(round(SAMPLE_RATE * 240.0 / bpm))
            beat = bar // 4
            # Rewind by whole bars (phase kept) so no run reaches the end.
            deck.jump(-(int(deck.position // bar) - 2) * bar)
            # Anywhere in the bar, and a settled baseline just before.
            _play(s, int(rng.integers(0, bar)))
            base = _play(s, 2 * bar)
            ref = np.mean([_bands_db(base[i * beat:(i + 1) * beat]) for i in range(8)], axis=0)
            harder = rng.random() < 0.7
            d = float(rng.uniform(0.5, 1.0)) * (1 if harder else -1)
            cli.apply_intent(s, Intent(action="set_energy", params={"direction": d}))
            after = _play(s, 9 * bar)
            latency = None
            for i in range(9 * 4):
                moved = np.abs(_bands_db(after[i * beat:(i + 1) * beat]) - ref).max()
                if moved >= AUDIBLE_DB:
                    latency = (i + 1) / 4.0  # bars from the request to the end of that beat
                    break
            assert latency is not None, f"run {run}: {d:+.2f} at {bpm} BPM never became audible"
            latencies.append(latency)
    finally:
        for s in sessions.values():
            s.shutdown()
    mean, worst = float(np.mean(latencies)), float(np.max(latencies))
    print(f"\ngo-harder latency over {len(latencies)} runs: mean {mean:.2f} bars, "
          f"p50 {np.median(latencies):.2f}, max {worst:.2f}")
    assert mean <= 8.0 and worst <= 8.0


def test_the_correction_is_logged_and_skipped_inside_a_blend(session):
    assert session.energy_correction(0.8) is not None
    assert any(e.get("event") == "energy_correction" for e in log_events(session))
    pending = session.scheduler.pending()
    assert pending and {session.scheduler.band_of(c) for c in pending} <= {"immediate", "short"}
    session.engine._trans_active = True
    try:
        assert session.energy_correction(0.8) is None
    finally:
        session.engine._trans_active = False


def test_the_cancel_word_backs_out_of_a_running_blend(session):
    from djai.cli import handle_override

    assert handle_override(session, "cancel") == "No transition to cancel."
    eng = session.engine
    session.cue_next(0.0, origin="test")
    _play(session, SAMPLE_RATE // 2)
    other = session.cued_deck()
    total = int(16 * 4 * SAMPLE_RATE * 60.0 / eng.master_bpm)
    eng.arm_transition_plan("bass_swap", total, eng.master_bpm)
    eng.submit(StartTransition(from_deck=session.live_deck, to_deck=other, total_frames=total))
    session._transition_armed = True
    _play(session, SAMPLE_RATE * 5)
    assert eng.transition_active
    reply = handle_override(session, "cancel")
    assert reply.startswith("Cancelled")
    _play(session, SAMPLE_RATE * 2)
    assert not eng.transition_active and not eng.deck(other).playing
    assert eng.deck(session.live_deck).gain.cur == 1.0
    assert any(e.get("event") == "transition_cancelled" for e in log_events(session))


# --- the feel clock: phase re-locks within a tested bound ----------------------------

import pytest  # noqa: E402

from djai import phrase  # noqa: E402
from djai.supervisor import DRIFT_IGNORE_MS, MONITOR_INTERVAL, SessionLog, Supervisor  # noqa: E402
from tests.test_engine import drive, loud_track  # noqa: E402

#: Measured 0.49-1.54 bars across +-8..60 ms (scratch/feel_clock_probe.py).
RELOCK_BOUND_BARS = 2.0


def _drift_ms(sup, eng, name: str) -> float:
    _, mb, pa, pb = eng.clock
    pairs = {"a": (eng.deck_a, pa), "b": (eng.deck_b, pb)}
    deck, pos = pairs[name]
    expected = sup._expected_beat(deck, pos, mb, pairs[eng.master_deck])
    return (phrase.beat_at_frame(deck.track.analysis, pos) - expected) * deck.track.analysis.beat_period * 1000


@pytest.mark.parametrize("offset_ms", [12.0, 18.0, 40.0, -40.0])
def test_the_feel_clock_relocks_phase_within_two_bars(tmp_path, offset_ms):
    """Nudge regime (5-20 ms) and resync regime (over 20 ms), both directions.

    "Locked" is inside the supervisor's own deadband, DRIFT_IGNORE_MS, and
    held for the rest of the run -- not touched once and lost again.
    """
    bpm = 124.0
    eng = Engine(blocksize=BLOCK, stretch=False)
    a, b = loud_track(200.0, bpm=bpm, seconds=120.0), loud_track(3000.0, bpm=bpm, seconds=120.0)
    for n, t in (("a", a), ("b", b)):
        p = tmp_path / f"{n}.wav"
        p.write_bytes(b"x")
        t.analysis.path, t.analysis.track_id = str(p), n
    eng.submit(LoadTrack(deck="a", track=a, master=True))
    eng.submit(LoadTrack(deck="b", track=b))
    drive(eng, 4, BLOCK)
    sched = Scheduler(eng)
    sup = Supervisor(eng, sched, [a.analysis, b.analysis], session_log=SessionLog(tmp_path))
    every = max(1, int(MONITOR_INTERVAL * SAMPLE_RATE / BLOCK))
    for i in range(40):
        sched.tick(eng.frames_played)
        drive(eng, 1, BLOCK)
        if i % every == 0:
            sup.check_drift()

    eng.deck_b.position += offset_ms / 1000.0 * SAMPLE_RATE
    drive(eng, 1, BLOCK)
    bar = SAMPLE_RATE * 240.0 / bpm
    t0, locked_at = eng.frames_played, None
    for i in range(int(8 * bar / BLOCK)):
        sched.tick(eng.frames_played + BLOCK)
        drive(eng, 1, BLOCK)
        if i % every == 0:
            sup.check_drift()
        if abs(_drift_ms(sup, eng, "b")) < DRIFT_IGNORE_MS:
            locked_at = locked_at or eng.frames_played
        else:
            locked_at = None
    assert locked_at is not None, "never re-locked"
    assert (locked_at - t0) / bar <= RELOCK_BOUND_BARS
    assert abs(_drift_ms(sup, eng, "b")) < DRIFT_IGNORE_MS
