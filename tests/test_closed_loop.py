"""Phase 4.1: the closed loop (SPEC §5).

The loop is measured through the real engine: synthetic audio on a real
Session, the output tap, the proxies, the loop, and corrections scheduled
through the Short band and played back through the deck's own EQ and filter.
"Simulated" in the criterion means the drops and spikes are injected -- a
filter left closed, a boosted EQ, a fader pushed -- not that the audio is.
"""

from __future__ import annotations

import numpy as np
import pytest

from djai import cli, room
from djai.commands import IMMEDIATE, SetEQ, SetFilter, SetGain
from djai.deck import CHANNELS, SAMPLE_RATE, LoadedTrack
from tests.synth import render_track
from tests.test_integration import log_events, make_analysis

BPM = 124.0
BAR = int(round(SAMPLE_RATE * 240.0 / BPM))
#: One autopilot tick (0.5 s) in 1024-frame blocks.
TICK_BLOCKS = 21


# --- the loop on its own ---------------------------------------------------------


def _feed(loop: room.EnergyLoop, values, start_bar: int = 0):
    return [loop.assess(v, start_bar + i) for i, v in enumerate(values)]


def test_the_loop_learns_its_baseline_then_waits_two_bars_before_answering():
    loop = room.EnergyLoop()
    got = _feed(loop, [-20.0] * 4)
    assert [v.status for v in got] == ["settling"] * 4 and loop.baseline == -20.0
    got = _feed(loop, [-25.0, -25.0], 4)
    assert got[0].direction is None and got[0].status == "watch"
    assert got[1].status == "drop" and got[1].direction > 0


def test_a_level_between_release_and_trigger_is_watched_not_answered():
    loop = room.EnergyLoop()
    _feed(loop, [0.0] * 4)
    got = _feed(loop, [2.0] * 30, 4)
    assert all(v.direction is None for v in got)
    assert {v.status for v in got} == {"watch"}


def test_the_loop_waits_for_its_last_move_and_is_bounded():
    loop = room.EnergyLoop()
    _feed(loop, [0.0] * 4)
    got = _feed(loop, [-20.0] * 60, 4)
    moves = [i for i, v in enumerate(got) if v.direction is not None]
    assert all(b - a >= room.EnergyLoop.REFRACTORY_BARS for a, b in zip(moves, moves[1:]))
    assert loop.direction <= room.EnergyLoop.MAX_DIRECTION
    steps = np.diff([0.0] + [got[i].direction for i in moves])
    assert np.all(np.abs(steps) <= room.EnergyLoop.MAX_STEP + 1e-9)
    # Out of reach: said so, not answered with more of the same.
    assert got[-1].status == "saturated"


# --- through the engine ----------------------------------------------------------

_AUDIO: dict[str, np.ndarray] = {}


def _music() -> np.ndarray:
    """A drum-machine track with bass and chords (tests.synth): a musical
    spectrum, so the filter and EQ bite as they do on real records -- white
    noise, with half its energy above 11 kHz, overstates the low-pass tenfold.
    Every bar has the same arrangement, so the analysis' flat bar energy is
    the truth."""
    if "a" not in _AUDIO:
        import tempfile

        import soundfile as sf

        path = render_track(__import__("pathlib").Path(tempfile.mkdtemp()) / "m.wav",
                            BPM, bars=110)
        audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
        _AUDIO["a"] = np.vstack([audio, np.zeros((2, CHANNELS), dtype=np.float32)])
    return _AUDIO["a"]



def _session(tmp_path, monkeypatch, name="s", mic=False) -> cli.Session:
    crate = []
    for i, (tid, cam) in enumerate((("live", "8A"), ("next", "9A"))):
        p = tmp_path / f"{name}-{tid}.wav"
        p.write_bytes(b"x")
        crate.append(make_analysis(f"{name}{tid}", BPM + 0.5 * i, cam, 0.3 + 0.3 * i, p))
    monkeypatch.setattr(
        cli, "load_track",
        lambda a: LoadedTrack(analysis=a, audio=_music()))
    monkeypatch.setattr("djai.config.TIME_STRETCH_ENABLED", False)
    monkeypatch.setattr("djai.config.CLOSED_LOOP_ENABLED", True)
    monkeypatch.setattr("djai.config.ROOM_MIC_ENABLED", mic)
    s = cli.Session(crate, log_dir=tmp_path / f"logs-{name}")
    s.start_first_track()
    return s


class Run:
    """Plays a session block by block, ticking the loop as the autopilot would,
    and keeping the per-bar record the assertions read."""

    def __init__(self, s: cli.Session, mic_noise: float | None = None, seed: int = 0):
        self.s = s
        self.buf = np.zeros((1024, CHANNELS), dtype=np.float32)
        self.blocks = 0
        self.bars: list[tuple[int, str, float, float | None]] = []
        self.mic_noise = mic_noise
        self.rng = np.random.default_rng(seed)

    @property
    def bar_count(self) -> int:
        return len(self.bars)

    def play_bars(self, n: float) -> None:
        eng, s = self.s.engine, self.s
        deck = eng.deck(s.live_deck)
        for _ in range(int(n * BAR / 1024)):
            # Stay inside the analysed region: rewind by whole bars, phase kept.
            if deck.position > 200 * SAMPLE_RATE:
                deck.jump(-(int(deck.position // BAR) - 8) * BAR)
            start = eng.frames_played
            s.scheduler.tick(start + 1024)
            eng.callback(self.buf, 1024, None, None)
            if self.mic_noise is not None:
                # The room: the system, a little late, plus crowd noise.
                heard = self.buf + self.mic_noise * self.rng.standard_normal(self.buf.shape)
                s.room.mic.feed(heard.astype(np.float32), start)
            self.blocks += 1
            if self.blocks % TICK_BLOCKS == 0:
                self._tick()

    def _tick(self) -> None:
        s = self.s
        monitor = s.room
        took = []
        orig = monitor.poll

        def spy(live, paused):
            rows = orig(live, paused)
            took.extend(rows)
            return rows

        monitor.poll = spy
        try:
            s._room_tick()
        finally:
            monitor.poll = orig
        for reading, verdict in took:
            self.bars.append((
                reading.bar, verdict.status if verdict else "-",
                reading.loudness_db, verdict.direction if verdict else None,
            ))

    def corrections(self, since: int = 0) -> list[tuple[int, float]]:
        return [(i, d) for i, (_, _, _, d) in enumerate(self.bars) if i >= since and d is not None]


def _inject(s: cli.Session, kind: str) -> None:
    deck = s.live_deck
    cmd = {
        "filter_drop": SetFilter(deck=deck, position=-0.5, execute_at=IMMEDIATE, origin="fault"),
        "eq_spike": SetEQ(deck=deck, mid=1.8, high=1.8, execute_at=IMMEDIATE, origin="fault"),
        "gain_spike": SetGain(deck=deck, gain=1.6, execute_at=IMMEDIATE, origin="fault"),
    }[kind]
    s.engine.submit(cmd)


def _back_to_normal(s: cli.Session, run: Run) -> None:
    """Neutral controls, and the loop re-learns: the start of each episode."""
    deck = s.engine.deck(s.live_deck)
    s.scheduler.cancel_all()
    for knob in (deck.eq_low, deck.eq_mid, deck.eq_high):
        knob.jump(1.0)
    deck.filter_pos.jump(0.0)
    deck.set_gain(1.0)
    s.room.handover()
    run.play_bars(6)


def _recovered_within(run: Run, since: int, bars: int) -> int | None:
    """Bars from ``since`` until the loop reads the level as working again and
    stays there for 4 bars, or None if it did not inside ``bars``."""
    rows = run.bars[since:since + bars]
    for i in range(len(rows) - 3):
        if all(r[1] == "ok" for r in rows[i:i + 4]):
            return i
    return None


@pytest.mark.parametrize("kind", ["filter_drop", "eq_spike", "gain_spike"])
def test_an_injected_drop_or_spike_is_corrected_within_one_phrase(tmp_path, monkeypatch, kind):
    s = _session(tmp_path, monkeypatch, kind)
    run = Run(s)
    try:
        run.play_bars(8)
        assert s.room.loop.baseline is not None
        since = run.bar_count
        _inject(s, kind)
        run.play_bars(40)
        fixed = _recovered_within(run, since, 32)
        assert fixed is not None, run.bars[since:]
        moves = run.corrections(since)
        assert moves, "the loop never answered"
        if kind == "gain_spike":
            # Nothing on the deck's EQ or filter moved: a real step, calmer.
            assert all(d < 0 for _, d in moves)
        else:
            # A control left where it should not be: put back, not fought.
            assert moves == [(moves[0][0], 0.0)]
            events = log_events(s)
            dec = [e for e in events if e.get("event") == "closed_loop"]
            assert "put back" in dec[0]["trigger"]
        # Through the Short band, and logged as a decision with its snapshot.
        events = log_events(s)
        corr = [e for e in events if e.get("event") == "energy_correction"
                and "closed loop" in e.get("trigger", "")]
        assert corr and "4-bar line" in corr[0]["action"]
        dec = [e for e in events if e.get("event") == "closed_loop"]
        assert dec and dec[0]["snapshot"] is not None and dec[0]["heuristic"] is True
    finally:
        s.shutdown()


def test_the_correction_is_scheduled_in_the_short_band(tmp_path, monkeypatch):
    s = _session(tmp_path, monkeypatch, "band")
    try:
        Run(s).play_bars(3)
        assert s.energy_correction(0.5, lead_bars=2.0, origin="closed loop")
        bands = {s.scheduler.band_of(c) for c in s.scheduler.pending()}
        assert bands == {"short"}
    finally:
        s.shutdown()


def test_a_drop_past_the_corrections_reach_is_reported_not_chased(tmp_path, monkeypatch):
    """A fader pulled well down: harder EQ lifts ~2 dB at most, so the loop
    goes to its bound once, says so, and does not oscillate."""
    s = _session(tmp_path, monkeypatch, "far")
    run = Run(s)
    try:
        run.play_bars(8)
        since = run.bar_count
        s.engine.submit(SetGain(deck=s.live_deck, gain=0.35, execute_at=IMMEDIATE, origin="fault"))
        run.play_bars(64)
        moves = run.corrections(since)
        assert moves and all(d > 0 for _, d in moves)
        assert len(moves) <= 3
        assert any(e.get("event") == "closed_loop_saturated" for e in log_events(s))
    finally:
        s.shutdown()


def _hour(s: cli.Session, run: Run, seed: int) -> list[dict]:
    """60 minutes of set time with an injected fault every 2.5-4 minutes."""
    rng = np.random.default_rng(seed)
    bars_per_min = BPM / 4.0
    episodes = []
    run.play_bars(8)
    while run.bar_count < 60 * bars_per_min:
        quiet = float(rng.uniform(2.5, 4.0)) * bars_per_min
        kind = str(rng.choice(["filter_drop", "eq_spike", "gain_spike"]))
        _back_to_normal(s, run)
        run.play_bars(quiet - 48)
        since = run.bar_count
        _inject(s, kind)
        run.play_bars(42)
        episodes.append({"kind": kind, "since": since,
                         "fixed": _recovered_within(run, since, 32),
                         "moves": run.corrections(since)})
    return episodes


def _reversals(moves) -> int:
    """Correction steps that change direction: the loop undoing itself."""
    steps = np.diff([0.0] + [d for _, d in moves])
    signs = np.sign(steps[np.abs(steps) > 1e-9])
    return int(np.sum(signs[1:] != signs[:-1]))


def test_an_hour_of_drops_and_spikes_each_corrected_within_a_phrase_without_oscillation(
    tmp_path, monkeypatch,
):
    s = _session(tmp_path, monkeypatch, "hour")
    run = Run(s)
    try:
        episodes = _hour(s, run, seed=20260923)
    finally:
        s.shutdown()
    fixed = [e["fixed"] for e in episodes]
    assert len(episodes) >= 14
    assert all(f is not None for f in fixed), episodes
    reversals = sum(_reversals(e["moves"]) for e in episodes)
    in_quiet = [
        (i, d) for i, d in run.corrections()
        if not any(e["since"] <= i < e["since"] + 42 for e in episodes)
    ]
    print(f"\n{len(episodes)} episodes over {run.bar_count} bars: recovered after "
          f"{np.mean(fixed):.1f} bars on average, worst {max(fixed)}; "
          f"{sum(len(e['moves']) for e in episodes)} correction(s), {reversals} reversal(s), "
          f"{len(in_quiet)} outside an episode")
    assert reversals == 0
    assert not in_quiet


def test_disabling_the_room_mic_changes_nothing_but_signal_quality(tmp_path, monkeypatch):
    runs = {}
    for mic in (False, True):
        s = _session(tmp_path, monkeypatch, f"mic{int(mic)}", mic=mic)
        run = Run(s, mic_noise=0.05 if mic else None, seed=3)
        try:
            run.play_bars(8)
            _inject(s, "filter_drop")
            run.play_bars(24)
            _inject(s, "eq_spike")
            run.play_bars(24)
            runs[mic] = (run, s.room.state())
        finally:
            s.shutdown()
    off, on = runs[False], runs[True]
    assert [(b, st, d) for b, st, _, d in off[0].bars] == [(b, st, d) for b, st, _, d in on[0].bars]
    assert off[1]["mic"] == "off" and "signal_quality" not in off[1]
    assert on[1]["mic"] == "on" and on[1]["room"]["proxy"] is True
    assert on[1]["signal_quality"] is not None and on[1]["signal_quality"] < 0.999


def test_the_proxies_are_labelled_and_measure_what_they_say():
    p = room.OutputProxies()
    bar_frames = 4 * 22050
    rng = np.random.default_rng(1)
    loud = 0.3 * rng.standard_normal((bar_frames * 4, 2))
    quiet = 0.03 * rng.standard_normal((bar_frames * 4, 2))
    readings = p.push(np.vstack([loud, quiet]), 0, lambda f: f / bar_frames)
    levels = [r.loudness_db for r in readings]
    assert len(readings) == 7
    assert levels[0] - levels[-1] == pytest.approx(20.0, abs=0.5)
    summary = p.summary()
    assert summary["proxy"] is True
    assert summary["energy_variance"] > 50 and summary["loudness_dynamics_db"] > 15


def test_the_callback_tap_only_copies_what_the_room_hears(tmp_path, monkeypatch):
    s = _session(tmp_path, monkeypatch, "tap")
    try:
        buf = np.zeros((1024, CHANNELS), dtype=np.float32)
        out = []
        for _ in range(8):
            s.engine.callback(buf, 1024, None, None)
            out.append(buf.copy())
        got = np.zeros((8 * 1024, CHANNELS), dtype=np.float32)
        s.room.tap.read_into(got, got.shape[0])
        assert np.array_equal(got, np.vstack(out))
    finally:
        s.shutdown()


def test_the_loop_leaves_the_decks_alone_when_a_blend_is_about_to_start(tmp_path, monkeypatch):
    from djai.commands import StartTransition

    s = _session(tmp_path, monkeypatch, "armed")
    try:
        Run(s).play_bars(3)
        at = s.engine.frames_played + 4 * BAR
        s.scheduler.submit(StartTransition(execute_at=at, origin="test"))
        assert s.energy_correction(-0.4, lead_bars=2.0, origin="closed loop") is None
        assert not [c for c in s.scheduler.pending() if c.origin == "closed loop"]
        # The operator's own "calmer" is theirs to make, blend or not.
        assert s.energy_correction(-0.4) is not None
    finally:
        s.shutdown()

