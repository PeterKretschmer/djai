"""SPEC §3 (Phase 2.3): the action language -- schema, repair, execution, fuzz.

THREADING CONTEXT: main thread (pytest). The scheduler is ticked and the audio
callback driven by hand; the autopilot is ticked once a bar in the fuzz run so
the plans land on a set that is really moving.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from djai import actions as act
from djai import cli, config
from djai.commands import SetPitch
from djai.deck import CHANNELS, SAMPLE_RATE, LoadedTrack
from djai.intent import IntentEngine
from tests.test_integration import make_analysis

BPM = 124.0
BEAT = SAMPLE_RATE * 60.0 / BPM


# --- a session of pure low tones, where any discontinuity is obvious --------------------


def _tone(freq: float, seconds: float = 240.0, level: float = 0.2) -> np.ndarray:
    """Integer Hz, so every 1 s tile holds whole cycles and tiling is seamless."""
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    one = level * np.sin(2 * np.pi * freq * t)
    sig = np.tile(one, int(seconds) + 1)[: int(seconds * SAMPLE_RATE)]
    audio = np.stack([sig, sig], axis=1).astype(np.float32)
    return np.vstack([audio, np.zeros((2, CHANNELS), dtype=np.float32)])


def _smooth_riser(n_frames: int, seed: int = 0) -> np.ndarray:
    """Stands in for the band-passed noise riser in the click test only.

    The riser's DSP runs on the control thread; what can click is how the
    callback ramps its level, and that path is untouched. Noise would hide a
    click from the detector; a low tone does not.
    """
    t = np.arange(max(1, int(n_frames))) / SAMPLE_RATE
    one = np.sin(2 * np.pi * 70.0 * t)
    return np.stack([one, one], axis=1).astype(np.float32)


@pytest.fixture
def tone_session(tmp_path, monkeypatch):
    freqs = {"alpha": 105.0, "beta": 147.0, "gamma": 126.0, "delta": 84.0}
    crate = []
    for i, (tid, freq) in enumerate(freqs.items()):
        p = tmp_path / f"{tid}.wav"
        p.write_bytes(b"x")
        crate.append(make_analysis(tid, BPM + 0.5 * (i % 2), ("8A", "9A", "8B", "7A")[i],
                                   0.3 + 0.2 * i, p))

    def fake_load(analysis):
        return LoadedTrack(analysis=analysis, audio=_tone(freqs[analysis.track_id]))

    monkeypatch.setattr(cli, "load_track", fake_load)
    monkeypatch.setattr(config, "TIME_STRETCH_ENABLED", False)
    monkeypatch.setattr("djai.glue.render_riser", _smooth_riser)
    monkeypatch.setattr("djai.engine.render_riser", _smooth_riser)
    s = cli.Session(crate, log_dir=tmp_path / "logs")
    s.start_first_track()
    _play(s, SAMPLE_RATE // 2)  # the first track lands on its deck
    yield s
    s.shutdown()


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


def _plan(strategy="none", actions=None, intent="test"):
    d = {"intent": intent, "strategy": strategy}
    if actions is not None:
        d["actions"] = actions
    return d


# --- schema and validator -----------------------------------------------------------------


def test_every_strategy_expands_to_schema_valid_actions():
    for name, (_desc, actions) in act.STRATEGIES.items():
        plan, errors = act.parse(_plan(name))
        assert plan is not None, (name, errors)
        assert plan.expanded and plan.actions == actions
        for a in actions:
            assert act.check(act.ACTION_SCHEMAS[a["kind"]], a) == [], (name, a)


@pytest.mark.parametrize("bad, why", [
    ({"intent": "x"}, "missing 'strategy'"),
    ({"intent": "x", "strategy": "lift", "extra": 1}, "unexpected 'extra'"),
    ({"intent": "x", "strategy": "wobble"}, "not one of"),
    (_plan("lift", [{"kind": "explode"}]), "kind must be one of"),
    (_plan("lift", [{"kind": "fader", "deck": "a"}]), "missing 'to'"),
    (_plan("lift", [{"kind": "fader", "deck": "a", "to": True}]), "expected a number"),
    (_plan("lift", [{"kind": "fader", "deck": "a", "to": "0.5"}]), "expected a number"),
    (_plan("lift", [{"kind": "fader", "deck": "a", "to": 1.5}]), "outside"),
    (_plan("lift", [{"kind": "pitch", "deck": "a", "percent": float("inf")}]), "must be finite"),
    (_plan("lift", [{"kind": "beat_jump", "deck": "a", "beats": 2.5}]), "expected an integer"),
    (_plan("lift", [{"kind": "beat_jump", "deck": "a", "beats": 10 ** 400}]), "outside"),
    (_plan("lift", [{"kind": "loop", "deck": "a", "beats": 3}]), "not one of"),
    (_plan("lift", [{"kind": "dynamics", "direction": 0.5, "deck": "a"}]), "unexpected 'deck'"),
    (_plan("lift", [{"kind": "dynamics", "direction": 0.1}] * 13), "more than 12"),
    ("{not json", "not JSON"),
    ("[" * 5000, "not JSON"),
    (None, "expected an object"),
])
def test_the_validator_refuses_what_the_schema_forbids(bad, why):
    plan, errors = act.parse(bad)
    assert plan is None
    assert any(why in e for e in errors), errors


# --- trajectories ------------------------------------------------------------------------


@pytest.mark.parametrize("shape", act.SHAPES)
def test_trajectories_land_exactly_and_move_one_way(shape):
    steps = act.trajectory(0.2, 1.0, 8.0, shape)
    assert steps[-1] == (8.0, 1.0)
    values = [v for _, v in steps]
    assert values == sorted(values)
    assert [o for o, _ in steps] == sorted(o for o, _ in steps)
    assert act.trajectory(0.3, 0.7, 0.0) == [(0.0, 0.7)]


def test_trajectories_are_tempo_relative(tone_session):
    """The same plan spans the same number of beats at any tempo."""
    s = tone_session
    _play(s, SAMPLE_RATE)
    plan = _plan("none", [{"kind": "filter", "deck": "live", "to": -0.5, "bars": 2}])
    out = act.execute(s, plan)
    span = out.commands[-1].execute_at - out.commands[0].execute_at
    beats = span / (SAMPLE_RATE * 60.0 / s.engine.master_bpm)
    assert beats == pytest.approx(7.75, abs=0.01)  # sixteenths over 8 beats: first at +1/4


# --- repair ---------------------------------------------------------------------------------


def _ctx(s):
    return act.context_of(s)


def test_repair_clamps_pitch_to_the_stretch_cap(tone_session):
    s = tone_session
    payload = _plan("none", [{"kind": "pitch", "deck": "live", "percent": 15}])
    plan, _ = act.parse(payload)
    actions, notes = act.repair(plan, _ctx(s))
    assert actions[0]["percent"] == pytest.approx(config.MAX_STRETCH_RATIO * 100)
    assert any("stretch cap" in n for n in notes)
    out = act.execute(s, payload)
    assert out.ok and not out.rejected
    assert max(abs(c.rate - 1.0) for c in out.commands if isinstance(c, SetPitch)) \
        <= config.MAX_STRETCH_RATIO + 1e-9


def test_repair_never_lets_the_room_go_silent(tone_session):
    s = tone_session
    plan, _ = act.parse(_plan("none", [{"kind": "fader", "deck": "live", "to": 0.0, "bars": 1}]))
    actions, notes = act.repair(plan, _ctx(s))
    assert actions[0]["to"] == act.MIN_LIVE_LEVEL and notes


def test_repair_drops_moves_on_a_deck_the_blend_owns(tone_session):
    s = tone_session
    s.engine._trans_active = True
    s.engine._trans_from = s.engine.deck(s.live_deck)
    try:
        plan, _ = act.parse(_plan("none", [
            {"kind": "eq", "deck": "live", "band": "low", "to": 0.0},
            {"kind": "fx", "effect": "riser"},
            {"kind": "fx", "effect": "volume_dip", "beats": 1},
        ]))
        actions, notes = act.repair(plan, _ctx(s))
    finally:
        s.engine._trans_active = False
        s.engine._trans_from = None
    assert [a["kind"] for a in actions] == ["fx"] and actions[0]["effect"] == "volume_dip"
    assert len(notes) == 2


def test_repair_drops_the_later_of_two_actions_on_one_control(tone_session):
    s = tone_session
    plan, _ = act.parse(_plan("none", [
        {"kind": "filter", "deck": "live", "to": -0.5, "bars": 2},
        {"kind": "fx", "effect": "filter_swell", "deck": "live", "beats": 4},  # same knob
        {"kind": "filter", "deck": "live", "to": 0.0, "bars": 1, "at_bar": 2},  # after: fine
        {"kind": "loop", "deck": "live", "beats": 4, "hold_bars": 2},
        {"kind": "fx", "effect": "reverse", "deck": "live", "beats": 8},  # playhead, overlaps
    ]))
    actions, notes = act.repair(plan, _ctx(s))
    assert [(a["kind"], a.get("effect")) for a in actions] == [
        ("filter", None), ("filter", None), ("loop", None)]
    assert sum("conflicts" in n for n in notes) == 2
    assert any("shortened" in n for n in notes)


def test_repair_drops_actions_on_an_empty_deck(tone_session):
    s = tone_session
    plan, _ = act.parse(_plan("none", [{"kind": "fader", "deck": "incoming", "to": 1.0}]))
    actions, notes = act.repair(plan, _ctx(s))
    assert actions == [] and "empty deck" in notes[0]


# --- execution ---------------------------------------------------------------------------------


def test_a_strategy_runs_as_one_cancellable_group_and_the_next_plan_supersedes_it(tone_session):
    s = tone_session
    _play(s, SAMPLE_RATE)
    first = act.execute(s, _plan("tension"))
    assert first.ok and len(first.commands) > 10
    tokens = {c.token for c in first.commands}
    assert len(tokens) == 1
    assert all(s.supervisor.validate(c) is None for c in first.commands)
    second = act.execute(s, _plan("breathe"))
    assert second.ok
    assert next(iter(tokens)).cancelled
    assert all(c.token is not next(iter(tokens)) for c in s.scheduler.pending())


def test_actions_on_two_decks_share_one_anchor(tone_session):
    s = tone_session
    s.cue_next(0.0, origin="test")
    _play(s, SAMPLE_RATE)
    out = act.execute(s, _plan("none", [
        {"kind": "eq", "deck": "live", "band": "low", "to": 0.0, "start": "next_bar"},
        {"kind": "eq", "deck": "incoming", "band": "low", "to": 0.0, "start": "next_bar"},
        {"kind": "eq", "deck": "incoming", "band": "high", "to": 1.2, "start": "next_bar",
         "at_bar": 1},
    ]))
    assert out.ok, out.summary()
    firsts = {}  # the first step of each action: (deck, band) -> frame
    for cmd in out.commands:
        band = "low" if cmd.low is not None else "high"
        firsts.setdefault((cmd.deck, band), cmd.execute_at)
    live, inc = s.live_deck, s.cued_deck()
    a, b, c = firsts[(live, "low")], firsts[(inc, "low")], firsts[(inc, "high")]
    assert a == b
    assert c - a == pytest.approx(4 * SAMPLE_RATE * 60.0 / s.engine.master_bpm, abs=1)


def test_a_cue_action_brings_the_track_in_at_the_ranked_mix_point(tone_session):
    s = tone_session
    for t in s.crate:  # give every track ranked regions to choose from
        t.mix_in_regions = [{"bar": 8, "seconds": 8 * 4 * t.beat_period, "quality": 0.9},
                            {"bar": 16, "seconds": 16 * 4 * t.beat_period, "quality": 0.8}]
    out = act.execute(s, _plan("none", [{"kind": "cue", "mix_point": 2}]))
    assert out.ok and s.cue_mix_point is None, "one-shot"
    cued = s._cued.analysis
    assert cued.mix_in_bar == 16.0
    assert all(t.mix_in_bar != 16.0 for t in s.crate), "the crate's entry is never mutated"


def test_the_fallback_decides_when_the_model_is_down_and_says_so(tone_session):
    s = tone_session
    down = IntentEngine(base_url="http://127.0.0.1:9")
    down.available = False
    out = act.act(s, "build some tension", down)
    assert out.ok and out.plan.strategy == "tension"
    assert out.fallback == "model unavailable"
    assert "fallback" in out.summary()
    assert act.act(s, "flibbertigibbet", None).plan.strategy == "none"
    reply = cli.handle_override(s, 'act {"intent": "x", "strategy": "calm"}')
    assert reply.startswith("calm:")


# --- the fuzz ------------------------------------------------------------------------------------

JUNK = [None, "", "x", True, False, -1, 0, 10 ** 30, -(10 ** 30), 1e308, -1e308,
        float("nan"), float("inf"), [], {}, [1, 2], {"a": 1}, "next_beat", 0.5]


def _valid_value(rng, sub):
    if "const" in sub:
        return sub["const"]
    if "enum" in sub:
        return sub["enum"][int(rng.integers(len(sub["enum"])))]
    if sub.get("type") == "integer":
        return int(rng.integers(int(sub["minimum"]), int(sub["maximum"]) + 1))
    return round(float(rng.uniform(sub["minimum"], sub["maximum"])), 3)


def _random_action(rng) -> dict:
    kind = list(act.ACTION_SCHEMAS)[int(rng.integers(len(act.ACTION_SCHEMAS)))]
    schema = act.ACTION_SCHEMAS[kind]
    out = {}
    for key, sub in schema["properties"].items():
        if key in schema["required"] or rng.random() < 0.5:
            out[key] = _valid_value(rng, sub)
    return out


def _mutate(rng, payload: dict):
    p = json.loads(json.dumps(payload))
    op = int(rng.integers(8))
    target = p["actions"][int(rng.integers(len(p["actions"])))] if p.get("actions") else p
    keys = list(target)
    if op == 0 and keys:
        target.pop(keys[int(rng.integers(len(keys)))])
    elif op == 1:
        target["bogus"] = JUNK[int(rng.integers(len(JUNK)))]
    elif op == 2 and keys:
        target[keys[int(rng.integers(len(keys)))]] = JUNK[int(rng.integers(len(JUNK)))]
    elif op == 3:
        p["actions"] = [_random_action(rng) for _ in range(int(rng.integers(13, 30)))]
    elif op == 4:
        p["strategy"] = JUNK[int(rng.integers(len(JUNK)))]
    elif op == 5:
        return [p]
    elif op == 6:
        return json.dumps(p)[: int(rng.integers(1, 40))]
    else:
        target["kind"] = JUNK[int(rng.integers(len(JUNK)))]
    return p


def _payload(rng):
    r = rng.random()
    strategy = list(act.STRATEGIES)[int(rng.integers(len(act.STRATEGIES)))]
    if r < 0.40:
        return _plan(strategy, [_random_action(rng) for _ in range(int(rng.integers(1, 6)))])
    if r < 0.75:
        return _mutate(rng, _plan(strategy, [_random_action(rng) for _ in range(int(rng.integers(1, 4)))]))
    if r < 0.85:
        return JUNK[int(rng.integers(len(JUNK)))] if rng.random() < 0.5 else \
            "".join(chr(int(c)) for c in rng.integers(32, 0x2FF, int(rng.integers(0, 60))))
    return _plan(strategy)


#: Second difference of the output, in full-scale units. The tones' own peak
#: here is about 1e-3 even with every EQ band and resonance at its limit; a
#: step in the waveform of even -40 dBFS reads as 1e-2.
CLICK_D2: float = 5e-3


def test_1000_random_payloads_never_raise_and_never_click(tone_session, capsys):
    s = tone_session
    rng = np.random.default_rng(20260923)
    bar = int(4 * SAMPLE_RATE * 60.0 / BPM)
    counts = {"valid": 0, "refused": 0, "repairs": 0, "rejected": 0, "commands": 0}
    worst, worst_at, peak = 0.0, None, 0.0
    tail = None  # the first bar has nothing before it to be discontinuous with
    underruns = s.engine.underruns
    for i in range(1000):
        out = act.execute(s, _payload(rng))
        assert isinstance(out, act.Outcome)
        assert not any(e.startswith("internal") for e in out.errors), out.errors
        if out.ok:
            counts["valid"] += 1
            counts["repairs"] += len(out.repairs)
            counts["rejected"] += len(out.rejected)
            counts["commands"] += len(out.commands)
        else:
            counts["refused"] += 1
        s._autopilot_tick()
        audio = _play(s, bar)
        assert np.isfinite(audio).all()
        joined = (audio if tail is None else np.concatenate([tail, audio]))[:, 0].astype(np.float64)
        d2 = np.abs(joined[2:] - 2 * joined[1:-1] + joined[:-2])
        if d2.max() > worst:
            worst, worst_at = float(d2.max()), (i, int(d2.argmax()))
        peak = max(peak, float(np.abs(audio).max()))
        tail = audio[-2:]
    with capsys.disabled():
        print(f"\n    fuzz: {counts}; worst second difference {worst:.2e} at {worst_at}; "
              f"peak {peak:.3f}; underruns {s.engine.underruns - underruns}")
    assert s.engine.underruns == underruns, "an exception reached the audio callback"
    assert counts["valid"] > 300 and counts["refused"] > 200
    assert worst < CLICK_D2, f"discontinuity at payload {worst_at}"


# --- regressions: one per root cause the fuzz found ----------------------------------------

from djai.commands import (  # noqa: E402
    BeatJump, LoadTrack, SetFilter, SetGain, SetLoop, SetMasterGain, SetRiser, StartTransition,
)
from djai.engine import Engine  # noqa: E402
from tests.test_deck import make_track  # noqa: E402


def _tone_engine(blocksize: int = 1024) -> Engine:
    e = Engine(blocksize=blocksize, stretch=False)
    base = make_track(n_frames=SAMPLE_RATE * 60, bpm=BPM)
    e.submit(LoadTrack(deck="a", track=LoadedTrack(analysis=base.analysis, audio=_tone(105.0, 60.0)),
                       master=True))
    e.submit(SetGain(deck="a", gain=1.0))
    return e


def _render(e: Engine, blocks: int, frames: int = 1024, at=None) -> np.ndarray:
    out, buf = [], np.zeros((frames, CHANNELS), dtype=np.float32)
    for k in range(blocks):
        if at and k in at:
            e.submit(at[k])
        e.callback(buf, frames, None, None)
        out.append(buf[:, 0].astype(np.float64).copy())
    return np.concatenate(out)


def _d2(x: np.ndarray) -> float:
    return float(np.abs(x[2:] - 2 * x[1:-1] + x[:-2]).max())


TONE_D2 = _d2(_tone(105.0, 2.0)[:, 0].astype(np.float64))


def test_a_beat_jump_is_crossfaded():
    """Was a raw playhead step: up to 0.33 full scale in one sample."""
    x = _render(_tone_engine(), 30, at={10: BeatJump(deck="a", frames=0.37 * BEAT)})
    assert _d2(x[1024:]) < 20 * TONE_D2


def test_a_loop_engaged_away_from_the_playhead_is_crossfaded():
    e = _tone_engine()
    x = _render(e, 30, at={10: SetLoop(deck="a", start_frame=2.3 * BEAT, length_frames=BEAT)})
    assert e.deck_a.loop_active
    assert _d2(x[1024:]) < 20 * TONE_D2


@pytest.mark.parametrize("position", [-0.6, -0.3, -0.08, 0.37, 0.8])
def test_the_filter_engages_and_moves_without_a_step(position):
    """The wet level is ramped per sample and a coefficient change is
    crossfaded; either alone left up to 2,400x the tone's own curvature."""
    steps = {10: SetFilter(deck="a", position=position, resonance=1.0),
             20: SetFilter(deck="a", position=position * 0.7),
             30: SetFilter(deck="a", position=0.0)}
    x = _render(_tone_engine(), 40, at=steps)
    assert _d2(x[1024:]) < 60 * TONE_D2


@pytest.mark.parametrize("frames", [256, 1024, 2048])
def test_riser_and_master_ramps_finish_inside_any_callback_length(frames):
    """Ramps were normalised to the configured block size: a shorter callback
    ramped halfway and stepped the rest (0.1 full scale in one sample)."""
    e = _tone_engine(blocksize=2048)
    blocks = 40 * 1024 // frames
    at = {blocks // 4: SetRiser(gain=0.3, riser=_smooth_riser(SAMPLE_RATE * 4)),
          blocks // 2: SetRiser(gain=0.0),
          blocks // 3: SetMasterGain(gain=0.5),
          2 * blocks // 3: SetMasterGain(gain=1.0)}
    x = _render(e, blocks, frames, at)
    assert _d2(x[2048:]) < 20 * TONE_D2


def test_the_engine_never_restarts_a_running_blend_or_loads_under_it():
    e = _tone_engine()
    other = LoadedTrack(analysis=make_track(n_frames=SAMPLE_RATE * 60, bpm=BPM).analysis,
                        audio=_tone(147.0, 60.0))
    e.submit(LoadTrack(deck="b", track=other, play=False))
    _render(e, 4)
    e.arm_transition_plan("bass_swap", int(64 * BEAT), BPM)
    e.submit(StartTransition(from_deck="a", to_deck="b", total_frames=int(64 * BEAT)))
    _render(e, 40)
    progress, pos_b = e._trans_frames, e.deck_b.position
    e.submit(LoadTrack(deck="b", track=other, start_frame=0, play=True))
    e.submit(StartTransition(from_deck="a", to_deck="b", total_frames=int(8 * BEAT)))
    _render(e, 1)
    assert e._trans_total == int(64 * BEAT) and e._trans_frames > progress
    assert e.deck_b.position > pos_b, "deck b kept its place: nothing was loaded under the blend"


def test_nothing_is_cued_over_the_deck_the_engine_just_handed_over_to(tone_session):
    """Between the engine finishing a blend and the next autopilot tick,
    `live_deck` still names the deck that stopped. A cue or an arm in that gap
    used to load the next track over the music (three routes in the fuzz)."""
    s = tone_session
    assert s.cue_next(0.0, origin="test")
    _play(s, SAMPLE_RATE // 2)
    old, new = s.live_deck, s.cued_deck()
    total = int(4 * BEAT)
    s.engine.arm_transition_plan("cut", total, BPM)
    s.engine.submit(StartTransition(from_deck=old, to_deck=new, total_frames=total))
    s._transition_armed = True
    _play(s, total + SAMPLE_RATE)  # the engine hands over; no tick has run
    assert not s.engine.transition_active and s.live_deck == old
    playing = s.engine.deck(new).track
    s.cue_next(0.0, origin="test")  # before any autopilot tick
    _play(s, SAMPLE_RATE // 4)
    assert s.live_deck == new
    assert s.engine.deck(new).track is playing and s.engine.deck(new).playing
