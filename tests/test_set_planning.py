"""SPEC §6 (Phase 3.3): set planning, narrative critic, persisted set state.

THREADING CONTEXT: main thread (pytest), no audio device. The crate is
synthetic (tests/test_integration.make_analysis); tempos spread over 100-140
BPM so the tempo arc has real gaps to route around.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from djai import cli, config, planner, selector
from djai.deck import CHANNELS, SAMPLE_RATE, LoadedTrack
from djai.setstate import SetState
from tests.test_integration import make_analysis

KEYS = [f"{n}{m}" for n in range(1, 13) for m in "AB"]


def _crate(tmp_path, n: int = 40, seed: int = 3):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        p = tmp_path / f"k{i:02d}.wav"
        p.write_bytes(b"x")
        bpm = round(rng.uniform(100.0, 140.0), 1)
        t = make_analysis(f"k{i:02d}", bpm, rng.choice(KEYS), round(rng.uniform(0.1, 0.9), 3), p)
        object.__setattr__(t, "title", f"Track {i:02d}")
        out.append(t)
    return out


def _tone(seconds: float = 240.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    sig = 0.1 * np.sin(2 * np.pi * 110.0 * t)
    return np.stack([sig, sig], axis=1).astype(np.float32)


TONE = _tone()


def _drive(s, blocks: int) -> None:
    buf = np.zeros((1024, CHANNELS), dtype=np.float32)
    for _ in range(blocks):
        s.scheduler.tick(s.engine.frames_played + 1024)
        s.engine.callback(buf, 1024, None, None)


@pytest.fixture
def planned(tmp_path, monkeypatch):
    crate = _crate(tmp_path)
    monkeypatch.setattr(cli, "load_track", lambda a: LoadedTrack(analysis=a, audio=TONE))
    monkeypatch.setattr(config, "TIME_STRETCH_ENABLED", False)
    s = cli.Session(crate, log_dir=tmp_path / "logs", cache_dir=tmp_path / "cache")
    s.attach_set_state(tmp_path / "state.json")
    yield s
    s.shutdown()


def _events(s, name=None):
    evs = [json.loads(line) for line in s.session_log.path.read_text().splitlines()]
    return [e for e in evs if name is None or e["event"] == name]


def _check_arc(plan, start_bpm=None):
    """Every planned pair meets inside the stretch cap, or is a bridged cue."""
    bpm = start_bpm
    for slot in plan.slots:
        if bpm and slot.tempo != "tempo_bridge":
            assert selector.tempo_path(bpm, slot.track.bpm, travel=True).blendable, slot
            assert abs(slot.stretch_pct) <= config.MAX_STRETCH_RATIO * 100 + 1e-6
            assert abs(slot.ride_pct) <= config.MAX_STRETCH_RATIO * 100 + 1e-6
        bpm = slot.track.bpm * (1 + slot.stretch_pct / 100.0)


def test_a_90_minute_plan_is_critic_scored_and_logged_before_playback(planned, capsys):
    s = planned
    plan = s.replan("planned")
    assert s.start_first_track()
    _drive(s, 4)
    events = [e["event"] for e in _events(s)]
    assert events.index("set_plan") < events.index("track_started"), "planned first"
    logged = _events(s, "set_plan")[0]
    assert logged["critic"]["score"] > 0 and logged["critic"]["source"] == "rules"
    assert logged["critic"]["cap_violations"] == 0
    assert logged["near"] and logged["mid"] and logged["full"], "three horizons"
    end = plan.slots[-1].start_min + planner.minutes_of(plan.slots[-1].track)
    assert end >= 90.0
    _check_arc(plan)
    assert s.history[0].track_id == plan.slots[0].track.track_id, "opens on the plan"
    with capsys.disabled():
        c = plan.critique
        print(f"\n    90-min plan: {len(plan.slots)} tracks to {end:.1f} min, critic "
              f"{c['score']}/10 (arc {c['arc_fit']}, clash {c['clash_rate']}, "
              f"callbacks {c['callbacks']}, risks {c['risks']}), cap violations "
              f"{c['cap_violations']}")


def test_the_autopilot_follows_the_plan(planned):
    s = planned
    s.replan("planned")
    assert s.start_first_track()
    _drive(s, 4)
    _drive(s, 4)
    for _ in range(3):
        expected = next(x for x in s.set_plan.slots if not x.cued).track.track_id
        s._cued = None
        assert s.cue_next(origin="test")
        assert s._cued.analysis.track_id == expected
        _drive(s, 2)
    assert not _events(s, "plan_deviation")


def test_a_cue_added_mid_set_is_honoured_by_the_revised_plan_at_once(planned):
    s = planned
    s.replan("planned")
    assert s.start_first_track()
    _drive(s, 4)
    s._cued = None
    assert s.cue_next(origin="test")
    before = len(_events(s, "set_plan"))
    # A track the plan did not have near the front.
    planned_ids = {x.track.track_id for x in s.set_plan.slots[:6]}
    target = next(t for t in s.crate if t.track_id not in planned_ids
                  and t.track_id not in s.played
                  and selector.tempo_path(s._playing_bpm(), t.bpm, travel=True).blendable)
    s.cue_track(target, "after", 2)
    after = _events(s, "set_plan")
    assert len(after) == before + 1, "revised immediately, not a track later"
    slots = s.set_plan.slots
    assert slots[2].track.track_id == target.track_id and slots[2].cued
    _check_arc(s.set_plan)


def test_an_off_plan_event_revises_the_plan_within_one_track(planned):
    s = planned
    s.replan("planned")
    assert s.start_first_track()
    _drive(s, 4)
    planned_ids = {x.track.track_id for x in s.set_plan.slots}
    stray = next(t for t in s.crate if t.track_id not in planned_ids and t.track_id not in s.played)
    s._forced_next = stray                      # the operator forces something else
    s._cued = None
    assert s.cue_next(origin="test")
    dev = _events(s, "plan_deviation")
    assert dev and stray.title in dev[-1]["action"]
    last = _events(s, "set_plan")[-1]
    assert last["trigger"].startswith("off-plan")
    assert stray.track_id not in {x.track.track_id for x in s.set_plan.slots}


def test_a_cue_beyond_the_cap_is_planned_as_a_tempo_bridge(planned):
    s = planned
    s.replan("planned")
    assert s.start_first_track()
    _drive(s, 4)
    bpm = s._playing_bpm()
    far = next(t for t in s.crate
               if not selector.tempo_path(bpm, t.bpm, travel=True).blendable
               and t.track_id not in s.played)
    s.cue_track(far, "next")
    assert s.cue_queue[0]["status"] == "needs_bridge"
    slot = s.set_plan.slots[0]
    assert slot.track.track_id == far.track_id and slot.tempo == "tempo_bridge"
    assert s.set_plan.critique["cap_violations"] == 0
    assert s.set_plan.critique["tempo_bridges"] == 1


def test_personas_shape_the_arc_and_the_transition_mode(planned):
    s = planned
    means = {}
    for name in ("warmup", "hype"):
        s.set_persona(name)
        plan = s.replan("persona test") if s.set_plan else s.replan("planned")
        means[name] = sum(x.intensity for x in plan.slots[:8]) / 8
    assert means["warmup"] < means["hype"] - 0.1, means
    s.transition_mode = "auto"
    s.set_persona("warmup")
    assert s._mode_for(0.5) == "invisible"
    s.set_persona("hype")
    assert s._mode_for(-0.5) == "showy"


def test_plan_fit_is_a_ranking_factor_in_suggestions(planned):
    s = planned
    s.replan("planned")
    assert s.start_first_track()
    _drive(s, 4)
    nxt = next(x for x in s.set_plan.slots if not x.cued).track
    rows = {r.track.track_id: r for r in s.suggest(40)}
    assert rows[nxt.track_id].plan_fit == 1.0
    assert rows[nxt.track_id].rank <= 3
    saved, s.set_plan = s.set_plan, None
    assert all(r.plan_fit == 0.0 for r in s.suggest(5)), "no plan, no fit"
    s.set_plan = saved


def test_the_narrative_critic_runs_without_a_model_and_takes_an_optional_note(planned):
    import threading

    s = planned
    assert s.intent_engine is None
    plan = s.replan("planned")
    assert plan.critique["source"] == "rules"
    done = threading.Event()

    class Model:
        available = True

        def critique_plan(self, summary):
            done.set()
            return {"score": 7.0, "note": "Good lift into the second peak."}, ""

    s.intent_engine = Model()
    s.replan("with model")
    assert done.wait(5)
    for _ in range(50):
        notes = _events(s, "set_plan_note")
        if notes:
            break
        import time
        time.sleep(0.02)
    assert notes and notes[-1]["llm_score"] == 7.0
    assert s.set_plan.critique["source"] == "rules", "the model's note is advisory"


def test_the_critic_zeroes_a_plan_that_breaks_the_cap(tmp_path):
    crate = _crate(tmp_path, 4)
    a, b = crate[0], crate[1]
    bad = planner.Plan("default", 90.0, [
        planner.Slot(a, 0.0, 0.5, 0.5, "direct", 0.0),
        planner.Slot(b, 4.0, 0.5, 0.5, "direct", 12.0),     # 12% stretch
    ])
    c = planner.critique(bad, planner.PERSONAS["default"])
    assert c["cap_violations"] == 1 and c["score"] == 0.0


def test_set_state_survives_a_process_kill(tmp_path):
    """A real process: plan, cue, queue two requests, then die hard."""
    root = Path(__file__).resolve().parents[1]
    state = tmp_path / "state.json"
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(root)!r})
        from pathlib import Path
        import numpy as np
        from djai import cli, config
        from djai.deck import LoadedTrack
        from tests.test_set_planning import _crate, TONE
        tmp = Path({str(tmp_path)!r})
        crate = _crate(tmp)
        cli.load_track = lambda a: LoadedTrack(analysis=a, audio=TONE)
        config.TIME_STRETCH_ENABLED = False
        s = cli.Session(crate, log_dir=tmp / "logs_a", cache_dir=tmp / "cache")
        s.attach_set_state(Path({str(state)!r}))
        s.set_persona("hype")
        s.replan("planned")
        s.start_first_track()
        s._cued = None
        s.cue_next(origin="test")
        s.cue_song("Track 30", "after", 3)
        s.cue_song("Track 31", "next")
        print("ids", s.history[0].track_id, s.history[1].track_id, flush=True)
        os._exit(9)          # no shutdown, no flush, no clean end
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 9, out.stderr[-2000:]
    played = out.stdout.split("ids", 1)[1].split()
    saved = SetState.load(state)
    assert not saved.ended
    crate = _crate(tmp_path)
    s = cli.Session(crate, log_dir=tmp_path / "logs_b", cache_dir=tmp_path / "cache")
    try:
        assert "Resumed" in s.attach_set_state(state)
        assert [e["title"] for e in s.cue_queue] == ["Track 30", "Track 31"]
        assert s.persona == "hype"
        assert set(played) <= s.played
        assert s.set_state.plan and s.set_state.plan["persona"] == "hype"
        plan = s.replan("planned")                 # cmd_play re-plans on resume
        titles = [x.track.title for x in plan.slots]
        assert "Track 31" in titles[:2] and "Track 30" in titles[:5]
        assert not set(played) & {x.track.track_id for x in plan.slots}
    finally:
        s.shutdown()
