"""Phase 5: a tempo journey instead of an 8% wall.

THREADING CONTEXT: main thread (pytest). Real Sessions on synthetic audio, no
device and no background threads.

The rule this replaces refused anything more than 8% away and cut instead. A
set that wants to travel from 100 BPM to 128 cannot do that in one mix, and
should not have to: it rides the tempo across several tracks, and where a track
sits half or double the playing tempo it needs no stretching at all.
"""

from __future__ import annotations

import pytest

from djai import cli, config, selector
from djai.commands import SetPitch, StartTransition
from djai.deck import LoadedTrack
from tests.test_engine import drive, loud_track
from tests.test_integration import log_events, make_analysis, seek_near_mix_out

BLOCK = 512


# --- how two tempos meet ----------------------------------------------------------


def test_a_close_tempo_is_a_straight_mix():
    path = selector.tempo_path(128.0, 130.0)
    assert path.technique == "direct"
    assert path.rate_b == pytest.approx(128.0 / 130.0)
    assert path.ride_percent == 0.0 and path.blendable


def test_a_gap_past_the_stretch_range_is_ridden_not_cut():
    path = selector.tempo_path(128.0, 140.0)
    assert path.technique == "ride"
    assert 0 < path.ride_percent <= selector.RIDE_PERCENT
    # What the incoming deck is asked to stretch stays inside the limit: the
    # ride covers the rest of the gap.
    assert abs(path.rate_b - 1.0) <= config.MAX_STRETCH_RATIO + 1e-9
    ridden = 128.0 * (1.0 + path.ride_percent)
    assert 140.0 * path.rate_b == pytest.approx(ridden, rel=1e-6)


def test_a_bigger_ride_is_given_more_bars():
    # Both of these need a ride: the deck's own stretch cannot reach either.
    small = selector.tempo_path(128.0, 141.0)
    large = selector.tempo_path(128.0, 147.0)
    assert small.technique == large.technique == "ride"
    assert large.ride_percent > small.ride_percent
    assert large.bars >= small.bars
    assert selector.RIDE_BARS_MIN <= small.bars <= selector.RIDE_BARS_MAX


def test_half_and_double_time_need_no_stretching_at_all():
    """87 under 174 is not a 50% tempo error, it is every other beat."""
    double = selector.tempo_path(174.0, 87.0)
    assert double.technique == "double" and double.ratio == 2.0
    assert double.rate_b == pytest.approx(1.0, abs=1e-6)

    half = selector.tempo_path(87.0, 174.0)
    assert half.technique == "half" and half.ratio == 0.5
    assert half.rate_b == pytest.approx(1.0, abs=1e-6)


def test_a_tempo_with_no_path_says_so():
    path = selector.tempo_path(100.0, 128.0)
    assert path.technique == "none" and not path.blendable
    assert "no blend path" in path.describe()


def test_the_technique_is_visible_in_the_reason():
    path = selector.tempo_path(128.0, 140.0)
    assert "ride" in path.describe() and "bars" in path.describe()


# --- the journey ------------------------------------------------------------------


def ladder(tmp_path, tempos):
    """A crate that is a ladder of tempos, with files the supervisor can see."""
    out = []
    for bpm in tempos:
        path = tmp_path / f"{int(bpm)}.wav"
        path.write_bytes(b"placeholder")
        out.append(make_analysis(f"t{int(bpm)}", float(bpm), "8A", 0.5, path))
    return out


def test_a_set_travels_from_100_to_128_without_a_single_cut(tmp_path):
    """The acceptance case: a crate with a gap in it, crossed by riding."""
    selector.forget_descriptions()
    crate = ladder(tmp_path, [100, 104, 107, 119, 123, 128])
    start = crate[0]

    plan = selector.plan_journey(crate, start, set(), steps=5, target_bpm=128.0)

    assert plan.steps, "no journey at all"
    assert all(step.tempo_path.blendable for step in plan.steps), plan.reason()

    tempos = [start.bpm] + [s.track.bpm for s in plan.steps]
    reached = next((i for i, bpm in enumerate(tempos) if bpm >= 127.0), None)
    assert reached is not None, f"never got to 128: {tempos}"
    # Climbing is what matters up to the point it arrives; what it does with
    # the tracks left over after that is a different question.
    climb = tempos[: reached + 1]
    assert climb == sorted(climb), f"the journey must climb: {climb}"


def test_the_journey_prefers_a_step_that_keeps_going(tmp_path):
    selector.forget_descriptions()
    crate = ladder(tmp_path, [120, 124, 128])
    plan = selector.plan_journey(crate, crate[0], set(), steps=2, target_bpm=128.0)
    assert plan.first.track.bpm == 124.0, plan.reason()


def test_without_a_target_the_journey_still_plans(tmp_path):
    selector.forget_descriptions()
    crate = ladder(tmp_path, [120, 124, 128])
    plan = selector.plan_journey(crate, crate[0], set(), steps=2)
    assert plan.steps


def test_a_track_that_cannot_be_reached_is_not_offered(tmp_path):
    selector.forget_descriptions()
    crate = ladder(tmp_path, [100, 128])
    ranked = selector.rank_candidates(crate, crate[0], set())
    assert ranked == [], "128 is not reachable from 100 in one mix"


def test_half_time_makes_a_distant_track_reachable(tmp_path):
    selector.forget_descriptions()
    crate = ladder(tmp_path, [174, 87])
    ranked = selector.rank_candidates(crate, crate[0], set())
    assert ranked and ranked[0].track.bpm == 87.0
    assert ranked[0].tempo_path.technique == "double"


# --- in a session -----------------------------------------------------------------


def build_session(tmp_path, monkeypatch, tempos):
    selector.forget_descriptions()
    crate = ladder(tmp_path, tempos)

    def fake_load(analysis):
        return LoadedTrack(analysis=analysis,
                           audio=loud_track(300.0, analysis.bpm, 240.0).audio)

    monkeypatch.setattr(cli, "load_track", fake_load)
    monkeypatch.setattr("djai.config.TIME_STRETCH_ENABLED", False)
    s = cli.Session(crate, log_dir=tmp_path / "logs")
    s.start_first_track()
    drive(s.engine, 4, BLOCK)
    return s


@pytest.fixture
def riding_session(tmp_path, monkeypatch):
    """A crate whose next track is a ride away, not a cut away."""
    s = build_session(tmp_path, monkeypatch, [128, 140])
    yield s
    s.shutdown()


def events(s, name):
    return [e for e in log_events(s) if e["event"] == name]


def test_the_cue_plans_a_ride_and_schedules_it(riding_session):
    s = riding_session
    seek_near_mix_out(s, bars_before=48.0)
    assert s.cue_next() is True

    assert s.tempo_plan is not None and s.tempo_plan.technique == "ride"
    assert events(s, "tempo_path"), "the path is logged"
    assert events(s, "tempo_ride"), "and the ride is scheduled"
    pitches = [c for c in s.scheduler.pending() if isinstance(c, SetPitch)]
    assert pitches, "no pitch steps queued"
    assert all(p.deck == s.live_deck for p in pitches)
    rates = [p.rate for p in pitches]
    assert rates == sorted(rates), "a ride walks one way"
    assert max(rates) <= 1.0 + config.MAX_STRETCH_RATIO + 1e-9


def test_the_ride_actually_moves_the_deck(riding_session):
    s = riding_session
    seek_near_mix_out(s, bars_before=48.0)
    s.cue_next()
    before = s.engine.deck(s.live_deck).rate

    for _ in range(600):
        s.scheduler.tick(s.engine.frames_played)
        drive(s.engine, 16, BLOCK)
        if s.engine.deck(s.live_deck).rate > before + 1e-4:
            break

    assert s.engine.deck(s.live_deck).rate > before, "the deck never moved"


def test_the_incoming_deck_is_cued_at_the_ridden_rate(riding_session):
    s = riding_session
    seek_near_mix_out(s, bars_before=48.0)
    s.cue_next()
    drive(s.engine, 4, BLOCK)
    idle = s.engine.deck(s.cued_deck())
    assert idle.track is not None
    assert idle.rate == pytest.approx(s.tempo_plan.rate_b, rel=1e-6)


def test_a_reachable_tempo_is_never_cut(riding_session):
    s = riding_session
    seek_near_mix_out(s, bars_before=48.0)
    s.cue_next()
    drive(s.engine, 2, BLOCK)
    assert s.arm_transition() is not None

    swap = next(c for c in s.scheduler.pending() if isinstance(c, StartTransition))
    assert swap.total_frames > s.engine.blocksize, "a blend, not a cut"
    assert s.last_transition_choice[0] != "cut"


def test_a_metric_pair_plays_at_its_own_speed_and_the_supervisor_agrees(
    tmp_path, monkeypatch
):
    """The half-time case end to end: no stretching, and no drift war."""
    s = build_session(tmp_path, monkeypatch, [174, 87])
    try:
        seek_near_mix_out(s, bars_before=40.0)
        assert s.cue_next() is True
        assert s.tempo_plan.technique == "double"

        drive(s.engine, 4, BLOCK)
        idle = s.engine.deck(s.cued_deck())
        assert idle.rate == pytest.approx(1.0, abs=1e-6), "half-time needs no stretch"
        assert idle.metric_ratio == 2.0
        # The supervisor must not try to pull it up to the master's number.
        assert s.supervisor.nominal_rate(idle) == pytest.approx(1.0, abs=1e-6)
        assert s.arm_transition() is not None, "a metric pair is blendable"
        assert events(s, "tempo_path")
    finally:
        s.shutdown()


# --- steering the set --------------------------------------------------------------


def test_the_operator_can_say_where_the_tempo_is_going(riding_session):
    s = riding_session
    assert "no tempo target" in cli.handle_override(s, "tempo").lower()

    reply = cli.handle_override(s, "tempo 128")
    assert s.tempo_target == 128.0 and "128" in reply
    assert [e for e in log_events(s) if e["event"] == "tempo_target"]

    assert "128" in cli.handle_override(s, "tempo")
    assert "Tempo target cleared." == cli.handle_override(s, "tempo off")
    assert s.tempo_target is None


def test_nonsense_and_impossible_targets_are_refused(riding_session):
    s = riding_session
    assert "not `tempo" in cli.handle_override(s, "tempo faster")
    assert "outside 60-200" in cli.handle_override(s, "tempo 400")
    assert s.tempo_target is None


def test_the_target_reaches_the_planner(riding_session, monkeypatch):
    """The cue plans toward the target, not just toward the best next track."""
    s = riding_session
    s.tempo_target = 140.0
    seen = {}
    real = cli.plan_journey

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "plan_journey", spy)
    seek_near_mix_out(s, bars_before=48.0)
    s.cue_next()
    assert seen.get("target_bpm") == 140.0


# --- the set has to actually move ---------------------------------------------------


def test_travelling_lets_the_incoming_track_keep_its_own_tempo():
    """A journey that pulls every track back to the opening tempo is not one."""
    hold = selector.tempo_path(128.0, 132.0)
    travel = selector.tempo_path(128.0, 132.0, travel=True)
    # Holding beat-matches the new track to what is playing; travelling rides
    # the playing deck up to it and leaves it at its own speed.
    assert hold.rate_b == pytest.approx(128.0 / 132.0)
    assert travel.rate_b == pytest.approx(1.0, abs=1e-6)
    assert travel.ride_percent > 0
    assert 132.0 * travel.rate_b == pytest.approx(132.0, abs=1e-6)


def test_a_gap_too_big_to_ride_still_meets_in_the_middle():
    path = selector.tempo_path(107.0, 120.0, travel=True)
    assert path.blendable and path.technique == "ride"
    assert path.ride_percent == pytest.approx(selector.RIDE_PERCENT, abs=1e-9)
    assert abs(path.rate_b - 1.0) <= config.MAX_STRETCH_RATIO + 1e-9
    # Part of the way there, which is what the next mix continues from.
    landed = 120.0 * path.rate_b
    assert 107.0 < landed < 120.0


def test_the_plan_climbs_from_the_tempo_it_will_be_playing_at(tmp_path):
    """Each step is planned from where the one before it left the set."""
    selector.forget_descriptions()
    crate = ladder(tmp_path, [100, 104, 107, 119, 123, 128])
    plan = selector.plan_journey(crate, crate[0], set(), steps=5, target_bpm=128.0)
    landed = [
        step.track.bpm * (step.tempo_path.rate_b or 1.0) for step in plan.steps
    ]
    assert landed == sorted(landed), f"the played tempo must climb: {landed}"
    assert landed[-1] > 120.0, f"the set never got near 128: {landed}"


def test_the_beat_clock_follows_the_deck_that_is_now_the_mix(riding_session):
    """Without this the night is pinned to whatever it opened at."""
    s = riding_session
    s.tempo_target = 140.0
    opened_at = s.engine.master_bpm
    seek_near_mix_out(s, bars_before=48.0)
    assert s.cue_next() is True
    drive(s.engine, 2, BLOCK)  # let the queued load reach the deck

    # Let the ride land first, as it does in a set: it is scheduled a phrase
    # or more before the transition it was planned for.
    for _ in range(6000):
        s.scheduler.tick(s.engine.frames_played)
        drive(s.engine, 16, BLOCK)
        s._autopilot_tick()
        if not s._ride_steps:
            break
    assert s.engine.master_bpm > opened_at + 1.0, "the clock did not follow the ride"
    assert s.arm_transition() is not None

    for _ in range(4000):
        s.scheduler.tick(s.engine.frames_played)
        drive(s.engine, 16, BLOCK)
        s._autopilot_tick()
        if s.live_deck != "a":
            break

    assert s.live_deck != "a", "the transition never finished"
    live = s.engine.deck(s.live_deck)
    effective = live.track.analysis.bpm * live.rate
    assert s.engine.master_bpm == pytest.approx(effective, rel=1e-6)
    assert s.engine.master_bpm > opened_at + 1.0, "the set did not travel"
    # The ride is what moved it, and each landed step is in the log.
    assert events(s, "tempo_ride_step")


def test_the_clock_follows_the_new_track_after_a_cut(tmp_path, monkeypatch):
    """A cut leaves the clock counting a track nobody is playing any more.

    That is what pinned a soak to the tempo it opened at: every later track was
    planned and stretched against a number that had not been true for an hour.
    """
    s = build_session(tmp_path, monkeypatch, [128, 100])
    try:
        opened_at = s.engine.master_bpm
        seek_near_mix_out(s, bars_before=8.0)
        assert s.cue_next() is True
        assert s.tempo_plan.technique == "none", "28 BPM apart has no blend path"
        drive(s.engine, 2, BLOCK)
        assert s.arm_transition() is not None

        for _ in range(9000):
            s.scheduler.tick(s.engine.frames_played)
            drive(s.engine, 16, BLOCK)
            s._autopilot_tick()
            if s.live_deck != "a":
                break

        assert s.live_deck != "a", "the cut never happened"
        live = s.engine.deck(s.live_deck)
        landed = live.track.analysis.bpm * live.rate
        assert landed != pytest.approx(opened_at, rel=1e-3), "same tempo, no test"
        assert s.engine.master_bpm == pytest.approx(landed, rel=1e-3)
        handover = events(s, "tempo_handover")
        assert handover, "the hand-over is logged"
        assert handover[0]["to_bpm"] == pytest.approx(landed, rel=1e-3)
    finally:
        s.shutdown()


def test_the_next_cue_is_planned_from_the_ridden_tempo(riding_session):
    """The bug this cost a soak to find: planning from the printed tempo."""
    s = riding_session
    live = s.engine.deck(s.live_deck)
    live.rate = 1.05  # as a ride would have left it
    s.engine.hand_master_to(s.live_deck)
    printed = live.track.analysis.bpm
    assert s._playing_bpm() == pytest.approx(printed * 1.05, rel=1e-6)
    # And the path the arm will use is measured against that, not the label.
    path = s.tempo_path_to(live.track.analysis)
    assert path.rate_b == pytest.approx(1.05, rel=1e-3)


def test_a_ride_does_not_outlive_the_deck_it_was_riding(riding_session):
    """Its steps would pitch a stopped deck, and move the clock to match."""
    s = riding_session
    s.tempo_target = 140.0
    seek_near_mix_out(s, bars_before=48.0)
    s.cue_next()
    assert s._ride_steps, "no ride to leave behind"
    drive(s.engine, 2, BLOCK)
    s.arm_transition()

    for _ in range(6000):
        s.scheduler.tick(s.engine.frames_played)
        drive(s.engine, 16, BLOCK)
        s._autopilot_tick()
        if s.live_deck != "a":
            break

    assert s.live_deck != "a", "the transition never finished"
    assert not s._ride_steps, "the old ride is still pending"
    assert not [
        c for c in s.scheduler.pending()
        if getattr(c, "origin", "") == "tempo_ride"
    ]


def test_the_ride_is_measured_at_the_speed_it_will_be_running(riding_session):
    """A ride speeds the deck up, so its mix-out arrives sooner than it looks."""
    s = riding_session
    s.tempo_target = 140.0
    # Just enough room at today's rate, not enough once the deck is riding.
    seek_near_mix_out(s, bars_before=selector.RIDE_MARGIN_BARS + 2.0)
    s.cue_next()
    scheduled = [e for e in events(s, "tempo_ride") if "pitch step" in e["action"]]
    for event in scheduled:
        bars = float(event["action"].split(" over ")[1].split(" bars")[0])
        assert bars <= 2.0, f"rode {bars} bars into a blend"
