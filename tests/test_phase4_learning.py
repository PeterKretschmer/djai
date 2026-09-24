"""Phase 4: a critic that measures distance, a journey, and what the room says.

THREADING CONTEXT: main thread (pytest). No real renders here except where a
test says so: the critic's search is exercised with a predictor written in the
test, so what is checked is the search, not the DSP under it.
"""

from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pytest

from djai import critic, feedback, selector, transition as tr
from djai.analysis import TrackAnalysis
from tests.test_integration import make_analysis
from tests.test_integration import session as _session_fixture

session = _session_fixture

SAMPLE_RATE = 44100


@pytest.fixture
def profile():
    """A reference distribution with the shape the real one has."""
    return {
        "n_transitions": 26,
        "measures": {
            "length_bars": {"median": 34.0, "iqr": [25.0, 37.0], "n": 26},
            "level_change_db": {"median": 0.07, "iqr": [-2.2, 0.96], "n": 26},
            "level_peak_db": {"median": -1.5, "iqr": [-2.6, -0.7], "n": 26},
            "low_dip_db": {"median": -18.2, "iqr": [-21.3, -12.9], "n": 26},
            "low_bump_db": {"median": 1.2, "iqr": [0.7, 3.2], "n": 26},
        },
    }


INSIDE = {"length_bars": 30.0, "level_change_db": 0.0, "level_peak_db": -1.5,
          "low_dip_db": -18.0, "low_bump_db": 1.0}


# --- the distance -----------------------------------------------------------------


def test_a_transition_inside_the_reference_range_scores_zero(profile):
    score, detail = critic.distance(INSIDE, profile)
    assert score == 0.0
    assert all(d["distance"] == 0.0 for d in detail.values())


def test_distance_grows_with_how_far_outside_it_is(profile):
    near = dict(INSIDE, low_dip_db=-25.0)
    far = dict(INSIDE, low_dip_db=-40.0)
    assert 0 < critic.distance(near, profile)[0] < critic.distance(far, profile)[0]


def test_one_silly_number_cannot_drown_the_rest(profile):
    absurd = dict(INSIDE, low_dip_db=-1e6)
    score, detail = critic.distance(absurd, profile)
    assert detail["low_dip_db"]["distance"] == critic.MAX_DISTANCE
    # One measure at the cap, the rest at zero: the mean over the measures
    # that were actually scored, not over every measure that could be.
    assert score <= critic.MAX_DISTANCE / len(detail) + 1e-9


def test_weights_move_what_the_distance_cares_about(profile):
    measures = dict(INSIDE, low_dip_db=-25.0, level_peak_db=2.0)
    plain, _ = critic.distance(measures, profile)
    heavy, _ = critic.distance(measures, profile, {"low_dip_db": 4.0})
    assert heavy > plain, "weighting the worse measure raises the distance"


def test_no_profile_means_no_opinion():
    assert critic.distance(INSIDE, None) == (0.0, {})


def test_the_verdict_is_numbers_in_lines_of_text(profile):
    """The local model is text-only: it never sees an array."""
    score, detail = critic.distance(dict(INSIDE, low_dip_db=-30.0), profile)
    text = critic.as_text(detail, score)
    lines = text.splitlines()
    assert lines[0].startswith("distance ")
    assert any(line.startswith("low_dip_db ") and "reference" in line for line in lines)
    assert critic.worst(detail) == "low_dip_db"


# --- the search -------------------------------------------------------------------


def test_the_search_walks_toward_the_reference_range(profile):
    """A deliberately long blend, with a predictor that says so."""
    start = dataclasses.replace(tr.preset_params("bass_swap"), length_bars=8.0)

    def predict(params):
        # Only the length matters here, and the reference likes ~30 bars.
        return dict(INSIDE, length_bars=float(params.length_bars))

    result = critic.search(start, profile, predict)
    assert result.improved
    assert result.score < result.start_score
    assert 25.0 <= result.params.length_bars <= 37.0
    assert result.steps and result.steps[0]["field"] == "length_bars"


def test_a_transition_already_inside_the_range_is_left_alone(profile):
    params = dataclasses.replace(tr.preset_params("bass_swap"), length_bars=30.0)
    result = critic.search(params, profile, lambda p: dict(INSIDE, length_bars=float(p.length_bars)))
    assert not result.improved
    assert result.params is params
    assert result.evaluations <= 2, "nothing to fix means almost no work"


def test_the_search_never_proposes_what_the_supervisor_refuses(profile):
    class Refuses:
        def validate_transition_params(self, schema):
            return None, "no"

    start = dataclasses.replace(tr.preset_params("bass_swap"), length_bars=8.0)
    result = critic.search(
        start, profile, lambda p: dict(INSIDE, length_bars=float(p.length_bars)),
        supervisor=Refuses(),
    )
    assert not result.improved and result.params is start


# --- measuring a render -----------------------------------------------------------


def fake_render(blend_gain=1.0, seconds=40.0, blend_s=20.0, low_hz=60.0):
    """A window with a steady tone either side and a quieter blend between."""
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    audio = 0.5 * np.sin(2 * np.pi * low_hz * t) + 0.2 * np.sin(2 * np.pi * 900 * t)
    t0 = int((seconds - blend_s) / 2 * SAMPLE_RATE)
    t1 = t0 + int(blend_s * SAMPLE_RATE)
    audio[t0:t1] *= blend_gain
    mix = np.stack([audio, audio], axis=1).astype(np.float32)
    return types.SimpleNamespace(
        mix=mix, transition_start_frame=t0, transition_frames=t1 - t0,
        bars=24.0, blocksize=2048, bpm_a=128.0, bpm_b=128.0, rate_b=1.0,
        a_start_deck_frame=0.0, b_anchor_output_frame=t0, b_anchor_deck_frame=0.0,
        envelope=[], grid=[], limiter_engaged=False,
    )


def test_a_blend_that_dips_is_measured_as_dipping():
    quiet = critic.measures_from_render(fake_render(blend_gain=0.25))
    steady = critic.measures_from_render(fake_render(blend_gain=1.0))
    assert quiet["low_dip_db"] < -6.0, quiet
    assert steady["low_dip_db"] > quiet["low_dip_db"] + 5.0
    assert quiet["level_peak_db"] < steady["level_peak_db"]


def test_a_render_with_no_steady_audio_is_not_measured():
    render = fake_render(seconds=40.0, blend_s=39.9)
    assert critic.measures_from_render(render) == {}


# --- the selector's description of a track ----------------------------------------


def track(tid, bpm=128.0, cam="8A", energy=0.5, rms=None, features=None, tmp=None):
    analysis = make_analysis(tid, bpm, cam, energy, tmp / f"{tid}.wav")
    if rms is not None:
        object.__setattr__(analysis, "beat_rms", list(rms))
    object.__setattr__(analysis, "intensity_features", features or {
        "percussive": 0.5, "flux": 2.5, "density": 4.0})
    return analysis


def test_groove_reads_the_shape_of_a_bar(tmp_path):
    selector.forget_descriptions()
    four_on_floor = track("straight", rms=[1.0, 0.3, 0.6, 0.3] * 40, tmp=tmp_path)
    offbeat = track("offbeat", rms=[0.3, 1.0, 0.3, 0.6] * 40, tmp=tmp_path)
    same = track("same", rms=[1.0, 0.3, 0.6, 0.3] * 40, tmp=tmp_path)

    assert selector.similarity(four_on_floor, same)["groove"] > 0.95
    assert selector.similarity(four_on_floor, offbeat)["groove"] < 0.9


def test_timbre_uses_the_stem_balance_when_it_is_there(tmp_path, monkeypatch):
    selector.forget_descriptions()
    from djai import stems as stems_mod

    balances = {
        "vocal": {"drums": 0.2, "bass": 0.2, "other": 0.2, "vocals": 0.4},
        "drummy": {"drums": 0.6, "bass": 0.3, "other": 0.08, "vocals": 0.02},
    }
    monkeypatch.setattr(stems_mod, "balance", lambda tid, cache: balances.get(tid))
    a = track("vocal", tmp=tmp_path)
    b = track("drummy", tmp=tmp_path)
    assert selector.timbre_vector(a, tmp_path)[:4] == (0.2, 0.2, 0.2, 0.4)
    assert selector.similarity(a, b, tmp_path)["timbre"] < 0.85


def test_a_track_is_most_like_itself(tmp_path):
    selector.forget_descriptions()
    a = track("one", tmp=tmp_path)
    assert selector.similarity(a, a) == {"timbre": 1.0, "groove": 1.0,
                                         "neighbourhood": 1.0}


# --- planning a journey ------------------------------------------------------------


@pytest.fixture
def crate(tmp_path):
    selector.forget_descriptions()
    # A ladder of tempos: 120 up to 132, plus one outlier nothing can follow.
    out = [track(f"t{int(bpm)}", bpm=bpm, tmp=tmp_path)
           for bpm in (120.0, 123.0, 126.0, 129.0, 132.0)]
    out.append(track("island", bpm=200.0, tmp=tmp_path))
    return out


def test_a_journey_plans_several_tracks_ahead(crate):
    plan = selector.plan_journey(crate, crate[0], set(), steps=3)
    assert len(plan.steps) == 3
    ids = [c.track.track_id for c in plan.steps]
    assert len(set(ids)) == 3, "a plan does not play the same track twice"
    assert plan.first is plan.steps[0]
    assert "journey" in plan.reason()


def test_a_journey_does_not_walk_into_a_dead_end(crate):
    """The point of looking ahead: a pick with nothing to follow it loses to
    one that keeps the set going."""
    plan = selector.plan_journey(crate, crate[0], set(), steps=3)
    assert plan.first.track.track_id != "island", plan.reason()


def test_a_journey_of_one_step_is_the_old_behaviour(crate):
    plan = selector.plan_journey(crate, crate[0], set(), steps=1)
    greedy = selector.select_next(crate, crate[0], set())
    assert plan.first.track.track_id == greedy.track.track_id


def test_played_tracks_stay_out_of_the_plan(crate):
    played = {"t123", "t126"}
    plan = selector.plan_journey(crate, crate[0], played, steps=3)
    assert not ({c.track.track_id for c in plan.steps} & played)


# --- feedback ----------------------------------------------------------------------


class Log:
    def __init__(self):
        self.rows = []

    def write(self, event, **fields):
        self.rows.append(dict(event=event, **fields))
        return self.rows[-1]


def test_a_verdict_is_written_with_everything_needed_to_learn_from_it():
    log = Log()
    feedback.record(log, feedback.WORKED, tracks=("a", "b"), style="bass_swap",
                    critic_score=0.4, worst_measure="low_dip_db",
                    similarity={"timbre": 0.9})
    row = log.rows[0]
    assert row["event"] == "operator_feedback" and row["verdict"] == "worked"
    assert row["track_pair"] == ["a", "b"] and row["worst_measure"] == "low_dip_db"
    assert row["critic_score"] == 0.4 and row["similarity"] == {"timbre": 0.9}


def test_an_unknown_verdict_is_refused():
    with pytest.raises(ValueError, match="verdict must be"):
        feedback.record(Log(), "meh")


def verdicts(worked, didnt, worst="low_dip_db", sim_worked=0.9, sim_didnt=0.4):
    rows = []
    for _ in range(worked):
        rows.append({"event": "operator_feedback", "verdict": "worked",
                     "worst_measure": "", "similarity": {"timbre": sim_worked}})
    for _ in range(didnt):
        rows.append({"event": "operator_feedback", "verdict": "didnt",
                     "worst_measure": worst, "similarity": {"timbre": sim_didnt}})
    return rows


def test_nothing_is_learned_from_too_little(tmp_path):
    learned = feedback.learn(verdicts(1, 1))
    assert learned["critic"] == {} and learned["selector"] == {}
    assert "needed before" in learned["summary"]["learning"]


def test_the_measure_blamed_for_bad_transitions_gains_weight():
    learned = feedback.learn(verdicts(3, 4))
    assert learned["critic"]["low_dip_db"] > 1.0
    assert learned["summary"]["didnt"] == 4


def test_a_similarity_that_separates_good_from_bad_gains_weight():
    learned = feedback.learn(verdicts(4, 4, sim_worked=0.9, sim_didnt=0.3))
    assert learned["selector"]["timbre"] > 1.0
    flat = feedback.learn(verdicts(4, 4, sim_worked=0.6, sim_didnt=0.6))
    assert flat["selector"]["timbre"] == pytest.approx(1.0, abs=0.01)


def test_weights_cannot_run_away():
    learned = feedback.learn(verdicts(0, 60))
    lo, hi = feedback.WEIGHT_RANGE
    assert all(lo <= w <= hi for w in learned["critic"].values())


def test_overrides_and_skips_are_counted_as_evidence():
    rows = verdicts(3, 3) + [
        {"event": "override", "trigger": "ui:button"},
        {"event": "forced_next", "track": "something else"},
        {"event": "commands_skipped", "action": "dropped 2"},
    ]
    summary = feedback.learn(rows)["summary"]
    assert summary["overrides"] == 3


def test_reading_real_logs_never_raises(tmp_path):
    (tmp_path / "session_20260101_000000.jsonl").write_text(
        '{"event": "operator_feedback", "verdict": "worked"}\nnot json\n',
        encoding="utf-8",
    )
    learned = feedback.weights_for(tmp_path)
    assert learned["summary"]["verdicts"] == 1

# --- in a running session ----------------------------------------------------------


def test_the_button_records_a_verdict_and_reloads_the_weights(session, tmp_path):
    """The operator's verdict reaches the log, and the weights are re-learned
    then and there rather than next time the program starts."""
    from djai.ui_server import UIServer
    from tests.test_integration import log_events

    ui = UIServer(session, intent_engine=None)
    reply = ui.handle_action({"type": "feedback", "verdict": "worked"})
    assert reply["ok"], reply

    rows = [e for e in log_events(session) if e["event"] == "operator_feedback"]
    assert rows and rows[-1]["verdict"] == "worked"
    assert "track_pair" in rows[-1]
    assert isinstance(session.feedback_summary, dict)
    assert session.feedback_summary.get("verdicts", 0) >= 1


def test_an_unknown_verdict_from_the_page_is_refused(session):
    from djai.ui_server import UIServer

    ui = UIServer(session, intent_engine=None)
    assert ui.handle_action({"type": "feedback", "verdict": "sideways"})["ok"] is False


def test_a_cue_plans_a_journey_and_logs_where_it_was_heading(session):
    from tests.test_integration import log_events, seek_near_mix_out

    seek_near_mix_out(session, bars_before=40.0)
    assert session.cue_next() is True

    assert session.journey is not None and session.journey.steps
    assert session.journey.first.track.track_id == session._cued.analysis.track_id
    logged = [e for e in log_events(session) if e["event"] == "journey"]
    if len(session.journey.steps) > 1:
        assert logged and logged[-1]["ahead"], "the plan beyond the next track is logged"
