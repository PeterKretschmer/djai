"""Transitions designed in a parameter space, with the model kept on a leash.

THREADING CONTEXT: main thread (pytest). The design worker is a real thread;
tests join it through the session's own non-blocking accessor.

The division these tests exist to defend: the model owns WHAT happens, the
engine owns WHEN. Nothing the model returns can carry a time, and nothing it
returns runs without the supervisor agreeing to it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from djai import transition as tr
from djai.commands import IMMEDIATE, StartTransition
from djai.deck import SAMPLE_RATE
from tests.test_engine import drive
from tests.test_integration import log_events
from tests.test_integration import session as _session_fixture

session = _session_fixture

BLOCK = 512

GOOD = {
    "length_bars": 16,
    "curve": "s_curve",
    "low_swap_bar": 8,
    "low_swap_bars": 2,
    "deck_a_high_rolloff": True,
    "deck_b_low_delay_bars": 4,
    "echo_bars": 2,
    "filter_sweep": "hp_out",
    "filter_resonance": 0.4,
    "entry_point": "mix_in",
    "intensity": 0.6,
    # The high-energy set, all off. Present because every field is required:
    # an omission is a rejection, not a default.
    "loop_out_bars": 0,
    "loop_halving": False,
    "beat_repeat_division": 0,
    "backspin_bars": 0,
    "double_drop_bars": 0,
    "align_mode": "phrase",
    # Effects (Phase 3), all off.
    "reverb_bars": 0,
    "brake_beats": 0,
    "riser_bars": 0,
    "vocal_aware": True,
}


class FakeModel:
    """Stands in for Ollama. Returns whatever the test wants, or fails."""

    def __init__(self, raw=None, reason="", available=True, calls=None):
        self.raw = raw
        self.reason = reason
        self.available = available
        self.calls = calls if calls is not None else []

    def design_transition(self, context, timeout_s=None):
        self.calls.append(context)
        if self.raw is None:
            return None, self.reason or "unavailable"
        return dict(self.raw), ""


# --- the parameter space ------------------------------------------------------


def test_every_preset_is_a_point_in_the_space():
    for style in tr.STYLES:
        p = tr.preset_params(style)
        assert p.name == style
        assert p.length_bars > 0
        assert p.curve in tr.CURVES
        assert p.filter_sweep in tr.FILTER_SWEEPS
        assert tr.INTENSITY_RANGE[0] <= p.intensity <= tr.INTENSITY_RANGE[1]
        assert p.echo_bars in range(tr.ECHO_BARS_RANGE[1] + 1)


def test_the_model_facing_length_floor_is_stricter_than_the_presets():
    """`echo_out` is a two-bar move, and the schema's floor is four.

    The range constrains what a MODEL may ask for, not what the rule-based
    path may do. Left as it is deliberately: `echo_out` is a signed-off style
    and stretching it to fit the schema would change how it sounds. The
    consequence, which is real, is that a model cannot reproduce `echo_out`
    exactly -- the shortest echo it can design is four bars.
    """
    assert tr.preset_params("echo_out").length_bars < tr.LENGTH_BARS_RANGE[0]
    for style in ("bass_swap", "cut", "filter_sweep"):
        p = tr.preset_params(style)
        lo, hi = tr.LENGTH_BARS_RANGE
        assert lo <= p.length_bars <= hi, f"{style} is outside the space"


def test_the_schema_a_model_returns_has_exactly_the_documented_fields():
    schema = tr.preset_params("bass_swap").to_schema()
    assert set(schema) == {
        "length_bars", "curve", "low_swap_bar", "low_swap_bars",
        "deck_a_high_rolloff", "deck_b_low_delay_bars", "echo_bars",
        "filter_sweep", "filter_resonance", "entry_point", "intensity",
        "loop_out_bars", "loop_halving", "beat_repeat_division",
        "backspin_bars", "double_drop_bars", "align_mode",
        "reverb_bars", "brake_beats", "riser_bars", "vocal_aware",
    }
    # Provenance is ours, not the model's.
    assert "name" not in schema


def test_no_field_in_the_space_can_carry_a_time():
    """The structural guarantee: there is nowhere to put a sample position.

    Every field is either a shape, a flag, or a count of bars measured from
    the start of the transition. None of them is a position in a track.
    """
    schema = tr.preset_params("bass_swap").to_schema()
    for field in schema:
        assert not any(
            word in field for word in ("frame", "sample", "second", "ms", "time")
        ), f"{field} looks like it could carry absolute timing"


def test_every_curve_builds_a_usable_envelope():
    for curve in tr.CURVES:
        p = tr.TransitionParams(length_bars=16, curve=curve, low_swap_bar=8)
        env = tr.build_envelope_from_params(p, SAMPLE_RATE * 8, BLOCK)
        assert np.isfinite(env).all()
        assert env.shape[1] == tr.N_COLS
        assert env[0, tr.FROM_GAIN] > 0.9, f"{curve} must start with A up"
        assert env[-1, tr.FROM_GAIN] == pytest.approx(0.0)
        assert env[-1, tr.TO_GAIN] == pytest.approx(1.0)
        assert float(env[:, tr.FROM_GAIN].max()) <= 1.001
        assert float(env[:, tr.TO_GAIN].max()) <= 1.001


def test_the_curves_are_actually_different_from_each_other():
    shapes = {}
    for curve in tr.CURVES:
        p = tr.TransitionParams(length_bars=16, curve=curve, low_swap_bar=8)
        env = tr.build_envelope_from_params(p, SAMPLE_RATE * 8, BLOCK)
        shapes[curve] = env[:, tr.TO_GAIN].copy()
    names = list(shapes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = float(np.max(np.abs(shapes[names[i]] - shapes[names[j]])))
            assert diff > 0.02, f"{names[i]} and {names[j]} are the same shape"


def test_the_low_swap_happens_where_it_was_asked_to():
    p = tr.TransitionParams(length_bars=16, curve="linear",
                            low_swap_bar=12, low_swap_bars=1)
    total = SAMPLE_RATE * 16
    env = tr.build_envelope_from_params(p, total, BLOCK)
    rows = env.shape[0] - 1

    def at_bar(bar):
        return env[min(rows - 1, int(rows * bar / 16.0))]

    assert at_bar(4)[tr.FROM_LOW] > 0.9, "A still has the bass before the swap"
    assert at_bar(4)[tr.TO_LOW] < 0.1
    assert at_bar(15)[tr.FROM_LOW] < 0.1, "and has lost it after"
    assert at_bar(15)[tr.TO_LOW] > 0.9


def test_holding_the_incoming_bass_out_actually_holds_it_out():
    p = tr.TransitionParams(length_bars=16, curve="linear",
                            low_swap_bar=0, deck_b_low_delay_bars=8)
    env = tr.build_envelope_from_params(p, SAMPLE_RATE * 16, BLOCK)
    rows = env.shape[0] - 1
    early = env[int(rows * 4 / 16.0)]
    late = env[int(rows * 12 / 16.0)]
    assert early[tr.TO_LOW] < 0.1, "B's bass must wait"
    assert late[tr.TO_LOW] > 0.9, "and then arrive"


def test_echo_bars_open_the_send_and_close_it_again():
    p = tr.TransitionParams(length_bars=16, curve="linear", echo_bars=4,
                            intensity=1.0)
    env = tr.build_envelope_from_params(p, SAMPLE_RATE * 16, BLOCK)
    send = env[:, tr.ECHO_SEND]
    assert send.max() > 0.3, "the send has to open"
    assert send[0] == pytest.approx(0.0), "and not at the start"
    assert send[-1] == pytest.approx(0.0), "and close before the end"


def test_no_echo_means_the_delay_line_is_never_touched():
    p = tr.TransitionParams(length_bars=16, echo_bars=0)
    env = tr.build_envelope_from_params(p, SAMPLE_RATE * 8, BLOCK)
    assert float(env[:, tr.ECHO_SEND].max()) == 0.0


def test_intensity_scales_depth_without_touching_length():
    total = SAMPLE_RATE * 16
    soft = tr.build_envelope_from_params(
        tr.TransitionParams(length_bars=16, filter_sweep="hp_out", intensity=0.2),
        total, BLOCK)
    hard = tr.build_envelope_from_params(
        tr.TransitionParams(length_bars=16, filter_sweep="hp_out", intensity=1.0),
        total, BLOCK)
    assert soft.shape == hard.shape, "intensity must not change the length"
    assert soft[:, tr.FROM_FILTER].max() < hard[:, tr.FROM_FILTER].max()


def test_a_designed_transition_renders_real_audio(session):
    """End to end: params -> envelope -> engine -> samples out.

    Drives the actual callback rather than inspecting the array, so this also
    covers the limiter sitting downstream of whatever a design asks for.
    """
    engine = session.engine
    engine.deck("b").attach(engine.deck("a").track, 0)

    params = tr.TransitionParams(
        length_bars=12, curve="s_curve", low_swap_bar=6, low_swap_bars=2,
        deck_a_high_rolloff=True, deck_b_low_delay_bars=2, echo_bars=2,
        filter_sweep="hp_out", intensity=0.8, name="llm",
    )
    total = SAMPLE_RATE * 6
    assert engine.arm_transition_params(params, total, 125.0) == "llm"
    engine.submit(
        StartTransition(
            from_deck="a", to_deck="b", total_frames=total,
            execute_at=IMMEDIATE, origin="test",
        )
    )

    out = drive(engine, total // BLOCK + 40, BLOCK)
    assert np.isfinite(out).all(), "the render produced non-finite samples"
    assert float(np.max(np.abs(out))) <= 0.99, "Part 1's ceiling still applies"
    assert float(np.max(np.abs(out))) > 0.01, "it produced no audio at all"
    assert not engine.transition_active, "it must finish"
    assert engine.deck_state("b").gain == pytest.approx(1.0, abs=0.05)
    for band in engine.deck_state("b").eq:
        assert band == pytest.approx(1.0, abs=0.05), "B must be handed a clean EQ"


def test_every_curve_renders_without_clipping(session):
    """Whatever shape is designed, the output stays under the ceiling."""
    engine = session.engine
    engine.deck("b").attach(engine.deck("a").track, 0)
    total = SAMPLE_RATE * 2

    for curve in tr.CURVES:
        params = tr.TransitionParams(
            length_bars=8, curve=curve, low_swap_bar=4, intensity=1.0, name="llm"
        )
        engine.arm_transition_params(params, total, 125.0)
        engine.submit(StartTransition(from_deck="a", to_deck="b",
                                      total_frames=total, execute_at=IMMEDIATE,
                                      origin="test"))
        out = drive(engine, total // BLOCK + 8, BLOCK)
        peak = float(np.max(np.abs(out)))
        assert peak <= 0.99, f"{curve} peaked at {peak:.4f}"
        assert engine.limiter_clips == 0


# --- validation ---------------------------------------------------------------


def test_a_well_formed_design_is_accepted(session):
    params, why = session.supervisor.validate_transition_params(GOOD, session.crate[1])
    assert params is not None and why == ""
    assert params.name == "llm"
    assert params.length_bars == 16
    assert params.curve == "s_curve"


@pytest.mark.parametrize(
    "field,value",
    [
        ("length_bars", 2),            # below the floor
        ("length_bars", 64),           # above the ceiling
        ("length_bars", "sixteen"),    # not a number
        ("curve", "wobble"),           # invented
        ("low_swap_bars", 0),
        ("low_swap_bars", 99),
        ("deck_b_low_delay_bars", -1),
        ("deck_b_low_delay_bars", 40),
        ("echo_bars", 9),
        ("filter_sweep", "phaser"),
        ("intensity", 1.7),
        ("intensity", -0.2),
        ("deck_a_high_rolloff", "yes"),
        ("low_swap_bar", 40),          # outside the transition's own length
        ("low_swap_bar", "half"),
    ],
)
def test_every_out_of_range_field_is_rejected(session, field, value):
    raw = dict(GOOD)
    raw[field] = value
    params, why = session.supervisor.validate_transition_params(raw, session.crate[1])
    assert params is None, f"{field}={value!r} should not have been accepted"
    assert field in why or "number" in why, why


def test_a_missing_field_is_rejected(session):
    raw = dict(GOOD)
    del raw["curve"]
    params, why = session.supervisor.validate_transition_params(raw, session.crate[1])
    assert params is None and "curve" in why


def test_something_that_is_not_an_object_is_rejected(session):
    for junk in ([1, 2, 3], "length_bars=16", None, 7):
        params, why = session.supervisor.validate_transition_params(junk, session.crate[1])
        assert params is None and why


def test_the_echo_may_fill_the_whole_blend_but_not_exceed_it(session):
    """The boundary case, and a note on why the other side is unreachable.

    `echo_bars` tops out at 4 and `length_bars` starts at 4, so a valid echo
    can never be longer than a valid blend. The guard is still there because
    it is cheap and the ranges are config, but it cannot fire through this
    door -- an out-of-range echo is caught by its own range check first.
    """
    short = dict(GOOD, length_bars=4, low_swap_bar=2)
    ok, why = session.supervisor.validate_transition_params(
        dict(short, echo_bars=4), session.crate[1]
    )
    assert ok is not None, f"a full-length echo is legal, but: {why}"

    params, why = session.supervisor.validate_transition_params(
        dict(short, echo_bars=5), session.crate[1]
    )
    assert params is None and "echo_bars" in why

    assert tr.ECHO_BARS_RANGE[1] <= tr.LENGTH_BARS_RANGE[0], (
        "if these ranges ever cross, the length guard becomes reachable"
    )


def test_a_hot_cue_that_does_not_exist_is_rejected(session):
    from djai import analysis as an

    track = session.crate[1]
    track.hot_cues = [an.make_hot_cue(1, track.mix_in, "mix in")]
    raw = dict(GOOD, entry_point="hot_cue_7")
    params, why = session.supervisor.validate_transition_params(raw, track)
    assert params is None
    assert "hot cue 7" in why and "[1]" in why


def test_a_hot_cue_that_does_exist_is_accepted(session):
    from djai import analysis as an

    track = session.crate[1]
    track.hot_cues = [
        an.make_hot_cue(1, track.mix_in, "mix in"),
        an.make_hot_cue(4, track.mix_in + 30, "drop"),
    ]
    raw = dict(GOOD, entry_point="hot_cue_4")
    params, why = session.supervisor.validate_transition_params(raw, track)
    assert params is not None, why
    assert params.hot_cue_index() == 4


def test_a_hot_cue_with_no_runway_after_it_is_rejected(session):
    """Regression, from a 30-minute run that produced dead air.

    A designed entry at a cue near the incoming track's own mix-out empties
    the deck shortly after the blend hands over, and the silence recovery is
    audible. Existing is not the same as usable, and the supervisor is the
    only thing standing between a model's choice and the room.
    """
    from djai import analysis as an

    track = session.crate[1]
    bar_s = 4 * 60.0 / track.bpm
    too_late = track.mix_out - (tr.MIN_ENTRY_RUNWAY_BARS - 8) * bar_s
    track.hot_cues = [
        an.make_hot_cue(1, track.mix_in, "mix in"),
        an.make_hot_cue(3, too_late, "drop"),
    ]
    params, why = session.supervisor.validate_transition_params(
        dict(GOOD, entry_point="hot_cue_3"), track
    )
    assert params is None
    assert "runway" in why or "bars before mix-out" in why


def test_a_hot_cue_with_room_behind_it_is_still_accepted(session):
    from djai import analysis as an

    track = session.crate[1]
    bar_s = 4 * 60.0 / track.bpm
    track.hot_cues = [an.make_hot_cue(3, 16 * bar_s, "drop")]
    params, why = session.supervisor.validate_transition_params(
        dict(GOOD, entry_point="hot_cue_3"), track
    )
    assert params is not None, why
    assert params.hot_cue_index() == 3


def test_both_routes_into_a_deck_use_the_same_runway_rule(session):
    """The bug appeared once through each route; the rule now lives once."""
    from djai import analysis as an

    track = session.crate[1]
    bar_s = 4 * 60.0 / track.bpm
    late = track.mix_out - (tr.MIN_ENTRY_RUNWAY_BARS - 4) * bar_s
    early = 16 * bar_s
    assert not tr.entry_has_runway(track, late)
    assert tr.entry_has_runway(track, early)

    # The rule-based chooser refuses it...
    track.hot_cues = [an.make_hot_cue(3, late, "drop")]
    assert tr._drop_cue(track) is None
    # ...and so does the supervisor, for a designed entry.
    params, _why = session.supervisor.validate_transition_params(
        dict(GOOD, entry_point="hot_cue_3"), track
    )
    assert params is None


def test_a_malformed_entry_point_is_rejected(session):
    for bad in ("hot_cue_", "hot_cue_x", "the drop", 4):
        raw = dict(GOOD, entry_point=bad)
        params, _why = session.supervisor.validate_transition_params(
            raw, session.crate[1]
        )
        assert params is None, f"{bad!r} should not have been accepted"


# --- choosing what actually runs ----------------------------------------------


def _rule_based(style: str) -> bool:
    """A preset, or a shape generated for the pair (SPEC §4) -- never the model's."""
    return style in tr.STYLES or style.startswith("gen_")


def _arm(session, style="auto"):
    session.transition_style = style
    assert session.cue_next(origin="test")
    drive(session.engine, 4, BLOCK)
    return session.arm_transition(origin="test")


def test_a_valid_design_is_what_runs(session):
    session.intent_engine = FakeModel(GOOD)
    _arm(session)
    style, rule = session.last_transition_choice
    assert style == "llm", rule
    assert "designed" in rule


def test_an_invalid_design_falls_back_to_the_preset_and_says_so(session):
    session.intent_engine = FakeModel(dict(GOOD, curve="wobble"))
    _arm(session)
    style, rule = session.last_transition_choice
    assert _rule_based(style), "a rejected design must not run"
    assert "rejected" in rule

    events = [e for e in log_events(session)
              if e["event"] == "transition_design_rejected"]
    assert events, "the rejection must be logged"
    assert events[-1]["params"]["curve"] == "wobble", "with the params it rejected"
    assert "curve" in events[-1]["reason"], "and the reason"


def test_no_model_means_the_presets_run(session):
    session.intent_engine = None
    _arm(session)
    style, rule = session.last_transition_choice
    assert _rule_based(style)
    assert "no design used" in rule


def test_a_model_that_has_gone_away_mid_session_changes_nothing(session):
    """Ollama stopped: the set carries on with presets, no interruption."""
    session.intent_engine = FakeModel(GOOD)
    _arm(session)
    assert session.last_transition_choice[0] == "llm"

    # It goes down between transitions.
    session.intent_engine.available = False
    session._transition_armed = False
    session._cued = None
    style = _arm(session) is not None
    assert style, "a transition still had to be armed"
    assert _rule_based(session.last_transition_choice[0])


def test_a_design_that_is_still_running_is_not_waited_for(session):
    """The scheduler never blocks on a model."""
    import threading

    release = threading.Event()

    class SlowModel(FakeModel):
        def design_transition(self, context, timeout_s=None):
            release.wait(timeout=5)
            return dict(GOOD), ""

    session.intent_engine = SlowModel(GOOD)
    _arm(session)
    style, rule = session.last_transition_choice
    assert _rule_based(style), "a slow design must not be used"
    assert "not ready in time" in rule
    release.set()


def test_the_autopilot_waits_for_a_design_before_arming(session):
    """Regression, from a 30-minute run where no design was ever used.

    Cueing and arming are one 0.5 s tick apart; a design takes seconds. So the
    answer was never back in time and every transition silently fell back to a
    preset -- the feature was inert and the logs said so only if you read them.
    The pre-roll window is around 105 s wide, so the wait costs nothing.
    """
    import threading

    release = threading.Event()

    class SlowModel(FakeModel):
        def design_transition(self, context, timeout_s=None):
            release.wait(timeout=5)
            return dict(GOOD), ""

    session.intent_engine = SlowModel(GOOD)
    assert session.cue_next(origin="test")
    drive(session.engine, 4, BLOCK)

    assert session.design_pending(), "it should be willing to wait"
    release.set()
    for _ in range(50):
        if not session.design_pending():
            break
        time.sleep(0.05)
    assert not session.design_pending(), "and stop waiting once it arrives"

    session.arm_transition(origin="test")
    assert session.last_transition_choice[0] == "llm"


def test_the_wait_gives_up_rather_than_arming_late(session, monkeypatch):
    """A design is never worth arriving late for."""
    import threading

    from djai import config

    never = threading.Event()

    class HungModel(FakeModel):
        def design_transition(self, context, timeout_s=None):
            never.wait(timeout=30)
            return dict(GOOD), ""

    monkeypatch.setattr(config, "TRANSITION_DESIGN_WAIT_S", 0.2)
    session.intent_engine = HungModel(GOOD)
    assert session.cue_next(origin="test")
    time.sleep(0.35)
    assert not session.design_pending(), "the deadline must end the wait"

    session.arm_transition(origin="test")
    assert _rule_based(session.last_transition_choice[0])
    never.set()


def test_no_design_in_flight_means_no_waiting(session):
    session.intent_engine = None
    assert session.cue_next(origin="test")
    assert not session.design_pending()


def test_an_explicit_style_request_is_not_second_guessed(session):
    session.intent_engine = FakeModel(GOOD)
    _arm(session, style="filter_sweep")
    style, rule = session.last_transition_choice
    assert style == "filter_sweep"
    assert "operator" in rule
    assert session.intent_engine.calls == [], "no design should have been asked for"


# --- safety beats everything ---------------------------------------------------


def test_a_weak_grid_forces_a_cut_over_a_32_bar_design(session):
    """The operator cues a weak-grid track by hand; safety still overrides.

    Since Phase 0A a grid this weak is quarantined, so autonomous selection
    will not reach for the track at all -- the only way it gets on a deck is
    the operator naming it, which quarantine deliberately still allows. That
    is exactly when this override has to hold.
    """
    session.intent_engine = FakeModel(dict(GOOD, length_bars=32, curve="s_curve"))
    for track in session.crate:
        object.__setattr__(track, "grid_confidence", 0.02)
    session._forced_next = session.crate[1]

    _arm(session)
    style, rule = session.last_transition_choice
    assert style == "cut", rule
    assert "safety override" in rule and "grid confidence" in rule


def test_a_wide_tempo_gap_forces_a_cut_over_a_design(session):
    """Tested at the decision, not through the autopilot.

    A track needing more than the stretch limit never reaches a transition:
    the supervisor rejects the load at cue time, well before anything is
    armed. So the tempo-gap override is exercised where it actually lives.
    """
    session.intent_engine = FakeModel(dict(GOOD, length_bars=32))
    choice = tr.TransitionChoice(
        style="cut", entry="mix_in", hot_cue_index=None,
        rule="BPM delta 14.0% above 8%",
    )
    params, source, note = session._params_for(
        choice, session.crate[1], safety_forced=True
    )
    assert params.name == "cut"
    assert source == "safety"
    assert "BPM delta" in note
    assert session.intent_engine.calls == [], "the design is not even consulted"


def test_the_design_is_discarded_not_merged_when_safety_fires(session):
    """A 32-bar blend must not become a 32-bar cut."""
    session.intent_engine = FakeModel(dict(GOOD, length_bars=32))
    choice = tr.TransitionChoice(
        style="cut", entry="mix_in", hot_cue_index=None,
        rule="grid confidence below 0.50",
    )
    params, _source, _note = session._params_for(
        choice, session.crate[1], safety_forced=True
    )
    assert params.length_bars == tr.LENGTH_BARS_RANGE[0]
    assert params == tr.preset_params("cut")


# --- context and logging -------------------------------------------------------


def test_the_designer_is_told_the_music_and_nothing_positional(session):
    model = FakeModel(GOOD)
    session.intent_engine = model
    session.energy_direction = -0.6
    _arm(session)

    assert model.calls, "the designer should have been asked"
    ctx = model.calls[0]
    assert set(ctx) == {"deck_a", "deck_b", "hot_cues", "energy_direction"}
    assert ctx["energy_direction"] == pytest.approx(-0.6)
    for deck in ("deck_a", "deck_b"):
        assert set(ctx[deck]) == {"bpm", "key", "energy"}
    blob = json.dumps(ctx)
    for word in ("frame", "sample", "position", "mix_out", "second"):
        assert word not in blob, f"context leaked {word!r} to the model"


def test_the_context_says_which_hot_cues_are_usable(session):
    """A yes/no, not a position: the model still gets no timing.

    Measured need: 5 of 6 rejections across 32 designs were the model naming a
    cue too near its track's mix-out, because nothing in the context told one
    cue from another. Adding the flag took the usable-params rate from 81% to
    91%.
    """
    from djai import analysis as an
    from djai import transition as tr_mod

    model = FakeModel(GOOD)
    session.intent_engine = model
    incoming = session.crate[1]
    bar_s = 4 * 60.0 / incoming.bpm
    incoming.hot_cues = [
        an.make_hot_cue(1, 16 * bar_s, "drop"),                       # room behind it
        an.make_hot_cue(2, incoming.mix_out - 4 * bar_s, "mix out"),  # none
    ]

    session.start_transition_design(incoming)
    session._design["thread"].join(timeout=5)
    ctx = model.calls[0]

    by_name = {c["name"]: c for c in ctx["hot_cues"]}
    assert by_name["hot_cue_1"]["usable"] is True
    assert by_name["hot_cue_2"]["usable"] is False
    assert tr_mod.entry_has_runway(incoming, 16 * bar_s)

    # And still no positions anywhere in what the model sees.
    blob = json.dumps(ctx)
    for word in ("sample_position", "seconds", "frame", "mix_out"):
        assert word not in blob, f"context leaked {word!r}"


def test_the_prompt_tells_the_model_to_use_only_usable_cues():
    from djai.intent import TRANSITION_PROMPT

    assert "usable" in TRANSITION_PROMPT


def test_every_transition_logs_what_was_asked_and_what_ran(session):
    session.intent_engine = FakeModel(GOOD)
    session.energy_direction = 0.8
    _arm(session)

    armed = [e for e in log_events(session) if e["event"] == "transition_armed"]
    assert armed
    last = armed[-1]
    assert last["requested_energy"] == pytest.approx(0.8)
    assert last["design_source"] == "llm"
    assert last["design_raw"]["curve"] == "s_curve"
    assert last["params_used"]["curve"] == "s_curve"
    assert last["params_used"]["length_bars"] == 16


def test_the_preset_path_logs_the_params_it_used_too(session):
    session.intent_engine = None
    _arm(session)
    armed = [e for e in log_events(session) if e["event"] == "transition_armed"]
    used = armed[-1]["params_used"]
    assert used["curve"] in tr.CURVES
    assert armed[-1]["design_source"] in ("preset", "generated")


# --- the prompt ----------------------------------------------------------------


def test_every_rejection_names_the_field_and_the_value(session):
    """No silent fallbacks, and no rejections you cannot act on.

    A constant fallback looks exactly like a model with no imagination, so a
    rejection has to say which field failed and what it held.
    """
    cases = [
        ("length_bars", 99),
        ("curve", "wobble"),
        ("low_swap_bars", 0),
        ("deck_b_low_delay_bars", 40),
        ("echo_bars", 9),
        ("filter_sweep", "phaser"),
        ("intensity", 3.0),
        ("entry_point", "hot_cue_9"),
        ("low_swap_bar", 900),
    ]
    for field, value in cases:
        params, why = session.supervisor.validate_transition_params(
            dict(GOOD, **{field: value}), session.crate[1]
        )
        assert params is None, f"{field}={value!r} was accepted"
        assert field in why, f"the reason does not name {field}: {why}"
        assert str(value).strip("'\"") in why, (
            f"the reason does not give the value {value!r}: {why}"
        )


def test_a_rejected_design_is_logged_with_its_params_and_reason(session):
    session.intent_engine = FakeModel(dict(GOOD, curve="wobble"))
    _arm(session)
    events = [e for e in log_events(session)
              if e["event"] == "transition_design_rejected"]
    assert events
    last = events[-1]
    assert last["params"]["curve"] == "wobble"
    assert "curve" in last["reason"] and "wobble" in last["reason"]
    assert last["fell_back_to"] in tr.STYLES


def test_the_session_counts_where_each_shape_came_from(session):
    session.intent_engine = FakeModel(GOOD)
    _arm(session)
    assert session.design_counts["llm"] == 1
    assert session.style_counts["llm"] == 1

    session._transition_armed = False
    session._cued = None
    session.intent_engine = FakeModel(dict(GOOD, curve="nonsense"))
    _arm(session)
    assert session.design_counts["rejected"] == 1
    assert sum(session.design_counts.values()) == 2


def test_the_counts_reach_the_ui(session):
    from djai.ui_server import UIServer

    server = UIServer(session, intent_engine=None)
    session.intent_engine = FakeModel(GOOD)
    _arm(session)

    trans = server.state()["transition"]
    assert trans["sources"] == {"llm": 1}
    assert trans["style_counts"] == {"llm": 1}
    json.dumps(trans)


def test_the_page_shows_the_diversity_counter(session):
    from fastapi.testclient import TestClient

    from djai.ui_server import UIServer

    body = TestClient(UIServer(session, intent_engine=None)._app).get("/").text
    assert "trans-diversity" in body
    js = (Path(__file__).resolve().parent.parent / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "style_counts" in js and "designed" in js


def test_the_examples_demonstrate_genuinely_different_shapes():
    """Eleven examples that are eleven shapes, not one shape eleven times.

    llama3.1:8b copies what it is shown. Examples that all look like a bass
    swap produce transitions that all sound like a bass swap. The first five
    are blends, the next three the high-energy moves, and the last three the
    Phase 3 effects -- one field each.
    """
    import re

    from djai.intent import TRANSITION_PROMPT

    raw = re.findall(r'\{"length_bars".*', TRANSITION_PROMPT)
    assert len(raw) == 11
    shapes = [json.loads(r) for r in raw]

    assert len({json.dumps(s, sort_keys=True) for s in shapes}) == 11
    blends, hot, effects = shapes[:5], shapes[5:8], shapes[8:]

    def no_effects(s):
        return s["reverb_bars"] == 0 and s["brake_beats"] == 0 and s["riser_bars"] == 0

    assert all(no_effects(s) for s in blends + hot), "effects are shown only by effect examples"
    assert any(s["reverb_bars"] > 0 and s["echo_bars"] > 0 for s in effects), "echo into reverb"
    assert any(s["brake_beats"] in (1, 2) for s in effects)
    assert any(s["riser_bars"] >= 4 and s["filter_sweep"] != "none" and s["echo_bars"] > 0
               for s in effects), "riser, and the filter+echo combination"
    assert all(s["vocal_aware"] is True for s in shapes)

    assert {s["curve"] for s in blends} == set(tr.CURVES), "every curve once"
    assert len({s["length_bars"] for s in blends}) == 5, "five different lengths"
    assert len({s["filter_sweep"] for s in blends}) == len(tr.FILTER_SWEEPS)
    assert sum(1 for s in blends if s["entry_point"].startswith("hot_cue")) >= 2
    assert any(s["low_swap_bar"] is None for s in blends), "null swap demonstrated"
    assert len({s["echo_bars"] for s in blends}) >= 3
    assert all(
        s["loop_out_bars"] == 0
        and not s["loop_halving"]
        and s["beat_repeat_division"] == 0
        and s["backspin_bars"] == 0
        and s["double_drop_bars"] == 0
        and s["align_mode"] == "phrase"
        for s in blends
    ), "a blend example must not demonstrate a high-energy field"

    assert any(s["loop_out_bars"] > 0 and s["loop_halving"] for s in hot)
    assert any(s["beat_repeat_division"] in (8, 16) for s in hot)
    assert any(s["double_drop_bars"] > 0 and s["align_mode"] == "drop" for s in hot)
    assert all(s["intensity"] >= 0.75 for s in hot), "high energy means high energy"


def test_omitted_fields_are_rejected_rather_than_defaulted(session):
    """Defaults must not quietly reconstruct a bass swap.

    Every field is required. A model that leaves one out gets a rejection
    naming it, not a shape that happens to be the default.
    """
    from djai.intent import TRANSITION_SCHEMA

    assert set(TRANSITION_SCHEMA["required"]) == set(TRANSITION_SCHEMA["properties"])
    for field in GOOD:
        raw = {k: v for k, v in GOOD.items() if k != field}
        params, why = session.supervisor.validate_transition_params(
            raw, session.crate[1]
        )
        assert params is None, f"a design missing {field} was accepted"
        assert field in why


def test_the_transition_prompt_stays_short():
    """llama3.1:8b degrades on long prompts; the brief's bar is 500 words."""
    from djai.intent import TRANSITION_PROMPT

    assert len(TRANSITION_PROMPT.split()) < 500


def test_the_prompt_carries_eleven_worked_examples():
    """Five blends from Part 3, the three high-energy ones, and three effects."""
    from djai.intent import TRANSITION_PROMPT

    assert TRANSITION_PROMPT.count('{"length_bars"') == 11


def test_the_decode_schema_pins_every_field():
    from djai.intent import TRANSITION_SCHEMA

    props = TRANSITION_SCHEMA["properties"]
    assert set(props) == set(tr.preset_params("bass_swap").to_schema())
    assert set(TRANSITION_SCHEMA["required"]) == set(props)
    assert props["curve"]["enum"] == list(tr.CURVES)
    assert props["filter_sweep"]["enum"] == list(tr.FILTER_SWEEPS)
