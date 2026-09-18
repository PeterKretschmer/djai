"""Selector and intent-parser tests.

THREADING CONTEXT: main thread (pytest). No network: the intent tests exercise
:func:`djai.intent.parse_response` directly, which is where every safety
property of the LLM boundary actually lives.
"""

from __future__ import annotations

import pytest

from djai.analysis import TrackAnalysis
from djai.intent import ACTIONS, parse_response
from djai.selector import key_relation, rank_candidates, select_next


def track(
    tid: str,
    bpm: float,
    camelot: str = "8A",
    energy: float = 0.5,
    confidence: float = 1.0,
) -> TrackAnalysis:
    return TrackAnalysis(
        track_id=tid,
        path=f"/tmp/{tid}.wav",
        title=tid,
        duration_s=300.0,
        bpm=bpm,
        beats=[0.0, 60.0 / bpm],
        downbeats=[0.0],
        key_name="A minor",
        camelot=camelot,
        beat_rms=[energy],
        energy=energy,
        grid_confidence=confidence,
    )


# --- Camelot -----------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("8A", "8A", "same"),
        ("8A", "8B", "relative"),
        ("8A", "9A", "adjacent"),
        ("8A", "7A", "adjacent"),
        ("12A", "1A", "adjacent"),  # the wheel wraps
        ("1A", "12A", "adjacent"),
        ("8A", "2A", "clash"),
        ("8A", "9B", "clash"),
        ("8A", "nonsense", "unknown"),
    ],
)
def test_key_relation(a, b, expected):
    assert key_relation(a, b) == expected


# --- selector ----------------------------------------------------------------


def test_rejects_tracks_outside_the_bpm_window():
    current = track("cur", 124.0)
    crate = [
        current,
        track("in_range", 128.0),  # +3.2%
        track("too_fast", 140.0),  # +12.9%
        track("too_slow", 100.0),
    ]
    ids = {c.track.track_id for c in rank_candidates(crate, current)}
    assert ids == {"in_range"}


def test_never_returns_the_current_track():
    current = track("cur", 124.0)
    assert select_next([current], current) is None


def test_skips_tracks_already_played():
    current = track("cur", 124.0)
    crate = [current, track("t1", 125.0), track("t2", 125.0)]
    chosen = select_next(crate, current, played={"t1"})
    assert chosen is not None and chosen.track.track_id == "t2"


def test_prefers_a_compatible_key_over_a_clashing_one():
    current = track("cur", 124.0, camelot="8A")
    crate = [
        current,
        track("clash", 124.0, camelot="2A", energy=0.5),
        track("compatible", 124.0, camelot="9A", energy=0.5),
    ]
    chosen = select_next(crate, current)
    assert chosen is not None and chosen.track.track_id == "compatible"
    assert chosen.key_relation == "adjacent"


def test_energy_direction_changes_the_choice():
    current = track("cur", 124.0, camelot="8A", energy=0.5)
    crate = [
        current,
        track("calm", 124.0, camelot="8A", energy=0.0),
        track("hot", 124.0, camelot="8A", energy=1.0),
    ]
    up = select_next(crate, current, energy_direction=1.0)
    down = select_next(crate, current, energy_direction=-1.0)
    assert up is not None and up.track.track_id == "hot"
    assert down is not None and down.track.track_id == "calm"
    assert up.track.track_id != down.track.track_id


def test_is_a_pure_function():
    current = track("cur", 124.0)
    crate = [current, track("t1", 125.0), track("t2", 126.0)]
    first = select_next(crate, current, energy_direction=0.3)
    for _ in range(5):
        again = select_next(crate, current, energy_direction=0.3)
        assert again is not None and first is not None
        assert again.track.track_id == first.track.track_id
        assert again.score == pytest.approx(first.score)


def test_returns_none_on_an_empty_crate():
    assert select_next([], track("cur", 124.0)) is None


def test_candidate_reason_is_human_readable():
    current = track("cur", 124.0, camelot="8A")
    chosen = select_next([current, track("t1", 125.0, camelot="9A")], current)
    assert chosen is not None
    reason = chosen.reason()
    assert "t1" in reason and "BPM" in reason and "9A" in reason


# --- intent parsing ----------------------------------------------------------


def test_parses_a_well_formed_response():
    intent = parse_response(
        '{"action": "next_track", "params": {"energy": 0.6}, "reply": "Cueing up."}'
    )
    assert intent.ok
    assert intent.action == "next_track"
    assert intent.energy_direction == pytest.approx(0.6)
    assert intent.reply == "Cueing up."


def test_extracts_json_wrapped_in_prose_or_fences():
    fenced = '```json\n{"action": "none", "params": {}, "reply": "ok"}\n```'
    assert parse_response(fenced).ok
    chatty = 'Sure! {"action": "none", "params": {}, "reply": "ok"} Hope that helps.'
    assert parse_response(chatty).ok


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json at all",
        "[1, 2, 3]",
        '"just a string"',
        '{"action": "next_track"',  # truncated
        '{"params": {}, "reply": "hi"}',  # no action
        '{"action": "launch_missiles", "params": {}, "reply": "hi"}',
        '{"action": "next_track", "params": "loud", "reply": "hi"}',
        '{"action": "next_track", "params": {}, "reply": 42}',
        '{"action": 7, "params": {}, "reply": "hi"}',
        "null",
    ],
)
def test_malformed_responses_degrade_to_none(raw):
    """Every malformed shape must become an inert `none`, never a command."""
    intent = parse_response(raw)
    assert not intent.ok, f"should have been rejected: {raw!r}"
    assert intent.action == "none"
    assert intent.error
    assert intent.raw == raw or raw == ""


def test_every_declared_action_parses():
    for action in ACTIONS:
        intent = parse_response(
            f'{{"action": "{action}", "params": {{}}, "reply": "ok"}}'
        )
        assert intent.ok and intent.action == action


def test_missing_params_defaults_to_empty():
    intent = parse_response('{"action": "skip_queued", "reply": "dropped"}')
    assert intent.ok and intent.params == {}


def test_energy_direction_is_clamped():
    assert parse_response(
        '{"action": "set_energy", "params": {"direction": 99}, "reply": ""}'
    ).energy_direction == pytest.approx(1.0)
    assert parse_response(
        '{"action": "set_energy", "params": {"direction": -99}, "reply": ""}'
    ).energy_direction == pytest.approx(-1.0)
    assert parse_response(
        '{"action": "set_energy", "params": {"direction": "loud"}, "reply": ""}'
    ).energy_direction == pytest.approx(0.0)


def test_hold_bars_is_clamped_with_a_sane_default():
    assert parse_response(
        '{"action": "hold_blend", "params": {"bars": 8}, "reply": ""}'
    ).hold_bars == 8
    assert parse_response(
        '{"action": "hold_blend", "params": {"bars": 9999}, "reply": ""}'
    ).hold_bars == 64
    assert parse_response(
        '{"action": "hold_blend", "params": {}, "reply": ""}'
    ).hold_bars == 16


def test_intent_never_carries_a_track_choice():
    """The model may steer energy; it may never name the track that plays.

    Track choice is the selector's job, so a hallucinated title has nowhere to
    go even if the model emits one.
    """
    intent = parse_response(
        '{"action": "next_track", "params": {"title": "Nonexistent Banger"},'
        ' "reply": "ok"}'
    )
    assert intent.ok
    assert intent.energy_direction == pytest.approx(0.0)
    # `title` survives in params but nothing in the pipeline reads it.
    assert "title" in intent.params

