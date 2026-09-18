"""The preview, wired into a live Session: what it may change, and when not.

THREADING CONTEXT: main thread (pytest). The Session is never started, so the
scheduler's pre-roll worker thread does not run; the hook is called directly
with the command the scheduler would have handed it, which makes every
commit path deterministic. One test runs the real preview end to end.
"""

from __future__ import annotations

import dataclasses

import pytest

from djai import cli, config, revise, transition as tr
from djai.commands import LoadTrack, StartTransition
from djai.deck import SAMPLE_RATE
from tests.test_integration import seek_near_mix_out
from tests.test_integration import session as _session_fixture
from tests.test_engine import drive

session = _session_fixture


def armed_pair(s):
    pending = s.scheduler.pending()
    load = next(c for c in pending if isinstance(c, LoadTrack))
    swap = next(c for c in pending if isinstance(c, StartTransition))
    return load, swap


@pytest.fixture
def armed(session):
    seek_near_mix_out(session, bars_before=64.0)
    session.cue_next(0.0)
    drive(session.engine, 2)
    assert session.arm_transition() is not None
    return session


def outcome_for(s, **changes):
    original = s._armed_preview["params"]
    new = dataclasses.replace(original, **changes)
    revisions = [
        revise.Revision(k, getattr(original, k), v, "test", "test")
        for k, v in changes.items()
    ]
    return revise.PreviewOutcome(
        status="revised", params=new, original=original, revisions=revisions,
        rounds=[{"round": 0, "params": original.to_schema(),
                 "measurements": {"loudness_dip_db": 5.0}}],
    )


def stub_preview(monkeypatch, outcome):
    calls = []

    def run(*a, **k):
        calls.append((a, k))
        return outcome
    monkeypatch.setattr(revise, "run_preview", run)
    return calls


# --- wiring ---------------------------------------------------------------------


def test_the_session_scheduler_carries_the_hook(session):
    assert session.scheduler._on_pre_roll == session._on_pre_roll
    assert session.scheduler._pre_roll_frames == int(config.PREVIEW_PRE_ROLL_S * SAMPLE_RATE)


def test_no_hook_and_no_thread_when_the_preview_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PREVIEW_ENABLED", False)
    monkeypatch.setattr(config, "TIME_STRETCH_ENABLED", False)
    s = cli.Session([], log_dir=tmp_path / "logs")
    try:
        assert s.scheduler._on_pre_roll is None
    finally:
        s.shutdown()


def test_arming_records_exactly_what_was_queued(armed):
    load, swap = armed_pair(armed)
    rec = armed._armed_preview
    assert rec["load"] is load and rec["swap"] is swap
    assert rec["bars"] > 0 and rec["previewed"] is False


# --- committing a revision --------------------------------------------------------


def test_a_shorter_revision_replaces_the_queued_transition(armed, monkeypatch):
    load, swap = armed_pair(armed)
    original_bars = armed._armed_preview["bars"]
    original_length = armed._armed_preview["params"].length_bars
    shorter = original_length - 4
    events = []
    armed.add_preview_listener(lambda kind, payload: events.append(kind))
    stub_preview(monkeypatch, outcome_for(armed, length_bars=shorter, curve="equal_power"))

    armed._on_pre_roll(swap)

    new_load, new_swap = armed_pair(armed)
    assert new_swap is not swap and new_load is not load
    assert new_swap.execute_at == swap.execute_at, "the start does not move"
    assert new_swap.total_frames < swap.total_frames, "the blend is shorter"
    # The armed length is not always the designed length (placement and any
    # held bars move it), so the revision carries over as a proportion.
    assert new_swap.total_frames == tr.transition_frames(
        armed._armed_preview["bpm"], SAMPLE_RATE,
        transition_bars=original_bars * shorter / original_length)
    assert new_load.start_frame == load.start_frame
    assert len(armed.scheduler) == 2, "exactly one pair queued, not two"
    # Armed through the params path under a new name, so the revision is not
    # silently rebuilt as the preset it started from.
    assert armed.engine._armed_plan[1].endswith("+preview")
    assert events == ["started", "finished"]
    assert "revised" in armed.last_preview.events[-1]


def test_a_revision_scales_a_length_placement_already_shortened(armed, monkeypatch):
    """Found live: a 24-bar design armed as 9 bars to fit before mix-out.

    Subtracting the revision's six bars from nine is fine; subtracting a
    twenty-bar cut is not, and it queued a transition of -56905 frames that only
    the supervisor stopped. The revision's proportion is what carries over.
    """
    _load, swap = armed_pair(armed)
    params = armed._armed_preview["params"]
    armed._armed_preview.update(
        params=dataclasses.replace(params, length_bars=24), bars=9.0)
    stub_preview(monkeypatch, outcome_for(armed, length_bars=4))
    armed._on_pre_roll(swap)
    _new_load, new_swap = armed_pair(armed)
    assert new_swap is not swap, armed.last_preview.events[-1]
    assert new_swap.total_frames == tr.transition_frames(
        armed._armed_preview["bpm"], SAMPLE_RATE, transition_bars=9.0 * 4 / 24)
    assert new_swap.total_frames > 0


def test_a_longer_revision_keeps_the_armed_length(armed, monkeypatch):
    """Placement was worked back from mix-out for the armed length."""
    _load, swap = armed_pair(armed)
    longer = min(tr.LENGTH_BARS_RANGE[1], armed._armed_preview["params"].length_bars + 8)
    stub_preview(monkeypatch, outcome_for(armed, length_bars=longer, curve="equal_power"))
    armed._on_pre_roll(swap)
    _new_load, new_swap = armed_pair(armed)
    assert new_swap.total_frames == swap.total_frames
    assert "mix-out" in armed.last_preview.events[-1]


def test_nothing_changes_too_close_to_the_hand_off(armed, monkeypatch):
    load, swap = armed_pair(armed)
    monkeypatch.setattr(config, "PREVIEW_COMMIT_MARGIN_S", 1e9)
    stub_preview(monkeypatch, outcome_for(armed, curve="equal_power"))
    armed._on_pre_roll(swap)
    assert armed_pair(armed) == (load, swap)
    assert "too close" in armed.last_preview.events[-1]


def test_a_transition_cancelled_meanwhile_is_not_resurrected(armed, monkeypatch):
    """A panic or skip empties the queue while the preview renders."""
    _load, swap = armed_pair(armed)

    def cancelled_during_render(*a, **k):
        armed.scheduler.cancel_all()
        return outcome_for(armed, curve="equal_power")

    monkeypatch.setattr(revise, "run_preview", cancelled_during_render)
    armed._on_pre_roll(swap)
    assert len(armed.scheduler) == 0, "the preview put a cancelled transition back"


def test_a_transition_re_armed_meanwhile_is_left_alone(armed, monkeypatch):
    load, swap = armed_pair(armed)

    def rearmed_during_render(*a, **k):
        armed._armed_preview = dict(armed._armed_preview)   # a new arm
        return outcome_for(armed, curve="equal_power")

    monkeypatch.setattr(revise, "run_preview", rearmed_during_render)
    armed._on_pre_roll(swap)
    assert armed_pair(armed) == (load, swap)


def test_the_replacement_is_not_previewed_again(armed, monkeypatch):
    _load, swap = armed_pair(armed)
    calls = stub_preview(monkeypatch, outcome_for(armed, curve="equal_power"))
    armed._on_pre_roll(swap)
    _new_load, new_swap = armed_pair(armed)
    armed._on_pre_roll(new_swap)        # the scheduler announces the new object
    assert len(calls) == 1


def test_a_grid_fault_becomes_a_cut_at_the_planned_hand_off(armed, monkeypatch):
    _load, swap = armed_pair(armed)
    outcome = revise.PreviewOutcome(
        status="cut", params=armed._armed_preview["params"],
        original=armed._armed_preview["params"], force_cut=True,
    )
    stub_preview(monkeypatch, outcome)
    armed._on_pre_roll(swap)
    new_load, new_swap = armed_pair(armed)
    assert new_swap.total_frames == armed.engine.blocksize
    assert new_swap.execute_at > swap.execute_at, "cut where the blend would have ended"
    assert new_load.execute_at == new_swap.execute_at
    assert armed.engine._armed_plan[1] == "cut"
    assert armed.last_transition_choice[0] == "cut"


def test_an_operator_style_is_not_previewed(session, monkeypatch):
    session.transition_style = "bass_swap"
    seek_near_mix_out(session, bars_before=64.0)
    session.cue_next(0.0)
    drive(session.engine, 2)
    assert session.arm_transition() is not None
    _load, swap = armed_pair(session)
    calls = stub_preview(monkeypatch, None)
    session._on_pre_roll(swap)
    assert calls == []


def test_a_manual_transition_is_not_previewed(armed, monkeypatch):
    calls = stub_preview(monkeypatch, None)
    stray = StartTransition(from_deck="a", to_deck="b", total_frames=1000,
                            execute_at=10**9, origin="operator")
    armed._on_pre_roll(stray)
    assert calls == []


# --- end to end -----------------------------------------------------------------


def test_the_real_preview_runs_on_a_live_session(armed):
    """No stubs: render, measure, rules, commit, log, on the synthetic set."""
    load, swap = armed_pair(armed)
    armed._on_pre_roll(swap)
    outcome = armed.last_preview
    assert outcome is not None
    assert outcome.status in ("committed", "revised", "reverted", "budget", "cut")
    assert outcome.rounds, "the transition was actually rendered and measured"
    new_load, new_swap = armed_pair(armed)
    if outcome.force_cut:
        assert new_swap.total_frames == armed.engine.blocksize
    else:
        assert new_swap.execute_at == swap.execute_at
    armed.session_log._file.flush()
    events = [line for line in armed.session_log.path.read_text(encoding="utf-8").splitlines()
              if '"preview' in line]
    assert any('"event": "preview"' in e for e in events)
    assert any('"event": "preview_commit"' in e for e in events)
