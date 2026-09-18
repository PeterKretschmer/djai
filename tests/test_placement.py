"""Mix points, backwards placement, and the mid-track regression guard.

THREADING CONTEXT: main thread (pytest). No audio device.

These cover the placement question -- *where* in a track the hand-off belongs --
as distinct from the alignment question `next_phrase_boundary` answers.
"""

from __future__ import annotations

import numpy as np
import pytest

from djai import analysis, phrase
from djai.analysis import TrackAnalysis
from djai.commands import StartTransition
from djai.deck import CHANNELS, SAMPLE_RATE, Deck, LoadedTrack

BEATS_PER_BAR = 4


def make_analysis(
    bpm: float = 120.0,
    bars: int = 128,
    first_downbeat: float = 0.0,
    mix_in_bar: float = 16.0,
    mix_out_bar: float = 96.0,
    title: str = "t",
) -> TrackAnalysis:
    period = 60.0 / bpm
    n = bars * BEATS_PER_BAR
    beats = [first_downbeat + i * period for i in range(n)]
    bar_s = BEATS_PER_BAR * period
    return TrackAnalysis(
        track_id=title,
        path=f"/tmp/{title}.wav",
        title=title,
        duration_s=beats[-1] + period,
        bpm=bpm,
        beats=beats,
        downbeats=beats[::BEATS_PER_BAR],
        key_name="A minor",
        camelot="8A",
        beat_rms=[0.5] * n,
        energy=0.5,
        grid_confidence=1.0,
        mix_in=first_downbeat + mix_in_bar * bar_s,
        mix_out=first_downbeat + mix_out_bar * bar_s,
        mix_in_bar=mix_in_bar,
        mix_out_bar=mix_out_bar,
        mix_points_estimated=False,
    )


def deck_at(a: TrackAnalysis, bar: float) -> Deck:
    deck = Deck("a")
    deck.track = LoadedTrack(
        analysis=a,
        audio=np.zeros((int(a.duration_s * SAMPLE_RATE) + 4, CHANNELS), np.float32),
    )
    deck.position = phrase.frame_at_bar(a, bar)
    deck.playing = True
    return deck


# --- mix point detection -----------------------------------------------------


def rms_trace(intro_bars, body_bars, outro_bars, level=1.0, quiet=0.05):
    beats = (intro_bars + body_bars + outro_bars) * BEATS_PER_BAR
    out = np.full(beats, level)
    out[: intro_bars * BEATS_PER_BAR] = quiet
    if outro_bars:
        out[-outro_bars * BEATS_PER_BAR :] = quiet
    return out


def beat_times(n, bpm=120.0, offset=0.0):
    return np.array([offset + i * 60.0 / bpm for i in range(n)])


def test_mix_in_skips_a_quiet_intro():
    trace = rms_trace(intro_bars=8, body_bars=64, outro_bars=8)
    beats = beat_times(trace.size)
    mix_in, mix_out, estimated = analysis._mix_points(beats, trace, 0, 120.0)
    assert not estimated
    bar_s = 2.0
    assert mix_in / bar_s == pytest.approx(8.0, abs=1.0)


def test_mix_out_lands_at_the_start_of_the_outro():
    trace = rms_trace(intro_bars=8, body_bars=64, outro_bars=16)
    beats = beat_times(trace.size)
    mix_in, mix_out, estimated = analysis._mix_points(beats, trace, 0, 120.0)
    assert not estimated
    bar_s = 2.0
    assert mix_out / bar_s == pytest.approx(72.0, abs=1.0)
    assert mix_out > mix_in


def test_mix_in_requires_eight_sustained_bars():
    """A single loud stab in an ambient intro must not read as the drop."""
    trace = rms_trace(intro_bars=16, body_bars=64, outro_bars=8)
    trace[4 * BEATS_PER_BAR : 5 * BEATS_PER_BAR] = 1.0  # one loud bar at bar 4
    beats = beat_times(trace.size)
    mix_in, _out, estimated = analysis._mix_points(beats, trace, 0, 120.0)
    assert not estimated
    assert mix_in / 2.0 >= 15.0, "latched onto a transient, not the sustained body"


def test_a_track_with_no_quiet_sections_mixes_out_at_the_end():
    trace = rms_trace(intro_bars=0, body_bars=64, outro_bars=0)
    beats = beat_times(trace.size)
    mix_in, mix_out, estimated = analysis._mix_points(beats, trace, 0, 120.0)
    assert not estimated
    assert mix_in == pytest.approx(0.0, abs=1e-6)
    assert mix_out / 2.0 == pytest.approx(63.0, abs=1.0)


def test_unreadable_energy_falls_back_and_flags_it():
    trace = np.zeros(64 * BEATS_PER_BAR)
    beats = beat_times(trace.size)
    mix_in, mix_out, estimated = analysis._mix_points(beats, trace, 0, 120.0)
    assert estimated is True
    assert mix_out > mix_in


def test_fallback_uses_16_and_32_bars():
    trace = np.zeros(128 * BEATS_PER_BAR)
    beats = beat_times(trace.size)
    mix_in, mix_out, estimated = analysis._mix_points(beats, trace, 0, 120.0)
    assert estimated
    assert mix_in / 2.0 == pytest.approx(16.0, abs=0.1)
    assert mix_out / 2.0 == pytest.approx(127.0 - 32.0, abs=1.5)


def test_mix_points_respect_the_downbeat_phase():
    trace = rms_trace(intro_bars=8, body_bars=48, outro_bars=8)
    beats = beat_times(trace.size)
    mix_in, mix_out, _ = analysis._mix_points(beats, trace, 2, 120.0)
    period = 0.5
    assert (mix_in / period - 2) % BEATS_PER_BAR == pytest.approx(0.0, abs=1e-6)
    assert (mix_out / period - 2) % BEATS_PER_BAR == pytest.approx(0.0, abs=1e-6)


def test_schema_version_was_bumped_for_mix_points():
    assert analysis.ANALYSIS_VERSION >= 3


def test_stale_sidecars_are_reported_not_silently_dropped(tmp_path, capsys):
    """Too old to bring forward: reported, never silently dropped."""
    import json

    a = make_analysis(title="old")
    d = json.loads(a.to_json())
    # Below every version in UPGRADABLE_FROM, so there is no path forward.
    d["analysis_version"] = min(analysis.UPGRADABLE_FROM + (analysis.ANALYSIS_VERSION,)) - 1
    (tmp_path / "old.json").write_text(json.dumps(d), encoding="utf-8")

    crate = analysis.load_crate(tmp_path)
    assert crate == []
    out = capsys.readouterr().out
    assert "predate analysis schema" in out
    assert "analyze" in out


def test_only_the_storage_only_schema_change_is_brought_forward(tmp_path, capsys):
    """v6 moved three lists to an array file; v5 changed the tempo itself.

    A v5 sidecar holds numbers measured the current way, only stored inline, so
    it is split and rewritten in place. Anything older still predates the tempo
    fix: no arithmetic on it recovers a number that came out of the audio, and
    reading one would mean trusting the tempo the v5 bump exists to fix.
    """
    import json

    # v7 is upgradable too: v8 only added structure fields that start empty.
    assert analysis.UPGRADABLE_FROM == (5, 6, 7), "only the schema changes that re-measure nothing upgrade"

    audio = tmp_path / "old.wav"
    audio.write_bytes(b"placeholder")
    a = make_analysis(title="old")
    for version in (3, 4):
        d = json.loads(a.to_json())
        d["analysis_version"] = version
        d["path"] = str(audio)
        (tmp_path / f"v{version}.json").write_text(json.dumps(d), encoding="utf-8")

    crate = analysis.load_crate(tmp_path)
    assert crate == [], "a stale tempo must never be read as authoritative"
    out = capsys.readouterr().out
    assert "predate analysis schema" in out
    assert "upgraded in place" not in out


# --- backwards placement -----------------------------------------------------


def test_transition_ends_at_mix_out():
    a = make_analysis(mix_out_bar=96.0)
    b = make_analysis(title="b", mix_in_bar=16.0)
    plan = phrase.plan_transition(deck_at(a, 40.0), b, 16.0)
    assert plan is not None and not plan.is_cut

    start_bar = phrase.bar_at_frame(a, plan.start_frame)
    end_bar = start_bar + plan.bars
    assert abs(end_bar - a.mix_out_bar) <= 2.0, (
        f"transition ends at bar {end_bar}, mix_out is {a.mix_out_bar}"
    )


def test_start_snaps_down_to_an_8_bar_boundary():
    a = make_analysis(mix_out_bar=95.0)  # 95 - 16 = 79 -> snaps down to 72
    b = make_analysis(title="b")
    plan = phrase.plan_transition(deck_at(a, 10.0), b, 16.0)
    start_bar = phrase.bar_at_frame(a, plan.start_frame)
    assert start_bar == pytest.approx(72.0, abs=0.01)
    assert start_bar % phrase.PLACEMENT_GRID_BARS == pytest.approx(0.0, abs=0.01)


def test_length_absorbs_the_snap_remainder_so_it_ends_on_mix_out():
    """Snapping the start down would otherwise finish up to 7 bars early."""
    for mix_out in (88.0, 89.0, 91.0, 93.0, 95.0, 96.0):
        a = make_analysis(mix_out_bar=mix_out)
        b = make_analysis(title="b")
        plan = phrase.plan_transition(deck_at(a, 10.0), b, 16.0)
        start_bar = phrase.bar_at_frame(a, plan.start_frame)
        assert start_bar % phrase.PLACEMENT_GRID_BARS == pytest.approx(0.0, abs=0.01)
        assert start_bar + plan.bars == pytest.approx(mix_out, abs=0.01)
        assert 16.0 <= plan.bars < 16.0 + phrase.PLACEMENT_GRID_BARS


def test_incoming_deck_enters_at_its_mix_in_never_zero():
    a = make_analysis()
    b = make_analysis(title="b", first_downbeat=3.0, mix_in_bar=24.0)
    plan = phrase.plan_transition(deck_at(a, 10.0), b, 16.0)
    assert plan.entry_frame_b == int(round(b.mix_in * SAMPLE_RATE))
    assert plan.entry_frame_b > 0
    assert phrase.bar_at_frame(b, plan.entry_frame_b) == pytest.approx(24.0, abs=0.01)


def test_late_selection_shortens_the_transition_and_logs_why():
    a = make_analysis(mix_out_bar=96.0)
    b = make_analysis(title="b")
    # Ideal start is bar 80; we are already at bar 84.
    plan = phrase.plan_transition(deck_at(a, 84.0), b, 16.0)
    assert plan is not None and not plan.is_cut
    start_bar = phrase.bar_at_frame(a, plan.start_frame)
    assert start_bar > 84.0
    assert plan.bars >= phrase.MIN_TRANSITION_BARS
    assert start_bar + plan.bars <= a.mix_out_bar + 0.01
    assert plan.reason and "shortened" in plan.reason


def test_no_room_left_plans_a_cut_and_logs_it(caplog):
    a = make_analysis(mix_out_bar=96.0)
    b = make_analysis(title="b")
    with caplog.at_level("WARNING"):
        plan = phrase.plan_transition(deck_at(a, 94.0), b, 16.0)
    assert plan is not None and plan.is_cut
    assert plan.bars == 0.0
    assert plan.reason and "cutting" in plan.reason
    assert any("no room to blend" in r.message for r in caplog.records)


def test_plan_never_returns_a_start_before_now():
    a = make_analysis(mix_out_bar=96.0)
    b = make_analysis(title="b")
    for now_bar in (0.0, 40.0, 79.0, 80.0, 88.0, 92.0):
        plan = phrase.plan_transition(deck_at(a, now_bar), b, 16.0)
        start_bar = phrase.bar_at_frame(a, plan.start_frame)
        assert start_bar >= now_bar - 0.01, f"start {start_bar} is behind {now_bar}"


def test_plan_returns_none_without_a_track():
    assert phrase.plan_transition(Deck("empty"), make_analysis(), 16.0) is None


def test_next_phrase_boundary_contract_is_unchanged():
    """It still answers alignment, on the 32-bar grid, forwards only."""
    a = make_analysis()
    deck = deck_at(a, 5.0)
    boundary = phrase.next_phrase_boundary(deck)
    assert phrase.bar_at_frame(a, boundary) == pytest.approx(32.0, abs=0.01)
    assert phrase.BARS_PER_PHRASE == 32
    exact = phrase.frame_at_bar(a, 32.0)
    assert phrase.bar_at_frame(a, phrase.next_phrase_boundary(deck, exact)) == (
        pytest.approx(64.0, abs=0.01)
    )


def test_bar_zero_is_the_first_downbeat_not_sample_zero():
    """A track with a silent lead-in still reports bar 0 at its first downbeat."""
    a = make_analysis(first_downbeat=3.0)
    assert phrase.bar_at_frame(a, a.first_downbeat * SAMPLE_RATE) == pytest.approx(
        0.0, abs=1e-6
    )
    assert phrase.bar_at_frame(a, 0.0) < -1.0, "sample 0 should be a negative bar"
    assert phrase.frame_at_bar(a, 0.0) == pytest.approx(3.0 * SAMPLE_RATE, abs=1.0)


# --- supervisor regression guard ---------------------------------------------


def test_supervisor_rejects_a_mid_track_transition(placement_rig):
    engine, supervisor, a, _b = placement_rig
    engine.deck_a.position = phrase.frame_at_bar(a, 8.0)  # ~8% in

    rejection = supervisor.validate(
        StartTransition(from_deck="a", to_deck="b", total_frames=SAMPLE_RATE * 10,
                        origin="llm")
    )
    assert rejection is not None
    assert "would start at" in rejection.reason
    assert "plan_transition" in rejection.reason


def test_supervisor_logs_both_positions_on_rejection(placement_rig, caplog):
    import json

    engine, supervisor, a, _b = placement_rig
    engine.deck_a.position = phrase.frame_at_bar(a, 4.0)
    supervisor.validate(
        StartTransition(from_deck="a", to_deck="b", total_frames=1000, origin="llm")
    )
    supervisor.log._file.flush()
    entries = [
        json.loads(line)
        for line in supervisor.log.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rejected = [e for e in entries if e["event"] == "command_rejected"]
    assert rejected
    action = rejected[-1]["action"]
    assert "%" in action and "mix-out" in action


def test_guard_allows_a_correctly_placed_blend_on_a_short_track(placement_rig):
    """A blend is a fixed number of bars, so on a short track it is a large
    fraction of the running time. 24 bars is 52 s; on a 103 s track that has to
    start at 45% however correctly it is placed. The guard is about placement
    being anchored to mix-out, not about absolute track length."""
    engine, supervisor, a, _b = placement_rig
    # 45% of the way in -- what a correct placement looks like on a short track.
    engine.deck_a.position = a.duration_s * 0.45 * SAMPLE_RATE

    assert supervisor.validate(
        StartTransition(from_deck="a", to_deck="b", total_frames=SAMPLE_RATE * 10)
    ) is None


def test_guard_still_catches_a_first_boundary_after_selection_placement(placement_rig):
    """The original bug: scheduled at the first phrase boundary after the track
    was picked. Bar 32 of a 128-bar track is 25% -- still well under the guard."""
    engine, supervisor, a, _b = placement_rig
    engine.deck_a.position = phrase.frame_at_bar(a, 32.0)
    rejection = supervisor.validate(
        StartTransition(from_deck="a", to_deck="b", total_frames=SAMPLE_RATE * 10,
                        origin="autopilot")
    )
    assert rejection is not None
    assert "plan_transition" in rejection.reason


def test_supervisor_accepts_a_properly_placed_transition(placement_rig):
    engine, supervisor, a, _b = placement_rig
    engine.deck_a.position = phrase.frame_at_bar(a, 80.0)
    assert supervisor.validate(
        StartTransition(from_deck="a", to_deck="b", total_frames=SAMPLE_RATE * 10)
    ) is None


@pytest.fixture
def placement_rig(tmp_path):
    from djai.commands import LoadTrack
    from djai.engine import Engine
    from djai.scheduler import Scheduler
    from djai.supervisor import SessionLog, Supervisor
    from tests.test_engine import drive

    a = make_analysis(title="a", bars=128, mix_out_bar=96.0)
    b = make_analysis(title="b", bars=128, mix_in_bar=16.0)
    for t in (a, b):
        p = tmp_path / f"{t.title}.wav"
        p.write_bytes(b"x")
        t.path = str(p)

    engine = Engine(blocksize=512)
    for name, t in (("a", a), ("b", b)):
        audio = np.zeros((int(t.duration_s * SAMPLE_RATE) + 4, CHANNELS), np.float32)
        engine.submit(
            LoadTrack(deck=name, track=LoadedTrack(analysis=t, audio=audio),
                      master=(name == "a"))
        )
    drive(engine, 2)

    supervisor = Supervisor(
        engine, Scheduler(engine), [a, b], session_log=SessionLog(tmp_path)
    )
    return engine, supervisor, a, b

