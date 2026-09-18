"""Supervisor tests: drift response at each threshold, and the validator.

THREADING CONTEXT: main thread (pytest). The monitor thread is never started;
``check_drift`` and ``validate`` are called directly so each threshold can be
exercised deterministically.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from djai import phrase
from djai import supervisor as sup
from djai.commands import LoadTrack, Resync, SetEQ, SetRate, StartTransition
from djai.deck import SAMPLE_RATE
from djai.engine import Engine
from djai.scheduler import Scheduler
from djai.supervisor import SessionLog, Supervisor
from tests.test_engine import drive, loud_track


@pytest.fixture
def rig(tmp_path):
    """Engine with two decks loaded, plus a scheduler and supervisor."""
    engine = Engine(blocksize=512)
    track_a = loud_track(200.0, bpm=120.0, seconds=300.0)
    track_b = loud_track(3000.0, bpm=120.0, seconds=300.0)
    track_b.analysis.track_id = "track_b"
    track_b.analysis.title = "track_b"
    # The validator checks that the audio file still exists on disk, so the
    # fixture's analyses have to point at real paths.
    for name, track in (("a", track_a), ("b", track_b)):
        p = tmp_path / f"track_{name}.wav"
        p.write_bytes(b"placeholder")
        track.analysis.path = str(p)

    engine.submit(LoadTrack(deck="a", track=track_a, master=True))
    engine.submit(LoadTrack(deck="b", track=track_b, play=False))
    engine.deck_a.gain.jump(1.0)
    drive(engine, 4)

    scheduler = Scheduler(engine)
    crate = [track_a.analysis, track_b.analysis]
    log = SessionLog(tmp_path)
    supervisor = Supervisor(engine, scheduler, crate, session_log=log)
    return engine, scheduler, supervisor, track_a, track_b


def read_log(supervisor: Supervisor) -> list[dict]:
    supervisor.log._file.flush()
    return [
        json.loads(line)
        for line in supervisor.log.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def shift_deck(engine: Engine, deck_name: str, ms: float) -> None:
    """Push a deck's playhead off the grid by a given number of milliseconds.

    Drives one callback afterwards: the supervisor reads ``engine.clock``, the
    consistent snapshot published by the audio callback, not the live
    attributes, so an injected shift is invisible until a block is rendered.
    """
    engine.deck(deck_name).position += ms / 1000.0 * SAMPLE_RATE
    drive(engine, 1)


# --- drift monitor -----------------------------------------------------------


def test_small_drift_is_ignored(rig):
    engine, scheduler, supervisor, *_ = rig
    supervisor.check_drift()  # establish the baseline
    scheduler.cancel_all()

    shift_deck(engine, "a", 2.0)  # under DRIFT_IGNORE_MS
    supervisor.check_drift()

    assert supervisor.interventions == 0
    assert not any(e["event"].startswith("drift_") for e in read_log(supervisor))


def test_medium_drift_nudges_the_rate_and_logs_it(rig):
    engine, scheduler, supervisor, *_ = rig
    supervisor.check_drift()
    scheduler.cancel_all()
    before = engine.deck_a.rate

    shift_deck(engine, "a", 10.0)  # between ignore and resync
    supervisor.check_drift()

    pending = scheduler.pending() + list(engine.queue.queue)
    rate_cmds = [c for c in pending if isinstance(c, SetRate)]
    assert rate_cmds, "a 10 ms drift should produce a rate nudge"
    # Running ahead means slowing down.
    assert rate_cmds[-1].rate < before
    assert abs(rate_cmds[-1].rate - before) / before <= sup.MAX_RATE_NUDGE + 1e-9

    entries = [e for e in read_log(supervisor) if e["event"] == "drift_nudge"]
    assert entries, "the intervention must be logged"
    entry = entries[-1]
    assert entry["deck"] == "a"
    assert "track_pair" in entry and "position_bars" in entry
    assert "trigger" in entry and "action" in entry
    assert entry["drift_ms"] == pytest.approx(10.0, abs=1.0)


def test_nudge_direction_reverses_when_running_behind(rig):
    engine, scheduler, supervisor, *_ = rig
    supervisor.check_drift()
    scheduler.cancel_all()
    before = engine.deck_a.rate

    shift_deck(engine, "a", -10.0)
    supervisor.check_drift()

    rate_cmds = [
        c for c in scheduler.pending() + list(engine.queue.queue)
        if isinstance(c, SetRate)
    ]
    assert rate_cmds and rate_cmds[-1].rate > before, "behind means speed up"


def test_large_drift_hard_resyncs_at_the_next_downbeat(rig):
    engine, scheduler, supervisor, *_ = rig
    supervisor.check_drift()
    scheduler.cancel_all()

    shift_deck(engine, "a", 50.0)  # over DRIFT_RESYNC_MS
    supervisor.check_drift()

    pending = scheduler.pending()
    resyncs = [c for c in pending if isinstance(c, Resync)]
    assert resyncs, "a 50 ms drift should hard-resync"
    assert not resyncs[0].is_immediate, "resync must wait for the downbeat"

    entries = [e for e in read_log(supervisor) if e["event"] == "drift_resync"]
    assert entries and entries[-1]["drift_ms"] == pytest.approx(50.0, abs=2.0)


def test_nudges_do_not_accumulate(rig):
    """Repeated nudges are relative to nominal, never to the bent rate."""
    engine, scheduler, supervisor, *_ = rig
    supervisor.check_drift()
    nominal = supervisor.nominal_rate(engine.deck_a)

    for _ in range(5):
        shift_deck(engine, "a", 10.0)
        supervisor.check_drift()
        for cmd in scheduler.tick(10**12):
            pass
        drive(engine, 1)  # let the engine apply the queued SetRate

    assert abs(engine.deck_a.rate - nominal) / nominal <= sup.MAX_RATE_NUDGE + 1e-9


def test_rate_is_restored_once_drift_clears(rig):
    engine, scheduler, supervisor, *_ = rig
    supervisor.check_drift()
    nominal = supervisor.nominal_rate(engine.deck_a)

    shift_deck(engine, "a", 10.0)
    supervisor.check_drift()
    scheduler.tick(10**12)
    drive(engine, 1)
    assert engine.deck_a.rate != nominal

    # Put it back on the grid; the supervisor should undo its correction.
    shift_deck(engine, "a", -10.0)
    supervisor.check_drift()
    scheduler.tick(10**12)
    drive(engine, 1)
    assert engine.deck_a.rate == pytest.approx(nominal)
    assert any(e["event"] == "drift_settled" for e in read_log(supervisor))


def test_recueing_the_same_track_rebaselines_the_phase(rig):
    """Regression: a 10-minute run reported 181 s of "drift" on a deck.

    The baseline was keyed on track id, so re-cueing the *same* track onto the
    same deck kept an offset captured three minutes and one playhead earlier.
    It is now keyed on Deck.load_seq, which changes on every attach.
    """
    engine, scheduler, supervisor, track_a, _ = rig
    supervisor.check_drift()
    scheduler.cancel_all()

    # Same track, fresh playhead, far from where it was.
    engine.submit(LoadTrack(deck="a", track=track_a, start_frame=120 * SAMPLE_RATE))
    drive(engine, 4)
    supervisor.check_drift()  # must re-baseline, not report a huge drift
    supervisor.check_drift()

    entries = read_log(supervisor)
    assert not any(e["event"] == "drift_resync" for e in entries), (
        "re-cueing the same track was mistaken for drift"
    )
    assert supervisor.interventions == 0


def test_implausible_drift_rebaselines_instead_of_resyncing(rig):
    engine, scheduler, supervisor, *_ = rig
    supervisor.check_drift()
    scheduler.cancel_all()

    shift_deck(engine, "a", 60_000.0)  # a full minute: not credible as drift
    supervisor.check_drift()

    assert not [c for c in scheduler.pending() if isinstance(c, Resync)]
    entries = [e for e in read_log(supervisor) if e["event"] == "baseline_reset"]
    assert entries, "an implausible reading should be logged as a baseline reset"

    # And the next check is quiet, because the baseline was recaptured.
    before = supervisor.interventions
    supervisor.check_drift()
    supervisor.check_drift()
    assert supervisor.interventions == before


def test_a_resync_in_flight_is_not_reissued_every_tick(rig):
    engine, scheduler, supervisor, *_ = rig
    supervisor.check_drift()
    scheduler.cancel_all()

    shift_deck(engine, "a", 60.0)
    for _ in range(10):
        supervisor.check_drift()

    resyncs = [c for c in scheduler.pending() if isinstance(c, Resync)]
    assert len(resyncs) == 1, f"queued {len(resyncs)} duplicate resyncs"


def test_a_stalled_stream_is_reported_once_and_on_recovery(rig, monkeypatch):
    """Regression: a real 10-minute run produced 481 s of audio in 630 s.

    PortAudio stopped calling back, reported no error, and left `underruns` at
    0 -- nothing in the program could tell the music had stopped.
    """
    engine, _scheduler, supervisor, *_ = rig
    monkeypatch.setattr(type(engine), "running", property(lambda self: True))

    clock = [1000.0]
    monkeypatch.setattr(sup.time, "monotonic", lambda: clock[0])

    drive(engine, 2)
    supervisor.check_stalled()  # establishes progress
    assert not supervisor.stalled

    clock[0] += sup.STALL_TIMEOUT_S + 0.5  # no callbacks in the meantime
    supervisor.check_stalled()
    assert supervisor.stalled
    stalls = [e for e in read_log(supervisor) if e["event"] == "stream_stalled"]
    assert len(stalls) == 1
    assert "no audio callback" in stalls[0]["trigger"]

    clock[0] += 10.0
    for _ in range(5):
        supervisor.check_stalled()
    assert len([e for e in read_log(supervisor) if e["event"] == "stream_stalled"]) == 1

    drive(engine, 1)  # the stream comes back
    supervisor.check_stalled()
    assert not supervisor.stalled
    assert any(e["event"] == "stream_resumed" for e in read_log(supervisor))


def test_baseline_aligns_bars_to_the_master_deck_not_the_free_counter(rig):
    """Regression: a +469 ms baseline error at every transition.

    `master_beat` is a counter that starts when a track loads and has no
    particular phase relative to the master deck's bar lines. Rounding the new
    deck's offset against *it* could land a beat out, and the monitor then
    resynced every bar without ever converging. The baseline must be the bar
    difference between the two decks.
    """
    engine, scheduler, supervisor, track_a, track_b = rig

    # Put the free counter deliberately out of phase with deck A's bars.
    engine.master_beat += 1.7
    drive(engine, 1)

    # Bar-aligned means the two decks share a bar *phase* -- a whole number of
    # bars apart, not each on an integer bar of its own grid.
    a, b = track_a.analysis, track_b.analysis
    bar_a = phrase.bar_at_frame(a, engine.deck_a.position)
    engine.deck_b.position = phrase.frame_at_bar(b, bar_a + 8.0)
    engine.deck_b.playing = True
    drive(engine, 1)

    supervisor.check_drift()  # captures the baseline
    scheduler.cancel_all()
    supervisor.check_drift()  # now measures against it

    assert supervisor.interventions == 0, (
        "a bar-aligned entry should read as zero drift regardless of the "
        "free counter's phase"
    )
    entries = [e for e in read_log(supervisor) if e["event"].startswith("drift_")]
    assert not entries, f"unexpected drift events: {entries}"


def test_a_deck_entering_off_the_bar_grid_is_detected(rig):
    """The flip side: real misalignment must still be caught and corrected."""
    engine, scheduler, supervisor, track_a, track_b = rig
    a, b = track_a.analysis, track_b.analysis

    bar_a = phrase.bar_at_frame(a, engine.deck_a.position)
    # A quarter of a bar out -- one beat, unmistakably wrong.
    engine.deck_b.position = phrase.frame_at_bar(b, bar_a + 8.25)
    engine.deck_b.playing = True
    drive(engine, 1)

    supervisor.check_drift()
    scheduler.cancel_all()
    supervisor.check_drift()

    assert supervisor.interventions >= 1
    entries = [e for e in read_log(supervisor) if e["event"].startswith("drift_")]
    assert entries, "a one-beat misalignment should be reported"
    # One beat at 120 BPM is 500 ms.
    assert abs(entries[-1]["drift_ms"]) == pytest.approx(500.0, rel=0.1)


def test_resync_uses_one_consistent_clock_snapshot(rig):
    """Regression: resyncs repeated every bar at -22 to -42 ms, never
    converging -- one 2048-frame block, from mixing the snapshot the drift was
    measured from with live reads of position and frames_played."""
    import inspect

    src = inspect.getsource(sup.Supervisor._hard_resync)
    assert "self.engine.frames_played" not in src, (
        "_hard_resync must use the snapshot's frame count, not a live read"
    )
    assert "next_downbeat(deck)" not in src, (
        "_hard_resync must pass the snapshot position to next_downbeat"
    )


def test_underruns_are_logged(rig):
    engine, scheduler, supervisor, *_ = rig
    engine.underruns = 3
    supervisor.check_underruns()
    entries = [e for e in read_log(supervisor) if e["event"] == "underrun"]
    assert entries and entries[-1]["total"] == 3


# --- command validator -------------------------------------------------------


def test_validator_rejects_a_track_not_in_the_cache(rig):
    engine, scheduler, supervisor, *_ = rig
    stranger = loud_track(500.0, bpm=120.0)
    stranger.analysis.track_id = "hallucinated"
    rejection = supervisor.validate(
        LoadTrack(deck="b", track=stranger, origin="llm")
    )
    assert rejection is not None
    assert "not in the analysis cache" in rejection.reason
    assert any(e["event"] == "command_rejected" for e in read_log(supervisor))


def test_validator_rejects_an_out_of_range_rate(rig):
    engine, scheduler, supervisor, track_a, track_b = rig
    rejection = supervisor.validate(
        LoadTrack(deck="b", track=track_b, rate=1.5, origin="llm")
    )
    assert rejection is not None and "stretch range" in rejection.reason


def test_validator_rejects_loading_onto_a_deck_mid_transition(rig):
    engine, scheduler, supervisor, track_a, track_b = rig
    engine.submit(
        StartTransition(from_deck="a", to_deck="b", total_frames=SAMPLE_RATE * 10)
    )
    drive(engine, 2)
    assert engine.transition_active

    rejection = supervisor.validate(LoadTrack(deck="b", track=track_b, origin="llm"))
    assert rejection is not None and "mid-transition" in rejection.reason


def test_validator_rejects_a_second_concurrent_transition(rig):
    engine, scheduler, supervisor, *_ = rig
    engine.submit(
        StartTransition(from_deck="a", to_deck="b", total_frames=SAMPLE_RATE * 10)
    )
    drive(engine, 2)
    rejection = supervisor.validate(
        StartTransition(from_deck="a", to_deck="b", total_frames=100, origin="llm")
    )
    assert rejection is not None and "already running" in rejection.reason


def test_validator_rejects_a_transition_onto_the_same_deck(rig):
    _, _, supervisor, *_ = rig
    rejection = supervisor.validate(
        StartTransition(from_deck="a", to_deck="a", total_frames=100, origin="llm")
    )
    assert rejection is not None and "same deck" in rejection.reason


def test_validator_rejects_out_of_range_eq(rig):
    _, _, supervisor, *_ = rig
    assert supervisor.validate(SetEQ(deck="a", low=17.0, origin="llm")) is not None
    assert supervisor.validate(SetEQ(deck="a", low="loud", origin="llm")) is not None
    assert supervisor.validate(SetEQ(deck="a", low=0.5, origin="llm")) is None


def test_validator_accepts_a_good_command(rig):
    engine, scheduler, supervisor, track_a, track_b = rig
    assert supervisor.validate(LoadTrack(deck="b", track=track_b, rate=1.02)) is None


def test_rejected_commands_never_reach_the_scheduler(rig):
    engine, scheduler, supervisor, *_ = rig
    stranger = loud_track(500.0, bpm=120.0)
    stranger.analysis.track_id = "hallucinated"

    rejection = supervisor.submit_validated(
        LoadTrack(deck="b", track=stranger, origin="llm")
    )
    assert rejection is not None
    assert len(scheduler) == 0
    assert engine.queue.qsize() == 0


def test_playback_continues_after_a_rejection(rig):
    """The whole point: a bad command must not disturb audio."""
    engine, scheduler, supervisor, *_ = rig
    before = float(np.max(np.abs(drive(engine, 20))))
    assert before > 0.1, "precondition: audio should be playing"

    stranger = loud_track(500.0, bpm=120.0)
    stranger.analysis.track_id = "hallucinated"
    supervisor.submit_validated(LoadTrack(deck="b", track=stranger, origin="llm"))

    after = float(np.max(np.abs(drive(engine, 20))))
    assert after == pytest.approx(before, rel=0.05), (
        "audio was disturbed by a rejected command"
    )


def test_session_log_records_every_required_field(rig, tmp_path):
    engine, scheduler, supervisor, *_ = rig
    supervisor.check_drift()
    scheduler.cancel_all()
    shift_deck(engine, "a", 12.0)
    supervisor.check_drift()

    entry = [e for e in read_log(supervisor) if e["event"] == "drift_nudge"][-1]
    for field in ("ts", "event", "track_pair", "position_bars", "trigger", "action"):
        assert field in entry, f"session log entry is missing {field!r}"
    assert supervisor.log.path.name.startswith("session_")
    assert supervisor.log.path.suffix == ".jsonl"
