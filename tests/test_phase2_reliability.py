"""Phase 2: the music goes on -- no blend partner, no model, no blocking.

THREADING CONTEXT: main thread (pytest). Real Sessions on synthetic audio, no
device and no background threads; the audio callback is driven directly and the
autopilot is ticked at the pace its own thread keeps.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from djai import cli
from djai.analysis import TrackAnalysis
from djai.commands import StartTransition
from djai.deck import LoadedTrack, TransportState
from djai.intent import IntentEngine
from tests.test_cue_regression import BLOCKS_PER_TICK, play_to_outro
from tests.test_engine import drive, loud_track
from tests.test_integration import log_events, make_analysis
from tests.test_integration import session as _session_fixture

session = _session_fixture

BLOCK = 512
#: Far enough apart that no rate can match them: 128 -> 200 is +56%.
SLOW_BPM, FAST_BPM = 128.0, 200.0


@pytest.fixture
def unmatched_session(tmp_path, monkeypatch):
    """A crate of two tracks whose tempos have no blend between them."""
    crate = []
    for tid, bpm, cam in (("slow", SLOW_BPM, "8A"), ("fast", FAST_BPM, "8A")):
        p = tmp_path / f"{tid}.wav"
        p.write_bytes(b"placeholder")
        crate.append(make_analysis(tid, bpm, cam, 0.5, p))

    def fake_load(analysis: TrackAnalysis) -> LoadedTrack:
        base = loud_track(200.0 + 500 * len(analysis.title), analysis.bpm, 240.0)
        return LoadedTrack(analysis=analysis, audio=base.audio)

    monkeypatch.setattr(cli, "load_track", fake_load)
    monkeypatch.setattr("djai.config.TIME_STRETCH_ENABLED", False)
    s = cli.Session(crate, log_dir=tmp_path / "logs")
    s.start_first_track()
    drive(s.engine, 4, BLOCK)
    yield s
    s.shutdown()


def events(s, name):
    return [e for e in log_events(s) if e["event"] == name]


# --- a tempo with no partner ------------------------------------------------------


def test_a_track_with_no_blend_partner_is_still_cued(unmatched_session):
    s = unmatched_session
    live = s.engine.deck(s.live_deck)

    assert s.cue_next() is True, "the last resort must find something"

    cued = s._cued
    assert cued is not None and cued.analysis.track_id != live.track.analysis.track_id
    orphan = events(s, "tempo_orphan")
    assert orphan and "cut" in orphan[-1]["action"]
    assert events(s, "track_cued")
    assert not events(s, "no_track_cued")


def test_the_unmatched_track_is_cued_at_its_own_tempo(unmatched_session):
    s = unmatched_session
    s.cue_next()
    drive(s.engine, 4, BLOCK)
    idle = s.engine.deck(s.cued_deck())
    assert idle.track is not None
    assert idle.rate == pytest.approx(1.0), "no rate could match; it plays its own"


def test_the_hand_over_is_a_one_block_cut_the_supervisor_accepts(unmatched_session):
    s = unmatched_session
    from tests.test_integration import seek_near_mix_out

    seek_near_mix_out(s, bars_before=40.0)
    s.cue_next()
    drive(s.engine, 2, BLOCK)

    assert s.arm_transition() is not None, "the supervisor must allow the cut"
    swap = next(c for c in s.scheduler.pending() if isinstance(c, StartTransition))
    assert swap.total_frames == s.engine.blocksize
    assert s.last_transition_choice[0] == "cut"
    assert not events(s, "command_rejected")


def test_the_room_hears_the_next_track_across_the_cut(unmatched_session):
    """The point of the whole thing: no silence where a blend cannot happen."""
    s = unmatched_session
    from tests.test_integration import seek_near_mix_out

    seek_near_mix_out(s, bars_before=40.0)
    before = s.live_deck
    quiet_run = worst_quiet = 0
    for _ in range(600):
        s._autopilot_tick()
        # The Session is not started here, so the scheduler thread is not
        # running: releasing what it holds is this loop's job.
        s.scheduler.tick(s.engine.frames_played)
        out = drive(s.engine, BLOCKS_PER_TICK // 8 or 1, BLOCK)
        if float(np.max(np.abs(out))) < 1e-4:
            quiet_run += 1
            worst_quiet = max(worst_quiet, quiet_run)
        else:
            quiet_run = 0
        if s.live_deck != before:
            break

    assert s.live_deck != before, "the cut never handed over"
    assert s.engine.deck(s.live_deck).transport is TransportState.PLAYING
    assert worst_quiet == 0, f"{worst_quiet} silent block(s) across the hand-over"
    assert events(s, "transition_complete")


# --- no model ---------------------------------------------------------------------


def test_cueing_and_arming_continue_with_the_model_dead(session):
    """Nothing in the cue path may wait on the LLM."""
    engine = IntentEngine(base_url="http://127.0.0.1:9", timeout_s=0.2)
    session.intent_engine = engine
    try:
        left = play_to_outro(session, session.live_deck)
    finally:
        engine.close()

    assert left is not None and left > 0.0, "cued before mix-out with no model"
    assert events(session, "track_cued")
    assert session.design_counts.get("model", 0) == 0


def test_a_design_that_never_answers_does_not_hold_up_the_transition(session):
    """A model that hangs is a slow answer, not a stopped mix."""
    import threading

    class Hanging:
        available = True
        model = "hanging"

        def design_transition(self, context):
            threading.Event().wait(30.0)  # never answers in time
            return None, "too late"

    session.intent_engine = Hanging()
    left = play_to_outro(session, session.live_deck)
    assert left is not None and left > 0.0

    armed = None
    for _ in range(400):
        session._autopilot_tick()
        drive(session.engine, BLOCKS_PER_TICK, BLOCK)
        if session._transition_armed:
            armed = True
            break
    assert armed, "the transition must arm on the preset rather than wait"


def test_the_model_is_warmed_without_holding_up_the_audio_stream():
    """Startup measured 62 s for a cold 8B load: a minute of unplayable room."""
    src = inspect.getsource(cli.cmd_play)
    assert "threading.Thread(target=warm" in src
    warm_call = src.index("intent_engine.warmup()")
    thread_start = src.index("threading.Thread(target=warm")
    assert src.index("def warm()") < warm_call < thread_start, (
        "warmup() must only be called from the background thread"
    )
    assert thread_start < src.index("session.start()"), (
        "the warm starts before the stream, and does not gate it"
    )
