"""End-to-end tests through the CLI layer, without a sound card or an API key.

THREADING CONTEXT: main thread (pytest). A real :class:`~djai.cli.Session` is
built but never ``start()``ed, so no device is opened and no background threads
run; the audio callback and each control-layer method are driven directly. That
makes the assertions deterministic and lets the LLM be replaced by a stub that
returns exactly the malformed output we want to defend against.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from djai import cli
from djai.analysis import TrackAnalysis
from djai.commands import LoadTrack, StartTransition
from djai.deck import SAMPLE_RATE, LoadedTrack
from djai.intent import Intent, parse_response
from tests.test_engine import drive, loud_track


def make_analysis(tid: str, bpm: float, camelot: str, energy: float, path) -> TrackAnalysis:
    period = 60.0 / bpm
    n = int(240 / period)
    bar_s = 4 * period
    total_bars = n // 4
    # Mix points matter now: placement is planned backwards from mix_out, and a
    # track without them reads as "no room to blend".
    mix_in_bar, mix_out_bar = 8.0, float(total_bars - 8)
    return TrackAnalysis(
        track_id=tid,
        path=str(path),
        title=tid,
        duration_s=240.0,
        bpm=bpm,
        beats=[i * period for i in range(n)],
        downbeats=[i * period for i in range(0, n, 4)],
        key_name="A minor",
        camelot=camelot,
        beat_rms=[energy] * n,
        energy=energy,
        grid_confidence=1.0,
        mix_in=mix_in_bar * bar_s,
        mix_out=mix_out_bar * bar_s,
        mix_in_bar=mix_in_bar,
        mix_out_bar=mix_out_bar,
        mix_points_estimated=False,
    )


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A real Session wired to synthetic audio, with no device and no threads."""
    crate = []
    for i, (tid, bpm, cam, energy) in enumerate(
        [
            ("opener", 124.0, "8A", 0.2),
            ("mid", 125.0, "9A", 0.5),
            ("banger", 126.0, "8B", 1.0),
        ]
    ):
        p = tmp_path / f"{tid}.wav"
        p.write_bytes(b"placeholder")
        crate.append(make_analysis(tid, bpm, cam, energy, p))

    def fake_load(analysis: TrackAnalysis) -> LoadedTrack:
        base = loud_track(200.0 + 500 * len(analysis.title), analysis.bpm, 240.0)
        return LoadedTrack(analysis=analysis, audio=base.audio)

    monkeypatch.setattr(cli, "load_track", fake_load)
    # No time stretching for these sessions. Their tracks are four-minute
    # synthetic tones at 124-126 BPM, so every cue asked the engine's worker to
    # phase-vocode a full track in the background -- gigabytes each, measured at
    # 6.0 GB for this file against 1.9 GB without. Nothing that borrows this
    # fixture tests stretching; tests/test_stretch.py does, on its own engines.
    # Read by the engine at construction, so it has to be set before Session().
    monkeypatch.setattr("djai.config.TIME_STRETCH_ENABLED", False)

    s = cli.Session(crate, log_dir=tmp_path / "logs")
    # Deliberately not s.start(): no OutputStream, no background threads.
    s.start_first_track()
    drive(s.engine, 4)
    yield s
    # The engine owns a stretch worker whose thread is started on construction
    # and keeps its cache of full-length stretched tracks alive until stopped.
    # Without this, every test that borrowed a session left one behind, and the
    # suite grew past the machine's memory. Shutdown is safe on a session that
    # was never started: each thread it joins is checked for first.
    s.shutdown()


def log_events(session: cli.Session) -> list[dict]:
    session.session_log._file.flush()
    return [
        json.loads(line)
        for line in session.session_log.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audio_level(session: cli.Session, blocks: int = 20) -> float:
    return float(np.max(np.abs(drive(session.engine, blocks))))


# --- the malformed-LLM-response criterion ------------------------------------


MALFORMED = [
    "I'd love to help! Let me cue something up for you.",  # prose, no JSON
    '{"action": "delete_everything", "params": {}, "reply": "ok"}',  # bad action
    '{"action": "next_track", "params": ["not", "an", "object"], "reply": "x"}',
    '{"action": "next_track", "params": {}',  # truncated
    "",
]


@pytest.mark.parametrize("raw", MALFORMED)
def test_malformed_llm_response_is_rejected_and_audio_continues(session, raw):
    before = audio_level(session)
    assert before > 0.1, "precondition: audio should be playing"

    intent = parse_response(raw)
    assert not intent.ok and intent.action == "none"

    # This is what the REPL does with a bad reply: log it, schedule nothing.
    session.session_log.write(
        "intent", trigger="user text", action="none", ok=False, error=intent.error
    )
    cli.apply_intent(session, intent)

    assert len(session.scheduler) == 0, "a malformed reply must schedule nothing"
    after = audio_level(session)
    assert after == pytest.approx(before, rel=0.05), "audio was disturbed"

    failures = [e for e in log_events(session) if e["event"] == "intent" and not e["ok"]]
    assert failures, "the parse failure must be logged"


def test_hallucinated_track_never_reaches_the_decks(session):
    """Even a well-formed reply cannot introduce a track that is not cached."""
    stranger = loud_track(900.0, 124.0, 240.0)
    stranger.analysis.track_id = "hallucinated"
    stranger.analysis.title = "Track That Does Not Exist"

    before = audio_level(session)
    rejection = session.supervisor.submit_validated(
        LoadTrack(deck="b", track=stranger, origin="llm")
    )
    assert rejection is not None
    assert "not in the analysis cache" in rejection.reason
    assert len(session.scheduler) == 0
    assert audio_level(session) == pytest.approx(before, rel=0.05)

    rejected = [e for e in log_events(session) if e["event"] == "command_rejected"]
    assert rejected and "rejected" in rejected[-1]["action"]


# --- the "more energy" criterion ---------------------------------------------


def test_more_energy_cues_a_different_track_and_reports_the_bar_count(session, capsys):
    """`something with more energy` -> a different next track + a bar count."""
    session.cue_next(0.0)  # what the autopilot would have picked on its own
    assert session._cued is not None
    neutral_title = session._cued.title

    session.scheduler.cancel_all()
    session._cued = None
    session.played.discard("banger")
    drive(session.engine, 2)

    intent = Intent(
        action="next_track",
        params={"energy": 1.0},
        reply="Going up - the next one is harder.",
    )
    cli.apply_intent(session, intent)
    out = capsys.readouterr().out

    cued = session._cued
    assert cued is not None
    assert cued.title == "banger", "high energy should pick the hardest track"
    assert cued.title != neutral_title, "should differ from the neutral pick"
    assert "Cued: banger" in out
    assert "bars" in out, f"no bar count printed: {out!r}"

    pending = session.scheduler.pending()
    assert any(isinstance(c, StartTransition) for c in pending), "no transition queued"
    swap = next(c for c in pending if isinstance(c, StartTransition))
    bars = session.bars_until(swap.execute_at)
    assert 0 < bars <= 32, f"bar count until the swap is implausible: {bars}"


def test_less_energy_picks_the_calmer_track(session):
    # The neutral opener is "mid" (energy 0.5, closest to the 0.5 default), so
    # asking for calmer must move to "opener".
    assert session.engine.deck_a.track.analysis.title == "mid"
    session.cue_next(-1.0, origin="user")
    assert session._cued is not None and session._cued.title == "opener"


def seek_near_mix_out(session, bars_before: float = 24.0) -> None:
    """Put the live deck where the autopilot would actually arm a transition."""
    from djai import phrase

    live = session.engine.deck(session.live_deck)
    a = live.track.analysis
    live.position = phrase.frame_at_bar(a, a.mix_out_bar - bars_before)
    drive(session.engine, 2)


def test_skip_queued_drops_pending_commands_without_touching_audio(session):
    seek_near_mix_out(session)
    session.cue_next(0.0)
    drive(session.engine, 2)  # the cue is applied by the audio thread
    session.arm_transition()
    assert len(session.scheduler) > 0

    before = audio_level(session)
    cli.apply_intent(session, Intent(action="skip_queued", reply="dropped"))
    assert len(session.scheduler) == 0
    assert audio_level(session) == pytest.approx(before, rel=0.05)
    assert any(e["event"] == "commands_skipped" for e in log_events(session))


def test_hold_blend_extends_the_transition(session):
    from djai import phrase, transition

    cli.apply_intent(session, Intent(action="hold_blend", params={"bars": 16}))
    assert session.hold_extra_bars == 16

    # Far enough out that the longer blend still fits before mix-out.
    seek_near_mix_out(session, bars_before=64.0)
    session.cue_next(0.0)
    drive(session.engine, 2)  # the cue is applied by the audio thread
    session.arm_transition()
    swap = next(
        c for c in session.scheduler.pending() if isinstance(c, StartTransition)
    )

    live = session.engine.deck(session.live_deck)
    a = live.track.analysis
    start_bar = phrase.bar_at_frame(
        a, phrase.frame_at_bar(a, a.mix_out_bar) - 0
    )
    bars = swap.total_frames / SAMPLE_RATE * session.engine.master_bpm / 60.0 / 4.0
    # The hold lengthens the blend, and it still ends on mix-out.
    assert bars >= transition.TRANSITION_BARS + 16 - 0.01
    assert bars < transition.TRANSITION_BARS + 16 + phrase.PLACEMENT_GRID_BARS
    assert start_bar == pytest.approx(a.mix_out_bar, abs=0.01)


# --- panic path --------------------------------------------------------------


def test_cut_is_immediate_and_bypasses_the_llm(session):
    assert audio_level(session) > 0.1
    cli.handle_panic(session, "cut")

    # Straight to the engine queue: nothing was scheduled or validated.
    assert len(session.scheduler) == 0
    assert session.engine.queue.qsize() == 1

    drive(session.engine, 1)
    assert audio_level(session, blocks=2) < 1e-6, "cut did not silence the mix"
    assert any(
        e["event"] == "panic" and e["trigger"] == "cut" for e in log_events(session)
    )


def test_killbass_is_immediate(session):
    cli.handle_panic(session, "killbass")
    drive(session.engine, 2)
    assert session.engine.deck_a.eq_low.target == pytest.approx(0.0)
    assert any(
        e["event"] == "panic" and e["trigger"] == "killbass"
        for e in log_events(session)
    )


def test_stop_ends_the_repl_and_stops_both_decks(session):
    assert cli.handle_panic(session, "stop") is True
    drive(session.engine, 2)
    assert not session.engine.deck_a.playing
    assert not session.engine.deck_b.playing
    assert session.engine.stop_requested


def test_panic_words_are_matched_before_anything_else():
    """The REPL must treat these as panic commands, not as prose for the LLM."""
    for word in ("cut", "killbass", "stop", "quit", "exit"):
        assert word in cli.PANIC_WORDS


# --- autopilot ---------------------------------------------------------------


def test_autopilot_arms_a_transition_near_the_end_of_a_track(session):
    live = session.engine.deck(session.live_deck)
    # Jump to just inside the lead window.
    lead = session.autopilot_lead_seconds()
    target = live.track.n_frames - int((lead - 5) * SAMPLE_RATE)
    live.position = float(target)
    drive(session.engine, 2)

    session._autopilot_tick()  # cues the next track
    assert session.has_cued_track()
    drive(session.engine, 2)
    session._autopilot_tick()  # arms the transition
    assert session._transition_armed, "autopilot did not arm a transition"
    assert any(e["event"] == "transition_armed" for e in log_events(session))


def test_autopilot_recovers_immediately_when_the_live_deck_runs_out(session):
    """Regression: the mix went silent for tens of seconds mid-run.

    When a track ended with no transition in flight, the autopilot still waited
    for the next 32-bar boundary of a deck that had stopped advancing, so the
    silence lasted until something else happened to move it.
    """
    live = session.engine.deck(session.live_deck)
    live.position = float(live.track.n_frames - 10)
    drive(session.engine, 3)
    assert live.ended

    session._autopilot_tick()  # cues
    assert session.has_cued_track()
    session._autopilot_tick()  # starts it immediately, no boundary wait
    drive(session.engine, 4)

    assert session.live_deck != "a" or session.engine.deck(session.live_deck).playing
    new_live = session.engine.deck(session.live_deck)
    assert new_live.playing and not new_live.ended
    assert audio_level(session) > 0.1, "still silent after recovery"
    assert any(e["event"] == "recovered_from_silence" for e in log_events(session))


def test_transition_fires_on_a_phrase_boundary_and_stays_phase_locked(session):
    """The headline claim, measured: the swap starts exactly on a placement-grid
    line and the two decks hold a constant beat phase all the way through it."""
    from djai import phrase

    live = session.engine.deck(session.live_deck)
    # Just inside the autopilot's lead window, measured to mix-out.
    a = live.track.analysis
    lead_bars = session.autopilot_lead_seconds() * a.bpm / 60.0 / 4.0
    live.position = phrase.frame_at_bar(a, a.mix_out_bar - lead_bars + 2.0)
    drive(session.engine, 2)

    session._autopilot_tick()  # cue
    drive(session.engine, 2)
    session._autopilot_tick()  # arm
    assert session._transition_armed

    assert any(isinstance(c, StartTransition) for c in session.scheduler.pending())
    out_analysis = live.track.analysis

    # Drive until the transition starts, releasing scheduled commands as the
    # scheduler thread would.
    started_at_bar = None
    for _ in range(20_000):
        session.scheduler.tick(session.engine.frames_played)
        drive(session.engine, 1)
        if session.engine.transition_active:
            started_at_bar = phrase.bar_at_frame(out_analysis, live.position)
            break
    assert started_at_bar is not None, "the transition never started"

    # (a) It began on a placement-grid line of the outgoing deck. Automatic
    # transitions snap to 8 bars, not 32: a 32-bar snap moved the start by up
    # to a minute, which is what left hand-offs anywhere from 84% to 102%.
    grid = phrase.PLACEMENT_GRID_BARS
    offset = started_at_bar % grid
    offset = min(offset, grid - offset)
    assert offset < 0.05, f"swap began {offset:.3f} bars off an {grid}-bar line"

    # (b) Phase lock: the beat offset between the decks must not wander.
    incoming = session.engine.deck(session.cued_deck())
    in_analysis = incoming.track.analysis
    offsets = []
    while session.engine.transition_active:
        session.scheduler.tick(session.engine.frames_played)
        drive(session.engine, 16)
        # Only while both decks are actually running: the last step of the
        # transition stops the outgoing deck, and comparing a frozen playhead
        # against a moving one is not a drift measurement.
        if live.playing and incoming.playing and not live.ended:
            offsets.append(
                phrase.beat_at_frame(out_analysis, live.position)
                - phrase.beat_at_frame(in_analysis, incoming.position)
            )
    assert len(offsets) > 50, "transition finished implausibly fast"

    spread_beats = max(offsets) - min(offsets)
    spread_ms = spread_beats * out_analysis.beat_period * 1000.0
    assert spread_ms < 1.0, (
        f"decks drifted {spread_ms:.3f} ms apart across the transition"
    )

    # (c) And the constant offset is a whole number of bars, so the decks are
    # bar-aligned rather than merely running at the same tempo.
    bars_apart = offsets[0] / phrase.BEATS_PER_BAR
    assert bars_apart == pytest.approx(round(bars_apart), abs=1e-3), (
        f"decks are {bars_apart:.4f} bars apart, not bar-aligned"
    )


def test_autopilot_lead_covers_two_phrases(session):
    """The lead must exceed one transition plus a full phrase of waiting."""
    from djai import transition

    one = transition.transition_frames(session.engine.master_bpm, SAMPLE_RATE) / SAMPLE_RATE
    assert session.autopilot_lead_seconds() > 2 * one


def test_session_state_blob_is_json_serialisable(session):
    blob = session.state_blob()
    json.dumps(blob)  # must not raise - this is what goes to the model
    assert set(blob["decks"]) == {"a", "b"}
    assert blob["decks"]["a"]["title"] == "mid"
    assert blob["decks"]["a"]["live"] is True
    assert "played_this_session" in blob
    assert "queued_commands" in blob


def test_model_state_is_exactly_the_five_fields_the_model_needs(session):
    """The local 8B gets a deliberately tiny blob, not the full session state."""
    blob = session.model_state()
    assert set(blob) == {"bpm", "key", "bars_in", "transition", "played"}
    json.dumps(blob)  # must not raise - this is what goes to Ollama

    assert isinstance(blob["bpm"], float)
    assert blob["bpm"] > 0
    assert blob["key"] == "9A"  # the fixture's opening track
    assert blob["transition"] is False
    assert blob["played"] >= 1

    # None of the noise the model cannot act on. "mid" is the live track's
    # title, so it doubles as a check that titles never reach the model.
    serialised = json.dumps(blob)
    for leaked in ("title", "gain", "eq", "queued", ".wav", "underruns", "mid"):
        assert leaked not in serialised, f"{leaked!r} should not reach the model"


def test_model_state_reports_a_running_transition(session):
    from djai.commands import StartTransition

    session.cue_next(0.0)
    drive(session.engine, 2)
    session.engine.submit(
        StartTransition(from_deck="a", to_deck="b", total_frames=SAMPLE_RATE * 30)
    )
    drive(session.engine, 2)
    assert session.model_state()["transition"] is True


def test_state_blob_never_leaks_file_paths(session):
    """The model gets titles and musical facts, not the filesystem."""
    blob = json.dumps(session.state_blob())
    assert ".wav" not in blob
    assert "placeholder" not in blob

