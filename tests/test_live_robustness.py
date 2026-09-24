"""Phase 5.1 (SPEC §9): live robustness.

This file is the failure-mode catalogue. Each mode has an intentional-sounding
recovery and an adversarial test that drives the real Session and callback on
pure tones (no device) and checks the audio stays continuous and click-free.

| Failure mode          | Recovery                                                     | Test |
|-----------------------|--------------------------------------------------------------|------|
| analysis failure      | track set aside, selector/queue carries on, music unbroken   | test_analysis_failure_* |
| corrupt file          | refused at decode, set aside, next candidate cued            | test_corrupt_file_* |
| sudden silence        | live deck runs dry: cued track hard-started at mix-in        | test_sudden_silence_* |
| device fault          | watchdog's own stream plays the fallback; handed back healthy| tests/test_device_fault.py, tests/test_fallback.py |
| MC interruption       | `mc on`: master dips -12 dB over a beat, arming and the      | test_mc_* |
|                       | closed loop hold; `mc off` ramps back over a bar             | |
| crowd shift           | energy correction on the live deck within 2 bars             | test_crowd_shift_* |
| model unavailable     | keyword overrides + deterministic selector; a full set runs  | test_a_full_set_runs_with_the_model_unavailable |
| tempo desync          | supervisor nudges, or resyncs on a downbeat                  | test_tempo_desync_* |
| cued file missing     | reported to the operator, entry skipped, queue continues     | test_a_missing_cued_file_* |

THREADING CONTEXT: main thread (pytest). Autopilot ticks are called by hand
every AUTOPILOT_TICK seconds of rendered audio, as the real thread would.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from djai import cli, config
from djai.commands import StartTransition
from djai.deck import SAMPLE_RATE, LoadedTrack, TransportState
from tests.test_song_cue import BAR, CLICK_D2, _crate, _drive, _tone

TICK_FRAMES = int(cli.AUTOPILOT_TICK * SAMPLE_RATE)


@pytest.fixture
def live(tmp_path, monkeypatch):
    crate, freqs = _crate(tmp_path)
    monkeypatch.setattr(cli, "load_track",
                        lambda a: LoadedTrack(analysis=a, audio=_tone(freqs[a.track_id])))
    monkeypatch.setattr(config, "TIME_STRETCH_ENABLED", False)
    s = cli.Session(crate, log_dir=tmp_path / "logs")
    s.crate = [crate[0]]
    assert s.start_first_track()
    s.crate = crate
    _drive(s, SAMPLE_RATE // 2)
    s.intent_engine = None          # the model is unavailable throughout
    # Invisible shapes only: a showy riser is noise by design, and noise would
    # swamp the second-difference click check. Risers have their own tests.
    s.transition_mode = "invisible"
    yield s
    s.shutdown()


def _run(s, seconds: float, until=None) -> np.ndarray:
    """Render ``seconds``, ticking the autopilot as its thread would."""
    out = []
    for _ in range(max(1, int(seconds * SAMPLE_RATE / TICK_FRAMES))):
        s._autopilot_tick()
        s.supervisor.check_stalled()   # the supervisor's own thread, by hand
        s.supervisor.check_drift()
        out.append(_drive(s, TICK_FRAMES))
        if until is not None and until():
            break
    return np.concatenate(out)


def assert_continuous(mix: np.ndarray, what: str) -> None:
    """No click, no clip, and no gap of 10 ms or more of near-silence."""
    x = mix[:, 0].astype(np.float64)
    d2 = float(np.abs(x[2:] - 2 * x[1:-1] + x[:-2]).max())
    assert d2 < CLICK_D2, f"{what}: discontinuity {d2:.2e}"
    assert np.abs(x).max() <= 1.0, f"{what}: peak {np.abs(x).max():.3f}"
    quiet = np.abs(x) < 1e-4
    run = longest = 0
    for q in quiet:
        run = run + 1 if q else 0
        longest = max(longest, run)
    assert longest < SAMPLE_RATE // 100, f"{what}: {longest / SAMPLE_RATE * 1000:.0f} ms gap"


def _events(s, name):
    return [r for r in map(json.loads, s.session_log.path.read_text().splitlines())
            if r.get("event") == name]


def _to_cue_window(s) -> None:
    """Render, automation held, until the live deck is inside its cue window."""
    s.freeze(True)
    live = s.engine.deck(s.live_deck)
    _run(s, 400, until=lambda: s._seconds_to_mix_out(live) <= s.autopilot_lead_seconds())


# --- human takeover and hand-back -------------------------------------------------


def test_takeover_withdraws_an_armed_blend_at_once_and_keeps_the_cue(live):
    s = live
    assert s.cue_next(origin="test")
    _drive(s, 4096)
    assert s.arm_transition(origin="test") is not None
    cued = s._cued
    reply = cli.handle_override(s, "hold")
    assert "frozen" in reply.lower()
    # Nothing the automation queued is left to fire: takeover took 0 bars.
    assert not any(isinstance(c, StartTransition) for c in s.scheduler.pending())
    assert not s._transition_armed
    assert s._cued is cued, "the cued track stays cued for the hand-back"
    assert _events(s, "transition_aborted")[-1]["trigger"] == "operator took over"
    assert_continuous(_run(s, 4 * BAR / SAMPLE_RATE), "after takeover")


def test_hand_back_rearms_within_one_bar(live, capsys):
    s = live
    _to_cue_window(s)
    assert not s._transition_armed and s.frozen
    start = s.engine.frames_played
    cli.handle_override(s, "resume")
    audio = _run(s, 2 * BAR / SAMPLE_RATE, until=lambda: s._transition_armed)
    bars = (s.engine.frames_played - start) / BAR
    assert s._transition_armed, "the autopilot never picked the set back up"
    assert bars <= 1.0, f"hand-back took {bars:.2f} bars"
    assert_continuous(audio, "hand-back")
    with capsys.disabled():
        print(f"\n    hand-back: armed {bars:.2f} bars after resume")


# --- co-pilot ---------------------------------------------------------------------


def _level_db(mix: np.ndarray) -> float:
    return 20 * np.log10(float(np.sqrt(np.mean(mix[:, 0].astype(np.float64) ** 2))) + 1e-12)


def test_copilot_proposes_and_blends_on_go(live):
    s = live
    assert "Co-pilot" in cli.handle_override(s, "mode assisted")
    _to_cue_window(s)
    s.freeze(False)
    live_deck = s.engine.deck(s.live_deck)
    _run(s, 4 * BAR / SAMPLE_RATE)
    assert s.has_cued_track() and not s._transition_armed, "proposed, not armed"
    assert s._seconds_to_mix_out(live_deck) > s.last_call_seconds()
    assert "next up" in " ".join(s.drain_notices())
    assert "Blending into" in cli.handle_override(s, "go")
    assert s._transition_armed


def test_copilot_with_no_answer_blends_at_the_last_call(live, capsys):
    s = live
    cli.handle_override(s, "mode assisted")
    _to_cue_window(s)
    s.freeze(False)
    live_deck = s.engine.deck(s.live_deck)
    audio = _run(s, 120, until=lambda: s._transition_armed)
    assert s._transition_armed, "the room would have gone quiet"
    left = s._seconds_to_mix_out(live_deck)
    assert left <= s.last_call_seconds()
    assert _events(s, "arm_last_call")
    audio = np.concatenate([audio, _run(s, 90, until=lambda: s.live_deck != "a"
                                        and not s.engine.transition_active)])
    assert s.live_deck == "b", "the blend completed"
    assert_continuous(audio, "co-pilot last call")
    with capsys.disabled():
        print(f"\n    co-pilot last call: armed {left:.1f} s before mix-out "
              f"(last call {s.last_call_seconds():.1f} s)")


# --- MC interruption ---------------------------------------------------------------


def test_mc_dips_the_master_over_a_beat_and_restores_it_over_a_bar(live, capsys):
    s = live
    before = _run(s, 2 * BAR / SAMPLE_RATE)
    assert "Mic" in cli.handle_override(s, "mc on")
    assert s.mc
    dip = _run(s, 2 * BAR / SAMPLE_RATE)
    held = _run(s, 2 * BAR / SAMPLE_RATE)
    cli.handle_override(s, "mc off")
    back = _run(s, 2 * BAR / SAMPLE_RATE)
    after = _run(s, 2 * BAR / SAMPLE_RATE)
    depth = _level_db(held) - _level_db(before)
    assert -13.0 < depth < -11.0, f"dip {depth:.1f} dB"
    assert abs(_level_db(after) - _level_db(before)) < 0.5
    assert_continuous(np.concatenate([before, dip, held, back, after]), "mc dip")
    assert _events(s, "mc_on") and _events(s, "mc_off")
    with capsys.disabled():
        print(f"\n    mc: dip {depth:.1f} dB, restored to {_level_db(after) - _level_db(before):+.2f} dB")


def test_mc_holds_the_blend_until_the_last_call_and_pauses_the_closed_loop(live):
    s = live
    _to_cue_window(s)
    s.freeze(False)
    cli.handle_override(s, "mc on")
    _run(s, 4 * BAR / SAMPLE_RATE)
    assert not s._transition_armed, "no blend under the MC before the last call"
    assert not _events(s, "closed_loop"), "the loop read the dip as a fault"
    cli.handle_override(s, "mc off")
    _run(s, 2 * BAR / SAMPLE_RATE, until=lambda: s._transition_armed)
    assert s._transition_armed, "arming is free again once the mic is off"


# --- the failure-mode catalogue ----------------------------------------------------


def _blend_through(s, seconds: float = 400) -> np.ndarray:
    """Render until one hand-over has completed, automation running."""
    start = s.live_deck
    return _run(s, seconds, until=lambda: s.live_deck != start
                and not s.engine.transition_active and not s._transition_armed)


def test_corrupt_file_is_set_aside_and_the_next_track_blends_in(live):
    s = live
    good = "t3"
    real = cli.load_track

    def load(a):
        if a.track_id != good:
            raise RuntimeError("Error opening file: Format not recognised")
        return real(a)
    cli.load_track = load
    try:
        audio = _blend_through(s)
    finally:
        cli.load_track = real
    assert s.engine.deck(s.live_deck).track.analysis.track_id == good
    assert _events(s, "track_unplayable"), "each refusal is logged by name"
    assert_continuous(audio, "corrupt file")


def test_analysis_failure_is_set_aside_and_the_set_carries_on(live):
    s = live
    # Every track but t3 has a failed analysis: no tempo, or no grid.
    for t in s.crate:
        if t.track_id not in ("t1", "t3"):
            object.__setattr__(t, "bpm", float("nan"))
    object.__setattr__(s._track_by_id("t4"), "bpm", 124.0)
    object.__setattr__(s._track_by_id("t4"), "beats", [])
    reply = cli.handle_override(s, "cue levels")
    assert "analysis failed" in reply and not s.cue_queue, reply
    audio = _blend_through(s)
    assert s.engine.deck(s.live_deck).track.analysis.track_id == "t3"
    refused = {e["track"] for e in _events(s, "track_unplayable")}
    assert "Don't Let Me Down" in refused and "Levels - Radio Edit" in refused
    assert_continuous(audio, "analysis failure")


def test_sudden_silence_live_deck_runs_dry(tmp_path, monkeypatch, capsys):
    crate, freqs = _crate(tmp_path)

    def load(a):
        # The opener's file is truncated to 40 s; its analysis says 240 s.
        seconds = 40.0 if a.track_id == "t1" else 240.0
        return LoadedTrack(analysis=a, audio=_tone(freqs[a.track_id], seconds))
    monkeypatch.setattr(cli, "load_track", load)
    monkeypatch.setattr(config, "TIME_STRETCH_ENABLED", False)
    s = cli.Session(crate, log_dir=tmp_path / "logs")
    try:
        s.crate = [crate[0]]
        assert s.start_first_track()
        s.crate = crate
        s.transition_mode = "invisible"
        audio = _run(s, 60, until=lambda: s.live_deck != "a" and s.engine.frames_played
                     > 45 * SAMPLE_RATE)
        assert s.live_deck == "b", "nothing took over"
        assert _events(s, "run_dry"), "covered before the audio stopped"
        assert not _events(s, "recovered_from_silence"), "the room heard the gap"
        assert_continuous(audio[SAMPLE_RATE:], "sudden silence")
    finally:
        s.shutdown()


def test_a_missing_cued_file_is_reported_and_the_queue_continues(live):
    s = live
    cli.handle_override(s, "cue levels")
    cli.handle_override(s, "cue dont let me down")
    missing = s._track_by_id("t2")
    real = cli.load_track

    def load(a):
        if a.track_id == "t2":
            raise FileNotFoundError(f"No such file: {a.path}")
        return real(a)
    cli.load_track = load
    try:
        s.drain_notices()
        audio = _blend_through(s)
    finally:
        cli.load_track = real
    notices = " ".join(s.drain_notices())
    assert missing.title in notices, notices
    assert s.engine.deck(s.live_deck).track.analysis.track_id == "t4", "queue continued"
    assert_continuous(audio, "missing cued file")


def test_crowd_shift_energy_correction_is_click_free(live):
    s = live
    _run(s, 4 * BAR / SAMPLE_RATE)
    s.energy_correction(1.0, origin="test")
    up = _run(s, 4 * BAR / SAMPLE_RATE)
    s.energy_correction(-1.0, origin="test")
    down = _run(s, 4 * BAR / SAMPLE_RATE)
    assert_continuous(np.concatenate([up, down]), "crowd shift")


def test_tempo_desync_is_corrected_without_a_click(live):
    s = live
    assert s.cue_next(origin="test")
    audio = [_run(s, 1)]
    armed = s._transition_armed
    _to_cue_window(s)
    s.freeze(False)
    audio.append(_run(s, 200, until=lambda: s.engine.transition_active))
    assert s.engine.transition_active, armed
    incoming = s.engine.deck(s.cued_deck())
    incoming.position += 0.030 * SAMPLE_RATE      # 30 ms off the grid mid-blend
    audio.append(_run(s, 8 * BAR / SAMPLE_RATE))
    assert s.supervisor.interventions, "the supervisor never noticed"
    assert_continuous(np.concatenate(audio[1:]), "tempo desync")


# --- a full set with the model unavailable ------------------------------------------


def test_a_full_set_runs_with_the_model_unavailable(tmp_path, monkeypatch, capsys):
    """An hour on a 24-track crate: keyword fallback, selector, blends. No model."""
    import httpx

    from djai import understanding
    from djai.intent import IntentEngine
    from tests.test_integration import make_analysis

    rng = np.random.default_rng(5)
    crate, freqs = [], {}
    for i in range(24):
        p = tmp_path / f"s{i}.wav"
        p.write_bytes(b"x")
        t = make_analysis(f"s{i}", float(rng.uniform(122.0, 128.0)),
                          f"{rng.integers(1, 13)}{'AB'[i % 2]}",
                          float(rng.uniform(0.3, 0.8)), p)
        object.__setattr__(t, "title", f"Set Track {i:02d}")
        understanding.derive(t)
        crate.append(t)
        freqs[t.track_id] = 80 + 5 * i
    monkeypatch.setattr(cli, "load_track",
                        lambda a: LoadedTrack(analysis=a, audio=_tone(freqs[a.track_id])))
    monkeypatch.setattr(config, "TIME_STRETCH_ENABLED", False)
    s = cli.Session(crate, log_dir=tmp_path / "logs")
    calls = []
    engine = IntentEngine()
    engine._client = httpx.Client(transport=httpx.MockTransport(
        lambda r: calls.append(r) or httpx.Response(500)))
    engine.available = False
    s.intent_engine = engine
    s.transition_mode = "invisible"
    requests = iter(["more energy", "bring it down a bit", "harder", "calmer",
                     "more energy", "next track"])
    try:
        assert s.start_first_track()
        _drive(s, 1024)                    # the LoadTrack lands on the next block
        handovers, tail, seconds = [], np.zeros((0, 2), np.float32), 0.0
        last = s.engine.deck(s.live_deck).track.analysis.title
        while seconds < 60 * 60:
            chunk = _run(s, 10)
            seconds += 10
            assert_continuous(np.concatenate([tail, chunk]), f"set at {seconds / 60:.1f} min")
            tail = chunk[-SAMPLE_RATE // 50:]
            now = s.engine.deck(s.live_deck).track.analysis.title
            if now != last and not s.engine.transition_active:
                handovers.append(now)
                last = now
                text = next(requests, None)
                if text is not None:      # the REPL route, model down
                    assert cli.handle_override(s, text) is None
                    intent = engine.interpret(text, s.model_state())
                    assert intent.fallback and intent.ok, (text, intent)
                    cli.apply_intent(s, intent)
        armed = _events(s, "transition_armed")
        cuts = [e for e in armed if e.get("is_cut")]
        assert len(handovers) >= 15, handovers
        assert not calls, "the model was called while unavailable"
        assert not cuts, [e["action"] for e in cuts]
        assert s.engine.underruns == 0
        with capsys.disabled():
            print(f"\n    model-down set: {seconds / 60:.0f} min, {len(handovers)} blends, "
                  f"0 cuts, 0 network calls, 0 gaps or clicks")
    finally:
        s.shutdown()
