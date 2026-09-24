"""Phase 3.2: song suggest and song cue.

THREADING CONTEXT: main thread (pytest). The scheduler is ticked and the audio
callback driven by hand on pure tones, so no device is opened and any click is
obvious to the second-difference check.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from djai import cli, config, songs, understanding
from djai import transition as tr
from djai.deck import CHANNELS, SAMPLE_RATE, LoadedTrack, TransportState
from djai.setstate import SetState
from tests.test_integration import make_analysis

BPM = 124.0
BAR = 4 * 60.0 / BPM * SAMPLE_RATE
CLICK_D2 = 5e-3


def _tone(freq: int, seconds: float = 240.0, level: float = 0.2) -> np.ndarray:
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    one = level * np.sin(2 * np.pi * freq * t)
    sig = np.tile(one, int(seconds) + 1)[: int(seconds * SAMPLE_RATE)]
    return np.stack([sig, sig], axis=1).astype(np.float32)


#: (id, title, bpm, camelot, energy, tone Hz)
CRATE = [
    ("t1", "Opener", 124.0, "8A", 0.30, 105),
    ("t2", "Levels - Radio Edit", 124.5, "9A", 0.40, 147),
    ("t3", "Levels x Slide - Medley Version", 125.0, "8B", 0.50, 126),
    ("t4", "Don't Let Me Down", 123.5, "7A", 0.45, 84),
    ("t5", "Sexy Bitch feat. Akon", 125.5, "8A", 0.55, 98),
    ("t6", "Heads Will Roll - A-Trak Remix Radio Edit", 124.0, "9B", 0.60, 110),
    ("t7", "Stereo Love X Where Have You Been", 126.0, "8A", 0.50, 120),
    ("fast", "Warp Speed", 150.0, "8A", 0.70, 90),
    ("ridge", "Bridge Ridge", 137.0, "8A", 0.50, 95),
    ("weak", "Shaky Grid", 124.0, "8A", 0.50, 101),
]


def _crate(tmp_path):
    out, freqs = [], {}
    for tid, title, bpm, cam, energy, hz in CRATE:
        p = tmp_path / f"{tid}.wav"
        p.write_bytes(b"x")
        t = make_analysis(tid, bpm, cam, energy, p)
        object.__setattr__(t, "title", title)
        understanding.derive(t)
        out.append(t)
        freqs[tid] = hz
    weak = next(t for t in out if t.track_id == "weak")
    object.__setattr__(weak, "grid_confidence", 0.02)   # quarantined
    return out, freqs


def _drive(s, frames: int, block: int = 1024) -> np.ndarray:
    out = np.zeros((frames, CHANNELS), dtype=np.float32)
    buf = np.zeros((block, CHANNELS), dtype=np.float32)
    done = 0
    while done < frames:
        n = min(block, frames - done)
        s.scheduler.tick(s.engine.frames_played + n)
        s.engine.callback(buf[:n], n, None, None)
        out[done:done + n] = buf[:n]
        done += n
    return out


@pytest.fixture
def cue_session(tmp_path, monkeypatch):
    crate, freqs = _crate(tmp_path)
    monkeypatch.setattr(cli, "load_track",
                        lambda a: LoadedTrack(analysis=a, audio=_tone(freqs[a.track_id])))
    monkeypatch.setattr(config, "TIME_STRETCH_ENABLED", False)
    s = cli.Session(crate, log_dir=tmp_path / "logs")
    # Open on a known track rather than the selector's pick.
    s.crate = [crate[0]]
    assert s.start_first_track()
    s.crate = crate
    _drive(s, SAMPLE_RATE // 2)
    s.intent_engine = None          # the model is unavailable throughout
    yield s
    s.shutdown()


def _events(s, name):
    return [json.loads(line) for line in s.session_log.path.read_text().splitlines()
            if json.loads(line).get("event") == name]


# --- lookup ------------------------------------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("levles", "Levels - Radio Edit"),                       # typo
    ("levels", "Levels - Radio Edit"),                       # core title beats a medley
    ("dont let me dwn", "Don't Let Me Down"),                # apostrophe + typo
    ("sexy bitch", "Sexy Bitch feat. Akon"),                 # missing feat. credit
    ("sexy bitch feat akon", "Sexy Bitch feat. Akon"),
    ("heads will roll", "Heads Will Roll - A-Trak Remix Radio Edit"),  # remix tag
    ("heads will roll remix", "Heads Will Roll - A-Trak Remix Radio Edit"),
    ("stereo love", "Stereo Love X Where Have You Been"),    # partial
    ("where have you been", "Stereo Love X Where Have You Been"),
    ("warp sped", "Warp Speed"),
])
def test_fuzzy_lookup_resolves_misspelled_and_partial_queries(tmp_path, query, expected):
    crate, _ = _crate(tmp_path)
    found = songs.lookup(crate, query)
    assert found.status == "match", (query, found)
    assert found.track.title == expected


def test_an_ambiguous_query_returns_candidates_not_a_guess(tmp_path):
    crate, _ = _crate(tmp_path)
    found = songs.lookup(crate, "love")   # nothing but "Stereo Love" -> match
    assert found.status == "match"
    crate.append(make_analysis("t8", 124.0, "8A", 0.5, tmp_path / "t8.wav"))
    object.__setattr__(crate[-1], "title", "My Love")
    found = songs.lookup(crate, "love")
    assert found.status == "ambiguous"
    assert {t.title for t in found.tracks} == {"My Love", "Stereo Love X Where Have You Been"}
    assert found.track is None


@pytest.mark.parametrize("query", ["zzzz", "bohemian rhapsody", "", "   "])
def test_an_unmatched_query_reports_no_match(tmp_path, query):
    crate, _ = _crate(tmp_path)
    assert songs.lookup(crate, query).status == "none"


def test_cue_never_queues_on_ambiguity_or_no_match(cue_session):
    s = cue_session
    s.crate.append(make_analysis("t8", 124.0, "8A", 0.5, Path("t8.wav")))
    object.__setattr__(s.crate[-1], "title", "My Love")
    reply = cli.handle_override(s, "cue love")
    assert "more than one" in reply and "My Love" in reply
    reply = cli.handle_override(s, "cue bohemian rhapsody")
    assert "No track matches" in reply
    assert s.cue_queue == []


# --- suggest -------------------------------------------------------------------


def test_suggest_populates_every_reason_and_excludes_played_and_quarantined(cue_session):
    s = cue_session
    s.played.add("t2")
    rows = s.suggest(5)
    assert len(rows) == 5
    ids = [r.track.track_id for r in rows]
    assert "t2" not in ids and "t1" not in ids, "played tracks are excluded"
    assert "weak" not in ids, "quarantined tracks are excluded"
    assert [r.rank for r in rows] == [1, 2, 3, 4, 5]
    assert [r.score for r in rows] == sorted((r.score for r in rows), reverse=True)
    for r in rows:
        d = r.as_dict()
        for field in ("bpm_gap_pct", "stretch_pct", "tempo_match", "key_relation",
                      "energy_delta", "similarity", "mix_quality", "plan_fit"):
            assert d[field] is not None and d[field] != "", (field, d)
        assert d["tempo_match"] in ("direct", "ride", "half", "double")
    text = cli.handle_override(s, "suggest 3")
    assert text.count("\n") == 2 and "mix point" in text and "stretch" in text


# --- cue modes -------------------------------------------------------------------


def test_play_next_is_the_next_cue(cue_session):
    s = cue_session
    assert "Queued Don't Let Me Down" in cli.handle_override(s, "cue dont let me down")
    assert s.cue_next(origin="test")
    assert s._cued.analysis.track_id == "t4"
    assert s.cue_queue == [], "a played cue leaves the queue"


def test_play_after_n_waits_for_n_tracks_and_lets_later_cues_through(cue_session):
    s = cue_session
    cli.handle_override(s, "cue stereo love after 2")
    cli.handle_override(s, "cue sexy bitch next")
    picks = []
    for _ in range(3):
        s._cued = None
        assert s.cue_next(origin="test")
        picks.append(s._cued.analysis.track_id)
    assert picks[0] == "t5", "the next cue is not held up by an after-N cue ahead of it"
    assert picks[1] not in ("t5", "t7"), "one autonomous track while it waits"
    assert picks[2] == "t7", "after exactly two tracks"


def test_cued_tracks_override_selection_in_queue_order(cue_session):
    s = cue_session
    for q in ("heads will roll", "levels", "sexy bitch"):
        cli.handle_override(s, f"cue {q}")
    order = []
    for _ in range(3):
        s._cued = None
        assert s.cue_next(origin="test")
        order.append(s._cued.analysis.track_id)
    assert order == ["t6", "t2", "t5"]


def test_queue_reorder_and_remove(cue_session):
    s = cue_session
    for q in ("heads will roll", "levels", "sexy bitch"):
        cli.handle_override(s, f"cue {q}")
    cli.handle_override(s, "queue move 3 1")
    cli.handle_override(s, "queue remove 2")
    assert [e["track_id"] for e in s.cue_queue] == ["t5", "t2"]
    assert "1. Sexy Bitch" in cli.handle_override(s, "queue")


def test_play_now_is_audible_within_8_bars_and_artifact_free(cue_session, capsys):
    s = cue_session
    _drive(s, int(20 * BAR))                  # well inside the track
    start = s.engine.frames_played
    reply = cli.handle_override(s, "cue levels now")
    assert "next 4-bar line" in reply, reply
    idle = s.cued_deck()
    audio, heard_at = [], None
    for _ in range(int(12 * BAR // 1024)):
        s._autopilot_tick()
        audio.append(_drive(s, 1024))
        deck = s.engine.deck(idle)
        if (heard_at is None and deck.transport is TransportState.PLAYING
                and deck.gain.cur > 0.1):
            heard_at = s.engine.frames_played
    assert heard_at is not None, ("never heard", s._play_now, s._transition_armed,
                                  [(e["event"], e.get("action")) for e in map(json.loads, s.session_log.path.read_text().splitlines())][-12:])
    bars = (heard_at - start) / BAR
    assert bars <= 8.0, f"audible after {bars:.1f} bars"
    mix = np.concatenate(audio)[:, 0].astype(np.float64)
    d2 = float(np.abs(mix[2:] - 2 * mix[1:-1] + mix[:-2]).max())
    assert d2 < CLICK_D2, f"discontinuity {d2:.2e}"
    assert np.abs(mix).max() <= 1.0
    armed = _events(s, "transition_armed")[-1]
    with capsys.disabled():
        print(f"\n    play now: audible after {bars:.2f} bars; worst second difference "
              f"{d2:.2e}; peak {np.abs(mix).max():.3f}; shape {armed.get('style')}")
    assert armed["design_source"] == "play_now"
    assert armed["params_used"]["length_bars"] <= 4
    assert _events(s, "play_now"), "logged, with the glue gesture"


# --- the stretch cap ---------------------------------------------------------------


def test_a_cue_beyond_the_cap_asks_for_a_bridge_and_stays_queued(cue_session):
    s = cue_session
    reply = cli.handle_override(s, "cue warp speed")
    assert "bridge tempo" in reply and "bridge track" in reply
    assert s.cue_queue[0]["status"] == "needs_bridge"
    # The music does not wait: the selector fills in, the cue stays put.
    assert s.cue_next(origin="test")
    assert s._cued.analysis.track_id != "fast"
    assert s.cue_queue[0]["track_id"] == "fast"
    assert s.cue_queue[0]["status"] == "needs_bridge"


def test_a_tempo_bridge_comes_in_on_an_echo_out_never_a_cut(cue_session):
    s = cue_session
    cli.handle_override(s, "cue warp speed")
    assert "tempo bridge" in cli.handle_override(s, "bridge tempo")
    assert s.cue_next(origin="test")
    assert s._cued.analysis.track_id == "fast"
    _drive(s, 4096)
    s.arm_transition(origin="test")
    style, rule = s.last_transition_choice
    assert style == "echo_out", rule
    assert "tempo bridge" in rule


def test_a_bridge_track_goes_in_front_of_the_cue(cue_session):
    s = cue_session
    cli.handle_override(s, "cue warp speed")
    reply = cli.handle_override(s, "bridge track")
    assert "goes first" in reply, reply
    first, second = s.cue_queue
    assert second["track_id"] == "fast" and second["bridge"] == "track"
    assert first["bridge_for"] == second["id"]
    bridge_bpm = s._track_by_id(first["track_id"]).bpm
    for leg in (s._cap_path(s._track_by_id(first["track_id"])),
                s._cap_path(s._track_by_id("fast"), from_bpm=bridge_bpm)):
        assert leg.blendable


def test_a_quarantined_cue_warns_and_gets_a_conservative_transition(cue_session):
    s = cue_session
    reply = cli.handle_override(s, "cue shaky grid")
    assert "Warning" in reply and "low grid confidence" in reply
    assert s.cue_queue[0]["warning"]
    assert s.cue_next(origin="test")
    _drive(s, 4096)
    s.arm_transition(origin="test")
    style, rule = s.last_transition_choice
    assert style == "echo_out", rule        # not a cut, not a long blend


# --- no model, and a killed process -------------------------------------------------


def test_suggest_and_cue_work_from_the_repl_with_no_model(cue_session):
    s = cue_session
    assert s.intent_engine is None
    assert "1." in cli.handle_override(s, "suggest")
    assert "Queued" in cli.handle_override(s, "cue levles")


def test_the_keyword_fallback_parses_a_song_request():
    from djai.intent import keyword_intent

    i = keyword_intent("play levels after 2 tracks")
    assert i.action == "cue_track"
    assert i.params == {"query": "levels", "mode": "after", "after": 2}
    assert keyword_intent("play something harder").action != "cue_track"


def test_the_cue_queue_survives_a_restart(cue_session, tmp_path):
    s = cue_session
    path = tmp_path / "state.json"
    s.attach_set_state(path)
    cli.handle_override(s, "cue levels")
    cli.handle_override(s, "cue warp speed")
    # No shutdown: the process "dies" here.
    fresh = cli.Session(s.crate, log_dir=tmp_path / "logs2")
    try:
        assert "Resumed" in fresh.attach_set_state(path)
        assert [e["track_id"] for e in fresh.cue_queue] == ["t2", "fast"]
        assert fresh.cue_queue[1]["status"] == "needs_bridge"
        assert "t1" in fresh.played
    finally:
        fresh.shutdown()
    assert SetState.load(path).ended, "a clean shutdown ends the set"


def test_state_file_is_never_torn_by_a_kill(tmp_path):
    """Kill a process mid-way through thousands of saves; the file still loads."""
    path = tmp_path / "state.json"
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from pathlib import Path
        from djai.setstate import SetState
        s = SetState(path=Path({str(path)!r}))
        i = 0
        while True:
            s.add_cue({{"track_id": f"t{{i}}", "title": "x" * 200, "mode": "next",
                       "after": 0, "status": "queued", "bridge": None,
                       "warning": None, "added_at": 0}})
            i += 1
            if i == 50:
                print("ready", flush=True)
    """)
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    import time
    time.sleep(0.3)
    proc.kill()
    proc.wait()
    state = SetState.load(path)
    assert len(state.cue_queue) >= 50
    assert [e["id"] for e in state.cue_queue] == list(range(1, len(state.cue_queue) + 1))
