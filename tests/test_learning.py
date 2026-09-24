"""Phase 4.2: learning and style (SPEC §8).

The criterion is a controlled replay: a recorded session of an operator
choosing among suggestions is learned from, the ranking the next session shows
changes because of it, and rolling the model back restores the old ranking
exactly -- every track, every score.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from djai import cli, feedback, selector, songs
from djai import analysis as an
from djai.deck import LoadedTrack
from tests.test_integration import log_events, make_analysis
from tests.test_integration import session as _session_fixture

session = _session_fixture

# --- the operator's words ------------------------------------------------------------


@pytest.mark.parametrize("text,label", [
    ("that worked", "worked"), ("That didn't work", "didnt"), ("too early", "too_early"),
    ("that was too late!", "too_late"), ("more like this", "more_like_this"),
    ("less like this", "less_like_this"),
])
def test_feedback_phrases_are_matched_deterministically(text, label):
    assert feedback.parse_phrase(text) == label


@pytest.mark.parametrize("text", [
    "play more like this one by Calvin Harris", "cue too early by someone", "go harder",
])
def test_a_song_request_is_never_taken_for_feedback(text):
    assert feedback.parse_phrase(text) is None


# --- a session to learn in --------------------------------------------------------------


def _crate(tmp_path):
    """Sixteen tracks at one tempo, spread over energy and key."""
    keys = ["8A", "9A", "8B", "3A", "7A", "10A", "2B", "8A"]
    crate = []
    for i in range(16):
        p = tmp_path / f"t{i:02d}.wav"
        p.write_bytes(b"x")
        crate.append(make_analysis(f"t{i:02d}", 124.0, keys[i % len(keys)],
                                   0.1 + 0.9 * ((i * 7) % 16) / 15.0, p))
    return crate


def _session(tmp_path, monkeypatch, logs, persona="default") -> cli.Session:
    from tests.test_integration import loud_track

    monkeypatch.setattr(cli, "load_track", lambda a: LoadedTrack(
        analysis=a, audio=loud_track(300.0, a.bpm, 240.0).audio))
    monkeypatch.setattr("djai.config.TIME_STRETCH_ENABLED", False)
    # Session logs are named to the second: two sessions opened inside one
    # would share a file, and tonight's layer would read last night's choices.
    time.sleep(max(0.0, 1.05 - (time.time() % 1.0)))
    s = cli.Session(_crate(tmp_path), log_dir=logs)
    s.persona = persona
    s.start_first_track()
    # The load lands on the next callback; until then nothing is "playing"
    # and the ranking has no current track to relate keys and timbre to.
    s.engine.callback(np.zeros((512, 2), dtype=np.float32), 512, None, None)
    s.reload_feedback()
    return s


def _ranking(s: cli.Session) -> list[tuple[str, float]]:
    return [(r.track.track_id, r.score) for r in s.suggest(8)]


def _operator_prefers_energy(s: cli.Session, rounds: int = 6) -> None:
    """The recorded behaviour: shown five, the operator cues the most
    energetic -- rarely the top of the list, which holds the energy level."""
    for _ in range(rounds):
        rows = s.suggest(5)
        pick = max(rows, key=lambda r: r.track.energy)
        s.cue_track(pick.track)
        s.played.add(pick.track.track_id)


def test_a_replayed_session_changes_the_ranking_and_rollback_restores_it_exactly(
    tmp_path, monkeypatch,
):
    logs = tmp_path / "logs"
    # Night one: a plain session, nothing chosen. Its log trains v1.
    s1 = _session(tmp_path, monkeypatch, logs)
    before = _ranking(s1)
    s1.shutdown()
    v1 = feedback.train(logs)

    # Night two: the operator keeps picking the most energetic suggestion.
    s2 = _session(tmp_path, monkeypatch, logs)
    assert s2._stable_model["version"] == v1["version"]
    assert _ranking(s2) == before  # v1 knows nothing yet: the same ranking
    _operator_prefers_energy(s2)
    events = [e for e in log_events(s2) if e["event"] in ("suggestions_shown", "cue_added")]
    assert any(e.get("from_suggestion") for e in events if e["event"] == "cue_added")
    s2.shutdown()

    # The replay: v2 learned from it, and a fresh session ranks differently.
    v2 = feedback.train(logs)
    assert v2["version"] == v1["version"] + 1
    learned = v2["global"]
    assert learned["summary"]["suggestion_choices"] >= 5
    # The energetic picks were rarely the best key matches: passing over the
    # key-compatible rows is what the pairwise comparison reads.
    assert learned["selector"]["key"] < 1.0
    s3 = _session(tmp_path, monkeypatch, logs)
    after = _ranking(s3)
    assert [t for t, _ in after] != [t for t, _ in before]
    # The most energetic tracks moved up.
    energy = {t.track_id: t.energy for t in s3.crate}
    top = lambda rows: np.mean([energy[t] for t, _ in rows[:3]])  # noqa: E731
    assert top(after) > top(before)

    # Rollback: the old model, and the old ranking, exactly.
    assert s3.model_text("rollback") == f"Model v{v1['version']} is current."
    assert _ranking(s3) == before
    s3.shutdown()
    s4 = _session(tmp_path, monkeypatch, logs)
    assert s4._stable_model["version"] == v1["version"]
    assert _ranking(s4) == before
    s4.shutdown()


def test_tonight_moves_the_weights_without_touching_the_stable_model(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    feedback.train(logs)  # an empty v1
    mdir = feedback.model_dir(logs)
    stored = {p.name: p.read_bytes() for p in mdir.iterdir()}
    s = _session(tmp_path, monkeypatch, logs)
    try:
        before = dict(s.selector_weights)
        reply = cli.handle_override(s, "more like this")
        assert reply.startswith("Noted: more like this")
        assert s.selector_weights["timbre"] > before.get("timbre", 1.0)
        lo, hi = feedback.NIGHT_RANGE
        assert all(lo <= w <= hi for w in s.selector_weights.values())
        rows = [e for e in log_events(s) if e["event"] == "operator_feedback"]
        assert rows[-1]["verdict"] == "more_like_this" and rows[-1]["persona"] == "default"
    finally:
        s.shutdown()
    assert {p.name: p.read_bytes() for p in mdir.iterdir()} == stored


def test_model_learn_folds_tonight_in_once(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    s = _session(tmp_path, monkeypatch, logs)
    try:
        cli.handle_override(s, "more like this")
        tonight = s.selector_weights["timbre"]
        assert s.model_text("learn").startswith("Trained model v")
        # Stable now holds tonight's label; the night layer starts over, so
        # the label is not counted twice.
        assert s.selector_weights["timbre"] == pytest.approx(tonight)
    finally:
        s.shutdown()


def test_preferences_are_learned_per_persona(tmp_path):
    def picks(persona, key):
        rows = []
        for i in range(6):
            chosen = {"key": 1.0, "energy": 0.2} if key == "key" else {"key": 0.1, "energy": 1.0}
            other = {"key": 0.5, "energy": 0.5}
            rows += [{"event": "suggestions_shown", "persona": persona,
                      "rows": [{"track_id": f"c{i}", "terms": chosen},
                               {"track_id": f"o{i}", "terms": other}]},
                     {"event": "cue_added", "persona": persona, "track_id": f"c{i}"}]
        return rows

    model = feedback.build(picks("guetta", "key") + picks("warmup", "energy"))
    guetta = feedback.effective(model, None, "guetta")["selector"]
    warmup = feedback.effective(model, None, "warmup")["selector"]
    assert guetta["key"] > 1.0 > guetta["energy"]
    assert warmup["energy"] > 1.0 > warmup["key"]
    # A persona with nothing of its own falls back to the global model.
    assert feedback.effective(model, None, "hype") == feedback.effective(
        {"global": model["global"]}, None)


def test_too_early_shortens_the_blend_within_bounds():
    rows = [{"event": "operator_feedback", "verdict": "too_early"}] * 5
    learned = feedback.learn(rows)
    assert learned["timing_bias_bars"] == -feedback.TIMING_MAX_BARS
    assert feedback.learn(rows[:1])["timing_bias_bars"] == 0  # one is not enough


def test_the_learned_timing_reaches_the_armed_blend(session, monkeypatch):
    """"Too early" twice: the blend handed to placement is 8 bars shorter, so
    -- planned back from mix-out -- it starts later. Where placement then puts
    it is placement's own business (tests/test_integration)."""
    from djai import phrase
    from tests.test_integration import drive, seek_near_mix_out

    asked, designed = [], []
    real = phrase.plan_transition
    monkeypatch.setattr(phrase, "plan_transition",
                        lambda deck, b, bars: asked.append(bars) or real(deck, b, bars))
    real_params = session._params_for

    def params_for(*a, **k):
        out = real_params(*a, **k)
        designed.append(out[0].length_bars)
        return out

    monkeypatch.setattr(session, "_params_for", params_for)

    seek_near_mix_out(session, bars_before=64.0)
    session.cue_next(0.0)
    drive(session.engine, 2)
    # An operator's style: the same preset both times, not a fresh design.
    session.transition_style = "bass_swap"
    assert session.arm_transition() is not None
    session.abort_armed_transition()  # which drops the cue with it
    for _ in range(2):
        cli.handle_override(session, "too early")
    assert session.timing_bias_bars == -8
    session.cue_next(0.0)
    drive(session.engine, 2)
    assert session.arm_transition() is not None
    assert asked[0] == designed[0]  # no bias: the design's own length
    length = designed[-1]
    assert designed == [designed[0]] * 2 and length - 8 >= phrase.MIN_TRANSITION_BARS
    assert asked[-1] == max(min(phrase.MIN_TRANSITION_BARS, length), length - 8)


def test_learned_weights_are_multipliers_on_the_defaults(tmp_path):
    """Regression: a learned 1.0 used to replace the default weight outright,
    so "no opinion" on timbre raised it from 0.6 to 1.0."""
    crate = _crate(tmp_path)
    plain = selector.rank_candidates(crate, crate[0], set())
    ones = selector.rank_candidates(crate, crate[0], set(),
                                    weights={k: 1.0 for k in feedback.PREFERENCE_TERMS})
    assert [(c.track.track_id, c.score) for c in plain] == [
        (c.track.track_id, c.score) for c in ones]
    assert set(plain[0].terms) >= {"bpm", "key", "energy", "confidence", "timbre"}


def test_suggestion_terms_are_what_the_score_adds_up(tmp_path):
    crate = _crate(tmp_path)
    rows = songs.suggest(crate, crate[0], set(), n=5)
    c = selector
    for r in rows:
        t = r.terms
        expect = (c.W_BPM * t["bpm"] + c.W_KEY * t["key"] + c.W_ENERGY * t["energy"]
                  + c.W_CONFIDENCE * t["confidence"] + c.W_TIMBRE * t["timbre"]
                  + c.W_GROOVE * t["groove"] + c.W_NEIGHBOURHOOD * t["neighbourhood"])
        penalty = r.score - expect
        assert penalty <= 1e-3  # only penalties are subtracted beyond the terms


# --- incremental analysis ------------------------------------------------------------


def test_adding_one_track_analyses_that_track_only(tmp_path, monkeypatch):
    from tests.synth import render_track

    music, cache = tmp_path / "music", tmp_path / "cache"
    for i, bpm in enumerate((122.0, 126.0)):
        render_track(music / f"t{i}.wav", bpm, bars=24, seed=i)
    an.analyze_folder(music, cache)
    sidecars = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in cache.iterdir()}

    calls = []
    real = an.analyze_file
    monkeypatch.setattr(an, "analyze_file", lambda path, *a, **k: calls.append(path) or real(path, *a, **k))
    render_track(music / "new.wav", 124.0, bars=24, seed=9)
    results, analyzed, skipped = an.analyze_folder(music, cache)

    assert [p.name for p in calls] == ["new.wav"]
    assert (analyzed, skipped, len(results)) == (1, 2, 3)
    for name, (data, mtime) in sidecars.items():
        p = cache / name
        assert p.read_bytes() == data and p.stat().st_mtime_ns == mtime, name
    new = [p for p in cache.iterdir() if p.name not in sidecars]
    assert new and all(json.loads(p.read_text("utf-8"))["title"] for p in new
                       if p.suffix == ".json")
