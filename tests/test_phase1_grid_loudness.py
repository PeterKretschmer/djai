"""Phase 1: beat grid correction, tempo ambiguity, loudness, limiter reporting.

THREADING CONTEXT: main thread (pytest). The callback is driven directly.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from djai import analysis as an
from djai import cli, config, selector
from djai.commands import LoadTrack
from djai.deck import SAMPLE_RATE, load_track
from tests import synth
from tests.test_engine import drive, loud_track
from tests.test_integration import make_analysis
from tests.test_integration import session as _session_fixture

session = _session_fixture

BLOCK = 512


# --- rebuilding a grid ------------------------------------------------------------


def test_halving_rebuilds_the_grid_at_half_tempo_on_the_same_bar_line(tmp_path):
    ta = make_analysis("t", 128.0, "8A", 0.5, tmp_path / "t.wav")
    first, n = ta.first_downbeat, len(ta.beats)
    an.regrid(ta, ta.bpm / 2, first)
    assert ta.bpm == pytest.approx(64.0)
    assert ta.first_downbeat == pytest.approx(first, abs=1e-4)
    assert len(ta.beats) == pytest.approx(n / 2, abs=2)
    assert np.allclose(np.diff(ta.beats), 60.0 / 64.0, atol=1e-4)
    assert len(ta.beat_rms) == len(ta.beats)
    assert ta.grid_manually_corrected and not ta.tempo_ambiguous


def test_doubling_rebuilds_the_grid_at_double_tempo(tmp_path):
    ta = make_analysis("t", 64.0, "8A", 0.5, tmp_path / "t.wav")
    an.regrid(ta, ta.bpm * 2, ta.first_downbeat)
    assert ta.bpm == pytest.approx(128.0)
    assert np.allclose(np.diff(ta.beats), 60.0 / 128.0, atol=1e-4)
    assert np.allclose(np.diff(ta.downbeats), 4 * 60.0 / 128.0, atol=1e-4)


def test_a_nudge_moves_every_bar_line_by_the_same_amount(tmp_path):
    ta = make_analysis("t", 128.0, "8A", 0.5, tmp_path / "t.wav")
    before = np.asarray(ta.downbeats[1:9])
    an.regrid(ta, ta.bpm, ta.first_downbeat + 0.010)
    after = np.asarray(ta.downbeats[1:9])
    assert np.allclose(after - before, 0.010, atol=1e-4)


def test_tap_tempo_needs_eight_steady_taps():
    steady = [i * 0.5 for i in range(8)]
    assert an.bpm_from_taps(steady) == pytest.approx(120.0)
    assert an.bpm_from_taps(steady[:7]) is None
    assert an.bpm_from_taps([0, 0.5, 1.2, 1.5, 2.3, 2.6, 3.4, 3.7]) is None


# --- corrections from the REPL ------------------------------------------------------


def _steady_clock(monkeypatch):
    counter = itertools.count()
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(counter) * 0.5)


def test_all_four_corrections_work_from_the_repl_and_persist(session, tmp_path, monkeypatch):
    session.cache_dir = tmp_path
    analysis = session.engine.deck("a").track.analysis
    original = analysis.bpm

    assert "Saved" in cli.handle_override(session, "grid a double")
    assert analysis.bpm == pytest.approx(original * 2)
    assert "Saved" in cli.handle_override(session, "grid a halve")
    assert analysis.bpm == pytest.approx(original)
    first = analysis.first_downbeat
    assert "Saved" in cli.handle_override(session, "grid a nudge 12")
    assert analysis.first_downbeat == pytest.approx(first + 0.012, abs=1e-4)

    _steady_clock(monkeypatch)
    replies = [cli.handle_override(session, "grid a tap") for _ in range(8)]
    assert replies[0].startswith("Tap 1/8")
    assert "Saved" in replies[-1], replies[-1]
    assert analysis.bpm == pytest.approx(120.0)

    reloaded = an.load_cached(analysis.track_id, tmp_path)
    assert reloaded is not None and reloaded.grid_manually_corrected
    assert reloaded.bpm == pytest.approx(120.0)
    assert reloaded.first_downbeat == pytest.approx(analysis.first_downbeat, abs=1e-4)


def test_tempo_changes_are_refused_mid_blend_but_a_nudge_is_not(session, tmp_path):
    session.cache_dir = tmp_path
    other = loud_track(330.0, 125.0, 60.0)
    session.engine.submit(LoadTrack(deck="b", track=other, play=True))
    drive(session.engine, 2, BLOCK)
    bpm = session.engine.deck("a").track.analysis.bpm
    for op in ("halve", "double"):
        assert cli.handle_override(session, f"grid a {op}").startswith("Refused")
    assert session.engine.deck("a").track.analysis.bpm == bpm
    assert "Saved" in cli.handle_override(session, "grid a nudge -5")


def test_an_empty_deck_has_no_grid_to_correct(session):
    assert cli.handle_override(session, "grid b halve").startswith("Refused")
    assert "grid [a|b]" in cli.handle_override(session, "grid")


def test_correcting_the_master_decks_tempo_retunes_the_clock(session, tmp_path):
    session.cache_dir = tmp_path
    before = session.engine.master_bpm
    cli.handle_override(session, "grid a double")
    assert session.engine.master_bpm == pytest.approx(before * 2, rel=1e-6)


# --- corrections from the UI -----------------------------------------------------------


def test_all_four_corrections_work_from_the_ui_and_persist(session, tmp_path, monkeypatch):
    from djai.ui_server import UIServer

    session.cache_dir = tmp_path
    server = UIServer(session, intent_engine=None)
    analysis = session.engine.deck("a").track.analysis
    original = analysis.bpm

    assert server.handle_action({"type": "grid", "deck": "a", "op": "double"})["ok"]
    assert analysis.bpm == pytest.approx(original * 2)
    assert server.handle_action({"type": "grid", "deck": "a", "op": "halve"})["ok"]
    assert analysis.bpm == pytest.approx(original)
    assert server.handle_action({"type": "grid", "deck": "a", "op": "nudge", "ms": -8})["ok"]

    _steady_clock(monkeypatch)
    for _ in range(8):
        ack = server.handle_action({"type": "grid", "deck": "a", "op": "tap"})
    assert ack["ok"] and analysis.bpm == pytest.approx(120.0), ack

    ack = server.handle_action({"type": "grid", "deck": "a", "op": "downbeat", "seconds": 1.0})
    assert ack["ok"], ack
    wave = server.waveform("a")
    assert wave["grid_corrected"] is True
    assert len(wave["beats"]) == len(analysis.beats)
    assert wave["first_downbeat"] == pytest.approx(analysis.first_downbeat, abs=1e-3)
    assert an.load_cached(analysis.track_id, tmp_path).grid_manually_corrected


def test_the_ui_refuses_a_grid_op_it_does_not_know(session):
    from djai.ui_server import UIServer

    ack = UIServer(session, intent_engine=None).handle_action(
        {"type": "grid", "deck": "a", "op": "triple"}
    )
    assert ack["ok"] is False


def test_the_page_has_grid_controls_a_drag_handle_and_a_review_badge():
    root = Path(__file__).resolve().parent.parent / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")
    for deck in ("a", "b"):
        assert f'id="gridctl-{deck}"' in html
    for needle in ('type: "grid"', '"halve"', '"double"', '"nudge"', '"tap"',
                   '"downbeat"', "REVIEW", "first_downbeat", "pointerdown"):
        assert needle in js, needle


def test_the_library_flags_tracks_that_need_review(session):
    from djai.ui_server import UIServer

    doubtful, clean = session.crate[1], session.crate[2]
    doubtful.tempo_ambiguous = True
    clean.tempo_ambiguous = False
    rows = {r["track_id"]: r for r in UIServer(session, intent_engine=None).library()}
    assert rows[doubtful.track_id]["review"] is True
    assert rows[clean.track_id]["review"] is False


# --- ambiguity -------------------------------------------------------------------------


def test_the_ambiguity_rule_only_applies_where_double_is_a_real_tempo():
    close = {"base": 0.50, "half": 0.30, "double": 0.46}
    assert an.tempo_ambiguity(80.0, close) is True
    assert an.tempo_ambiguity(128.0, close) is False, "256 is not a tempo anyone mixes at"
    assert an.tempo_ambiguity(80.0, {"base": 0.50, "double": 0.30}) is False


def test_a_half_time_reading_is_flagged_for_review_not_accepted(tmp_path):
    path = tmp_path / "half.wav"
    synth.render_track(path, bpm=150.0, root="A", minor=True, bars=32, seed=4)
    reading = make_analysis("half", 75.0, "8A", 0.5, path)  # read at half its tempo
    reading.grid_confidence = 0.6
    reading.lufs = -14.0  # so only the tempo scores are measured here
    reading.played_peak_dbtp = -3.0
    an.complete_measurements(reading, path)
    scores = reading.tempo_scores
    assert scores["double"] >= config.TEMPO_AMBIGUITY_RATIO * scores["base"], scores
    assert reading.tempo_ambiguous
    assert reading.review_needed


def test_a_corrected_grid_is_never_flagged_again(tmp_path):
    ta = make_analysis("t", 75.0, "8A", 0.5, tmp_path / "t.wav")
    ta.tempo_ambiguous = True
    assert ta.review_needed
    an.regrid(ta, 150.0, ta.first_downbeat)
    assert not ta.review_needed


def test_a_manual_correction_survives_forced_reanalysis(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    synth.render_track(library / "t.wav", bpm=124.0, root="A", minor=True, bars=24, seed=1)
    cache = tmp_path / "cache"
    (ta,), _, _ = an.analyze_folder(library, cache)

    an.regrid(ta, ta.bpm * 2, ta.first_downbeat + 0.004)
    an.write_sidecar(ta, cache)
    corrected_bpm, corrected_first = ta.bpm, ta.first_downbeat

    (again,), analyzed, _ = an.analyze_folder(library, cache, force=True)
    assert analyzed == 1, "it really was re-analysed"
    assert again.grid_manually_corrected
    assert again.bpm == pytest.approx(corrected_bpm, rel=1e-6)
    assert again.first_downbeat == pytest.approx(corrected_first, abs=2e-3)
    reloaded = an.load_cached(again.track_id, cache)
    assert reloaded.bpm == pytest.approx(corrected_bpm, rel=1e-6)


def test_the_selector_ranks_a_track_needing_review_below_an_otherwise_better_one(tmp_path):
    current = make_analysis("now", 124.0, "8A", 0.5, tmp_path / "now.wav")
    clean = make_analysis("clean", 124.0, "8A", 0.5, tmp_path / "clean.wav")
    doubtful = make_analysis("doubtful", 124.0, "8A", 0.5, tmp_path / "doubtful.wav")
    clean.grid_confidence = 0.9      # slightly worse on its own merits
    doubtful.tempo_ambiguous = True  # but this one needs review
    ranked = selector.rank_candidates([current, doubtful, clean], current, set(), 0.0)
    assert [c.track.track_id for c in ranked] == ["clean", "doubtful"]


# --- loudness ----------------------------------------------------------------------------


def test_loudness_gain_is_applied_once_at_load(tmp_path):
    path = tmp_path / "tone.wav"
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE)
    sf.write(str(path), np.stack([tone, tone], axis=1).astype(np.float32), SAMPLE_RATE)
    ta = make_analysis("tone", 120.0, "8A", 0.5, path)
    plain = load_track(ta).audio
    ta.track_gain_db = -6.0
    quieter = load_track(ta).audio
    assert float(np.max(np.abs(quieter))) == pytest.approx(
        float(np.max(np.abs(plain))) * 10 ** (-6.0 / 20.0), rel=1e-4
    )


def test_the_gain_reaches_the_target_but_never_past_the_true_peak_cap():
    assert an.loudness_gain_db(-8.0, -3.0) == pytest.approx(config.LOUDNESS_TARGET_LUFS + 8.0)
    # Loud and peaky: the peak cap wins over the loudness target.
    assert an.loudness_gain_db(-10.0, 5.0) == pytest.approx(
        config.LOUDNESS_MAX_TRUE_PEAK_DBTP - 5.0
    )
    # The peak as a deck plays it wins when the EQ's phase rotation raises it.
    # A -12 LUFS track needs only -2 dB for the target; the file's peak alone
    # would allow that, but a +2 dBTP played peak forces -3 dB.
    assert an.loudness_gain_db(-12.0, -3.0) == pytest.approx(-2.0)
    assert an.loudness_gain_db(-12.0, -3.0, 2.0) == pytest.approx(
        config.LOUDNESS_MAX_TRUE_PEAK_DBTP - 2.0
    )
    assert an.loudness_gain_db(None, None) == 0.0
    assert an.loudness_gain_db(float("nan"), None) == 0.0


def test_loudness_is_measured_at_analysis(tmp_path):
    path = tmp_path / "t.wav"
    synth.render_track(path, bpm=124.0, root="A", minor=True, bars=16, seed=2)
    lufs, true_peak, played_peak = an.measure_loudness(path)
    assert -40.0 < lufs < 0.0
    assert math.isfinite(true_peak) and math.isfinite(played_peak)
    assert an.loudness_gain_db(lufs, true_peak, played_peak) <= (
        config.LOUDNESS_TARGET_LUFS - lufs + 1e-6
    )


def test_the_gain_keeps_the_peak_a_deck_actually_plays_under_the_cap(tmp_path):
    """The Phase 1 run's failure, reproduced on purpose.

    A heavily clipped master, the way a mastering limiter pins a pop record.
    The deck's EQ is flat in level but rotates phase, which re-exposes those
    peaks: capping gain on the file's true peak let a real track reach 1.025
    into the limiter. Capping on the peak as played must not.
    """
    from djai.deck import Deck

    rng = np.random.default_rng(7)
    n = SAMPLE_RATE * 6
    t = np.arange(n) / SAMPLE_RATE
    thump = np.sin(2 * np.pi * 55 * t) * (np.mod(t, 0.5) < 0.12)
    mix = 3.0 * thump + 0.45 * rng.standard_normal(n) + 2.0 * np.sin(2 * np.pi * 3000 * t)
    mix = np.clip(mix, -0.98, 0.98).astype(np.float32)
    path = tmp_path / "limited.wav"
    sf.write(str(path), np.stack([mix, mix], axis=1), SAMPLE_RATE, subtype="FLOAT")

    lufs, true_peak, played_peak = an.measure_loudness(path)
    assert played_peak > true_peak, (
        f"this test needs a master the EQ raises: file {true_peak:.2f}, played {played_peak:.2f}"
    )

    ta = make_analysis("limited", 120.0, "8A", 0.5, path)
    ta.track_gain_db = an.loudness_gain_db(lufs, true_peak, played_peak)
    deck = Deck("probe")
    deck.attach(load_track(ta), 0)
    deck.playing = True
    deck.gain.jump(1.0)
    peak = 0.0
    for _ in range(n // 2048 - 1):
        peak = max(peak, float(np.max(np.abs(deck.read(2048)))))
    ceiling = 10 ** (config.LOUDNESS_MAX_TRUE_PEAK_DBTP / 20)
    assert peak <= ceiling + 0.01, f"the deck played {peak:.4f}, over the cap {ceiling:.4f}"


# --- limiter reporting ------------------------------------------------------------------


def test_limiting_deeper_than_one_db_is_counted_and_logged(tmp_path):
    from djai.scheduler import Scheduler
    from djai.supervisor import SessionLog, Supervisor
    from tests.test_limiter import hot_engine

    engine = hot_engine(1.6)
    supervisor = Supervisor(engine, Scheduler(engine), [], session_log=SessionLog(tmp_path))
    drive(engine, 40, BLOCK)
    assert engine.limiter_heavy_blocks > 0
    supervisor.check_limiter()
    supervisor.log.close()
    entries = [json.loads(line) for line in
               supervisor.log.path.read_text(encoding="utf-8").splitlines() if line]
    hits = [e for e in entries if e["event"] == "limiter_reduction"]
    assert hits, "limiting past 1 dB must be logged"
    assert hits[-1]["deepest_reduction_db"] >= config.LIMITER_LOG_REDUCTION_DB


def test_normal_levels_never_trip_the_limiter_report():
    from tests.test_limiter import hot_engine

    engine = hot_engine(0.8)
    drive(engine, 100, BLOCK)
    assert engine.limiter_heavy_blocks == 0
