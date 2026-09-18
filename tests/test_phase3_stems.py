"""Phase 3: separated stems, cached offline, swapped in on a bar line.

THREADING CONTEXT: main thread (pytest). No separation runs here -- the model
is a 300 MB download and minutes of GPU time -- so the cache is written with
known synthetic stems and every number is checked against what was written.
One test does run the real model, and skips when it is not installed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from djai import cli, phrase
from djai import stems as stems_mod
from djai.commands import StartTransition, SwapStems
from djai.deck import SAMPLE_RATE, LoadedTrack
from djai.engine import Engine
from tests.test_engine import drive, loud_track
from tests.test_integration import log_events, make_analysis, seek_near_mix_out
from tests.test_integration import session as _session_fixture

session = _session_fixture

BLOCK = 512


def write_stems(analysis, cache_dir, gain=1.0, seconds=None, level=None):
    """A cache entry whose four stems are one steady tone each."""
    seconds = seconds or analysis.duration_s
    level = level or {"drums": 0.4, "bass": 0.3, "other": 0.2, "vocals": 0.1}
    folder = stems_mod.stems_dir(analysis.track_id, cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    manifest = {"model": stems_mod.MODEL_NAME, "track_id": analysis.track_id,
                "title": analysis.title, "sample_rate": SAMPLE_RATE,
                "duration_s": seconds, "gain": gain, "stems": {}}
    for i, name in enumerate(stems_mod.STEM_NAMES):
        wave = (level[name] * np.sin(2 * np.pi * (110 * (i + 1)) * t)).astype(np.float32)
        data = np.stack([wave, wave], axis=1)
        path = folder / f"{name}.flac"
        sf.write(str(path), data * gain, SAMPLE_RATE, subtype="PCM_16")
        manifest["stems"][name] = {"peak": float(level[name]), "bytes": path.stat().st_size}
    (folder / "stems.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def loaded(analysis, seconds=None):
    base = loud_track(300.0, analysis.bpm, seconds or analysis.duration_s)
    return LoadedTrack(analysis=analysis, audio=base.audio)


@pytest.fixture
def track(tmp_path):
    p = tmp_path / "t.wav"
    p.write_bytes(b"placeholder")
    return make_analysis("stemmy", 128.0, "8A", 0.5, p)


# --- the cache ------------------------------------------------------------------


def test_stems_are_found_by_the_tracks_content_hash(track, tmp_path):
    assert stems_mod.cached(track.track_id, tmp_path) is None
    write_stems(track, tmp_path)
    manifest = stems_mod.cached(track.track_id, tmp_path)
    assert manifest is not None and set(manifest["stems"]) == set(stems_mod.STEM_NAMES)


def test_a_cache_entry_from_another_model_is_not_used(track, tmp_path):
    write_stems(track, tmp_path)
    path = stems_mod.manifest_path(track.track_id, tmp_path)
    manifest = json.loads(path.read_text())
    manifest["model"] = "something-else"
    path.write_text(json.dumps(manifest))
    assert stems_mod.cached(track.track_id, tmp_path) is None


def test_a_missing_stem_file_invalidates_the_entry(track, tmp_path):
    write_stems(track, tmp_path)
    (stems_mod.stems_dir(track.track_id, tmp_path) / "bass.flac").unlink()
    assert stems_mod.cached(track.track_id, tmp_path) is None


# --- building a mix --------------------------------------------------------------


def test_a_mix_sums_only_the_stems_it_names(track, tmp_path):
    write_stems(track, tmp_path)
    full = loaded(track)
    instrumental = stems_mod.load_mix(full, tmp_path, "instrumental")
    acapella = stems_mod.load_mix(full, tmp_path, "acapella")

    assert instrumental is not None and acapella is not None
    # The vocals tone (110*4 Hz at 0.1) is in the acapella and not the rest.
    def energy_at(audio, freq):
        seg = audio[SAMPLE_RATE: 2 * SAMPLE_RATE, 0]
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        freqs = np.fft.rfftfreq(seg.size, 1 / SAMPLE_RATE)
        return float(spec[np.argmin(np.abs(freqs - freq))])

    assert energy_at(acapella.audio, 440.0) > 10 * energy_at(instrumental.audio, 440.0)
    assert energy_at(instrumental.audio, 110.0) > 10 * energy_at(acapella.audio, 110.0)


def test_the_stored_gain_is_undone_so_levels_come_back(track, tmp_path):
    write_stems(track, tmp_path, gain=1.0)
    plain = stems_mod.load_mix(loaded(track), tmp_path, "acapella")
    write_stems(track, tmp_path, gain=0.5)
    scaled = stems_mod.load_mix(loaded(track), tmp_path, "acapella")
    a = float(np.max(np.abs(plain.audio)))
    b = float(np.max(np.abs(scaled.audio)))
    assert b == pytest.approx(a, rel=0.02), "a stem written with headroom plays as loud"


def test_a_mix_is_never_louder_than_the_master_it_came_from(track, tmp_path):
    """Stems can be far louder than the limited mix they came from: one track's
    drums peaked at 14.2, another spends 14.5 s above 1.0. Handing that to the
    limiter would be a level the track never had, and the swap would jump."""
    write_stems(track, tmp_path, level={"drums": 3.0, "bass": 2.0,
                                        "other": 1.0, "vocals": 0.5}, gain=0.25)
    full = loaded(track)
    original_peak = float(np.max(np.abs(full.audio)))
    mix = stems_mod.load_mix(full, tmp_path, "instrumental")
    assert float(np.max(np.abs(mix.audio))) <= original_peak * 1.001


def test_a_quieter_mix_is_left_quiet(track, tmp_path):
    """An acapella is quieter than the master, and must stay that way."""
    write_stems(track, tmp_path, level={"drums": 0.5, "bass": 0.5,
                                        "other": 0.5, "vocals": 0.05})
    full = loaded(track)
    mix = stems_mod.load_mix(full, tmp_path, "acapella")
    assert float(np.max(np.abs(mix.audio))) < 0.2 * float(np.max(np.abs(full.audio)))


def test_a_mix_keeps_the_tracks_own_analysis_so_a_deck_will_take_it(track, tmp_path):
    write_stems(track, tmp_path)
    full = loaded(track)
    mix = stems_mod.load_mix(full, tmp_path, "no_bass")
    assert mix.analysis is full.analysis
    assert mix.audio.shape == full.audio.shape
    assert mix.audio.dtype == np.float32


def test_no_stems_means_no_mix(track, tmp_path):
    assert stems_mod.load_mix(loaded(track), tmp_path, "instrumental") is None


def test_an_unknown_mix_is_a_programming_error(track, tmp_path):
    write_stems(track, tmp_path)
    with pytest.raises(ValueError, match="unknown stem mix"):
        stems_mod.load_mix(loaded(track), tmp_path, "just_the_triangle")


# --- the swap ---------------------------------------------------------------------


def test_the_engine_swaps_a_stem_mix_under_the_playhead(track, tmp_path):
    write_stems(track, tmp_path)
    engine = Engine(blocksize=BLOCK)
    full = loaded(track, seconds=30.0)
    engine.submit(cli.LoadTrack(deck="a", track=full, master=True))
    engine.deck_a.gain.jump(1.0)
    drive(engine, 4, BLOCK)
    position = engine.deck_a.position
    mix = stems_mod.load_mix(full, tmp_path, "acapella")

    engine.submit(SwapStems(deck="a", mix="acapella", track=mix))
    drive(engine, 4, BLOCK)

    assert engine.deck_a.track is mix, "the deck is playing the stem mix"
    assert engine.deck_a.position > position, "and the playhead carried on"
    assert engine.deck_a.track.analysis is track
    engine.stop()


def test_a_mix_of_another_track_is_refused(track, tmp_path):
    other = make_analysis("elsewhere", 124.0, "9A", 0.5, tmp_path / "o.wav")
    engine = Engine(blocksize=BLOCK)
    mine = loaded(track, seconds=30.0)
    engine.submit(cli.LoadTrack(deck="a", track=mine, master=True))
    drive(engine, 2, BLOCK)

    engine.submit(SwapStems(deck="a", mix="acapella", track=loaded(other, seconds=30.0)))
    drive(engine, 2, BLOCK)

    assert engine.deck_a.track is mine, "a mix for other music must not land"
    engine.stop()


# --- transitions ------------------------------------------------------------------


def stem_all(session, cache_dir):
    for analysis in session.crate:
        write_stems(analysis, cache_dir, seconds=analysis.duration_s)


def test_a_vocal_clash_becomes_a_stem_move_not_a_cut(session, tmp_path, monkeypatch):
    """The brief's rule: vocals that would clash are moved out by stem, and the
    blend survives."""
    session.cache_dir = tmp_path
    stem_all(session, tmp_path)
    for analysis in session.crate:
        object.__setattr__(analysis, "vocal_bars", [[0, int(analysis.mix_out_bar)]])
        object.__setattr__(analysis, "vocal_fraction", 0.9)

    seek_near_mix_out(session, bars_before=40.0)
    session.cue_next()
    drive(session.engine, 2, BLOCK)
    assert session.arm_transition() is not None

    events = [e["event"] for e in log_events(session)]
    assert "vocal_clash_avoided" in events
    assert "vocal_clash_fallback" not in events, "no cut"
    assert session.last_transition_choice[0] != "cut"
    moves = [m for m in session.stem_moves if m["mix"] == "instrumental"]
    assert moves and moves[0]["deck"] == session.live_deck


def test_an_acapella_goes_over_the_incoming_instrumental(session, tmp_path):
    """Deck A is singing, deck B has no vocals of its own: the voice carries
    over the incoming track rather than both playing full."""
    session.cache_dir = tmp_path
    stem_all(session, tmp_path)
    live = session.engine.deck(session.live_deck).track.analysis
    object.__setattr__(live, "vocal_bars", [[0, int(live.mix_out_bar)]])
    object.__setattr__(live, "vocal_fraction", 0.9)
    for analysis in session.crate:
        if analysis is not live:
            object.__setattr__(analysis, "vocal_bars", [])
            object.__setattr__(analysis, "vocal_fraction", 0.0)

    seek_near_mix_out(session, bars_before=40.0)
    session.cue_next()
    drive(session.engine, 2, BLOCK)
    assert session.arm_transition() is not None

    moves = {(m["deck"], m["mix"]) for m in session.stem_moves}
    assert (session.live_deck, "acapella") in moves
    assert (session.cued_deck(), "instrumental") in moves
    assert (session.cued_deck(), "original") in moves, "and it gets its master back"


def test_the_planned_moves_reach_the_scheduler_as_bar_aligned_swaps(session, tmp_path):
    session.cache_dir = tmp_path
    stem_all(session, tmp_path)
    seek_near_mix_out(session, bars_before=40.0)
    session.cue_next()
    drive(session.engine, 2, BLOCK)
    session.arm_transition()
    assert session.stem_moves, "a long blend gets a bass hand-over at least"

    # The worker decodes four FLACs per mix, which takes seconds of wall time,
    # so this waits on the clock rather than on a tight loop that never yields.
    import time

    deadline = time.monotonic() + 30.0
    swaps: list = []
    while time.monotonic() < deadline:
        swaps = [c for c in session.scheduler.pending() if isinstance(c, SwapStems)]
        if len(swaps) >= len(session.stem_moves):
            break
        time.sleep(0.1)
    assert swaps, "no stem swap was queued"
    swap = next(c for c in session.scheduler.pending() if isinstance(c, StartTransition))
    for cmd in swaps:
        assert cmd.track is not None and not cmd.is_immediate
        assert cmd.execute_at >= swap.execute_at, "a move lands inside the blend"
    kinds = {c.mix for c in swaps}
    assert "no_bass" in kinds, f"expected a bass hand-over, got {kinds}"


def test_the_bass_hand_over_mix_has_no_bass(session, tmp_path):
    session.cache_dir = tmp_path
    stem_all(session, tmp_path)
    live = session.engine.deck(session.live_deck)
    mix = stems_mod.load_mix(live.track, tmp_path, "no_bass")
    seg = mix.audio[SAMPLE_RATE: 2 * SAMPLE_RATE, 0]
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    freqs = np.fft.rfftfreq(seg.size, 1 / SAMPLE_RATE)
    bass_bin = float(spec[np.argmin(np.abs(freqs - 220.0))])   # the bass stem's tone
    drums_bin = float(spec[np.argmin(np.abs(freqs - 110.0))])  # the drums stem's tone
    assert drums_bin > 10 * bass_bin


def test_without_stems_a_transition_is_planned_exactly_as_before(session, tmp_path):
    """Stems are an improvement to reach for, never a dependency."""
    session.cache_dir = tmp_path  # empty: no stems for anything
    seek_near_mix_out(session, bars_before=40.0)
    session.cue_next()
    drive(session.engine, 2, BLOCK)

    assert session.arm_transition() is not None
    assert session.stem_moves == []
    assert not [c for c in session.scheduler.pending() if isinstance(c, SwapStems)]


def test_stems_can_be_turned_off(session, tmp_path):
    session.cache_dir = tmp_path
    stem_all(session, tmp_path)
    session.stems_enabled = False
    assert not session.stems_ready(session.crate[0])
    seek_near_mix_out(session, bars_before=40.0)
    session.cue_next()
    drive(session.engine, 2, BLOCK)
    session.arm_transition()
    assert session.stem_moves == []


# --- the real model ----------------------------------------------------------------


@pytest.mark.skipif(not stems_mod.available(), reason="torch/torchaudio not installed")
def test_the_real_model_separates_a_synthetic_track(tmp_path):
    """Slow (GPU), and the only test that loads the 319 MB model."""
    from tests import synth

    path = synth.render_track(tmp_path / "real.wav", bpm=128.0, bars=8, seed=4)
    analysis = make_analysis("real", 128.0, "8A", 0.5, path)
    object.__setattr__(analysis, "path", str(path))
    manifest = stems_mod.precompute(analysis, tmp_path)

    assert set(manifest["stems"]) == set(stems_mod.STEM_NAMES)
    assert 0.0 < manifest["gain"] <= 1.0
    folder = stems_mod.stems_dir(analysis.track_id, tmp_path)
    for name in stems_mod.STEM_NAMES:
        data, sr = sf.read(str(folder / f"{name}.flac"), always_2d=True)
        assert sr == 44100 and data.shape[1] == 2
        assert np.max(np.abs(data)) <= 1.0, "stored stems never clip"
    assert stems_mod.cached(analysis.track_id, tmp_path) is not None
