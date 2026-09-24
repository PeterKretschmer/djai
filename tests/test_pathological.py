"""Phase 1.3 / SPEC §10: every bad input has a behaviour, and it is tested.

THREADING CONTEXT: main thread (pytest); the callback is driven directly and
no audio device is opened.

The rule: a bad file, a bad sample or a bad grid may cost you that *track*. It
may never cost you the *set*. Each case below states what happens, so that
"what does it do with a truncated file" has an answer other than "let's see".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from djai import cli
from djai.commands import LoadTrack
from djai.deck import CHANNELS, SAMPLE_RATE, LoadedTrack, load_track
from djai.engine import Engine
from tests.test_engine import drive, loud_track
from tests.test_integration import make_analysis

BLOCK = 512


def analysis_at(path: Path, bpm: float = 128.0):
    return make_analysis(path.stem, bpm, "8A", 0.5, path)


def write_tone(path: Path, seconds: float = 240.0) -> Path:
    tone = (0.3 * np.sin(np.linspace(0, 8000, int(SAMPLE_RATE * seconds)))
            ).astype(np.float32)
    sf.write(str(path), np.stack([tone, tone], axis=1), SAMPLE_RATE)
    return path


# --- files that decode, and what they decode to -------------------------------


@pytest.mark.parametrize("name, make, expect_frames", [
    ("zero_length", lambda p: sf.write(str(p), np.zeros((0, 2), np.float32),
                                       SAMPLE_RATE), 2),
    ("one_sample", lambda p: sf.write(str(p), np.zeros((1, 2), np.float32),
                                      SAMPLE_RATE), 3),
    ("silence", lambda p: sf.write(str(p), np.zeros((SAMPLE_RATE, 2), np.float32),
                                   SAMPLE_RATE), SAMPLE_RATE + 2),
])
def test_a_degenerate_file_loads_to_a_defined_length(tmp_path, name, make,
                                                     expect_frames):
    """Empty and near-empty files load rather than raise. The two extra frames
    are the interpolation guard every loaded track carries."""
    path = tmp_path / f"{name}.wav"
    make(path)
    track = load_track(analysis_at(path))
    assert track.audio.shape == (expect_frames, CHANNELS)
    assert np.isfinite(track.audio).all()


@pytest.mark.parametrize("channels", [1, 2, 8])
def test_any_channel_count_becomes_stereo(tmp_path, channels):
    path = tmp_path / f"ch{channels}.wav"
    data = (np.zeros(SAMPLE_RATE, np.float32) if channels == 1
            else np.zeros((SAMPLE_RATE, channels), np.float32))
    sf.write(str(path), data, SAMPLE_RATE)
    assert load_track(analysis_at(path)).audio.shape[1] == CHANNELS


@pytest.mark.parametrize("file_rate", [8000, 22050, 48000, 192000])
def test_any_sample_rate_is_resampled_to_the_engine_rate(tmp_path, file_rate):
    """A file at another rate must arrive at SAMPLE_RATE or it plays at the
    wrong speed and pitch for the whole track."""
    path = tmp_path / f"sr{file_rate}.wav"
    sf.write(str(path), np.zeros((file_rate, 2), np.float32), file_rate)
    track = load_track(analysis_at(path))
    # one second in, one second out, whatever the file said
    assert abs(track.audio.shape[0] - (SAMPLE_RATE + 2)) <= 2


def test_a_truncated_file_loads_what_survived(tmp_path):
    path = write_tone(tmp_path / "truncated.wav", seconds=2.0)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])
    track = load_track(analysis_at(path))
    assert 0 < track.audio.shape[0] < SAMPLE_RATE * 2
    assert np.isfinite(track.audio).all()


def test_non_finite_samples_in_a_file_are_replaced_with_silence(tmp_path):
    """A NaN that reaches the mix latches the master limiter's gain to NaN and
    the set never recovers, so it is scrubbed at the door."""
    sig = np.zeros((SAMPLE_RATE, 2), np.float32)
    sig[100] = np.nan
    sig[200] = np.inf
    path = tmp_path / "nonfinite.wav"
    sf.write(str(path), sig, SAMPLE_RATE)
    assert np.isfinite(load_track(analysis_at(path)).audio).all()


@pytest.mark.parametrize("name, payload", [
    ("not_audio", b"this is not a wav file at all"),
    ("empty", b""),
])
def test_a_file_that_is_not_audio_raises_rather_than_returning_junk(tmp_path,
                                                                    name, payload):
    path = tmp_path / f"{name}.wav"
    path.write_bytes(payload)
    with pytest.raises(Exception):
        load_track(analysis_at(path))


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        load_track(analysis_at(tmp_path / "gone.wav"))


# --- a bad sample may not take the master down --------------------------------


def test_a_nan_in_a_deck_cannot_latch_the_limiter(tmp_path):
    """The backstop, tested where it matters: a track whose audio went bad
    *after* loading -- a stretch that produced NaN, an FX divide by zero."""
    base = loud_track(200.0, bpm=128.0)
    audio = base.audio.copy()
    audio[5000:5010] = np.nan
    audio[6000:6010] = np.inf
    engine = Engine(blocksize=BLOCK)
    engine.submit(LoadTrack(deck="a", track=LoadedTrack(analysis=base.analysis,
                                                        audio=audio),
                            play=True, master=True, origin="t"))
    engine.deck_a.gain.jump(0.8)
    out = drive(engine, 60)

    assert np.isfinite(out).all(), "non-finite audio reached the output"
    assert np.isfinite(engine.limiter_gain), "the limiter gain latched"
    assert engine.nonfinite_blocks > 0, "the scrub never ran, so this proves nothing"
    engine.stop()


def test_the_limiter_recovers_after_the_bad_audio_passes():
    """Not merely finite: back to unity gain, so the set is not left quiet."""
    base = loud_track(200.0, bpm=128.0)
    audio = base.audio.copy()
    audio[1000:1010] = np.nan
    engine = Engine(blocksize=BLOCK)
    engine.submit(LoadTrack(deck="a", track=LoadedTrack(analysis=base.analysis,
                                                        audio=audio),
                            play=True, master=True, origin="t"))
    engine.deck_a.gain.jump(0.5)
    drive(engine, 200)
    assert engine.limiter_gain == pytest.approx(1.0, abs=0.01)
    engine.stop()


# --- one bad file may not cost the set ----------------------------------------


@pytest.fixture
def crate_with_a_broken_file(tmp_path):
    paths = []
    for name, bpm, broken in (("good_a", 124.0, False),
                              ("broken", 125.0, True),
                              ("good_b", 126.0, False)):
        path = tmp_path / f"{name}.wav"
        if broken:
            path.write_bytes(b"not a wav file")
        else:
            write_tone(path)
        paths.append(make_analysis(name, bpm, "8A", 0.5, path))
    return paths


def test_an_unreadable_opening_track_does_not_stop_the_set_starting(
        tmp_path, crate_with_a_broken_file):
    """It used to raise straight out of start_first_track: no set at all."""
    session = cli.Session(crate_with_a_broken_file, log_dir=tmp_path / "logs")
    try:
        assert session.start_first_track() is True
        drive(session.engine, 4)
        playing = session.engine.deck("a").track
        assert playing is not None and playing.analysis.title != "broken"
        assert "broken" in session._unplayable
    finally:
        session.shutdown()


def test_an_unreadable_track_is_set_aside_rather_than_retried_for_ever(
        tmp_path, crate_with_a_broken_file):
    """Nothing marks a failed track played, so without this the selector ranks
    the same broken file first on every tick and the set stalls."""
    session = cli.Session(crate_with_a_broken_file, log_dir=tmp_path / "logs")
    try:
        session.start_first_track()
        drive(session.engine, 4)
        assert session.cue_next(origin="autopilot") is True
        assert session._cued is not None
        assert session._cued.title != "broken"
        assert "broken" in session._unplayable
    finally:
        session.shutdown()


def test_an_unplayable_track_is_reported_by_name_with_a_reason(
        tmp_path, crate_with_a_broken_file):
    """A set that quietly stops choosing something must say why on disk."""
    session = cli.Session(crate_with_a_broken_file, log_dir=tmp_path / "logs")
    try:
        session.start_first_track()
        session.session_log._file.flush()
        text = session.session_log.path.read_text(encoding="utf-8")
        assert "track_unplayable" in text
        assert "broken" in text
    finally:
        session.shutdown()


def test_a_forced_unreadable_track_is_refused_not_dropped(
        tmp_path, crate_with_a_broken_file):
    """The operator asked for it by name, so they are told it cannot play --
    an operator request is never silently dropped."""
    session = cli.Session(crate_with_a_broken_file, log_dir=tmp_path / "logs")
    try:
        session.start_first_track()
        drive(session.engine, 4)
        session._forced_next = crate_with_a_broken_file[1]
        assert session.cue_next(origin="operator") is False
        assert any("will not decode" in n or "set aside" in n
                   for n in session.drain_notices())
    finally:
        session.shutdown()
