"""Phase 2: key lock and the resonant filter, measured.

THREADING CONTEXT: main thread (pytest). Decks and the engine callback are
driven directly; the stretch worker is a real thread where an Engine is built
with time-stretch enabled.

The Rubber Band stretch itself is covered by tests/test_stretch.py, whose pitch,
ratio and fallback tests now run against it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from djai import analysis, cli
from djai import deck as deck_mod
from djai import transition as tr
from djai.commands import LoadTrack
from djai.deck import SAMPLE_RATE, Deck, LoadedTrack, time_stretch
from djai.engine import Engine
from djai.render import render_transition
from tests import synth
from tests.test_designed_transitions import GOOD
from tests.test_engine import drive, loud_track
from tests.test_integration import session as _session_fixture
from tests.test_stretch import RATE, musical, shift_cents, wait_for_stretch
from tests.test_transition_library import BLOCK, _heap_slope, _start

session = _session_fixture

BIG = 2048


# --- helpers -----------------------------------------------------------------


def _playing_deck(track: LoadedTrack, resonance: float = 0.0) -> Deck:
    d = Deck("x")
    d.attach(track)
    d.playing = True
    d.gain.jump(1.0)
    d.set_filter(0.0, resonance)
    return d


def _run(d: Deck, blocks: int, positions=None) -> np.ndarray:
    out = []
    for i in range(blocks):
        if positions is not None:
            d.set_filter(float(positions[i]))
        out.append(d.read(BIG)[:, 0].copy())
    return np.concatenate(out)


def _band_ratio_db(x: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    """Energy in [lo, hi] Hz as a fraction of all energy, in dB."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / SAMPLE_RATE)
    band = spec[(freqs >= lo_hz) & (freqs <= hi_hz)].sum()
    return float(10.0 * np.log10(max(band, 1e-30) / max(spec.sum(), 1e-30)))


def _noise_track(seconds: float, seed: int = 0) -> LoadedTrack:
    n = int(SAMPLE_RATE * seconds)
    rng = np.random.default_rng(seed)
    sig = (0.25 * rng.standard_normal(n)).astype(np.float32)
    audio = np.stack([sig, sig], axis=1)
    audio = np.vstack([audio, np.zeros((2, 2), dtype=np.float32)])
    return LoadedTrack(analysis=loud_track(100.0, seconds=seconds).analysis, audio=audio)


# --- the filter: range, bypass, stability, zipper ----------------------------


def test_every_biquad_in_the_table_is_stable():
    """Both poles inside the unit circle, for every knob and resonance step."""
    table, wet = deck_mod.filter_tables()
    a1 = table[..., 0, 4]
    a2 = table[..., 0, 5]
    assert np.all(np.abs(a2) < 1.0)
    assert np.all(np.abs(a1) < 1.0 + a2)
    assert len(wet) == deck_mod.FILTER_STEPS
    assert wet[0] == 0.0 and wet[-1] == 1.0


def test_the_detent_is_a_true_bypass():
    track = loud_track(440.0, seconds=3.0)
    filtered = _playing_deck(track, resonance=1.0)
    plain = Deck("plain")
    plain.attach(track)
    plain.playing = True
    plain.gain.jump(1.0)
    for _ in range(20):
        assert np.array_equal(filtered.read(BIG), plain.read(BIG))


@pytest.mark.parametrize(
    "position, freq",
    [(-1.0, 3000.0), (1.0, 200.0)],
    ids=["low-pass fully left kills 3 kHz", "high-pass fully right kills 200 Hz"],
)
def test_the_knob_at_either_end_removes_what_it_should(position, freq):
    d = _playing_deck(loud_track(freq, seconds=4.0))
    d.filter_pos.jump(position)
    out = _run(d, 40)[20 * BIG:]
    ref = 0.5 / np.sqrt(2.0)
    level_db = 20.0 * np.log10(np.sqrt(np.mean(out ** 2)) / ref)
    assert level_db < -40.0, f"only {level_db:.1f} dB down"


def test_the_passband_is_left_at_unity_without_resonance():
    d = _playing_deck(loud_track(100.0, seconds=4.0), resonance=0.0)
    d.filter_pos.jump(-0.5)                   # low-pass near 775 Hz
    out = _run(d, 40)[20 * BIG:]
    level_db = 20.0 * np.log10(np.sqrt(np.mean(out ** 2)) / (0.5 / np.sqrt(2.0)))
    assert abs(level_db) < 0.5


def test_resonance_raises_the_response_at_the_cutoff():
    position = -0.5
    tone = deck_mod.filter_cutoff_hz(position)
    levels = []
    for res in (0.0, 1.0):
        d = _playing_deck(loud_track(tone, seconds=4.0), resonance=res)
        d.filter_pos.jump(position)
        out = _run(d, 40)[20 * BIG:]
        levels.append(np.sqrt(np.mean(out ** 2)))
    gain_db = 20.0 * np.log10(levels[1] / levels[0])
    assert gain_db > 6.0, f"resonance adds only {gain_db:.1f} dB at the cutoff"


def test_full_range_sweeps_at_maximum_resonance_stay_stable():
    """Detent to fully left, across to fully right and back, at resonance 1."""
    positions = np.concatenate([
        np.linspace(0.0, -1.0, 120),
        np.linspace(-1.0, 1.0, 240),
        np.linspace(1.0, 0.0, 120),
    ])
    d = _playing_deck(_noise_track(seconds=len(positions) * BIG / SAMPLE_RATE + 1.0),
                      resonance=1.0)
    out = _run(d, len(positions), positions)
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out))) < 4.0, "resonance ran away"
    # Back in the detent, the state is cleared and the deck is bypassed again.
    assert d._flt_side == -1


@pytest.mark.parametrize(
    "tone, positions, band",
    [
        (200.0, np.linspace(0.0, -0.6, 20), (2000.0, 20000.0)),
        (5000.0, np.linspace(0.0, 0.6, 20), (20.0, 1500.0)),
    ],
    ids=["low-pass sweep over a 200 Hz tone", "high-pass sweep over a 5 kHz tone"],
)
def test_a_resonant_sweep_has_no_zipper_noise(monkeypatch, tone, positions, band):
    """Zipper noise is energy the input does not have: stepped coefficients
    splatter a pure tone across the spectrum. Measured in a band far from the
    tone, against the same sweep with coefficients stepped once per block.

    A quick sweep, 0.93 s across 60% of the knob, because that is where
    stepping shows. Measured: 64-frame steps -65.5 dB against -51.5 dB per
    block; over 7.4 s the two are -75.8 and -70.0 dB."""

    def sweep() -> np.ndarray:
        d = _playing_deck(loud_track(tone, seconds=len(positions) * BIG / SAMPLE_RATE + 1.0),
                          resonance=1.0)
        return _run(d, len(positions), positions)

    smooth = _band_ratio_db(sweep(), *band)
    monkeypatch.setattr(deck_mod, "FILTER_SUBBLOCK", 1 << 20)
    stepped = _band_ratio_db(sweep(), *band)
    assert smooth < -60.0, f"sweep artefacts at {smooth:.1f} dB"
    assert stepped - smooth > 10.0, (
        f"per-sub-block stepping ({smooth:.1f} dB) should beat per-block "
        f"({stepped:.1f} dB) -- otherwise the metric is not seeing zipper noise"
    )


def _median_slope(engine, repeats: int = 3) -> float:
    """Median of repeated slope measurements.

    tracemalloc sees every thread, not just the one driving the callback. In a
    full single-process run another thread once allocated and freed about
    82 MB inside one measurement window, which read as -205,560 B/block for
    the baseline and +205,632 for the sweep: a single event, not the callback.
    Such an event lands in at most one of three windows, so the median
    discards it, while a genuine per-block allocation shows in all three.
    """
    return float(np.median([_heap_slope(engine) for _ in range(repeats)]))


def test_a_resonant_filter_sweep_allocates_nothing_per_block(session):
    """Same standard as the echo send: growth per callback against a bass swap."""
    engine = session.engine
    engine.deck("b").attach(engine.deck("a").track, 0)

    # Three slopes of 1,200 blocks each is ~42 s of audio: inside the 60 s
    # transitions, so the knob is still moving when the last one ends.
    _start(session, "bass_swap", SAMPLE_RATE * 60)
    drive(engine, 4, BLOCK)
    baseline = _median_slope(engine)

    _start(session, "filter_sweep", SAMPLE_RATE * 60)
    drive(engine, 8, BLOCK)
    assert engine.deck("a").filter_pos.cur > 0.0, "the knob should be moving"
    sweeping = _median_slope(engine)
    assert engine.transition_active, "the sweep must still be running at the end"
    assert engine.deck("a")._flt_side == 1, "and filtering, high-pass"
    if sweeping > baseline + 64:
        pytest.fail(
            f"filter sweep grows {sweeping:.1f} B/block vs {baseline:.1f} for a bass swap.\n"
            f"Top growth by file over 800 blocks (any thread):\n{_growth_sites(engine)}"
        )


def _growth_sites(engine, blocks: int = 800, top: int = 10) -> str:
    """Where the heap grew while the callback ran, by file and line.

    Diagnostic only, for a failure message. tracemalloc records every thread,
    so this is what tells a per-block allocation in the audio path apart from
    another thread allocating while the measurement happens to run.
    """
    import gc
    import threading
    import tracemalloc

    buf = np.zeros((BLOCK, 2), dtype=np.float32)
    gc.collect()
    tracemalloc.start(8)
    before = tracemalloc.take_snapshot()
    for _ in range(blocks):
        engine.callback(buf, BLOCK, None, None)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    stats = after.compare_to(before, "traceback")[:top]
    lines = []
    for stat in stats:
        frames = " <- ".join(
            f"{Path(f.filename).name}:{f.lineno}" for f in list(stat.traceback)[:4]
        )
        lines.append(f"  {stat.size_diff:+,} B in {stat.count_diff:+} blocks: {frames}")
    alive = sorted(t.name for t in threading.enumerate())
    return "\n".join(lines) + f"\nThreads alive: {alive}"


def test_the_callback_computes_no_filter_coefficients():
    """A guard on the code: the table is indexed, never built, per block."""
    import inspect

    src = inspect.getsource(Deck._run_filter) + inspect.getsource(Deck.read)
    for forbidden in ("np.cos", "np.sin", "math.", "butter(", "filter_tables(",
                      "filter_cutoff_hz(", "filter_q("):
        assert forbidden not in src, f"the audio path computes {forbidden}"


# --- key lock ------------------------------------------------------------------


def test_key_lock_toggles_per_deck_between_stretched_and_resampled():
    engine = Engine(blocksize=512)
    try:
        track = musical(seconds=12.0)
        engine.submit(LoadTrack(deck="a", track=track, rate=RATE, master=True))
        assert wait_for_stretch(engine, track, RATE) is not None
        engine.submit(LoadTrack(deck="a", track=track, rate=RATE, master=True))
        engine.submit(LoadTrack(deck="b", track=track, rate=RATE))
        drive(engine, 2)
        assert engine.deck_a.track.is_stretched and engine.deck_b.track.is_stretched

        assert engine.set_key_lock("a", False) == "off"
        drive(engine, 2)
        assert engine.deck_a.track is track, "deck A back on the source: resampling"
        assert engine.deck_b.track.is_stretched, "deck B untouched"
        assert engine.deck_state("a").key_lock is False
        assert engine.deck_state("b").key_lock is True

        # The copy is still cached, so turning it back on swaps it straight in.
        assert engine.set_key_lock("a", True) == "stretching"
        drive(engine, 2)
        assert engine.deck_a.track.is_stretched
        assert engine.deck_a.track.source is track
    finally:
        engine.stop()


def test_key_lock_on_waits_for_a_copy_and_then_swaps_it_in():
    engine = Engine(blocksize=512)
    try:
        engine.set_key_lock("a", False)
        drive(engine, 1)
        track = musical(seconds=8.0)
        engine.submit(LoadTrack(deck="a", track=track, rate=RATE, master=True))
        drive(engine, 1)
        time.sleep(0.3)
        assert engine._stretcher.requested == 0, "key lock off asks for no copy"
        assert not engine.deck_a.track.is_stretched

        assert engine.set_key_lock("a", True) == "stretching"
        deadline = time.time() + 60.0
        while not engine.deck_a.track.is_stretched and time.time() < deadline:
            drive(engine, 1)
            time.sleep(0.05)
        assert engine.deck_a.track.is_stretched, "the copy never swapped in"
    finally:
        engine.stop()


def test_a_key_lock_swap_crossfades_rather_than_clicking():
    track = musical(seconds=8.0)
    stretched = time_stretch(track, RATE)
    d = _playing_deck(stretched)
    d.set_rate(RATE)
    before = [d.read(512)[:, 0].copy() for _ in range(40)]
    position = d.position
    d.swap_audio(track)
    after = [d.read(512)[:, 0].copy() for _ in range(40)]
    assert d.position == pytest.approx(position + 40 * 512 * RATE)

    out = np.concatenate(before + after)
    steps = np.abs(np.diff(out))
    seam = 40 * 512
    normal = float(np.max(steps[: seam - 1]))
    at_swap = float(np.max(steps[seam - 2: seam + deck_mod._LOOP_XFADE + 2]))
    assert at_swap <= normal * 1.5, f"swap step {at_swap:.4f} vs normal {normal:.4f}"


def test_swap_audio_refuses_a_different_track():
    a, b = musical(seconds=3.0), musical(seconds=3.0, seed=5)
    d = _playing_deck(a)
    d.swap_audio(b)
    assert d.track is a


# --- controls: REPL, UI, page ---------------------------------------------------


def _release(session) -> None:
    session.scheduler.tick(session.engine.frames_played)
    drive(session.engine, 2, BLOCK)


def test_keylock_and_filter_work_from_the_repl(session):
    reply = cli.handle_override(session, "keylock a off")
    assert reply.startswith("Deck A key lock off"), reply
    _release(session)
    assert session.engine.deck("a").key_lock is False
    assert "Key lock: A off, B on" in cli.handle_override(session, "keylock")

    reply = cli.handle_override(session, "filter a -0.5 res 0.8")
    assert "low-pass" in reply, reply
    _release(session)
    state = session.engine.deck_state("a")
    assert state.filter == pytest.approx(-0.5)
    assert state.filter_resonance == pytest.approx(0.8)

    assert cli.handle_override(session, "filter b 0.7").startswith("Deck B filter: high-pass")
    assert cli.handle_override(session, "filter a 3").startswith("Refused")
    assert cli.handle_override(session, "filter a off") == "Deck A filter off."
    # Plain English still reaches the model.
    assert cli.handle_override(session, "filter this one out") is None
    assert cli.handle_override(session, "filter") is None


def test_keylock_and_filter_work_from_the_ui(session):
    from djai.ui_server import UIServer

    ui = UIServer(session, intent_engine=None)
    ack = ui.handle_action({"type": "filter", "deck": "b", "position": 0.6})
    assert ack["ok"] and "high-pass" in ack["text"], ack
    ack = ui.handle_action({"type": "keylock", "deck": "b", "on": False})
    assert ack["ok"], ack
    bad = ui.handle_action({"type": "filter", "deck": "b", "position": "loud"})
    assert not bad["ok"]
    _release(session)
    deck = ui.state()["decks"]["b"]
    assert deck["filter"] == pytest.approx(0.6)
    assert deck["key_lock"] is False
    assert "filter_resonance" in deck


def test_the_page_has_a_filter_knob_and_key_lock_per_deck():
    root = Path(__file__).resolve().parent.parent / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")
    for d in ("a", "b"):
        assert f'id="filt-{d}"' in html and 'min="-1"' in html
        assert f'id="keylock-{d}"' in html
    assert 'type: "filter"' in js and 'type: "keylock"' in js
    assert "key_lock" in js and "FILTER_SNAP" in js


# --- the schema --------------------------------------------------------------


def test_filter_resonance_is_validated_by_the_supervisor(session):
    supervisor = session.supervisor
    for bad in (1.5, -0.1, "loud", True, None):
        params, why = supervisor.validate_transition_params(
            dict(GOOD, filter_resonance=bad), session.crate[1]
        )
        assert params is None, f"filter_resonance={bad!r} was accepted"
        assert "filter_resonance" in why

    params, why = supervisor.validate_transition_params(
        dict(GOOD, filter_resonance=0.8), session.crate[1]
    )
    assert params is not None, why
    assert params.filter_resonance == pytest.approx(0.8)
    env = tr.build_envelope_from_params(params, SAMPLE_RATE * 8, 512)
    assert float(env[:-1, tr.FILTER_RES].max()) == pytest.approx(0.8)
    assert float(env[:-1, tr.FROM_FILTER].max()) > 0.5


def test_lp_in_opens_the_incoming_deck_filter():
    p = tr.TransitionParams(length_bars=16, filter_sweep="lp_in", intensity=0.8,
                            filter_resonance=0.3)
    env = tr.build_envelope_from_params(p, SAMPLE_RATE * 16, 512)
    assert env[0, tr.TO_FILTER] == pytest.approx(-0.8)
    assert env[-1, tr.TO_FILTER] == 0.0
    assert np.all(np.diff(env[:, tr.TO_FILTER]) >= -1e-7), "only ever opens"
    assert np.all(env[:, tr.FROM_FILTER] == 0.0)


def test_the_prompt_examples_all_carry_filter_resonance():
    import json

    from djai.intent import TRANSITION_PROMPT, TRANSITION_SCHEMA

    lines = [ln for ln in TRANSITION_PROMPT.splitlines() if ln.startswith('{"length_bars"')]
    assert len(lines) == 11  # 8 from Phase 2, plus Phase 3's three effect examples
    for ln in lines:
        assert "filter_resonance" in json.loads(ln)
    prop = TRANSITION_SCHEMA["properties"]["filter_resonance"]
    assert (prop["minimum"], prop["maximum"]) == tr.FILTER_RESONANCE_RANGE


# --- key lock is audible in a render -------------------------------------------


@pytest.fixture(scope="module")
def keylock_renders(tmp_path_factory):
    folder = tmp_path_factory.mktemp("keylock_tracks")
    synth.render_track(folder / "a.wav", bpm=120.0, root="A", minor=True, bars=24, seed=3)
    synth.render_track(folder / "b.wav", bpm=127.0, root="E", minor=True, bars=24, seed=4)
    a = analysis.analyze_file(folder / "a.wav")
    b = analysis.analyze_file(folder / "b.wav")
    out = tmp_path_factory.mktemp("keylock_out")
    on = render_transition(a, b, out / "on", style="cut", key_lock=True)
    off = render_transition(a, b, out / "off", style="cut", key_lock=False)
    return a, b, on, off


def test_key_lock_is_audible_in_the_rendered_wavs(keylock_renders):
    import soundfile as sf

    a, b, on, off = keylock_renders
    assert on.b_stretched and not off.b_stretched
    assert on.wav_path.exists() and off.wav_path.exists()

    # The last three bars are deck B alone.
    n = int(3 * 4 * 60.0 / a.bpm * SAMPLE_RATE)
    on_tail = sf.read(str(on.wav_path), dtype="float32")[0][-n:, 0]
    off_tail = sf.read(str(off.wav_path), dtype="float32")[0][-n:, 0]

    expected = 1200.0 * np.log2(on.rate_b)
    measured = shift_cents(on_tail, off_tail)
    assert abs(expected) > 60.0, "the pair should need an audible tempo change"
    assert abs(measured - expected) <= 25.0, (
        f"resampled tail should sit {expected:+.0f} cents from the key-locked "
        f"one, measured {measured:+.0f}"
    )

    # And key lock keeps deck B at its own pitch.
    source = sf.read(str(b.path), dtype="float32")[0][:, 0]
    own = source[len(source) // 2: len(source) // 2 + n]
    assert abs(shift_cents(own, on_tail)) <= 15.0
