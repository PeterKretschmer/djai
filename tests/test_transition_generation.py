"""SPEC §4 (Phase 3.1): contextual, seeded transition generation.

THREADING CONTEXT: main thread (pytest). Renders are headless
(`render.render_preview`), so no audio device is opened.

What is measured here, and how:

* determinism -- same context and seed, same parameters;
* every generated transition passes the supervisor's parameter check;
* diversity over a simulated 90-minute set -- no shape inside
  `DIVERSITY_WINDOW` of itself;
* 200 random pairs rendered artifact-free -- finite, never over full scale,
  and no discontinuity (second difference) on pure-tone decks, where any
  click is obvious;
* invisible vs showy -- a PROXY for "audibly distinct": spectral flux inside
  the transition relative to the bars before it, on synthetic music.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from djai import cli, config, render
from djai import transition as tr
from djai.deck import CHANNELS, SAMPLE_RATE, LoadedTrack
from djai.supervisor import Supervisor
from tests.test_integration import make_analysis

CAMELOT = [f"{n}{m}" for n in range(1, 13) for m in "AB"]

#: Same bar the Phase 2.3 fuzz uses: a -40 dBFS step reads as 1e-2.
CLICK_D2: float = 5e-3


def _supervisor() -> Supervisor:
    return Supervisor(engine=None, scheduler=None, crate=[])


def _accept(sup, track_b=None, track_a=None):
    def accept(params):
        ok, why = sup.validate_transition_params(params.to_schema(), track_b, track_a)
        return ok is not None, why
    return accept


def _random_context(rng: random.Random) -> tr.PairContext:
    return tr.PairContext(
        base_style=rng.choice(("bass_swap", "filter_sweep", "echo_out")),
        bpm_residual=rng.choice((0.0, 0.0, 0.01, 0.04)),
        energy_delta=rng.uniform(-0.3, 0.3),
        keys_compatible=rng.random() < 0.6,
        mix_quality=rng.choice((None, rng.random())),
        outro_falling=rng.random() < 0.2,
        drops=False,
    )


def test_generation_is_deterministic_in_its_seed():
    ctx = tr.PairContext(energy_delta=0.2, keys_compatible=False, mix_quality=0.7)
    for mode in tr.MODES:
        a = tr.generate_transition(ctx, mode, 1234, ["blend/s_curve/mid/dry"])
        b = tr.generate_transition(ctx, mode, 1234, ["blend/s_curve/mid/dry"])
        assert a == b
    seeds = {tr.generate_transition(ctx, "invisible", s).params for s in range(20)}
    assert len(seeds) > 3, "different seeds should reach different points"


def test_context_steers_the_family():
    """Hand-set weights, checked for direction only."""
    rising = tr.PairContext(energy_delta=0.3)
    falling = tr.PairContext(energy_delta=-0.3, outro_falling=True)
    count = lambda ctx, fams: sum(  # noqa: E731
        tr.generate_transition(ctx, "showy", s).params.name[4:] in fams for s in range(40)
    )
    assert count(rising, {"riser", "loop_roll", "beat_repeat", "backspin"}) > 30
    assert count(falling, {"echo_out", "reverb_out", "filter_echo", "brake"}) > 30
    clash = tr.PairContext(keys_compatible=False)
    fams = [tr.generate_transition(clash, "invisible", s).params.name for s in range(40)]
    assert fams.count("gen_filter_blend") > 30, "a key clash reaches for the filter"


def test_every_generated_transition_passes_the_supervisor():
    sup = _supervisor()
    rng = random.Random(7)
    for i in range(500):
        ctx = _random_context(rng)
        g = tr.generate_transition(ctx, rng.choice(tr.MODES + ("auto",)), i)
        assert g is not None
        ok, why = sup.validate_transition_params(g.params.to_schema())
        assert ok is not None, (g.params, why)


def test_a_refused_candidate_is_never_returned():
    ctx = tr.PairContext()
    g = tr.generate_transition(ctx, "showy", 3, accept=lambda p: (p.echo_bars == 0, ""))
    assert g is not None and g.params.echo_bars == 0
    assert tr.generate_transition(ctx, "showy", 3, accept=lambda p: (False, "no")) is None


def test_no_shape_repeats_inside_the_window_over_a_90_minute_set(capsys):
    """~3.5 minutes a track: 26 transitions in 90 minutes, modes as auto picks them."""
    rng = random.Random(90)
    sup = _supervisor()
    history: list[str] = []
    minutes = 0.0
    while minutes < 90.0:
        ctx = _random_context(rng)
        mode = "showy" if ctx.energy_delta > 0.05 else "invisible"
        g = tr.generate_transition(ctx, mode, len(history), history, _accept(sup))
        history.append(g.shape)
        minutes += rng.uniform(3.0, 4.0)
    for i, shape in enumerate(history):
        window = history[max(0, i - tr.DIVERSITY_WINDOW):i]
        assert shape not in window, f"{shape} repeated at transition {i}"
    worst = max(history.count(s) for s in history) / len(history)
    with capsys.disabled():
        print(f"\n    90-min set: {len(history)} transitions, {len(set(history))} shapes, "
              f"most common share {worst:.0%}")
    assert worst <= 0.2


def test_the_session_records_shapes_and_does_not_repeat_one(tone_session_gen):
    s = tone_session_gen
    shapes = []
    for _ in range(3):
        assert s.cue_next(origin="test")
        _drive(s, 4)
        assert s.arm_transition(origin="test") is not None, s._report_no_cue
        shapes.append(s.shape_history[-1])
        s.abort_armed_transition()
        s._cued = None
        s._transition_armed = False
    assert len(set(shapes)) == 3, shapes
    logged = [e for e in _events(s) if e["event"] == "transition_generated"]
    assert len(logged) == 3 and all("seed" in e for e in logged)


# --- rendering -------------------------------------------------------------------


def _tone(freq: int, seconds: float = 200.0, level: float = 0.2) -> np.ndarray:
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    one = level * np.sin(2 * np.pi * freq * t)
    sig = np.tile(one, int(seconds) + 1)[: int(seconds * SAMPLE_RATE)]
    return np.stack([sig, sig], axis=1).astype(np.float32)


def _smooth_riser(n_frames: int, seed: int = 0) -> np.ndarray:
    """A low tone for the riser, as in test_actions: noise would hide a click."""
    t = np.arange(max(1, int(n_frames))) / SAMPLE_RATE
    one = np.sin(2 * np.pi * 70.0 * t)
    return np.stack([one, one], axis=1).astype(np.float32)


def _render_pair(ta, tb, audio_a, audio_b, params):
    la = LoadedTrack(analysis=ta, audio=audio_a)
    lb = LoadedTrack(analysis=tb, audio=audio_b)
    bar = 4 * 60.0 / ta.bpm * SAMPLE_RATE
    at = ta.first_downbeat * SAMPLE_RATE + 48 * bar
    return render.render_preview(ta, tb, la, lb, params, at, tb.mix_in * SAMPLE_RATE)


def test_200_random_pairs_render_artifact_free(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("djai.engine.render_riser", _smooth_riser)
    sup = _supervisor()
    rng = random.Random(200)
    worst, worst_at, peak = 0.0, None, 0.0
    families = set()
    for i in range(200):
        bpm_a = rng.uniform(118.0, 130.0)
        ta = make_analysis(f"a{i}", bpm_a, rng.choice(CAMELOT), rng.uniform(0.2, 0.9),
                           tmp_path / f"a{i}.wav")
        tb = make_analysis(f"b{i}", bpm_a * rng.uniform(0.95, 1.05), rng.choice(CAMELOT),
                           rng.uniform(0.2, 0.9), tmp_path / f"b{i}.wav")
        ctx = _random_context(rng)
        mode = tr.MODES[i % 2]
        g = tr.generate_transition(ctx, mode, i, (), _accept(sup, tb, ta))
        families.add(g.params.name)
        r = _render_pair(ta, tb, _tone(rng.randint(60, 180)), _tone(rng.randint(60, 180)),
                         g.params)
        mix = r.mix[r.blocksize:, 0].astype(np.float64)  # block 0 starts from silence
        assert np.isfinite(mix).all(), g.params
        d2 = float(np.abs(mix[2:] - 2 * mix[1:-1] + mix[:-2]).max())
        if d2 > worst:
            worst, worst_at = d2, (i, g.params.name)
        peak = max(peak, float(np.abs(r.mix).max()))
    with capsys.disabled():
        print(f"\n    200 pairs, {len(families)} families: worst second difference "
              f"{worst:.2e} at {worst_at}; peak {peak:.3f}")
    assert peak <= 1.0
    assert worst < CLICK_D2, f"discontinuity at pair {worst_at}"


def _logspec(mono: np.ndarray, frame: int = 2048) -> np.ndarray:
    n = mono.size // frame
    spec = np.fft.rfft(mono[: n * frame].reshape(n, frame) * np.hanning(frame), axis=1)
    return np.log1p(100.0 * np.abs(spec))


def test_invisible_and_showy_are_distinct(tmp_path, capsys):
    """PROXY for "audibly distinct": departure from a plain crossfade.

    Each generated transition is rendered next to a plain equal-power blend of
    the same pair and length, and the mean log-spectral difference over the
    window is taken. Invisible should sit close to a clean blend; showy should
    not. Measured on synthetic music (tests/synth.py), not on real tracks.
    """
    import soundfile as sf

    from tests import synth

    tracks = []
    for k, (root, seed) in enumerate((("A", 1), ("C", 2), ("E", 3))):
        path = synth.render_track(tmp_path / f"s{k}.wav", 124.0, root=root, bars=96, seed=seed)
        audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
        tracks.append((make_analysis(f"s{k}", 124.0, "8A", 0.5, path), audio))
    sup = _supervisor()
    rng = random.Random(5)
    dist = {m: [] for m in tr.MODES}
    for i in range(12):
        (ta, aa), (tb, ab) = rng.sample(tracks, 2)
        ctx = _random_context(rng)
        for mode in tr.MODES:
            g = tr.generate_transition(ctx, mode, i, (), _accept(sup, tb, ta))
            plain = tr.TransitionParams(
                length_bars=g.params.length_bars, curve="equal_power",
                low_swap_bar=int(g.params.length_bars // 2), name="plain",
            )
            r = _render_pair(ta, tb, aa, ab, g.params)
            q = _render_pair(ta, tb, aa, ab, plain)
            t0 = r.transition_start_frame // 2048
            t1 = (r.transition_start_frame + r.transition_frames) // 2048 + 2
            got = _logspec(r.mix.mean(axis=1).astype(np.float64))[t0:t1]
            ref = _logspec(q.mix.mean(axis=1).astype(np.float64))[t0:t1]
            dist[mode].append(float(np.abs(got - ref).mean()))
    inv, show = np.array(dist["invisible"]), np.array(dist["showy"])
    with capsys.disabled():
        print(f"\n    departure from a plain crossfade (proxy): invisible median "
              f"{np.median(inv):.3f} [{inv.min():.3f}-{inv.max():.3f}], showy median "
              f"{np.median(show):.3f} [{show.min():.3f}-{show.max():.3f}]")
    assert inv.max() < show.min(), "the two modes' ranges must not overlap"


# --- a session on tones -------------------------------------------------------------


def _drive(s, blocks: int) -> None:
    buf = np.zeros((1024, CHANNELS), dtype=np.float32)
    for _ in range(blocks):
        s.scheduler.tick(s.engine.frames_played + 1024)
        s.engine.callback(buf, 1024, None, None)


def _events(s):
    import json

    return [json.loads(line) for line in s.session_log.path.read_text().splitlines()]


@pytest.fixture
def tone_session_gen(tmp_path, monkeypatch):
    freqs = {"alpha": 105, "beta": 147, "gamma": 126, "delta": 84, "eps": 98}
    crate = []
    for i, tid in enumerate(freqs):
        p = tmp_path / f"{tid}.wav"
        p.write_bytes(b"x")
        crate.append(make_analysis(tid, 124.0 + 0.5 * (i % 2), ("8A", "9A", "8B", "7A", "8A")[i],
                                   0.3 + 0.1 * i, p))

    monkeypatch.setattr(cli, "load_track",
                        lambda a: LoadedTrack(analysis=a, audio=_tone(freqs[a.track_id])))
    monkeypatch.setattr(config, "TIME_STRETCH_ENABLED", False)
    s = cli.Session(crate, log_dir=tmp_path / "logs")
    s.start_first_track()
    _drive(s, 30)
    yield s
    s.shutdown()
