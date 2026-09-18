"""Phase 3: musical intelligence, measured.

THREADING CONTEXT: main thread (pytest). Analysis runs for real on synthetic
tracks whose structure and vocals are known by construction; the engine is
driven through the offline renderer.

Section labels are checked against ten tracks annotated by construction
(tests/synth.py:STRUCTURED_LAYOUTS). Real-track annotations are not available
to this suite; see BLOCKERS.md.
"""

from __future__ import annotations

import csv
import dataclasses
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from djai import analysis as an
from djai import cli, selector
from djai import transition as tr
from djai.commands import IMMEDIATE, StartTransition
from djai.deck import SAMPLE_RATE, Deck, LoadedTrack
from djai.render import render_transition
from tests import synth
from tests.test_designed_transitions import GOOD
from tests.test_integration import log_events
from tests.test_integration import session as _session_fixture

session = _session_fixture


# --- structure and vocals, against ground truth -------------------------------


@pytest.fixture(scope="module")
def annotated(tmp_path_factory):
    folder = tmp_path_factory.mktemp("structured")
    out = []
    for i, ((layout, vocal), bpm) in enumerate(
        zip(synth.STRUCTURED_LAYOUTS, synth.STRUCTURED_BPMS)
    ):
        path = folder / f"struct_{i:02d}.wav"
        labels, vocals = synth.structured_track(
            path, bpm, layout, vocal, seed=i, root_midi=45 + i % 5
        )
        out.append((layout, labels, vocals, an.analyze_file(path)))
    return out


def _per_bar(ta: an.TrackAnalysis, n: int) -> list[str | None]:
    labels: list[str | None] = [None] * n
    for s in ta.sections:
        for bar in range(max(0, s["start_bar"]), min(n, s["end_bar"])):
            labels[bar] = s["label"]
    return labels


def test_section_labels_match_ten_annotated_tracks(annotated, capsys):
    rows, recalls = [], []
    for i, (layout, truth, _vocals, ta) in enumerate(annotated):
        predicted = _per_bar(ta, len(truth))
        accuracy = float(np.mean([p == t for p, t in zip(predicted, truth)]))
        true_bounds = [k for k in range(1, len(truth)) if truth[k] != truth[k - 1]]
        found = [s["start_bar"] for s in ta.sections[1:]]
        hits = sum(any(abs(t - f) <= 2 for f in found) for t in true_bounds)
        recalls.append((hits, len(true_bounds)))
        rows.append((i, accuracy, hits, len(true_bounds)))
        assert accuracy >= 0.85, (
            f"track {i}: {accuracy:.0%} of bars labelled right\n"
            f"  truth {layout}\n  found {[(s['label'], s['start_bar'], s['end_bar']) for s in ta.sections]}"
        )
    mean = float(np.mean([r[1] for r in rows]))
    hit, total = sum(h for h, _ in recalls), sum(t for _, t in recalls)
    with capsys.disabled():
        print(f"\n    sections: mean bar accuracy {mean:.1%}, boundaries {hit}/{total} within 2 bars")
        for i, acc, h, t in rows:
            print(f"      track {i}: {acc:.1%} bars, boundaries {h}/{t}")
    assert mean >= 0.95
    assert hit / total >= 0.9


def test_vocal_bars_match_the_annotated_vocals(annotated, capsys):
    tp = fp = fn = 0
    for _layout, _truth, vocals, ta in annotated:
        flagged = [False] * len(vocals)
        for a, b in ta.vocal_bars:
            for bar in range(max(0, a), min(len(vocals), b)):
                flagged[bar] = True
        tp += sum(v and f for v, f in zip(vocals, flagged))
        fp += sum(f and not v for v, f in zip(vocals, flagged))
        fn += sum(v and not f for v, f in zip(vocals, flagged))
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    with capsys.disabled():
        print(f"\n    vocals: precision {precision:.1%}, recall {recall:.1%} ({tp} tp, {fp} fp, {fn} fn)")
    assert precision >= 0.9
    assert recall >= 0.85


def test_v8_fields_are_measured_and_round_trip(annotated, tmp_path):
    ta = annotated[0][3]
    assert ta.analysis_version == 8
    assert ta.sections and ta.intensity is not None and ta.vocal_fraction is not None
    an.write_sidecar(ta, tmp_path)
    loaded = an.load_cached(ta.track_id, tmp_path)
    assert loaded.sections == ta.sections
    assert loaded.vocal_bars == ta.vocal_bars
    assert loaded.intensity == pytest.approx(ta.intensity)


def test_a_v7_entry_upgrades_in_place_with_structure_empty(annotated, tmp_path):
    ta = dataclasses.replace(annotated[0][3], sections=[], vocal_bars=[],
                             vocal_fraction=None, intensity=None,
                             intensity_features={}, analysis_version=7)
    path = an.write_sidecar(ta, tmp_path)
    data, status = an.read_sidecar(path)
    assert status == "upgraded"
    assert data["analysis_version"] == an.ANALYSIS_VERSION
    assert data["intensity"] is None, "measured later by analyze, never invented"


# --- intensity ------------------------------------------------------------------


def _write(path: Path, mono: np.ndarray) -> Path:
    sf.write(str(path), np.stack([mono, mono], axis=1).astype(np.float32), SAMPLE_RATE)
    return path


def test_intensity_ranks_a_quiet_banger_above_a_loud_ambient_track(tmp_path, capsys):
    rng = np.random.default_rng(7)
    n = SAMPLE_RATE * 30
    drums = np.zeros(n)
    spb = 60 / 132
    kick = synth._kick(SAMPLE_RATE)
    for beat in range(int(30 / spb)):
        i = int(beat * spb * SAMPLE_RATE)
        synth._add(drums, kick, i)
        for d in range(4):
            synth._add(drums, synth._burst(rng, 0.05, 0.01, 0.3, SAMPLE_RATE, bright=True),
                       i + int(d * spb / 4 * SAMPLE_RATE))
        if beat % 2:
            synth._add(drums, synth._burst(rng, 0.16, 0.05, 0.6, SAMPLE_RATE), i)
    drums *= 0.03 / np.max(np.abs(drums))          # -30 dBFS peak
    t = np.arange(n) / SAMPLE_RATE
    pad = sum(np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * t + k))
              for k, f in enumerate((220, 277.2, 329.6, 440)))
    pad = pad + 0.02 * rng.standard_normal(n)
    pad *= 0.89 / np.max(np.abs(pad))              # -1 dBFS peak

    quiet = an.analyze_file(_write(tmp_path / "quiet_banger.wav", drums))
    loud = an.analyze_file(_write(tmp_path / "loud_ambient.wav", pad))
    with capsys.disabled():
        print(f"\n    intensity: quiet banger {quiet.intensity:.3f} {quiet.intensity_features}, "
              f"loud ambient {loud.intensity:.3f} {loud.intensity_features}")
    assert quiet.intensity > loud.intensity + 0.3


# --- the selector's set arc --------------------------------------------------------


def _crate(n: int = 24) -> list[an.TrackAnalysis]:
    from tests.test_selector_intent import track as make

    rng = np.random.default_rng(3)
    keys = ["8A", "9A", "8B", "7A", "9B", "10A"]
    crate = []
    for i in range(n):
        t = make(f"t{i:02d}", float(125 + rng.uniform(-2, 2)))
        t.camelot = keys[i % len(keys)]
        t.intensity = float(i) / (n - 1)
        t.artist = f"artist {i}"
        t.vocal_fraction = 0.0
        crate.append(t)
    return crate


def _sequence(crate, phase: str, picks: int = 8) -> list[str]:
    current, played, history, out = crate[len(crate) // 2], set(), [], []
    played.add(current.track_id)
    history.append(current)
    for _ in range(picks):
        c = selector.select_next(crate, current, played, set_phase=phase, history=history)
        out.append(c.track.track_id)
        played.add(c.track.track_id)
        history.append(c.track)
        current = c.track
    return out


def test_warmup_and_peak_choose_different_sequences():
    crate = _crate()
    by_id = {t.track_id: t for t in crate}
    warmup = _sequence(crate, "warmup")
    peak = _sequence(crate, "peak")
    assert warmup != peak
    mean_warm = np.mean([by_id[i].intensity for i in warmup])
    mean_peak = np.mean([by_id[i].intensity for i in peak])
    assert mean_warm < 0.45 < 0.6 < mean_peak, (mean_warm, mean_peak)


def test_build_rises_and_cooldown_falls():
    crate = _crate()
    by_id = {t.track_id: t for t in crate}
    start = crate[len(crate) // 2].intensity
    build = [by_id[i].intensity for i in _sequence(crate, "build", 5)]
    cool = [by_id[i].intensity for i in _sequence(crate, "cooldown", 5)]
    assert build[-1] > start and cool[-1] < start


def _pair(**changes):
    crate = _crate(6)
    current = crate[0]
    a = dataclasses.replace(crate[1], track_id="clean", camelot="8A", artist="someone",
                            intensity=0.5)
    b = dataclasses.replace(a, track_id="penalised", **changes)
    return current, [current, a, b]


def test_the_same_artist_within_five_tracks_is_penalised():
    current, crate = _pair(artist="Repeat")
    history = [dataclasses.replace(current, artist="Repeat")]
    ranked = selector.rank_candidates(crate, current, history=history)
    assert [c.track.track_id for c in ranked][0] == "clean"
    assert any("artist" in p for c in ranked for p in c.penalties)


def test_the_same_key_within_three_tracks_is_penalised():
    current, crate = _pair(camelot="5A")
    history = [dataclasses.replace(current, camelot="5A", track_id="old")]
    ranked = selector.rank_candidates(crate, current, history=history)
    penalised = next(c for c in ranked if c.track.track_id == "penalised")
    assert any("key" in p for p in penalised.penalties)


def test_back_to_back_vocal_led_tracks_are_penalised():
    current, crate = _pair(vocal_fraction=0.8)
    current = dataclasses.replace(current, vocal_fraction=0.9)
    ranked = selector.rank_candidates(crate, current, history=[current])
    penalised = next(c for c in ranked if c.track.track_id == "penalised")
    clean = next(c for c in ranked if c.track.track_id == "clean")
    assert "vocal-led back to back" in penalised.penalties
    assert penalised.score < clean.score


# --- the supervisor: vocal clashes and the start floor ----------------------------


def test_a_vocal_clash_is_rejected_and_logged(session):
    a = dataclasses.replace(session.crate[0], vocal_bars=[[40, 80]])
    b = dataclasses.replace(session.crate[1], vocal_bars=[[0, 40]])
    env = tr.build_envelope("bass_swap", SAMPLE_RATE * 40, 2048, 125.0)
    reason = session.supervisor.check_vocal_clash(
        a, b, start_bar_a=48, entry_bar_b=0, bars=24, envelope=env, style="bass_swap"
    )
    assert reason is not None and "vocal clash" in reason
    events = [e for e in log_events_sup(session) if e["event"] == "vocal_clash_rejected"]
    assert events and events[-1]["clash_bars"] >= 1.0


def log_events_sup(session):
    log = session.supervisor.log
    log._file.flush()
    return [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_no_clash_while_the_incoming_low_band_is_still_cut(session):
    """The brief's exception: B's vocal over A's is allowed while B's bass is out."""
    a = dataclasses.replace(session.crate[0], vocal_bars=[[40, 80]])
    # Deck B sings only over the first part of the window, where a bass swap
    # keeps its low band cut.
    env = tr.build_envelope("bass_swap", SAMPLE_RATE * 40, 2048, 125.0)
    rows = len(env) - 1
    cut_until = next(i for i in range(rows) if env[i, tr.TO_LOW] >= 0.5) / rows * 24
    b = dataclasses.replace(session.crate[1], vocal_bars=[[0, int(cut_until) - 1]])
    assert session.supervisor.check_vocal_clash(
        a, b, start_bar_a=48, entry_bar_b=0, bars=24, envelope=env
    ) is None


def test_vocals_on_only_one_deck_never_clash(session):
    a = dataclasses.replace(session.crate[0], vocal_bars=[[0, 400]])
    b = dataclasses.replace(session.crate[1], vocal_bars=[])
    env = tr.build_envelope("bass_swap", SAMPLE_RATE * 40, 2048, 125.0)
    assert session.supervisor.check_vocal_clash(a, b, 48, 0, 24, env) is None


def _start_at_fraction(session, fraction: float, drop_aligned: bool):
    engine = session.engine
    track = engine.deck("a").track
    engine.deck("b").attach(track, 0)
    engine.deck("a").position = fraction * track.analysis.duration_s * SAMPLE_RATE
    rejection = session.supervisor.validate(StartTransition(
        from_deck="a", to_deck="b", total_frames=SAMPLE_RATE * 10,
        execute_at=IMMEDIATE, origin="autopilot", drop_aligned=drop_aligned,
    ))
    return None if rejection is None else rejection.reason


def test_the_start_guard_is_relaxed_for_drop_aligned_transitions_only(session):
    assert "Minimum is 40%" in (_start_at_fraction(session, 0.30, False) or "")
    assert _start_at_fraction(session, 0.30, True) is None
    assert "Minimum is 25%" in (_start_at_fraction(session, 0.20, True) or "")


# --- placement from structure -------------------------------------------------------


def _deck_with(ta: an.TrackAnalysis, position: float = 0.0) -> Deck:
    d = Deck("a")
    d.attach(LoadedTrack(analysis=ta, audio=np.zeros((16, 2), dtype=np.float32)))
    d.position = position
    return d


def test_outro_over_intro_is_the_default_placement(session):
    base_a, base_b = session.crate[0], session.crate[1]
    total_bars = int(base_a.duration_s / (4 * 60 / base_a.bpm))
    outro_start = int(total_bars * 0.8)
    a = dataclasses.replace(base_a, sections=[
        {"label": "drop", "start_bar": 0, "end_bar": outro_start},
        {"label": "outro", "start_bar": outro_start, "end_bar": total_bars},
    ])
    b = dataclasses.replace(base_b, sections=[
        {"label": "intro", "start_bar": 0, "end_bar": 16},
        {"label": "drop", "start_bar": 16, "end_bar": 64},
    ])
    plan = tr_plan(_deck_with(a), b, 8.0)
    assert plan.reason and plan.reason.startswith("outro over intro")
    from djai import phrase

    assert plan.start_frame == pytest.approx(phrase.frame_at_bar(a, outro_start), abs=1)
    assert plan.entry_frame_b == int(round(max(0.0, phrase.frame_at_bar(b, 0))))


def tr_plan(deck, track_b, bars):
    from djai import phrase

    return phrase.plan_transition(deck, track_b, bars)


def test_placement_falls_back_to_mix_out_without_sections(session):
    a = dataclasses.replace(session.crate[0], sections=[])
    plan = tr_plan(_deck_with(a), session.crate[1], 16.0)
    assert not (plan.reason or "").startswith("outro over intro")


def test_drop_to_drop_engages_when_both_tracks_have_labelled_drops(session):
    live = session.engine.deck("a")
    base = live.track.analysis
    bar_s = 4 * 60 / base.bpm
    total_bars = int(base.duration_s / bar_s)
    drop_bar = int(total_bars * 0.6)
    a = dataclasses.replace(base, hot_cues=[], sections=[
        {"label": "build", "start_bar": drop_bar - 8, "end_bar": drop_bar},
        {"label": "drop", "start_bar": drop_bar, "end_bar": drop_bar + 16},
    ])
    live.track = LoadedTrack(analysis=a, audio=live.track.audio)
    live.position = 0.0
    b_base = session.crate[1]
    b = dataclasses.replace(b_base, hot_cues=[], sections=[
        {"label": "intro", "start_bar": 0, "end_bar": 32},
        {"label": "drop", "start_bar": 32, "end_bar": 64},
    ], mix_out=b_base.duration_s)
    params = tr.preset_params("bass_swap")
    plan = session._plan_drop_aligned(live, b, params, 16.0)
    assert plan is not None, "both tracks have a drop section"
    assert plan.reason.startswith("drops aligned")
    from djai import phrase

    assert phrase.bar_at_frame(a, plan.start_frame) + 16 == pytest.approx(drop_bar, abs=0.01)
    assert phrase.bar_at_frame(b, plan.entry_frame_b) + 16 == pytest.approx(32, abs=0.01)


# --- the schema ------------------------------------------------------------------------


@pytest.mark.parametrize("field, bad", [
    ("reverb_bars", 5), ("reverb_bars", 1.5), ("brake_beats", 3), ("brake_beats", True),
    ("riser_bars", 3), ("riser_bars", 9), ("vocal_aware", "yes"), ("vocal_aware", None),
])
def test_effect_fields_are_validated(session, field, bad):
    params, why = session.supervisor.validate_transition_params(
        dict(GOOD, **{field: bad}), session.crate[1]
    )
    assert params is None and field in why


def test_a_brake_and_a_backspin_together_are_refused(session):
    params, why = session.supervisor.validate_transition_params(
        dict(GOOD, brake_beats=2, backspin_bars=1), session.crate[1]
    )
    assert params is None and "brake_beats" in why


def test_valid_effects_are_carried_into_the_envelope(session):
    raw = dict(GOOD, reverb_bars=3, riser_bars=6, echo_bars=2, vocal_aware=False)
    params, why = session.supervisor.validate_transition_params(raw, session.crate[1])
    assert params is not None, why
    assert (params.reverb_bars, params.riser_bars, params.vocal_aware) == (3, 6, False)
    env = tr.build_envelope_from_params(params, SAMPLE_RATE * 16, 512, 125.0)
    assert env[:-1, tr.REVERB_SEND].max() > 0.3
    assert env[:-1, tr.RISER_GAIN].max() > 0.1
    assert env[-1, tr.REVERB_SEND] == 0.0 and env[-1, tr.RISER_GAIN] == 0.0


def test_every_prompt_example_passes_the_supervisor(session):
    from djai.intent import TRANSITION_PROMPT

    lines = [ln for ln in TRANSITION_PROMPT.splitlines() if ln.startswith('{"length_bars"')]
    assert len(lines) == 11
    effect_examples = 0
    for ln in lines:
        raw = json.loads(ln)
        if raw["entry_point"] != "mix_in" or raw["align_mode"] == "drop":
            continue  # these need cues on a real track pair
        params, why = session.supervisor.validate_transition_params(raw, session.crate[1])
        assert params is not None, f"{ln}\n{why}"
        effect_examples += bool(raw["reverb_bars"] or raw["brake_beats"] or raw["riser_bars"])
    assert effect_examples == 3


# --- effects render offline with continuous automation -------------------------------


@pytest.fixture(scope="module")
def effect_renders(tmp_path_factory):
    folder = tmp_path_factory.mktemp("effect_tracks")
    synth.render_track(folder / "a.wav", bpm=126.0, root="A", minor=True, bars=40, seed=11)
    synth.render_track(folder / "b.wav", bpm=126.0, root="E", minor=True, bars=40, seed=12)
    a = an.analyze_file(folder / "a.wav")
    b = an.analyze_file(folder / "b.wav")
    out = tmp_path_factory.mktemp("effect_renders")
    return {
        style: render_transition(a, b, out / style, style=style, key_lock=False)
        for style in ("reverb_out", "brake", "noise_riser", "filter_echo")
    }


def _rows(result):
    with gzip.open(result.envelope_path, "rt", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


#: Largest step allowed between consecutive blocks, per automated column. The
#: riser's is its whole level: it is meant to stop dead on the hand-over, and
#: the engine ramps that final step across one block rather than jumping.
MAX_STEP = {"a_gain": 0.2, "a_rate": 0.1, "a_filter": 0.1, "reverb_send": 0.2,
            "riser_gain": tr.RISER_LEVEL + 1e-6}


@pytest.mark.parametrize("style, column, engaged", [
    ("reverb_out", "reverb_send", lambda v: max(v) > 0.5),
    ("brake", "a_rate", lambda v: min(v) < 0.1),
    ("noise_riser", "riser_gain", lambda v: max(v) > 0.1),
    ("filter_echo", "a_filter", lambda v: max(v) > 0.5),
])
def test_each_effect_renders_offline_with_continuous_automation(effect_renders, style, column, engaged):
    result = effect_renders[style]
    rows = [r for r in _rows(result) if r["in_transition"] == "1"]
    assert rows
    gains = [float(r["a_gain"]) for r in rows]
    for name in ("a_gain", column):
        values = [float(r[name]) for r in rows]
        steps = np.abs(np.diff(values))
        for i, step in enumerate(steps):
            if step <= MAX_STEP[name]:
                continue
            # The one allowed jump: deck A's knob and rate are handed back to
            # neutral on the final block, when A is already silent. Anything
            # audible, and any jump earlier than the hand-over, fails.
            handover = i == len(steps) - 1 and name in ("a_rate", "a_filter")
            assert handover and gains[i] <= 0.01, (
                f"{style}: {name} jumps {step:.3f} at block {i} of {len(steps)} "
                f"with deck A at gain {gains[i]:.3f}"
            )
    assert engaged([float(r[column]) for r in rows]), f"{style} never engaged"
    assert np.all(np.isfinite(result.mix))
    assert float(np.max(np.abs(result.mix))) <= 0.99


def test_the_reverb_tail_rings_after_the_send_closes(effect_renders):
    result = effect_renders["reverb_out"]
    rows = _rows(result)
    last_send = max(i for i, r in enumerate(rows) if float(r["reverb_send"]) > 0)
    # Deck A is silent after the transition; B is there, so compare against the
    # same stretch rendered without reverb: filter_echo has no reverb.
    block = result.blocksize
    start = (last_send + 2) * block
    tail = result.mix[start:start + SAMPLE_RATE // 2]
    dry = effect_renders["filter_echo"].mix[start:start + SAMPLE_RATE // 2]
    assert tail.size and float(np.sqrt(np.mean((tail - dry[: tail.shape[0]]) ** 2))) > 1e-3


# --- set phase: REPL, UI, intent ------------------------------------------------------


def test_set_phase_from_the_repl(session):
    assert cli.handle_override(session, "phase peak").startswith("Set phase: peak")
    assert session.set_phase == "peak"
    assert cli.handle_override(session, "phase loud").startswith("Rejected")
    assert session.set_phase == "peak"
    assert "Set phase is peak" in cli.handle_override(session, "phase")
    assert cli.handle_override(session, "phase auto").startswith("Set phase off")


def test_set_phase_from_the_ui(session):
    from djai.ui_server import UIServer

    ui = UIServer(session, intent_engine=None)
    ack = ui.handle_action({"type": "phase", "phase": "cooldown"})
    assert ack["ok"], ack
    state = ui.state()["transition"]
    assert state["set_phase"] == "cooldown"
    assert set(state["set_phases"]) == set(selector.SET_PHASES) | {"auto"}
    root = Path(__file__).resolve().parent.parent / "static"
    assert 'id="phase-pick"' in (root / "index.html").read_text(encoding="utf-8")
    assert 'type: "phase"' in (root / "app.js").read_text(encoding="utf-8")


def test_set_phase_from_the_intent_layer(session):
    from djai.intent import ACTIONS, RESPONSE_SCHEMA, SYSTEM_PROMPT, keyword_intent

    assert "set_phase" in ACTIONS and "set_phase" in SYSTEM_PROMPT
    assert RESPONSE_SCHEMA["properties"]["params"]["properties"]["phase"]["enum"] == list(
        selector.SET_PHASES
    )
    for text, phase in (("take it to peak time", "peak"), ("time to cool down", "cooldown"),
                        ("keep it a warm up", "warmup")):
        intent = keyword_intent(text)
        assert intent is not None and intent.action == "set_phase", text
        assert intent.params["phase"] == phase
        cli.apply_intent(session, intent)
        assert session.set_phase == phase


def test_the_effect_styles_are_named_to_the_model_and_the_keywords():
    from djai.intent import SYSTEM_PROMPT, keyword_intent

    for style, text in (("reverb_out", "wash it out in reverb"), ("brake", "brake the record"),
                        ("noise_riser", "throw a noise riser on it"),
                        ("filter_echo", "filter and echo it out")):
        assert style in SYSTEM_PROMPT
        intent = keyword_intent(text)
        assert intent is not None and intent.params.get("style") == style, text
