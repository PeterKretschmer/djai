"""Phase 1.2 / SPEC §7: the log is replayable, and the numbers are honest.

THREADING CONTEXT: main thread (pytest), plus a writer thread in the timing
test. No audio device is opened.
"""

from __future__ import annotations

import gzip
import json
import threading
import time

import numpy as np
import pytest

from djai import evaluation as ev
from djai.commands import LoadTrack
from djai.deck import SAMPLE_RATE
from djai.engine import Engine
from djai.supervisor import LOG_SCHEMA_VERSION, SessionLog
from tests.test_engine import drive, loud_track

BLOCK = 512


@pytest.fixture
def log(tmp_path):
    made = SessionLog(tmp_path)
    yield made
    made.close()


def snapshot_of(engine: Engine):
    return engine.features.latest()


@pytest.fixture
def engine():
    made = Engine(blocksize=BLOCK)
    made.submit(LoadTrack(deck="a", track=loud_track(200.0, bpm=128.0),
                          play=True, master=True, origin="t"))
    drive(made, 8)
    yield made
    made.stop()


# --- the log says what version it is ------------------------------------------


def test_a_log_declares_its_schema_version_on_the_first_line(log):
    first = json.loads(log.path.read_text(encoding="utf-8").splitlines()[0])
    assert first["event"] == "log_opened"
    assert first["schema_version"] == LOG_SCHEMA_VERSION


def test_a_log_from_the_future_is_refused_rather_than_misread(tmp_path):
    path = tmp_path / "future.jsonl"
    path.write_text(
        json.dumps({"event": "log_opened", "schema_version": 99}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ev.LogVersionError) as excinfo:
        ev.replay(path)
    assert "99" in str(excinfo.value)


def test_a_log_with_no_header_reads_as_version_one(tmp_path):
    """Logs written before the version existed still open."""
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps({"event": "track_started"}) + "\n", encoding="utf-8")
    assert ev.replay(path).schema_version == 1


# --- decisions carry what caused them -----------------------------------------


def test_a_decision_records_the_snapshot_it_was_taken_on(log, engine):
    did = log.decision("track_cued", snapshot_of(engine), "cue t128", deck="b")
    session = ev.replay(log.path)
    decision = session.decisions[0]
    assert decision.decision_id == did
    assert decision.kind == "track_cued"
    assert decision.action == "cue t128"
    assert decision.fields["deck"] == "b"
    assert decision.snapshot_value("master_bpm") == pytest.approx(128.0, abs=0.01)
    assert decision.snapshot_value("a", "playing") is True


def test_a_decision_without_audio_context_is_still_a_decision(log):
    """Before the first block there is no snapshot, and that is not an error."""
    log.decision("track_cued", None, "opening track")
    decision = ev.replay(log.path).decisions[0]
    assert decision.snapshot is None
    assert decision.snapshot_value("master_bpm") is None


def test_an_outcome_closes_the_window_over_its_decision(log, engine):
    did = log.decision("track_cued", snapshot_of(engine), "cue t128")
    drive(engine, 8)
    log.outcome(did, snapshot_of(engine), action="handed over")
    decision = ev.replay(log.path).decisions[0]
    assert decision.resolved
    assert decision.outcome["action"] == "handed over"


def test_an_unresolved_decision_is_visible_as_unresolved(log, engine):
    log.decision("track_cued", snapshot_of(engine), "cue t128")
    assert ev.replay(log.path).decisions[0].resolved is False


def test_decision_records_keep_their_own_event_name(log, engine):
    """Additive by design: existing readers must not have to learn a new one."""
    log.decision("track_cued", snapshot_of(engine), "cue")
    events = [r["event"] for r in ev.read_records(log.path)]
    assert "track_cued" in events


# --- replay -------------------------------------------------------------------


def test_a_session_replays_to_the_same_decision_sequence(log, engine):
    for i in range(5):
        drive(engine, 4)
        log.decision("track_cued", snapshot_of(engine), f"cue track {i}")
    first = ev.replay(log.path)
    second = ev.replay(log.path)
    assert ev.compare_sequences(first, second) == []
    assert [d.action for d in first.decisions] == [f"cue track {i}" for i in range(5)]


def test_a_divergent_sequence_is_reported_with_its_index(tmp_path, log, engine):
    log.decision("track_cued", snapshot_of(engine), "cue A")
    log.decision("track_cued", snapshot_of(engine), "cue B")
    original = ev.replay(log.path)

    other = SessionLog(tmp_path / "other")
    try:
        other.decision("track_cued", snapshot_of(engine), "cue A")
        other.decision("track_cued", snapshot_of(engine), "cue DIFFERENT")
    finally:
        other.close()
    diverged = ev.compare_sequences(original, ev.replay(other.path))
    assert len(diverged) == 1 and diverged[0].startswith("[1]")


def test_a_log_torn_by_a_kill_still_replays(log, engine):
    """A killed session leaves a half-written last line. That is the case the
    log exists for, so it must not be the case that breaks the reader."""
    log.decision("track_cued", snapshot_of(engine), "cue A")
    log.close()
    with log.path.open("ab") as fh:
        fh.write(b'{"event": "track_cued", "action": "cue ')  # killed mid-write
    session = ev.replay(log.path)
    assert len(session.decisions) == 1
    assert session.decisions[0].action == "cue A"


def test_a_rotated_gzipped_log_replays(tmp_path, log, engine):
    log.decision("track_cued", snapshot_of(engine), "cue A")
    log.close()
    packed = tmp_path / "rotated.jsonl.gz"
    with gzip.open(packed, "wb") as out:
        out.write(log.path.read_bytes())
    assert len(ev.replay(packed).decisions) == 1


# --- metrics ------------------------------------------------------------------


def test_proxy_metrics_are_labelled_as_proxies(log, engine):
    for i in range(4):
        drive(engine, 4)
        log.decision("track_cued", snapshot_of(engine), f"cue {i}")
    found = ev.metrics(ev.replay(log.path))
    assert found["decisions"].proxy is False
    assert found["tempo_variance"].proxy is True
    assert "(proxy)" in str(found["tempo_variance"])
    assert "(proxy)" not in str(found["decisions"])


def test_the_proxy_label_survives_a_baseline_diff(log, engine):
    for i in range(4):
        drive(engine, 4)
        log.decision("track_cued", snapshot_of(engine), f"cue {i}")
    found = ev.metrics(ev.replay(log.path))
    deltas = {d.name: d for d in ev.diff(found, found)}
    assert deltas["tempo_variance"].proxy is True
    assert "(proxy)" in str(deltas["tempo_variance"])
    assert deltas["decisions"].change == 0.0


def test_a_diff_reports_metrics_that_appeared_or_vanished():
    base = {"a": ev.Metric("a", 1.0, proxy=False)}
    now = {"b": ev.Metric("b", 2.0, proxy=True)}
    deltas = {d.name: d for d in ev.diff(base, now)}
    assert deltas["a"].current is None and "gone" in str(deltas["a"])
    assert deltas["b"].baseline is None and "new" in str(deltas["b"])


# --- the scenario suite --------------------------------------------------------


def test_every_scenario_the_brief_names_exists():
    names = {s.name for s in ev.SCENARIOS}
    assert names == {
        "clean_two_track", "multi_track_differing_bpm", "forced_energy_drop",
        "forced_analysis_failure", "rapid_cueing",
    }


def test_every_scenario_states_what_must_hold():
    for scenario in ev.SCENARIOS:
        assert scenario.expectations, f"{scenario.name} asserts nothing"
        assert scenario.bpms and scenario.minutes > 0
    assert ev.scenario("rapid_cueing").minutes == 10.0
    with pytest.raises(KeyError):
        ev.scenario("nope")


# --- the invariant: logging stays off the audio thread -------------------------


def test_logging_at_a_real_rate_does_not_change_callback_timing(tmp_path, engine):
    """SPEC §7's condition, at the rate a set actually logs.

    A real session logs a handful of decisions per track -- call it one every
    few seconds at the very most. This writes ten a second, which is already
    orders of magnitude faster than that, and measures what the callback
    notices.
    """
    buf = np.zeros((BLOCK, 2), dtype=np.float32)

    def timed(n: int) -> float:
        t0 = time.perf_counter()
        for _ in range(n):
            engine.callback(buf, BLOCK, None, None)
        return (time.perf_counter() - t0) / n

    quiet = timed(400)
    writing = SessionLog(tmp_path / "realistic")
    stop = threading.Event()

    def at_ten_per_second() -> None:
        i = 0
        while not stop.is_set():
            writing.decision("track_cued", engine.features.latest(), f"cue {i}")
            i += 1
            stop.wait(0.1)

    thread = threading.Thread(target=at_ten_per_second, daemon=True)
    thread.start()
    try:
        noisy = timed(400)
    finally:
        stop.set()
        thread.join(2.0)
        writing.close()

    cost_us = (noisy - quiet) * 1e6
    block_us = BLOCK / SAMPLE_RATE * 1e6
    print(f"\ncallback at a realistic log rate: {noisy * 1e6:.1f} us vs "
          f"{quiet * 1e6:.1f} us quiet ({cost_us:+.1f} us, "
          f"{100 * cost_us / block_us:+.2f}% of budget)")
    assert abs(cost_us) < 0.02 * block_us, (
        f"logging moved callback timing by {cost_us:+.1f} us at a real rate"
    )


def test_logging_hard_stays_inside_the_block_budget(tmp_path, engine):
    """The adversarial bound, stated as what it is.

    A writer thread hammers the log as fast as it can -- thousands of lines a
    second, which no set produces. The log flushes every line and the writer
    holds the GIL to do it, so this does cost the callback measurable time.
    What matters is that it stays a small fraction of the block budget even
    then: the callback is slowed, never starved.
    """
    buf = np.zeros((BLOCK, 2), dtype=np.float32)

    def timed(n: int) -> float:
        t0 = time.perf_counter()
        for _ in range(n):
            engine.callback(buf, BLOCK, None, None)
        return (time.perf_counter() - t0) / n

    quiet = timed(400)

    writing = SessionLog(tmp_path / "busy")
    stop = threading.Event()

    def hammer() -> None:
        i = 0
        while not stop.is_set():
            writing.decision("track_cued", engine.features.latest(), f"cue {i}")
            i += 1

    thread = threading.Thread(target=hammer, daemon=True)
    thread.start()
    try:
        noisy = timed(400)
    finally:
        stop.set()
        thread.join(2.0)
        writing.close()

    cost_us = (noisy - quiet) * 1e6
    block_us = BLOCK / SAMPLE_RATE * 1e6
    print(f"\ncallback with a log under load: {noisy * 1e6:.1f} us vs "
          f"{quiet * 1e6:.1f} us quiet ({cost_us:+.1f} us, "
          f"{100 * cost_us / block_us:+.2f}% of budget)")
    assert abs(cost_us) < 0.10 * block_us, (
        f"logging moved callback timing by {cost_us:+.1f} us"
    )
