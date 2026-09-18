"""Correct a transition from what the preview measured, then commit it.

THREADING CONTEXT: a **worker thread**, the same one :mod:`djai.preview` renders
on. Never the audio thread, never the scheduler's tick loop.

Two passes, in this order and never the other way round:

1. **Rules.** Deterministic, authoritative, and the only ones trusted for
   safety. Each maps one measurement over its threshold onto one correction.
   They cannot fail, cannot time out, and do not need a model to be running.
2. **The model, optionally, once.** Given the numbers and the rules' answer as
   text, it may propose something better. Everything it returns goes through
   `supervisor.py` unchanged; anything that fails validation is dropped and the
   rules' answer stands. It is never consulted about safety and never gets the
   last word by default.

The whole loop is wrapped in guarantees that matter more than any improvement
it might make:

* at most :data:`config.PREVIEW_MAX_ROUNDS` revisions, a hard cap
* a wall-clock budget; over it, commit what is current and log it
* a revision that makes ``loudness_dip_db`` or ``low_end_overlap_bars`` worse is
  reverted to what preceded it
* any failure at all -- render, measurement, model, arithmetic -- commits the
  ORIGINAL parameters

None of those paths can delay the transition. The loop runs inside the pre-roll
window and its output is a set of parameters; if it has not finished in time,
the parameters that were already armed are the ones that play.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from djai import config, preview as preview_mod, transition as tr
from djai.analysis import TrackAnalysis
from djai.deck import LoadedTrack

log = logging.getLogger(__name__)

#: Measurements on which a revision must not make things worse. Anything else
#: may move either way -- trading a little midrange clash for a flat level is a
#: reasonable thing for a revision to do; making the level sag deeper is not.
GUARDED = ("loudness_dip_db", "low_end_overlap_bars")

#: How much worse counts as worse, per guarded measurement. Below this is noise
#: in the measurement rather than a real regression.
WORSE_BY = {"loudness_dip_db": 0.25, "low_end_overlap_bars": 0.5}


@dataclass
class Revision:
    """One field changed, by one named rule, for one stated reason."""

    field: str
    before: Any
    after: Any
    rule: str
    because: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class PreviewOutcome:
    """What the loop decided, and everything it saw on the way there.

    ``status`` is one of:
      ``skipped``   preview is off, or there was nothing to preview
      ``committed`` measured, nothing needed changing
      ``revised``   measured and changed; ``revisions`` says what and why
      ``reverted``  changed, re-measured worse, and put back
      ``budget``    ran out of time; ``params`` is whatever was current
      ``failed``    could not render or measure; ``params`` is the ORIGINAL
      ``cut``       the grids do not agree; the transition must be a cut
    """

    status: str
    params: Any
    original: Any
    force_cut: bool = False
    rounds: list[dict[str, Any]] = field(default_factory=list)
    revisions: list[Revision] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    budget_ms: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.revisions) and self.status in ("revised", "budget")

    def summary(self) -> dict[str, Any]:
        """The compact form: what the UI shows and the session log records."""
        # A round committed on budget has no measurements of its own -- that is
        # what "unverified" means -- so the final numbers are the last round
        # that was actually measured. Found by the soak: reading None here
        # threw inside the pre-roll hook for every budget-committed preview.
        measured = [r["measurements"] for r in self.rounds if r.get("measurements")]
        first = measured[0] if measured else {}
        last = measured[-1] if measured else {}
        return {
            "status": self.status,
            "force_cut": self.force_cut,
            "rounds": len(self.rounds),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "budget_ms": self.budget_ms,
            "revisions": [r.as_dict() for r in self.revisions],
            "events": list(self.events),
            "first": _headline(first),
            "final": _headline(last),
        }


def _headline(m: dict[str, Any]) -> dict[str, Any]:
    """The six numbers worth showing on a screen in a dark room."""
    keys = (
        "peak_dbfs", "integrated_lufs", "loudness_dip_db", "low_end_overlap_bars",
        "vocal_overlap_bars", "spectral_clash", "transient_density_ratio",
        "phase_coherence",
    )
    return {k: m[k] for k in keys if k in m}


# --- the rules ------------------------------------------------------------------


def apply_rules(
    measurements: dict[str, Any], params, track_b: TrackAnalysis | None = None
) -> tuple[Any, list[Revision], bool]:
    """Deterministic corrections. Pure: same numbers in, same parameters out.

    Returns ``(params, revisions, force_cut)``. ``params`` is a new object; the
    one passed in is never mutated. ``track_b`` is only read, for which hot
    cues the incoming track actually has.
    """
    revisions: list[Revision] = []
    out = params

    def change(name: str, value: Any, rule: str, because: str) -> None:
        nonlocal out
        before = getattr(out, name)
        if before == value:
            return
        out = dataclasses.replace(out, **{name: value})
        revisions.append(Revision(name, before, value, rule, because))

    def value(key: str, default: float = 0.0) -> float:
        got = measurements.get(key)
        return float(got) if isinstance(got, (int, float)) else default

    # A grid that does not agree with itself is not a transition problem, and
    # no parameter here can fix it. Checked first, and it ends the matter.
    coherence = measurements.get("phase_coherence")
    if isinstance(coherence, (int, float)) and (
        coherence < config.PREVIEW_MIN_PHASE_COHERENCE
    ):
        return out, revisions, True

    # 1. The level sags through the blend: the curve is subtracting rather than
    #    blending. equal_power is the shape that does not; failing that, spend
    #    less time in the sag.
    dip = value("loudness_dip_db")
    if dip > config.PREVIEW_MAX_LOUDNESS_DIP_DB:
        why = f"loudness_dip_db {dip:.2f} over {config.PREVIEW_MAX_LOUDNESS_DIP_DB}"
        if out.curve != "equal_power":
            change("curve", "equal_power", "loudness_dip", why)
        else:
            lo, _hi = tr.LENGTH_BARS_RANGE
            shorter = max(float(lo), round(float(out.length_bars) * 0.75))
            change("length_bars", shorter, "loudness_dip", why)

    # 2. Two basslines at once. Hand the low band over sooner, or faster.
    overlap = value("low_end_overlap_bars")
    if overlap > config.PREVIEW_MAX_LOW_OVERLAP_BARS:
        why = (
            f"low_end_overlap_bars {overlap:.0f} over "
            f"{config.PREVIEW_MAX_LOW_OVERLAP_BARS:.0f}"
        )
        lo_bars, hi_bars = tr.LOW_SWAP_BARS_RANGE
        if out.low_swap_bar is None:
            # No managed swap at all is the extreme case of "too late": the two
            # low bands simply run together for the whole blend. A swap a
            # quarter of the way in is the earliest sensible place that still
            # lets the incoming track arrive first.
            change(
                "low_swap_bar", max(1, int(float(out.length_bars) * 0.25)),
                "low_overlap", why + " with no managed swap",
            )
            change("low_swap_bars", lo_bars, "low_overlap", why)
        elif out.low_swap_bar > 1:
            change("low_swap_bar", max(1, out.low_swap_bar // 2), "low_overlap", why)
        elif out.low_swap_bars > lo_bars:
            change(
                "low_swap_bars", max(lo_bars, out.low_swap_bars // 2),
                "low_overlap", why,
            )

    # 3. Two singers at once, when the design said to care. Entering at a
    #    later marker is what actually moves deck B's vocal away from deck A's;
    #    holding its low band back only delays the bass, so it is the fallback
    #    for a track with no later cue to go to.
    vocals = value("vocal_overlap_bars")
    if vocals > 0 and out.vocal_aware:
        why = f"vocal_overlap_bars {vocals:.0f}"
        later = _later_hot_cue(track_b, out.entry_point) if track_b is not None else None
        _lo_delay, hi_delay = tr.DECK_B_LOW_DELAY_RANGE
        if later is not None:
            change("entry_point", later, "vocal_overlap", why + ", entering later")
        elif out.deck_b_low_delay_bars < hi_delay:
            change(
                "deck_b_low_delay_bars",
                min(hi_delay, out.deck_b_low_delay_bars + 2),
                "vocal_overlap", why,
            )

    # 4. Both midranges in the same place: thin deck A out at the top, or sweep
    #    it out of the way.
    clash = value("spectral_clash")
    if clash > config.PREVIEW_MAX_SPECTRAL_CLASH:
        why = f"spectral_clash {clash:.3f} over {config.PREVIEW_MAX_SPECTRAL_CLASH}"
        if not out.deck_a_high_rolloff:
            change("deck_a_high_rolloff", True, "spectral_clash", why)
        elif out.filter_sweep == "none":
            change("filter_sweep", "hp_out", "spectral_clash", why)

    # 5. Too hot into the limiter. Depth is what intensity scales, so that is
    #    what comes down; nothing about the timing moves.
    peak = measurements.get("peak_dbfs")
    if isinstance(peak, (int, float)) and peak > config.PREVIEW_MAX_PEAK_DBFS:
        lo_i, _hi_i = tr.INTENSITY_RANGE
        change(
            "intensity", max(float(lo_i), round(float(out.intensity) * 0.8, 3)),
            "peak", f"peak_dbfs {peak:.2f} over {config.PREVIEW_MAX_PEAK_DBFS}",
        )

    return out, revisions, False


def _later_hot_cue(track_b: TrackAnalysis, entry_point: str) -> str | None:
    """The first hot cue in the incoming track positioned after the current entry.

    Only a cue the track really has, and only one that leaves enough of the
    track to play out -- the same two checks the supervisor applies to a cue a
    model names, so a rule revision can never be rejected downstream for
    pointing at nothing. None means there is no later entry to move to.
    """
    from djai.analysis import cue_seconds

    cues = sorted(
        (cue_seconds(c), int(c.get("index", -1))) for c in (track_b.hot_cues or [])
    )
    current_s = float(track_b.mix_in or 0.0)
    if entry_point.startswith("hot_cue_"):
        try:
            wanted = int(entry_point.rsplit("_", 1)[1])
        except ValueError:
            wanted = -1
        current_s = next((s for s, i in cues if i == wanted), current_s)
    bar_s = 4 * 60.0 / track_b.bpm if track_b.bpm > 0 else 2.0
    for seconds, index in cues:
        if seconds >= current_s + bar_s and tr.entry_has_runway(track_b, seconds):
            return f"hot_cue_{index}"
    return None


def is_worse(before: dict[str, Any], after: dict[str, Any]) -> str | None:
    """Has a revision made a guarded measurement worse? Returns which, or None."""
    for key in GUARDED:
        was, now = before.get(key), after.get(key)
        if not isinstance(was, (int, float)) or not isinstance(now, (int, float)):
            continue
        if float(now) > float(was) + WORSE_BY[key]:
            return f"{key} {was} -> {now}"
    return None


# --- the loop -------------------------------------------------------------------


def run_preview(
    state: "preview_mod.DeckAState",
    track_b: TrackAnalysis,
    loaded_b: LoadedTrack,
    entry_frame_b: float,
    params,
    intent_engine: Any = None,
    supervisor: Any = None,
    session_log: Any = None,
    budget_ms: float | None = None,
    max_rounds: int | None = None,
    blocksize: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> PreviewOutcome:
    """Render, measure, revise, commit. Never raises, never blocks a transition.

    Every return path yields a usable :class:`PreviewOutcome` whose ``params``
    are safe to arm. The caller's contract is simply: arm ``outcome.params``,
    and if ``outcome.force_cut`` is set, arm a cut instead.
    """
    budget = config.PREVIEW_BUDGET_MS if budget_ms is None else float(budget_ms)
    rounds_allowed = (
        config.PREVIEW_MAX_ROUNDS if max_rounds is None else int(max_rounds)
    )
    rounds_allowed = max(0, min(rounds_allowed, config.PREVIEW_MAX_ROUNDS))

    started = clock()
    outcome = PreviewOutcome(
        status="skipped", params=params, original=params, budget_ms=budget
    )

    def finish(status: str) -> PreviewOutcome:
        outcome.status = status
        outcome.elapsed_ms = (clock() - started) * 1000.0
        _log_preview(session_log, outcome)
        return outcome

    def spent_ms() -> float:
        return (clock() - started) * 1000.0

    if not config.PREVIEW_ENABLED:
        outcome.events.append("preview disabled")
        return finish("skipped")

    def take(current) -> dict[str, Any] | None:
        """One render and measurement, or None with the reason recorded."""
        try:
            return preview_mod.preview_transition(
                state, track_b, current, loaded_b, entry_frame_b,
                blocksize=blocksize,
            )
        except preview_mod.PreviewError as exc:
            outcome.events.append(f"preview failed at {exc.stage}: {exc.detail}")
            log.warning("preview failed (%s); committing original parameters", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - a preview must never escape
            outcome.events.append(f"preview raised {type(exc).__name__}: {exc}")
            log.warning("preview raised %s; committing original parameters", exc)
            return None

    # The budget is checked BEFORE each render, never during one: a render that
    # has started is cheaper to finish than to unpick, and the guarantee being
    # kept is "do not start work that cannot land in time".
    # Before the first render there is no measured cost yet, so estimate it
    # from how much audio the window holds. Without this a 100 ms budget still
    # starts a full render and finishes a second and a half over.
    bpm = float(getattr(state.analysis, "bpm", 0.0) or 0.0)
    window_s = (
        (float(params.length_bars) + 2.0 * config.PREVIEW_CONTEXT_BARS) * 240.0 / bpm
        if bpm > 0 else 0.0
    )
    estimate_ms = window_s * config.PREVIEW_MS_PER_AUDIO_S
    if spent_ms() + estimate_ms > budget:
        outcome.events.append(
            f"budget {budget:.0f} ms exceeded: a preview of this window costs "
            f"~{estimate_ms:.0f} ms; committing the parameters as armed"
        )
        return finish("budget")

    first_started = clock()
    measurements = take(params)
    #: What one render-and-measure costs on this machine, right now. A later
    #: render is only started if this much time is still left: "checked before
    #: the render" is not enough on its own, because a render started with
    #: 500 ms of budget left finishes two seconds over it.
    round_cost_ms = (clock() - first_started) * 1000.0
    if measurements is None:
        outcome.params = outcome.original
        return finish("failed")
    outcome.rounds.append({"round": 0, "params": params.to_schema(),
                           "measurements": measurements, "revisions": []})

    current, current_measurements = params, measurements

    for round_index in range(1, rounds_allowed + 1):
        revised, revisions, force_cut = apply_rules(
            current_measurements, current, track_b
        )
        if force_cut:
            outcome.force_cut = True
            outcome.events.append(
                f"phase_coherence {current_measurements.get('phase_coherence')} below "
                f"{config.PREVIEW_MIN_PHASE_COHERENCE}: forcing a cut"
            )
            outcome.params = current
            return finish("cut")

        rule_revisions = list(revisions)

        # The optional second opinion, once, only when a measurement actually
        # breached its threshold, and only with the time the rules' own
        # verification does not need. Measured on the crate: asking it about
        # every transition, including the ones nothing was wrong with, took
        # 20 of 20 previews over a 4000 ms budget.
        llm_note = None
        if (
            config.PREVIEW_LLM_REVISION
            and intent_engine is not None
            and supervisor is not None
            and round_index == 1
            and revisions
        ):
            # A quarter of margin on the reserved render: a render's cost varies
            # from one call to the next, and a model that uses every millisecond
            # it is offered otherwise leaves the verification exactly on the
            # edge of the budget -- measured, 5 of 20 revisions went unverified.
            left = budget - spent_ms() - 1.25 * round_cost_ms
            revised, llm_note = _llm_pass(
                revised, current_measurements, rule_revisions, revisions,
                intent_engine, supervisor, track_b, state.analysis,
                left, outcome,
            )
            if llm_note:
                outcome.events.append(llm_note)

        if not revisions:
            outcome.events.append("measurements within thresholds; nothing to revise")
            outcome.params = current
            return finish("committed" if round_index == 1 else "revised")

        # A revision exists. It is safe to commit unverified -- the rules only
        # ever move a parameter toward its safe end -- so running out of budget
        # here means committing it, not discarding it.
        if spent_ms() + round_cost_ms > budget:
            outcome.revisions.extend(revisions)
            outcome.params = revised
            outcome.events.append(
                f"budget {budget:.0f} ms: {spent_ms():.0f} ms spent and a "
                f"verifying render costs ~{round_cost_ms:.0f} ms; committing the "
                f"rule-revised parameters unverified"
            )
            outcome.rounds.append({
                "round": round_index, "params": revised.to_schema(),
                "measurements": None,
                "revisions": [r.as_dict() for r in revisions],
            })
            return finish("budget")

        round_started = clock()
        after = take(revised)
        round_cost_ms = max(round_cost_ms, (clock() - round_started) * 1000.0)
        if after is None:
            # The revision could not be verified. The rules are still
            # authoritative, so their answer stands and the failure is logged.
            outcome.revisions.extend(revisions)
            outcome.params = revised
            return finish("revised")

        regression = is_worse(current_measurements, after)
        outcome.rounds.append({
            "round": round_index, "params": revised.to_schema(),
            "measurements": after,
            "revisions": [r.as_dict() for r in revisions],
            "reverted": bool(regression),
        })
        if regression:
            outcome.events.append(f"revision made it worse ({regression}); reverted")
            outcome.params = current
            return finish("reverted")

        outcome.revisions.extend(revisions)
        current, current_measurements = revised, after

    outcome.params = current
    return finish("revised" if outcome.revisions else "committed")


def _llm_pass(
    rule_revised, measurements, rule_revisions, revisions,
    intent_engine, supervisor, track_b, track_a, left_ms, outcome,
):
    """One optional model revision, fully validated. Returns (params, note).

    The rules' answer is what is passed in and what comes back on any failure.
    Nothing here can widen what the supervisor accepts.
    """
    if left_ms <= 100:
        return rule_revised, "no budget left for a model revision"
    timeout_s = min(config.PREVIEW_LLM_TIMEOUT_S, left_ms / 1000.0)
    context = {
        "params": rule_revised.to_schema(),
        "measurements": _headline(measurements),
        "rules_applied": [
            {"field": r.field, "from": r.before, "to": r.after, "why": r.because}
            for r in rule_revisions
        ],
    }
    # A hard deadline, enforced here rather than trusted to the HTTP client:
    # httpx applies a timeout to each of connect, write and read separately, so
    # "2 seconds" can take several times that. The call runs on its own daemon
    # thread and is simply abandoned if it has not answered in time; whatever it
    # eventually returns goes nowhere.
    box: dict[str, Any] = {}

    def ask() -> None:
        try:
            box["result"] = intent_engine.revise_transition(context, timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001 - the model is never load-bearing
            box["error"] = exc

    worker = threading.Thread(target=ask, name="djai-preview-llm", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        return rule_revised, f"model revision abandoned after {timeout_s * 1000:.0f} ms"
    if "error" in box:
        exc = box["error"]
        return rule_revised, f"model revision raised {type(exc).__name__}: {exc}"
    raw, reason = box.get("result", (None, "no answer"))
    if raw is None:
        return rule_revised, f"model revision unavailable ({reason})"

    validated, why = supervisor.validate_transition_params(raw, track_b, track_a)
    if validated is None:
        return rule_revised, f"model revision rejected by the supervisor: {why}"

    # Accepted. Record what it moved, so the log says which pass did what.
    for f in dataclasses.fields(rule_revised):
        if f.name == "name":
            continue
        before, after = getattr(rule_revised, f.name), getattr(validated, f.name)
        if before != after:
            revisions.append(
                Revision(f.name, before, after, "llm", "model revision, validated")
            )
    return validated, "model revision accepted"


def _log_preview(session_log: Any, outcome: PreviewOutcome) -> None:
    """One record per preview: the file that says whether this is helping.

    Everything goes in -- the parameters that went in, every measurement of
    every round, every revision with the rule that made it, what was finally
    committed, and the time it all took.
    """
    if session_log is None:
        return
    try:
        session_log.write(
            "preview",
            status=outcome.status,
            force_cut=outcome.force_cut,
            elapsed_ms=round(outcome.elapsed_ms, 1),
            budget_ms=outcome.budget_ms,
            original=outcome.original.to_schema(),
            committed=outcome.params.to_schema(),
            revisions=[r.as_dict() for r in outcome.revisions],
            rounds=outcome.rounds,
            events=list(outcome.events),
        )
    except Exception:  # noqa: BLE001 - logging must not break a transition
        log.debug("preview logging failed", exc_info=True)
