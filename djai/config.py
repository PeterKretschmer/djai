"""Runtime configuration, all overridable by environment variable.

THREADING CONTEXT: read at import time on the main thread. Nothing here is
mutated after startup, so any thread may read these values.

Every setting is a module-level constant so the audio and intent layers can
import them without passing a config object around.
"""

from __future__ import annotations

import os

__all__ = [
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_TIMEOUT_S",
    "OLLAMA_WARMUP_TIMEOUT_S",
    "AUDIO_BLOCKSIZE",
    "TRANSITION_BARS",
    "ECHO_MAX_SECONDS",
    "TRANSITION_MIN_GRID_CONFIDENCE",
    "TRANSITION_DESIGN_WAIT_S",
    "LIMITER_KNEE",
    "LIMITER_RELEASE_MS",
    "LIMITER_SUBBLOCK",
    "LIMITER_LOG_REDUCTION_DB",
    "LOUDNESS_TARGET_LUFS",
    "LOUDNESS_MAX_TRUE_PEAK_DBTP",
    "TEMPO_AMBIGUITY_RATIO",
    "TIME_STRETCH_ENABLED",
    "MAX_STRETCH_RATIO",
    "STRETCH_DEADBAND",
    "STRETCH_CACHE_SIZE",
    "KEY_LOCK_DEFAULT",
    "FILTER_KNOB_RESONANCE",
    "CUE_RING_BLOCKS",
    "RECORD_ENABLED",
    "RECORD_DIR",
    "RECORD_RETENTION",
    "RECORD_RING_SECONDS",
    "FALLBACK_ENABLED",
    "FALLBACK_STALL_MS",
    "FALLBACK_UNDERRUN_BURST",
    "FALLBACK_WATCHDOG_HZ",
    "PREFLIGHT_MIN_GRID_CONFIDENCE",
    "PREFLIGHT_MIN_DISK_MB",
    "PREFLIGHT_MAX_DECODE_CHECKS",
    "PREVIEW_ENABLED",
    "PREVIEW_BUDGET_MS",
    "PREVIEW_MS_PER_AUDIO_S",
    "PREVIEW_PRE_ROLL_S",
    "PREVIEW_COMMIT_MARGIN_S",
    "PREVIEW_MAX_ROUNDS",
    "PREVIEW_CONTEXT_BARS",
    "PREVIEW_LLM_REVISION",
    "PREVIEW_LLM_TIMEOUT_S",
    "PREVIEW_MAX_LOUDNESS_DIP_DB",
    "PREVIEW_MAX_LOW_OVERLAP_BARS",
    "PREVIEW_MAX_SPECTRAL_CLASH",
    "PREVIEW_MAX_PEAK_DBFS",
    "PREVIEW_MIN_PHASE_COHERENCE",
    "PREVIEW_BAND_PRESENT_FRAC",
]


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


#: Where the local Ollama server is listening. Override with OLLAMA_BASE_URL.
OLLAMA_BASE_URL: str = _env_str("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

#: The local model used for intent parsing. Override with OLLAMA_MODEL.
OLLAMA_MODEL: str = _env_str("OLLAMA_MODEL", "llama3.1:8b")

#: Hard ceiling on one inference request. Past this the fallback chain takes
#: over -- a slow answer is worth less than a responsive prompt.
OLLAMA_TIMEOUT_S: float = _env_float("OLLAMA_TIMEOUT_S", 8.0)

#: The startup warmup gets far longer: a cold load of an 8B model off disk into
#: VRAM is a multi-second stall, and it happens once.
OLLAMA_WARMUP_TIMEOUT_S: float = _env_float("OLLAMA_WARMUP_TIMEOUT_S", 60.0)

#: Frames per audio callback. 2048 at 44.1 kHz is ~46 ms of headroom per block,
#: which is what lets the stream ride out the CPU and GPU spikes that local
#: inference causes on the same machine. It is still well inside the 100 ms
#: budget for a panic `cut`. Override with AUDIO_BLOCKSIZE.
AUDIO_BLOCKSIZE: int = _env_int("AUDIO_BLOCKSIZE", 2048)

#: Length of a transition, in bars. 24 bars is ~45 s at 128 BPM.
#:
#: Measured, not chosen: across 26 marked transitions in two professional sets,
#: every single one ran longer than the 16 bars this used to be, the shortest
#: being 22.4. The reference that transfers is the Peak Hour Tech House set --
#: 128.0 BPM against this crate's 127 median, same dance/pop family -- whose
#: blends measure 25.68 bars (IQR 25.6-26.8). 24 is the nearest musical length.
#:
#: The other set (Four Tet, long-form instrumental electronica) measures 37
#: bars; that one is real but does not transfer to vocal pop, where a longer
#: blend means more of two vocals over each other. See docs/tuning.md.
#: Override with TRANSITION_BARS.
TRANSITION_BARS: float = _env_float("TRANSITION_BARS", 24.0)

#: Below this grid confidence a blend is not attempted and the switch is made
#: instant instead. Deliberately separate from the preflight floor: preflight
#: asks "is this track analysable at all" and fails the whole run, while this
#: asks "can a 24-bar blend hold phase against this grid" for one transition.
#: On the v5 scale, tracks that detect cleanly score 0.42-0.64 and the ones
#: whose tempo is doubtful score 0.17-0.24, so this sits between them.
TRANSITION_MIN_GRID_CONFIDENCE: float = _env_float(
    "TRANSITION_MIN_GRID_CONFIDENCE", 0.25
)

#: How long the autopilot will hold off arming a transition while a design is
#: still being written, in seconds.
#:
#: Measured: llama3.1:8b answers in 1.1-3.7 s on this machine, and the
#: autopilot ticks every 0.5 s -- so without a deliberate wait the design was
#: never ready and every transition silently fell back to a preset. The lead
#: window is around 105 s, so waiting a few seconds costs nothing; the wait is
#: additionally cut short if the outgoing track is running out.
TRANSITION_DESIGN_WAIT_S: float = _env_float("TRANSITION_DESIGN_WAIT_S", 12.0)

#: Master limiter. Gain reduction begins here and approaches the ceiling
#: asymptotically, so the onset is gradual rather than a corner. Measured
#: need: a beatmatched blend of two loudness-war masters peaks at 1.48, and
#: the sources alone already exceed 1.0, so something has to give and it
#: should not be a hard clip.
#:
#: RAISED from 0.70 to 0.90 in Phase 1. With every track normalised to
#: LOUDNESS_TARGET_LUFS the limiter is true-peak safety only. Measured before
#: the change: a 30-minute offline set had it reducing gain on 38,071 of 38,759
#: blocks, 23,038 of them by more than 1 dB -- gain-riding, not protection.
LIMITER_KNEE: float = _env_float("LIMITER_KNEE", 0.90)

#: How fast the limiter lets go after a loud passage, in milliseconds. Long
#: enough not to pump on every kick, short enough not to duck a whole phrase.
LIMITER_RELEASE_MS: float = _env_float("LIMITER_RELEASE_MS", 150.0)

#: Gain is recomputed every this many frames. 64 is ~1.5 ms: short enough that
#: the gain envelope is smooth to the ear, long enough that the per-block cost
#: stays trivial. Attack is instantaneous within a sub-block, which is what
#: makes the ceiling a guarantee rather than a hope -- no lookahead, and so no
#: added latency on a monitor path a DJ is cueing against.
LIMITER_SUBBLOCK: int = _env_int("LIMITER_SUBBLOCK", 64)

#: Longest echo the `echo_out` delay line has to hold, in seconds. The ring is
#: allocated once at this size so no tempo ever needs a bigger one mid-set: a
#: dotted eighth at 60 BPM is 0.75 s, and 2 s leaves room for the tail.
ECHO_MAX_SECONDS: float = _env_float("ECHO_MAX_SECONDS", 2.0)

#: Limiter gain reduction beyond this many dB is logged. With loudness
#: normalisation in place, it indicates a bug rather than a hot master.
LIMITER_LOG_REDUCTION_DB: float = _env_float("LIMITER_LOG_REDUCTION_DB", 1.0)

#: Integrated loudness (BS.1770, LUFS) every track is normalised to at load.
LOUDNESS_TARGET_LUFS: float = _env_float("LOUDNESS_TARGET_LUFS", -14.0)

#: No track is played so loud that its true peak passes this, in dBTP. Measured
#: on the 96-track library: every track needs cutting to reach the target
#: (median -6 dB), yet 14 would still peak above -1 dBTP there -- decoded MP3s
#: overshoot -- so those are cut further and land somewhat under the target.
#:
#: The peak capped is the higher of the file's true peak and its true peak as a
#: deck plays it, through the three-band EQ at unity. That EQ is flat in level
#: but rotates phase: a 20-minute run measured a track reaching 1.025 into the
#: limiter while its file's true peak, after gain, was -2.6 dBTP. The
#: crossovers had raised it 3.1 dB, and capping on the file alone let it through.
LOUDNESS_MAX_TRUE_PEAK_DBTP: float = _env_float("LOUDNESS_MAX_TRUE_PEAK_DBTP", -1.0)

#: A track is flagged for review when the grid at twice its detected tempo
#: scores at least this fraction of the detected grid's onset agreement -- and
#: twice the tempo is itself a tempo the detector searches. See
#: :func:`djai.analysis.tempo_ambiguity` for why the second condition exists.
TEMPO_AMBIGUITY_RATIO: float = _env_float("TEMPO_AMBIGUITY_RATIO", 0.85)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


#: Match tempo by pitch-preserving time-stretch rather than by resampling.
#: Resampling a track 6% moves it a measured 100 cents -- a full semitone --
#: which is fine in a prototype and not fine in a club. Set to 0 to fall back to
#: resampling everywhere.
TIME_STRETCH_ENABLED: bool = _env_bool("TIME_STRETCH_ENABLED", True)

#: Widest tempo change we will ask for, as a fraction. Beyond this the selector
#: should never have offered the track, and the stretch is refused.
MAX_STRETCH_RATIO: float = _env_float("MAX_STRETCH_RATIO", 0.08)

#: Below this much tempo difference, stretching is not worth the CPU: 0.2% is
#: under 4 cents, which is inaudible.
STRETCH_DEADBAND: float = _env_float("STRETCH_DEADBAND", 0.002)

#: How many stretched copies to keep. Each is a full decoded track -- measured
#: at 72 MB for 3.6 minutes and 81 MB for four, at 44.1 kHz stereo float32 --
#: so this is a memory ceiling, not a hit-rate tuning knob: only the cued track
#: and the live one are ever needed. Two, not four, since Phase 2: four could
#: hold 323 MB of copies for no gain, on a 16 GB machine that also runs an 8B
#: model and has had the OS kill processes for low memory.
STRETCH_CACHE_SIZE: int = _env_int("STRETCH_CACHE_SIZE", 2)

#: Key lock: a deck plays at its original pitch whatever its tempo, using the
#: stretched copy. Per deck, and ON when a deck starts; turning it off on one
#: deck reverts that deck to resampling. Override the starting state with
#: KEY_LOCK_DEFAULT.
KEY_LOCK_DEFAULT: bool = _env_bool("KEY_LOCK_DEFAULT", True)

#: Resonance of each deck's filter knob when an operator turns it, 0.0-1.0. A
#: designed transition carries its own `filter_resonance`.
FILTER_KNOB_RESONANCE: float = _env_float("FILTER_KNOB_RESONANCE", 0.3)


# --- Phase 5: club hardening -------------------------------------------------

#: Blocks of slack in the cue ring, when cue runs on its own device. The two
#: streams have independent clocks, so the ring absorbs their drift; 8 blocks is
#: ~371 ms at 2048 frames, far more than two sound cards diverge over a set,
#: and pre-listen latency is set by how full the ring runs, not by its size.
CUE_RING_BLOCKS: int = _env_int("CUE_RING_BLOCKS", 8)

#: Record the master output. OFF by default; opt in with `play --record`.
#: Measured: nine sessions had taken 1.35 GB as 16-bit WAV (~635 MB/hour).
#: Recordings are now FLAC -- lossless and smaller. A killed process leaves a
#: FLAC recoverable up to its last fraction of a second (see SessionRecorder).
RECORD_ENABLED: bool = _env_bool("RECORD_ENABLED", False)

#: Where session recordings land. One file per run, named for its start time.
RECORD_DIR: str = _env_str("RECORD_DIR", "recordings")

#: How many FLAC session recordings to keep. Starting a new one deletes the
#: oldest beyond this. Only ``set_*.flac`` files count, so WAV recordings made
#: before recordings were FLAC are never removed automatically: `python -m djai
#: clean` lists them and deletes them only when the operator confirms.
RECORD_RETENTION: int = _env_int("RECORD_RETENTION", 5)

#: Seconds of audio the recording ring holds. The writer thread only has to keep
#: up on average; this is how long it may stall (a disk hiccup, a virus scanner)
#: before frames are dropped from the FILE. Audio itself is never affected --
#: the callback overwrites the ring and moves on.
RECORD_RING_SECONDS: float = _env_float("RECORD_RING_SECONDS", 8.0)

#: Arm the fallback player. It exists for one job: never let the room hear
#: silence because a Python-side thread died.
FALLBACK_ENABLED: bool = _env_bool("FALLBACK_ENABLED", True)

#: How long the callback may go without completing a block before the watchdog
#: calls the engine dead. Must clear a normal block period with room for an OS
#: scheduling hiccup: 46 ms per block at 2048, so 400 ms is ~9 blocks.
FALLBACK_STALL_MS: float = _env_float("FALLBACK_STALL_MS", 400.0)

#: Underruns inside one watchdog tick that count as "missing deadlines
#: repeatedly". A single underrun is system noise; a burst is a dying stream.
FALLBACK_UNDERRUN_BURST: int = _env_int("FALLBACK_UNDERRUN_BURST", 8)

#: Watchdog polls per second. The acceptance bar is audible fallback within
#: 500 ms, and the stall threshold already spends 400 ms of that, so the poll
#: interval has to be small next to what is left.
FALLBACK_WATCHDOG_HZ: float = _env_float("FALLBACK_WATCHDOG_HZ", 20.0)

#: Grid confidence below which preflight refuses a cached track. A weak grid
#: means the beat positions are guesses, and every transition is placed on them.
#: RECALIBRATED for analysis schema v5, where grid_confidence changed meaning.
#: It used to measure how uniform the detected beat intervals were, which a
#: grid laid down at a made-up tempo scores near 1.0 on. It now measures how
#: much of the track's onset energy actually lands on the grid, which real
#: music -- full of syncopation and off-beat hats -- scores far lower on by
#: construction. Measured across 96 tracks: v4 ran 0.53-1.00 with a median of
#: 0.82; v5 runs 0.17-0.64 with a median of 0.36.
#:
#: This one is a floor for "the analysis is broken", not a quality bar, because
#: preflight fails the whole run if any track is below it.
PREFLIGHT_MIN_GRID_CONFIDENCE: float = _env_float(
    "PREFLIGHT_MIN_GRID_CONFIDENCE", 0.15
)

#: Free disk preflight insists on, in MB. A session log is tiny; a recording is
#: ~635 MB/hour, so this is roughly a four-hour night with room to spare.
PREFLIGHT_MIN_DISK_MB: int = _env_int("PREFLIGHT_MIN_DISK_MB", 3000)

#: How many cached tracks preflight fully decodes. Decoding a whole crate takes
#: minutes; 0 means every track, which is the thorough pre-gig run.
PREFLIGHT_MAX_DECODE_CHECKS: int = _env_int("PREFLIGHT_MAX_DECODE_CHECKS", 0)

# --- the transition preview loop ----------------------------------------------
#
# Render the planned transition in the pre-roll window, measure the result, and
# revise the parameters before it runs -- a DJ cueing in headphones. Every
# setting here is a safety valve: the transition fires on time whatever the
# preview does or fails to do.

#: Whether to preview at all. Off makes every other setting here inert.
PREVIEW_ENABLED: bool = _env_bool("PREVIEW_ENABLED", True)

#: Total wall-clock budget for one preview, in milliseconds, covering every
#: render, measurement and revision. Exceeded means: stop, commit whatever
#: parameters are current, and log a budget-exceeded event.
#:
#: 4000 ms buys one measured round on this machine. A render costs roughly
#: 1.0-1.6 s per pass at the live blocksize (measured: 24 bars of audio, both
#: decks, 28x realtime at blocksize 2048) and a round is two passes -- the mix,
#: and deck B alone so the two can be told apart. Raising this to ~8000 buys the
#: second round that verifies a revision actually helped.
PREVIEW_BUDGET_MS: float = _env_float("PREVIEW_BUDGET_MS", 4000.0)

#: What a preview round costs per second of audio in its window, in ms, before
#: one has been measured: render plus measurement. The loop will not start a
#: first render that this says cannot finish inside the budget. Measured on
#: this machine at blocksize 2048: 0.8-1.2 s render and 0.55-0.85 s measurement
#: for a 16-bar window plus 8 bars of context (~45 s), i.e. ~30-45 ms per second.
PREVIEW_MS_PER_AUDIO_S: float = _env_float("PREVIEW_MS_PER_AUDIO_S", 40.0)

#: How far ahead of a transition's hand-off the preview is started, in seconds.
#: The autopilot usually arms well before this, so in practice the preview
#: starts as soon as the transition is armed; the horizon only matters for a
#: transition armed with less lead than this.
PREVIEW_PRE_ROLL_S: float = _env_float("PREVIEW_PRE_ROLL_S", 30.0)

#: The preview's parameters are only swapped in while the hand-off is at least
#: this far away. Closer than that, the armed transition stands untouched: a
#: late change is a race with the scheduler, and losing it is worse than not
#: making the change.
PREVIEW_COMMIT_MARGIN_S: float = _env_float("PREVIEW_COMMIT_MARGIN_S", 1.0)

#: Hard cap on revision rounds, per the brief. Not overridable upward: two is
#: the limit the design allows, and a third round would be a change of design
#: rather than a change of setting.
PREVIEW_MAX_ROUNDS: int = min(2, _env_int("PREVIEW_MAX_ROUNDS", 2))

#: Bars rendered either side of the transition window, so the preview can see
#: what the blend left behind as well as what it did.
PREVIEW_CONTEXT_BARS: float = _env_float("PREVIEW_CONTEXT_BARS", 4.0)

#: Whether the optional single LLM revision pass runs. The rule-based pass is
#: always authoritative and runs first; this only ever proposes.
PREVIEW_LLM_REVISION: bool = _env_bool("PREVIEW_LLM_REVISION", True)

#: How long the LLM revision pass may take. It is given whatever is left of the
#: budget, capped here, and a timeout simply means the rules' answer stands.
PREVIEW_LLM_TIMEOUT_S: float = _env_float("PREVIEW_LLM_TIMEOUT_S", 2.0)

# --- revision thresholds ------------------------------------------------------
# What counts as wrong. Each is the trigger for one deterministic correction in
# revise.py; see the table there.

#: Short-term loudness may sag this far below the two sources' average before
#: the crossfade curve is judged wrong. 3 dB is the point a dip stops reading as
#: dynamics and starts reading as a mistake.
PREVIEW_MAX_LOUDNESS_DIP_DB: float = _env_float("PREVIEW_MAX_LOUDNESS_DIP_DB", 3.0)

#: Bars in which both decks carry significant sub-200 Hz energy. Above this the
#: bass swap is not doing its job.
PREVIEW_MAX_LOW_OVERLAP_BARS: float = _env_float("PREVIEW_MAX_LOW_OVERLAP_BARS", 2.0)

#: Correlation of the two decks' 200 Hz - 2 kHz spectra during the overlap,
#: above which the midrange is muddy.
PREVIEW_MAX_SPECTRAL_CLASH: float = _env_float("PREVIEW_MAX_SPECTRAL_CLASH", 0.6)

#: Master peak ceiling for the rendered window, in dBFS.
PREVIEW_MAX_PEAK_DBFS: float = _env_float("PREVIEW_MAX_PEAK_DBFS", -1.0)

#: Cross-correlation of the two decks' beat positions, below which the grids do
#: not agree. That is a grid fault, not a transition fault, so it forces a cut
#: rather than a revision.
PREVIEW_MIN_PHASE_COHERENCE: float = _env_float("PREVIEW_MIN_PHASE_COHERENCE", 0.5)

#: A deck's band energy counts as "present" above this fraction of its own peak
#: in the window. Relative, so a quiet breakdown is not read as silence.
PREVIEW_BAND_PRESENT_FRAC: float = _env_float("PREVIEW_BAND_PRESENT_FRAC", 0.18)

