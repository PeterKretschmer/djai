# djai - an AI-mixed DJ (v1)

A real-time two-deck audio engine that beatmatches and crossfades automatically,
steered by typing plain English at a terminal prompt. Inference is local
(llama3.1:8b via Ollama) — no API key, nothing leaves the machine.

```
$ python -m djai analyze ./tracks/
$ python -m djai play

Now playing: 03_driver  125.0 BPM  1B (B major)
djai> something with more energy
Going up - cueing a harder track for the next phrase.
Cued: 09_stepper (128.0 BPM)
Bass swap fires in 23 bars.
djai> cut
cut.
```

## The one idea

An LLM takes seconds to answer. An audio callback has milliseconds. Those two
facts cannot be reconciled by making the model faster, so this program never
tries. It separates them completely:

```
     REPL thread            scheduler thread          audio thread
  +----------------+      +------------------+    +------------------+
  | your text      |      | commands sorted  |    | callback():      |
  |      |         |      | by musical time  |    |  drain queue     |
  | intent.py -----+-----="                  |    |  read both decks |
  | (Ollama, ~0.5s)| cmd  |  fires at the    |---=|  mix + EQ + gain |
  |      |         |      |  next 32-bar     |  Q |  write output    |
  | supervisor     |      |  phrase boundary |    |                  |
  |  .validate()   |      +------------------+    | no LLM, no I/O,  |
  +----------------+                              | no locks, no log |
                                                  +------------------+
```

Latency is hidden by **quantization, not speed**. Your sentence is parsed,
resolved to a structured command, and queued to fire on the next 32-bar phrase
boundary - up to a minute of music away. A model that takes three seconds is
still early.

Running the model on the same box adds a second problem the API version did not
have: inference spikes CPU and GPU on the machine that is also rendering audio.
That is handled by giving the callback more headroom (2048-frame blocks, ~46 ms)
and by leaving CPU threads free for it (`num_thread: 6` of 16). The architecture
above did not change.

The three layers are strictly separated, and the separation is enforced by what
each layer is allowed to touch:

| Layer | Module | Thread | May touch audio? |
|---|---|---|---|
| Chat / intent | `intent.py` | REPL | never - returns a description, not a command |
| Command queue | `commands.py`, `scheduler.py` | scheduler | only by enqueueing |
| Audio engine | `deck.py`, `engine.py`, `transition.py` | audio | is the audio |
| Oversight | `supervisor.py` | monitor | only via the scheduler |

Every module's docstring states its threading context.

## Install

Inference is local. There is no API key and nothing leaves the machine.

**1. Install Ollama** from [ollama.com/download](https://ollama.com/download),
then pull the model and confirm the server is up:

```bash
ollama pull llama3.1:8b
ollama list                                  # llama3.1:8b should appear
curl http://localhost:11434/api/tags         # should return JSON
```

On Windows the tray app starts the server automatically; `ollama serve` runs it
by hand. `ollama ps` shows what is resident and whether it is on the GPU —
`100% GPU` is what you want.

**2. Install the package:**

```bash
pip install -e .
```

Python 3.11+. Dependencies: `numpy`, `scipy`, `sounddevice`, `soundfile`,
`librosa`, `pyloudnorm`, `httpx`, `fastapi`, `uvicorn`.

`scipy` runs the deck's IIR filters - the EQ crossovers and the resonant filter
- through `sosfilt`, and measures true peak at analysis time. It already ships
as a hard dependency of `librosa`; it is listed explicitly because the code
imports it directly.

Key lock uses the **Rubber Band** command-line tool (4.0.0, R3 engine), vendored
for Windows under `djai/bin/rubberband/` with its GPL licence. Elsewhere, install
`rubberband` so it is on `PATH`. If it is missing or fails, decks resample to
tempo, which moves pitch, and the failure is logged.

Ollama not running? `analyze`, `play`, the panic commands and `state` all still
work — typed English falls back to keyword matching.

### Settings

All are environment variables, all optional:

| Variable | Default | What it does |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | where the server is |
| `OLLAMA_MODEL` | `llama3.1:8b` | the model to use |
| `OLLAMA_TIMEOUT_S` | `8` | ceiling on one request; past it, keywords take over |
| `OLLAMA_WARMUP_TIMEOUT_S` | `60` | ceiling on the startup load |
| `AUDIO_BLOCKSIZE` | `2048` | frames per audio callback |
| `RECORD_ENABLED` | `0` | record every `play` session even without `--record` |
| `RECORD_RETENTION` | `5` | FLAC recordings kept; starting one deletes the oldest beyond this |
| `LOUDNESS_TARGET_LUFS` | `-14` | integrated loudness every track is normalised to at load |
| `LOUDNESS_MAX_TRUE_PEAK_DBTP` | `-1` | no track is played so loud its true peak passes this |
| `TEMPO_AMBIGUITY_RATIO` | `0.85` | how close the double-tempo grid must score to flag a track for review |
| `LIMITER_LOG_REDUCTION_DB` | `1` | limiting deeper than this is logged as a bug |
| `TIME_STRETCH_ENABLED` | `1` | `0` turns stretching off everywhere: every deck resamples |
| `KEY_LOCK_DEFAULT` | `1` | whether each deck starts with key lock on |
| `FILTER_KNOB_RESONANCE` | `0.3` | resonance of the filter knob when an operator turns it (0-1) |

## Use

```bash
python -m djai analyze ./tracks/      # offline; writes ./cache/<hash>.json + .npz
python -m djai analyze ./tracks/ --force
python -m djai devices                # list usable output devices
python -m djai play                   # start the engine and the REPL
python -m djai play --device 2 --blocksize 512
python -m djai play --record          # also record the master to recordings/*.flac
python -m djai render "A" "B" --keep-renders   # offline render, kept in renders/
python -m djai clean                  # list reclaimable disk space; asks before deleting
python -m djai import rekordbox rekordbox.xml   # grids, hot cues, playlists from a Rekordbox XML export
python -m djai import serato "E:/_Serato_"      # crates from a _Serato_ folder; grids and cues from each file's tags
python -m djai import rekordbox lib.xml --relocate "D:/Music=E:/Music"   # a library that has moved
```

Disk use stays bounded by default. Recording is off unless `--record` is given,
writes FLAC, and keeps the newest `RECORD_RETENTION` sessions; WAV recordings
from older versions are only ever removed by `clean`, on confirmation. `render`
writes to a scratch folder that is removed on exit unless `--keep-renders` or
`--out` is given, and its CSVs are gzipped. Session logs rotate at 10 MB and
keep five gzipped predecessors, and repeated supervisor interventions within a
second collapse into one entry with a `count`.

At the prompt:

| Input | Effect |
|---|---|
| `cut` | master gain to zero, **immediately** - never sent to the LLM |
| `killbass` | low band to zero on both decks, immediately |
| `stop` | stop both decks and exit |
| `state` | print deck states without calling the LLM |
| `grid [a\|b] halve` / `double` / `nudge <ms>` / `tap` / `downbeat <s>` | correct a track's beat grid; saved, and never overwritten by analysis. Tempo changes are refused mid-blend; `tap` sets the tempo from 8+ steady taps |
| `keylock [a\|b] on` / `off` | key lock per deck: original pitch at any tempo, or resampling; `keylock` alone reports both |
| `filter [a\|b] <-1..1> [res <0..1>]` / `filter [a\|b] off` | the deck's filter knob: negative sweeps a low-pass down (to 60 Hz), positive a high-pass up (to 12 kHz), 0 is off |
| `phase warmup` / `build` / `peak` / `cooldown` / `auto` | set arc: which intensity the next track picks aim for; `phase` alone reports it |
| `style reverb_out` / `brake` / `noise_riser` / `filter_echo` | effect transitions: echo into a reverb tail, a 2-beat turntable stop, an 8-bar noise riser, a filter sweep with an echo |
| `loop [a\|b] in` / `out` / `exit` / `halve` / `double` / `<beats>` | loops. `<beats>` sets an auto-loop from the next beat. `exit` resumes where straight playback would be, in phase |
| `roll [a\|b] 1/8` … `4` / `off` | loop roll, in bars, from the beat just played. `off` lets go in phase |
| `jump [a\|b] ±1` / `±4` / `±8` / `±16` | beat jump, in bars. With quantize on, it fires on the next beat |
| `pitch [a\|b] <±pct>` / `reset` | pitch fader, ±8%. A deck moved by hand is not drift-corrected until it is synced |
| `sync [a\|b]` | match the deck's tempo to the master, then shift it into phase |
| `quantize on` / `off` | snap loops, rolls and jumps to the beat grid (on by default) |
| anything else | interpreted by the model, queued to the next phrase boundary |

The model can answer with exactly six actions: `next_track`, `set_energy`,
`hold_blend`, `skip_queued`, `describe_state`, `none`. Anything else is a parse
failure and becomes an inert `none`.

## What the layers actually do

### `analysis.py` - offline, cached, never runs during playback

Per track: BPM, beat grid, downbeats, key (Camelot + name), per-beat RMS, and a
`grid_confidence` in 0-1. Two files per track: `cache/<hash>.json` holds the
scalars, mix points and hot cues, and `cache/<hash>.npz` holds the beat grid,
downbeats and per-beat RMS as float32 arrays - 63% smaller than the old all-JSON
sidecar across a 96-track library. Both carry the schema version, and both are
checked before anything in them is used. A v5 sidecar is split into the two
files in place without re-reading audio. The hash is over file **content**
(size + head + tail), so renaming or moving a track still hits the cache while
editing it misses.

Each track's integrated loudness (BS.1770) and true peak are measured too - the
true peak both of the file and of the track as a deck plays it, through the
three-band EQ at unity. That EQ is flat in level but rotates phase, which lifts
the peaks of heavily limited masters by 2-3 dB. The deck applies a gain at load
that brings the track to `LOUDNESS_TARGET_LUFS`, cut further where needed to
keep the higher of those two peaks under `LOUDNESS_MAX_TRUE_PEAK_DBTP`, so the
master limiter only catches true peaks.
A track is flagged for review - a badge in the library, and a lower rank in the
selector - when its grid confidence is weak, or when a grid at twice its tempo
agrees nearly as well as the detected one and twice the tempo is plausible.
Corrections made with `grid`, or with the grid controls and bar-1 handle in the
web UI, set `grid_manually_corrected`, which analysis never overwrites. A cache
from an older version is upgraded in place; run `analyze` once afterwards to
measure loudness and tempo scores for tracks that lack them.

Three things here are less obvious than they look:

- **Tempo octaves.** `librosa.beat.beat_track` routinely returns a half, double
  or dotted multiple of the real tempo - on the test crate it reported 80.7 BPM
  for a 122 BPM track. The fix is a comb-filter re-score of each plausible
  multiple against the onset envelope, weighted by a log-normal prior centred on
  128 BPM.
- **Beat-frame quantisation.** librosa reports beats on a 23 ms frame grid, so
  the median inter-beat interval lands on 123.05 BPM for a 125.000 BPM track no
  matter how clean the audio is. Tempo and phase are instead recovered from a
  single-bin DFT of the onset envelope, which localises both far below frame
  resolution (measured error on the synthetic crate: **within 0.01 BPM on 12 of
  12**, worst case 0.006 BPM on the deliberately drifting track). The same
  quantisation inflates interval variance, so `grid_confidence` subtracts the
  known quantisation variance before taking the CV - otherwise every track
  scores ~0.85 regardless of how tight it actually is.
- **A uniform grid is stored, not the raw detections.** The decks change tempo by
  resampling at a constant rate, so a constant-tempo grid is the only grid they
  can actually hold phase against. `grid_confidence` is what tells you whether
  that model fits the track.

Key detection is Krumhansl-Schmuckler over a chroma computed from the *harmonic*
component above C3 - kick drums have strong pitched tails in the bottom octaves
(a 45 Hz tail reads as F#1) that otherwise dominate the estimate. It resolves the
tonic reliably but, like all KS implementations, confuses a minor key with its
relative major. That confusion is harmless here: relative keys share a Camelot
number and the selector already treats them as compatible.

### `deck.py` - one track, one playhead, gain, 3-band EQ, filter, rate

**Key lock.** Each deck has it, and it is on by default. When a track is cued at
a new tempo, a stretch worker runs Rubber Band's R3 engine over the whole track.
The deck then plays that copy at the track's own pitch. Measured at +6 %:

- R3 shifted pitch 0.00 cents; resampling shifts it +100.
- R3 kept 0.72 of a click's energy in its first 5 ms, against 0.65 for the
  librosa phase vocoder it replaced.
- It took 16.6 s for a 209 s track, which the cue's pre-roll absorbs.

Turning key lock off swaps the original back under the playhead, crossfaded
over 64 frames. Until a copy is ready, or if Rubber Band fails, the deck
resamples.

**Filter.** A resonant biquad sits after the EQ on one centre-detented knob.
Turning left sweeps a low-pass from 20 kHz down to 60 Hz; turning right sweeps a
high-pass from 20 Hz up to 12 kHz; resonance runs from 0 (flat) to 1 (Q 4).

All coefficients are built once into a table: 1024 knob steps × 17 resonance
steps. The callback only indexes that table. While the knob moves it steps the
coefficients every 64 frames and carries the filter state across. Inside the
detent the filter is bypassed bit-for-bit. `filter_sweep` and designed `hp_out`
/ `lp_in` transitions drive this knob.

Every working buffer is preallocated and every numpy operation writes through
`out=`.

The EQ uses **Linkwitz-Riley 4th-order crossovers**, and it is worth saying why,
because the cheaper design is wrong in a way that passes the obvious test. One
lowpass per crossover with subtraction for the upper bands reconstructs the input
*exactly* at unity gain - the bands telescope back algebraically, so a flat-EQ
test passes perfectly. But a Butterworth shifts phase, so `raw - low` does not
cancel low frequencies. Measured: at 60 Hz through a 250 Hz lowpass the magnitude
is 1.00 and the phase is about -54 degrees, so the "kill" leaves **62 % of the
bass amplitude**. LR4 branches are in phase at their crossover and sum to an
allpass: flat EQ preserves the magnitude spectrum (within 0.09 dB) and a bass
kill genuinely removes bass (-49 dB at 60 Hz).

Filter coefficients stay `float64` on purpose - a 250 Hz crossover at 44.1 kHz
sits at a normalised frequency of 0.011, close enough to DC that `float32`
coefficients visibly degrade the response.

### `engine.py` - the callback

Owns both decks and the master beat clock. Drains a `queue.Queue` of commands at
the top of every callback, mixes, applies the transition automation, writes the
output. No logging, no I/O, no allocation beyond what scipy's `sosfilt` returns
(it has no `out=` parameter). Exceptions are caught and turned into silence: a
dropout is survivable, a torn-down stream is not.

It publishes `engine.clock` - `(frames_played, master_beat, pos_a, pos_b)` as a
single tuple rebound once per callback. Reading those four attributes separately
from another thread is a race: the callback advances deck positions before the
beat counter, so a monitor that reads between them sees a full block of phantom
skew - 11.6 ms at 512 frames, squarely inside the drift monitor's nudge band.
That bug was real, and the snapshot is what prevents it.

A command released late is compensated at the cue point. The scheduler polls
every 2 ms and commands apply at the top of a callback, so a cue lands up to a
block after its musical moment; the deck is started where it *would* have been
had it fired on time. Without that, every transition began with a systematic
12-24 ms phase error that the drift monitor then had to hard-resync away (34
resyncs in a 10-minute run; zero after).

### `transition.py` - the one transition

A 32-bar bass-swap crossfade, all bar counts as constants at the top:

```
bar   0    4    8   12   16   20   24   28   32
      |----|----|----|----|----|----|----|----|
B in  gain 0 -> 1    |  |    |
B low ===============0===| /--1================
A low ===============1===|-\ 0
A gain ========================1 -> 0
```

Bars 0-8 the incoming deck fades up with its low band cut, so two tracks play
over one bassline. Bars 8-16 steady. Bars 16-20 the bass swaps, linearly and
complementary so the two low bands always sum to 1. Bars 20-32 the outgoing deck
fades out. Gains are equal-power; the bass swap is linear.

Measured through a full transition: the two decks hold a **constant offset of
255.999983 beats - exactly 64 bars** - with under 0.01 ms of wander.

### `supervisor.py` - makes failures legible

**Drift monitor**, every 100 ms: compares each deck's actual beat position to
where the master clock says it should be. Under 5 ms ignored, 5-20 ms a
fractional rate nudge, over 20 ms a hard resync at the next downbeat.

Several details make it behave rather than oscillate:

- Corrections are computed relative to the deck's **nominal** rate, never its
  current one, and are removed once drift clears. A proportional controller with
  no way home overshoots and limit-cycles; the first version did exactly that.
- A `SetRate` on the master deck deliberately does **not** retune the master
  clock. Correcting a deck *against* a reference must not move the reference, or
  drift can never read as zero.
- The phase baseline is keyed on `Deck.load_seq`, not on the track id. Re-cueing
  the *same* track is still a new playhead; keying on the id made a stale offset
  look like 181 seconds of drift, resynced forever and never fixed.
- Drift beyond ~2 bars is not treated as drift at all. It means our own baseline
  is stale, so the baseline is recaptured and the anomaly logged.

The non-master deck's baseline is snapped to the nearest whole bar, so a sloppy
cue registers as a real error instead of being adopted as correct.

**Stall check** - a third check, beyond the two v1 asks for. A 10-minute run on
this machine's *default* output device produced 481 s of audio in 630 s of wall
clock: PortAudio stopped calling the callback, reported no error, and left the
underrun count at zero. Nothing else in the program could tell that the music had
stopped. It reports and logs; it does not try to restart the stream. Delete
`check_stalled` and its line in `_run` if you want the literal two.

**Command validator**: nothing from the LLM reaches the scheduler without
passing `validate()` - the track must be in the cache, the file must still exist,
the BPM delta must be inside +-8 %, the target deck must not be mid-transition,
EQ gains must be in range. Rejections are logged with a reason and playback is
untouched.

Everything lands in `logs/session_<timestamp>.jsonl`, one JSON object per line,
each with the track pair, position in bars, trigger and action:

```json
{"ts":"2026-09-07T19:00:41.062+00:00","event":"drift_nudge","deck":"b",
 "track_pair":["07_late","02_rolling"],"position_bars":7.83,
 "trigger":"drift -13.9 ms","action":"rate 1.029310 -> 1.028670","drift_ms":-13.9}
```

### `intent.py` - the only LLM call, and it is local

One `POST /api/chat` to Ollama per typed line. Single-turn, stateless, no
history, `"stream": false`. Three things make an 8B model safe to put here:

- **Decoding is schema-constrained.** The request carries a literal JSON schema
  in `format`, with the six valid actions as an `enum`. An action outside the
  six is not something the model is discouraged from emitting — it is
  unrepresentable.
- **The prompt is flat.** One line of role, six actions with their params, three
  worked examples, under 400 words, no nested sections and no instructions to
  reason. llama3.1:8b follows a list and examples far better than prose about
  how to behave.
- **The state blob is five fields** — BPM, key, bars in, whether a transition is
  running, tracks played. Everything else was noise it could not act on: it may
  not name tracks and cannot set gains.

`"keep_alive": "30m"` keeps the model resident in VRAM, because a cold reload is
a multi-second stall. `"num_thread": 6` of 16 hardware threads is deliberate —
the headroom left over is where the audio callback lives. Do not raise it.

**Failure is not an error path.** A local 8B times out and malforms more often
than a hosted API did, so on timeout, connection error, bad HTTP, malformed
body, or an action outside the six, the request falls back:

1. keyword-match the raw text against the six actions, taking a direction from
   words like *harder* / *chill*;
2. failing that, an inert `none` carrying the raw text that failed.

Either way the reason is logged to the session jsonl and the prompt stays
usable. Whatever survives all of this still goes through
`supervisor.validate()` before it can reach the scheduler — unchanged, and more
necessary than before.

Measured on this machine (RTX 3060, model resident): **15/15 correct actions**
across realistic phrasings, median latency **0.50 s**, cold load ~49 s once.

### `selector.py` - track choice, with no LLM in it

A pure function. BPM within +-6 %, Camelot-adjacent key, not already played,
energy closest to the requested direction. **The model never names a track** - it
supplies at most an energy direction between -1 and +1. A hallucinated title has
nowhere to go.

### Library import

`python -m djai import rekordbox|serato <path>` brings in another program's work. It uses only the standard library.

- **Rekordbox:** reads the XML export (File → Export Collection in xml format).
  - Tracks come from their `file://` locations, playlists from the folder tree (a nested playlist is named like `Sets / Peak Hour`).
  - The grid comes from each track's first `TEMPO` marker. `Battito` gives the marker's beat in the bar, so bar 1 lands where Rekordbox has it.
  - Hot cues and loops come from `POSITION_MARK` 0–7. Memory cues have no slot here, so they are counted and skipped.
- **Serato:** reads every `.crate` in `_Serato_/Subcrates` as a playlist.
  - Each track's grid and cues come from the "Serato BeatGrid" and "Serato Markers2" ID3 tags inside MP3s.
  - Track paths in a crate are relative to the drive the `_Serato_` folder is on. Use `--root` if the library is somewhere else.
- **Where imported data goes:**
  - An imported grid is marked as corrected by a person, so `analyze --force` never replaces it.
  - Imported cues replace any cue at the same slot.
  - A track the cache does not know yet is analysed on the imported grid, unless you pass `--no-analyze`.
  - Playlists go to `cache/library/playlists.json`.
- **Output:** every file gets one line saying whether it was imported, updated, missing, not analysed or failed, with its grid, cue count and playlists.

### Waveforms

- **Colours:** each deck's waveform is drawn in the EQ's own three bands: low in blue, mid in orange, high in white.
- **Cache:** the bands are computed once per track from the unstretched audio, so they stay on the beat grid when key lock stretches a deck. Each is kept at four resolutions: a whole-track overview, and about 25, 100 and 400 points per second. They are stored in `cache/waveforms/`, and a cached waveform is only used if its version matches.
- **Zoom:** the − / + buttons, or the mouse wheel, go from the whole track down to exactly one bar. Once zoomed in, the waveform scrolls under a fixed playhead.
- **Overlays:**
  - section strip (intro, build, drop, breakdown, outro)
  - every beat, once they are at least 4 px apart
  - downbeats and 8-bar phrase lines
  - numbered hot cues, with drops in red
  - mix points
  - the draggable bar-1 handle, which works at any zoom
- **Drawing:** both decks redraw every animation frame, with the playhead moved on between the 20 Hz state updates.
- **Frame timing:** the page measures its own frame intervals and draw times. It reports them to the server every 5 s (`ui_frames` in the state feed, logged about once a minute) and exposes them to the browser console as `window.djaiFrameStats()`.

### Performance controls

Loops, rolls, beat jump, the pitch fader, sync and quantize work the same from the REPL and the web UI. On the page they are the per-deck LOOP / ROLL / JUMP / PITCH rows and the QUANT button. Both surfaces run the same Session methods, and every position is worked out from the beat grid on the control thread; the audio thread only assigns.

- **Slip loops.** Every loop and roll exit resumes where straight playback would have been. That keeps the deck in phase and on the master's bar lines, whatever the loop's length or however many laps it ran. Halving and doubling move the loop's end, never its start.
- **Quantize (default on).** Auto-loops, loop exits and beat jumps wait for the deck's next beat line. Loop in/out points snap to the nearest beat, and a roll starts from the beat just played. A beat jump is always a whole number of bars, so it never moves the phase.
- **Manual control holds the automation.** Every deck gesture freezes the autopilot and shows HELD. `resume`, or RESUME on the page, hands it back.
- **Transitions win.** A gesture on a deck that is part of a running transition is refused and logged (`manual_refused`), because the crossfade is driving that deck against a fixed length. Quantize can still be toggled.
- **Pitch and sync.**
  - A deck whose pitch fader was moved by hand is left alone by drift correction.
  - Moving the master deck's fader moves the master clock, so the other deck follows it.
  - `sync` hands a deck back: it sets the master's tempo, shifts the deck into phase by under half a beat, and re-baselines the drift monitor.
- **Panic is unaffected.** CUT, KILL BASS and STOP go straight to the engine. They still act within one block, loops and pitch included.

### Musical structure and the set arc

`analyze` measures three things per track, stored as cache schema v8. Entries from v5–v7 upgrade in place, and `analyze` fills in what is missing without touching the tempo or grid.

**Sections.**
- Labels are intro, build, drop, breakdown and outro, stored as bar ranges.
- Boundaries come from a checkerboard novelty over the bar-by-bar self-similarity of MFCCs and chroma, plus the level change at each bar line, weighted toward 8- and 4-bar phrase lines.
- The first and last of three or more segments are the intro and outro.
- A segment within 3 dB of the loudest is a drop. One rising at least 0.2 dB/bar into a drop is a build. Anything else is a breakdown.

**Vocals.**
- Vocal bar ranges come from harmonic/percussive separation, restricted to 200 Hz–4 kHz.
- A bar counts as vocal when that band's harmonic content keeps moving (normalised flux above 0.11) and fills the band (energy share above 0.5).

**Intensity.**
- One 0–1 score, measured on loudness-normalised audio, so a quiet banger outranks a loud ambient track.
- Weights: percussive share 45%, onset flux 35%, onset density 20%.

**How placement uses them:**
- **Outro over intro** is the default: the blend starts where deck A's outro starts, and deck B enters at its intro.
- **Drop to drop:** when both tracks have labelled drops and the style is auto, the drops are lined up instead. Only these drop-aligned transitions may start from 25% of the outgoing track; everything else keeps the 40% guard.
- **Vocal clashes are refused.** The supervisor rejects, and logs, any transition where both decks are audible with vocals for a bar or more while the incoming low band is open. A vocal-aware design first tries moving deck A earlier or deck B later by 8 or 16 bars. Failing that, the mix cuts at the planned hand-off.

**How track selection uses them:**
- The selector takes a set phase:
  - **warmup** aims for intensity at or below 0.45
  - **peak** aims for 0.6 or above
  - **build** and **cooldown** step 0.08 up or down from the current track
- It penalises the same artist within 5 tracks, the same key within 3, and two vocal-led tracks back to back.
- The artist comes from an `Artist - Title` file name.

**Effect sends.**
- Echo and reverb take each deck after its EQ and filter, but before its fader. An echo of a filtered deck therefore echoes the filtered sound.
- Each send is capped at the headroom deck A's dry level leaves, so deck A plus its sends never exceed unity.
- **Echo repeats ring out.**
  - The send only decides what goes into the delay line. Repeats come back at a fixed level (`ECHO_RETURN`, 0.8) and fade by feedback.
  - So they keep sounding after the send closes, and past the hand-over into the incoming track, until they are 60 dB down.
  - A tail that is still ringing carries into the next transition.
  - CUT and STOP silence it at once.
- Measured before these rules: with one effect style forced for 20 minutes, the master reached 1.202 into the limiter for `echo_out`, 1.091 for `filter_echo` and 1.026 for `reverb_out`.
- Measured after: every style stays at or below 0.891, with no limiting.

**Effects.** They run in the callback with preallocated buffers:
- **Reverb:** an 8-line feedback delay network. Every delay is at least one block long, so it runs block-wise. It is fed by deck A and its echo, and its tail rings about 3 s.
- **Brake:** the outgoing deck's rate falls to zero over 1–2 beats.
- **Noise riser:** rendered on the control thread when the transition is armed. The callback only plays it back at the envelope's level.

Measured on 10 synthetic tracks with known structure: 99% of bars labelled correctly, boundaries 44/45 within 2 bars, vocal precision 100% and recall 95%. Real-track accuracy is not yet measured against human annotations.

## Testing

```bash
python -m pytest -q          # 169 tests, ~2.5 minutes
```

The audio callback is driven directly in tests rather than by PortAudio, so
timing assertions are deterministic and no sound card is needed. `tests/synth.py`
renders synthetic 4/4 tracks with known BPM, key and phrase structure, including
one with deliberate 2.5 % tempo drift so `grid_confidence` has something real to
discriminate.

## Known limits in v1

- **Key lock needs the stretch to finish first.** A 3.5-minute track takes about
  17 s to stretch. A load that arrives sooner, such as an immediate manual
  load, resamples until key lock is toggled again. Resampling at +-6 % moves
  pitch about a semitone.
- **Key mode is unreliable.** Tonic detection is solid; major/minor resolves to
  the relative key on ambiguous material. Camelot-compatible either way.
- **Command release is not sample-accurate.** The scheduler polls at 2 ms, so a
  command fires within ~15 ms of its musical position, and the cue point is
  corrected for that lateness rather than the release being made exact.
- **Drift is measured against the model, not the music.** The monitor compares
  playheads to the analysed grid. It catches cue error, rate mismatch and command
  jitter; it cannot notice a track whose own tempo wanders away from its grid,
  which would need live beat tracking in the audio path (out of scope).
- **Pick your output device deliberately.** `play` falls back across host APIs
  until something opens, because on this machine PortAudio's default host API
  enumerated zero devices while WDM-KS worked. The device it lands on is not
  always the one that stays healthy - run `python -m djai devices` and pass
  `--device` if a stall is reported.
- **The first startup after a reboot is slow.** Loading llama3.1:8b off cold
  disk measured ~49 s here; `Loading model...` blocks on it, up to 60 s. After
  that `keep_alive` holds it in VRAM for 30 minutes and requests are ~0.5 s. A
  warmup timeout is not treated as failure — the server is reachable and still
  loading, so only the first command pays.
- **An 8B is a weaker reader than a frontier model.** It reliably picks the
  right action from the six, but nuance in the `reply` sentence is thinner and
  it occasionally reaches for `next_track` where `set_energy` was meant. Both
  are valid actions and the supervisor validates either, so the cost is a
  slightly wrong choice, not a broken mix.
- **One transition, no stems, no MIDI, no GUI, no live input.**

## Layout

```
djai/
  analysis.py     offline analysis + JSON/npz cache         (main thread)
  deck.py         playhead, resampler, LR4 3-band EQ,       (AUDIO THREAD)
                  resonant filter; Rubber Band stretch      (worker thread)
  bin/rubberband/ vendored Rubber Band CLI (GPL)
  engine.py       the callback, two decks, master clock     (AUDIO THREAD)
  phrase.py       sample <-> beat <-> bar <-> 32-bar phrase (main/monitor)
  commands.py     the command vocabulary                    (built off-thread)
  scheduler.py    pending commands, released on time        (scheduler thread)
  transition.py   the 32-bar bass swap                      (AUDIO THREAD)
  selector.py     next-track choice, pure                   (main thread)
  intent.py       the only LLM call, to local Ollama        (REPL thread)
  config.py       settings, all env-overridable             (import time)
  supervisor.py   drift monitor, validator, session log     (monitor thread)
  cli.py          REPL, autopilot, subcommands              (main thread)
  housekeeping.py recording retention, what `clean` removes (main thread)
tests/
  synth.py                 synthetic crate generator
  test_analysis.py         cache behaviour, tempo, grid, key
  test_deck.py             EQ, resampling, real-time safety
  test_engine.py           mixing, transition, scheduler, phrase
  test_supervisor.py       drift thresholds, validator, logging
  test_selector_intent.py  selection rules, malformed-LLM handling
  test_intent_ollama.py    Ollama transport, fallback chain, config
  test_integration.py      end-to-end through the CLI layer
```
