# Tuning against reference sets

Phase 2. Purpose: measure real DJ transitions and let their medians set our
parameter values, instead of the numbers I picked by ear.

## Status: `TRANSITION_BARS` 16 -> 24, applied at n=26

Applied on your instruction at **26 measured transitions, below the 30 the phase
asks for**. Recorded here so the provenance is not lost: the evidence is two
professional sets, not thirty transitions across many, and the number should be
revisited if more reference material arrives.

Two consequences of the change are measured and reported at the end of this
document. Neither is a defect in the change; both are things the longer blend
now collides with.

## What was measured

| Set | n | Median tempo | Median blend |
|---|---|---|---|
| *Four Tet @ Lightning in a Bottle 2025* | 15 | 129.0 BPM | **37.07 bars** (69 s) |
| *Peak Hour Tech House* (John Summit, Chris Lake, Dom Dolla, Mau P) | 11 | 128.0 BPM | **25.68 bars** (48 s) |
| **Pooled** | **26** | 128.6 BPM | 34.29 bars |

26 measured, **0 rejected**.

Neither markers file named a file that exists on disk (`four_tet_lib_2025.mp3`,
`peak_hour_tech_house.mp3`). Both were resolved by matching words in the
filename — 75% and 100% overlap respectively — and the substitution is printed,
not made silently.

## The pooled median is the wrong number

The distribution is **bimodal**, with a clean gap:

```
22.4 24.5 25.3 25.6 25.6 25.6 25.7 25.7 25.7 27.8 28.8 28.8   <- tech house
                          |  gap  |
33.9 34.7 36.3 36.5 36.7 36.7 37.1 37.4 38.4 38.7 38.7 39.0 39.7 39.9   <- Four Tet
```

The pooled median of **34.29 bars lands inside that gap** — it describes
neither DJ. Per-set medians are the only meaningful summary, and the profile now
reports `by_set` alongside the pooled figures for exactly this reason.

| | Median | Nearest musical length | Off by |
|---|---|---|---|
| Four Tet | 37.07 bars | 40 | 2.93 bars |
| Tech house | 25.68 bars | **24** | 1.68 bars |

The tech house set is the tighter measurement — IQR of 1.2 bars, with 6 of its
11 blends marked at exactly 48.0 s.

## The finding that matters

**All 26 measured transitions are longer than our 16 bars. The shortest observed
is 22.4.** Our current value sits outside the entire observed range, across two
DJs and two genres.

## What each measurement justifies

### `length_bars` — 16 → 24, APPLIED

The tech house set is the reference that transfers: **128.0 BPM against your
crate's 127 median**, same broad dance/pop family, and by far the more
consistent of the two. It says 25.68 bars, nearest musical length **24**.

Four Tet's 37 bars is real but does not transfer — long-form instrumental
electronica, where a 70-second blend is idiomatic and vocals are not competing.

I withdraw my earlier "keep 16" recommendation. It rested on one set from one
genre; with a second set at your own tempo, in your own genre, saying 25.7
bars, holding a number I invented by ear is no longer defensible. **24 bars**
(45 s at 128 BPM, 21% of a median 214 s track) is the value the evidence
supports.

The vocal-overlap concern I raised is not gone, but it is smaller at 24 bars
than at the 32 I was arguing against, and the blend now ends at `mix_out` — the
outgoing track's outro — which is where pop vocals are sparsest.

### `level_peak_db` — confirms Phase 1, with n=26 across two DJs

**The mix essentially never gets louder than a single deck.** Max across all 26
is **+0.77 dB**; only **2 of 26** exceed 0 dB at all; median −1.54 dB.

Our equal-power crossfade holds combined power at exactly 0 dB, sitting right at
the top of measured practice. The shape it replaced measured **+5 to +9 dB** and
clipped 26–45% of blocks. That is now confirmed against two independent
professional sets.

No change. The value is vindicated, not merely unrefuted.

### `bpm_delta_pct` — cannot tune what it appears to tune

Median 0.50%, IQR 0.34–0.68, consistent across both sets. It is tempting to map
onto `selector.BPM_TOLERANCE` (6%) or `supervisor.MAX_RATE_DELTA` (8%). **That
is a category error.** This measures how far the *master tempo of the mix* moves
across a transition — by which point the DJ has already beatmatched the two
records. Our tolerances govern how far apart two tracks' *native* tempos may be
before we consider them mixable, which is not observable in a mixed recording.

No change, and this should stay unchanged even at n=100.

### `low_dip_db` — real, consistent, still not actionable here

Both sets show a pronounced low-end dip: −18.95 dB (Four Tet), −15.67 dB (tech
house). Our design holds low-end energy flat.

Not acted on, for two reasons unchanged from before: the dips **do not line up**
(each falls at a different point in the blend, so the elementwise-median
trajectory stays within ±3.6 dB and there is no consistent position to encode),
and changing low-band behaviour is an **algorithm** change, which Phase 2
explicitly excludes. It is also not separable from a musical breakdown in one of
the records without stem separation.

Worth revisiting as a Phase 3+ algorithm question, with a note that it now has
two-set support behind it.

## What the change actually produces, measured across all 96 crate tracks

| | |
|---|---|
| Blend length placed | **median 28.0 bars** (range 24–31), 51.9 s |
| Ends within 2 bars of `mix_out` | **96/96** |
| Cuts (no room to blend) | **0/96** |
| Start position | median 73.1% of the track |
| Starts below 70% | 27/96 |
| Starts below 50% | **2/96** |

### Consequence 1: placed length is 28 bars, not 24

`plan_transition` snaps the start down to the 8-bar grid and then stretches the
blend so it still ends exactly on `mix_out`. A 24-bar request therefore places
as 24–31 bars, median 28 — about 2.3 bars above the tech house reference of
25.68.

This is the placement contract working as designed and signed off: *at least*
the requested length, always ending at the outro. If you would rather the placed
median sit on the measured 25.68, the fix is to snap the start to the **nearest**
8-bar line instead of downwards, which would place 20–27 bars and still end on
`mix_out`. That changes placement logic, so I have not done it.

### Consequence 2: the two shortest tracks now collide with the supervisor

The 50% guard added in the placement phase rejects any transition starting
before half the outgoing track. A 52-second blend on a 103-second track
necessarily starts at 45%:

| Track | Duration | Start | Blend |
|---|---|---|---|
| 54_Levels_x_Slide_-_Medley_Version | 102.9 s | **45.1%** | 28 bars / 53 s |
| 17_There_She_Goes | 121.9 s | **49.4%** | 31 bars / 58 s |

On those two the automatic transition is rejected, the autopilot retries every
0.5 s until the track ends, and `_recover_from_silence` hard-starts the next
track — a cut rather than a blend. No dead air, but no blend either, plus
repeated rejection entries in the session log.

At 16 bars this did not happen.

**Fixed:** `supervisor.MIN_TRANSITION_START_FRACTION` lowered **0.50 → 0.40**.
The guard is about placement being anchored to mix-out, not about a track's
absolute length; a 45% start that still ends exactly on mix-out is not the fault
it was written to catch. It stays tight enough to catch the original bug, which
placed hand-offs at the first phrase boundary after selection — bar 32 of a
130-bar track is 25%.

After the change, across all 96 tracks: **0 rejected**, minimum start 45.1%,
median 73.1%. A deliberate bar-20 placement (19.7%) is still caught.

The alternative — capping the blend at a fraction of the outgoing track's
length, so short tracks get proportionally shorter blends — is musically better
and remains open. It is an algorithm change, so it belongs in a later phase.

## How the measurement was made trustworthy

Blend length was originally *inferred* from how the mix's timbre and harmony
change. Validated against constructed ground truth it was out by up to **+38
bars**, and BPM delta by **−15**. Start/end markers removed the inference
entirely: length is `end − start`, and the tempo and level references sit in
stretches that provably contain one track each.

The inference still runs as `fit_r2`, a quality signal only. Its median is
**0.50** across these 26 — independent confirmation that inferring blend width
from a mixed recording does not work, even when the true answer is known.

Four bugs were found and fixed getting here: the analysis window was narrower
than a 32-bar blend, so both reference stretches sat inside the blend; widening
it put them inside the *neighbouring* transitions; the low-band envelope was
smoothed over 0.1 s, so its "dip" tracked the gap between kick drums; and the
fourth was in the validation rig rather than the tool.

## Current values

| Config | Value | Source | Measurement | Status |
|---|---|---|---|---|
| `config.TRANSITION_BARS` | **24.0** | measured, Phase 2 | 25.68 bars (tech house), n=11 of 26 | **changed from 16.0** |
| `phrase.PLACEMENT_GRID_BARS` | 8 | Phase 1 | no grid alignment found | unchanged |
| `transition.BASS_SWAP_AT` | 0.5 | Phase 1 | dips vary in position | unchanged |
| `selector.BPM_TOLERANCE` | 0.06 | v1 | not measurable from a mix | unchanged |
| `supervisor.MAX_RATE_DELTA` | 0.08 | v1 | not measurable from a mix | unchanged |

## Running it

```bash
python -m djai reference ./sets ./sets            # a folder of markers files
python -m djai reference ./sets ./sets/markers.txt --out reference_profile.json
```

The markers argument may be a single `.txt` or a folder of them, one per set.
Markers may give a centre per transition, or a start and an end. **Start/end
pairs are detected, not declared**: if every line carries two increasing
timestamps less than 300 s apart, the file is read as boundaries. Prefer
boundaries — they are what makes the length measurement trustworthy.

```
# file            start       end
set_one.mp3       2:12        3:00
set_one.mp3       14:24       15:30
```

`reference_profile.json` holds median, IQR, min and max per measure both pooled
and `by_set`, the grid-alignment histogram, the elementwise-median low-band
trajectory, every per-transition row, and every rejected marker with its reason.
It warns when fewer than 30 transitions were usable.
