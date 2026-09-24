# djai — an AI-mixed DJ

A real-time two-deck DJ program in Python. It beatmatches, phrase-aligns and
crossfades between tracks on its own, and you steer it by typing plain English
at a prompt — "something with more energy", "bring the bass back in", "cut". A
local llama3.1:8b model running under Ollama turns that text into commands; no
API key is needed and nothing leaves your machine. The design problem it solves
is that an LLM answers in seconds while an audio callback has milliseconds, so
inference is kept entirely off the audio path: the engine never blocks on the
model, and if the model is slow or absent, typed input falls back to keyword
matching and the music keeps playing. There is a DJ-controller-style web UI
(`--ui`) alongside the terminal REPL.

For the full architecture write-up — the threading model, the scheduler, the
transition library, the supervisor and the reasoning behind each — see
**[docs/DESIGN.md](docs/DESIGN.md)**.

## Prerequisites

- **Python 3.11+** (developed on 3.12)
- **Ollama** — install from [ollama.com/download](https://ollama.com/download), then:
  ```bash
  ollama pull llama3.1:8b
  ollama list                              # llama3.1:8b should appear
  curl http://localhost:11434/api/tags     # should return JSON
  ```
  On Windows the tray app starts the server; otherwise run `ollama serve`.
  `ollama ps` shows whether the model is resident on the GPU.
- **Rubber Band CLI** (optional but recommended) — used for pitch-preserving
  time stretch (key lock). Install it so that `rubberband` is on your `PATH`:
  macOS `brew install rubberband`, Debian/Ubuntu `apt install rubberband-cli`,
  Windows: download from [breakfastquay.com/rubberband](https://breakfastquay.com/rubberband/).
  Without it, decks resample to match tempo instead, which shifts pitch. The
  program still runs and logs the fallback.
- **An audio output device.** `python -m djai devices` lists what it can see.

## Install

```bash
git clone https://github.com/PeterKretschmer/djai/
cd djai-github
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .                 # or: pip install -r requirements.txt
```

`requirements.txt` holds exact pinned versions; `pyproject.toml` holds the
looser ranges. Either works.

Optional extras:

```bash
pip install -e ".[dev]"          # pytest
pip install -e ".[beats]"        # Beat This! grids + HDemucs stems (several GB)
```

The `beats` extra pulls in PyTorch. For GPU support install it from the CUDA
index first — and note that stem separation also needs `torchaudio`:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[beats]"
```

## Where to put your music

Create a `tracks/` folder in the project root and put your audio files in it
(`.mp3`, `.wav`, `.flac`, `.m4a`). It is gitignored, so nothing you drop there
will be committed.

```
djai-github/
├── tracks/        <- your music goes here
├── cache/         <- analysis results, created automatically
├── logs/          <- session logs, created automatically
├── recordings/    <- only if you use --record
└── renders/       <- only if you use render --keep-renders
```

Analyse the library once before the first play. This is offline, writes a
`.json` + `.npz` per track into `cache/`, and is the slow step:

```bash
python -m djai analyze ./tracks/
```

Any path works — `tracks/` is just the default convention.

## Run

```bash
python -m djai devices                 # list usable output devices
python -m djai play                    # engine + terminal REPL
python -m djai play --ui               # also serve the web UI
python -m djai play --device 2 --blocksize 512
python -m djai play --record           # record the master to recordings/
```

Then type at the prompt:

```
Now playing: 03_driver  125.0 BPM  1B (B major)
djai> something with more energy
Going up - cueing a harder track for the next phrase.
Cued: 09_stepper (128.0 BPM)
Bass swap fires in 23 bars.
djai> cut
cut.
```

`cut` is the panic path and is always honoured immediately, even mid-inference.

## Configuration

Everything is optional environment variables — see
[`.env.example`](.env.example) for all 52 with their defaults, and
[docs/tuning.md](docs/tuning.md) for what is worth tuning. The ones you are
most likely to touch:

| Variable | Default | What it does |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | where the Ollama server is |
| `OLLAMA_MODEL` | `llama3.1:8b` | the model to use |
| `OLLAMA_TIMEOUT_S` | `8` | ceiling on one request; past it, keywords take over |
| `AUDIO_BLOCKSIZE` | `2048` | frames per audio callback; lower is tighter but riskier |
| `LOUDNESS_TARGET_LUFS` | `-14` | loudness every track is normalised to at load |
| `TIME_STRETCH_ENABLED` | `1` | `0` makes every deck resample instead |
| `KEY_LOCK_DEFAULT` | `1` | whether decks start with key lock on |

The program reads the process environment directly; it does not parse `.env`
itself. Export the variables, or use your shell's / IDE's `.env` support.

## Tests

The suite is large and memory-hungry; run it a file at a time rather than all
at once:

```bash
python -m pytest tests/test_deck.py
```

## Licence

The Rubber Band CLI, used at run time for key lock, is GPL-2.0-or-later and is
not distributed with this repository — install it separately as described above.
