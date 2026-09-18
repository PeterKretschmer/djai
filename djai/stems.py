"""Stem separation, offline, and the mixes the decks play.

THREADING CONTEXT: the ``stems`` subcommand separates (slow, GPU, blocking) and
:func:`load_mix` builds a mix on a **control thread** -- the cue, never the
audio thread and never the audio callback. Separation during playback is
forbidden: it is minutes of work per set and it would be holding the GIL while
a room waits.

What is stored, and why it is stored that way
---------------------------------------------
Four stems per track -- drums, bass, other, vocals -- under
``<cache>/stems/<track_id>/``, keyed by the same content hash the analysis
cache uses, so renaming or moving a file still hits it. FLAC 16-bit, ~20 MB a
minute for all four: the 105-track crate measured 6.5 GB and 20.3 minutes of
GPU, against 29 GB had it been float32 WAV.

One gain for the whole track is applied before writing and undone on load.
Separation routinely puts more energy in a stem than the limited mix has --
measured peaks of 1.36, 1.79 and 1.68 on three tracks -- so 16-bit without
headroom clipped tens of thousands of samples. Five dB off a 96 dB range still
leaves ~91 dB of signal to noise, which no room hears.

The stems do NOT sum back to the original exactly: measured -18.8 to -26.5 dB
of residual. That is the model, not the storage, and it is why a mix built here
is used for a transition's few bars rather than as the track's normal audio.

Model: torchaudio's HDEMUCS_HIGH_MUSDB_PLUS (83.6 M parameters, 319 MB), which
needs no dependency beyond the PyTorch that Phase 1's beat tracker already
installs. Measured 11 s per track across the crate on an RTX 3060 (3.3-5.3 s
with nothing else running) at 755 MiB of VRAM.
Its weights were trained on MUSDB-HQ, whose audio is non-commercial: fine for
a personal crate, a licensing question if this is ever distributed.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from djai.deck import CHANNELS, SAMPLE_RATE, LoadedTrack

log = logging.getLogger(__name__)

#: The bundle this cache was built with. A different model invalidates an entry.
MODEL_NAME = "HDEMUCS_HIGH_MUSDB_PLUS"

#: Stem names, in the order the model returns them.
STEM_NAMES: tuple[str, ...] = ("drums", "bass", "other", "vocals")

#: Seconds of audio per forward pass, and the crossfaded overlap between them.
#: Ten-second segments kept peak VRAM at 755 MiB, so the model fits beside an
#: 8B language model on a 12 GB card -- though the precompute run stops Ollama
#: anyway, being offline work.
SEGMENT_S: float = 10.0
OVERLAP_S: float = 0.1

#: Below this stored gain -- more than about 12 dB of headroom -- a track is
#: written 24-bit. Measured: 1 track in 105 needs it, at 1.5x the size.
DEEP_HEADROOM_GAIN: float = 0.25

#: Named mixes, as the stems they sum. These are what a deck actually plays
#: during a stem move: one array, built here, swapped in at a bar line.
MIXES: dict[str, tuple[str, ...]] = {
    "full": STEM_NAMES,
    "instrumental": ("drums", "bass", "other"),
    "acapella": ("vocals",),
    "no_bass": ("drums", "other", "vocals"),
    "no_drums": ("bass", "other", "vocals"),
    "drums_only": ("drums",),
    #: What is left of a track once the next one carries both the low end and
    #: the groove: the end of a progressive strip-down, full -> no_bass -> this.
    "vocals_and_other": ("other", "vocals"),
}

_model = None
_model_lock = threading.Lock()


def available() -> bool:
    """Is separation possible here? Imports nothing."""
    return (
        importlib.util.find_spec("torchaudio") is not None
        and importlib.util.find_spec("torch") is not None
    )


def stems_dir(track_id: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / "stems" / track_id


def manifest_path(track_id: str, cache_dir: Path) -> Path:
    return stems_dir(track_id, cache_dir) / "stems.json"


def cached(track_id: str, cache_dir: Path) -> dict | None:
    """The manifest for a track's stems, or None if they are missing or stale."""
    path = manifest_path(track_id, cache_dir)
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("model") != MODEL_NAME:
        return None
    folder = stems_dir(track_id, cache_dir)
    if any(not (folder / f"{name}.flac").is_file() for name in STEM_NAMES):
        return None
    return manifest


def _load_model():
    """The separation model, loaded once per process. Imports torch."""
    global _model
    import torch
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS as bundle

    with _model_lock:
        if _model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = bundle.get_model().to(device).eval()
            _model = (model, device, list(model.sources))
    return _model


def separate(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict[str, np.ndarray]:
    """Split ``(frames, channels)`` audio into named stems. SLOW, blocking.

    Segmented so peak memory does not scale with the track, with the segment
    edges crossfaded: a hard join leaves a click at every seam.
    """
    import torch

    model, device, sources = _load_model()
    mix = torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32))
    # The model was trained on standardised input; it is undone below.
    mean, std = float(mix.mean()), float(mix.std()) or 1.0
    mix = (mix - mean) / std

    seg = int(SEGMENT_S * sample_rate)
    overlap = int(OVERLAP_S * sample_rate)
    out = torch.zeros(len(sources), *mix.shape)
    weight = torch.zeros(mix.shape[-1])
    ramp = torch.linspace(0.0, 1.0, overlap) if overlap else None
    start = 0
    while start < mix.shape[-1]:
        end = min(start + seg, mix.shape[-1])
        with torch.no_grad():
            got = model(mix[:, start:end].to(device)[None])[0].cpu()
        win = torch.ones(end - start)
        if ramp is not None and start > 0:
            win[:overlap] = ramp
        if ramp is not None and end < mix.shape[-1]:
            win[-overlap:] = ramp.flip(0)
        out[:, :, start:end] += got * win
        weight[start:end] += win
        if end == mix.shape[-1]:
            break
        start = end - overlap
    out = out / weight.clamp(min=1e-8) * std + mean
    return {name: out[i].numpy().T.copy() for i, name in enumerate(sources)}


def precompute(analysis, cache_dir: Path, force: bool = False) -> dict:
    """Separate one track into the cache and return its manifest. SLOW.

    Skips a track whose stems are already there for this model unless forced.
    """
    cache_dir = Path(cache_dir)
    if not force:
        existing = cached(analysis.track_id, cache_dir)
        if existing is not None:
            return existing

    audio, sample_rate = sf.read(analysis.path, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, CHANNELS, axis=1)
    audio = audio[:, :CHANNELS]
    parts = separate(audio, sample_rate)

    # One shared gain, so the stems still sum to the mix after the round trip.
    # A stem can be much louder than the limited master it came from -- other
    # stems cancel against it -- and it is not a stray sample: one track's drums
    # spend 14.5 s above 1.0, peaking at 5.0, another peaks at 14.2. So the gain
    # follows the true peak, and a track that needs more than 12 dB of it is
    # written 24-bit rather than spending that much of a 16-bit floor.
    loudest = max(float(np.max(np.abs(part))) for part in parts.values())
    gain = 1.0 / loudest if loudest > 1.0 else 1.0
    subtype = "PCM_24" if gain < DEEP_HEADROOM_GAIN else "PCM_16"
    folder = stems_dir(analysis.track_id, cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": MODEL_NAME,
        "track_id": analysis.track_id,
        "title": analysis.title,
        "sample_rate": sample_rate,
        "duration_s": round(audio.shape[0] / sample_rate, 2),
        "gain": round(gain, 6),
        "subtype": subtype,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stems": {},
    }
    for name in STEM_NAMES:
        part = parts[name]
        path = folder / f"{name}.flac"
        sf.write(str(path), part * gain, sample_rate, subtype=subtype)
        manifest["stems"][name] = {
            "peak": round(float(np.max(np.abs(part))), 4),
            "bytes": path.stat().st_size,
        }
    manifest_path(analysis.track_id, cache_dir).write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    return manifest


def load_mix(
    track: LoadedTrack, cache_dir: Path, mix: str
) -> LoadedTrack | None:
    """A named mix of a cued track's stems, ready to swap under the playhead.

    CONTROL THREAD: decodes FLAC and sums. None when the stems are missing, or
    when the mix would be silent. The result carries the SAME analysis object,
    which is what lets :meth:`djai.deck.Deck.swap_audio` accept it -- the grid,
    the cues and the mix points are the track's, whatever is playing.
    """
    names = MIXES.get(mix)
    if names is None:
        raise ValueError(f"unknown stem mix {mix!r}; one of {sorted(MIXES)}")
    analysis = track.analysis
    manifest = cached(analysis.track_id, Path(cache_dir))
    if manifest is None:
        return None
    folder = stems_dir(analysis.track_id, Path(cache_dir))
    gain = float(manifest.get("gain", 1.0)) or 1.0
    total: np.ndarray | None = None
    for name in names:
        data, _sr = sf.read(str(folder / f"{name}.flac"), dtype="float32", always_2d=True)
        if total is None:
            total = data
        else:
            n = min(total.shape[0], data.shape[0])
            total = total[:n] + data[:n]
    if total is None:
        return None
    audio = np.ascontiguousarray(total / gain, dtype=np.float32)
    # A mix of stems can be louder than the master it came from, which would
    # hand the limiter a level the track never had and make the swap jump. It
    # may be quieter -- an acapella is -- but never louder.
    mix_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    original_peak = float(np.max(np.abs(track.audio))) if track.audio.size else 0.0
    if mix_peak > original_peak > 0.0:
        audio *= original_peak / mix_peak
    # The deck reads two guard frames past the end when interpolating, and the
    # original was padded the same way by load_track.
    want = track.audio.shape[0]
    if audio.shape[0] < want:
        audio = np.pad(audio, ((0, want - audio.shape[0]), (0, 0)))
    else:
        audio = audio[:want]
    return LoadedTrack(analysis=analysis, audio=audio)
