"""Synthetic 4/4 test tracks with known BPM, key and phrase structure.

THREADING CONTEXT: main thread (test fixtures / manual verification only).

These exist so the engine, analyser and transition can be exercised without a
music library. Each track is a plain four-on-the-floor loop: kick on every
beat, clap on 2 and 4, closed hat on the off-eighths, a root-note bassline, and
a triad pad that changes on every bar -- which is what gives the downbeat
estimator something real to lock onto.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

BEATS_PER_BAR = 4

_NOTE_OFFSET = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
}


def _midi_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _env(n: int, attack: int, decay: float, sr: int) -> np.ndarray:
    """Percussive envelope: short linear attack, exponential decay."""
    e = np.exp(-np.arange(n) / (decay * sr))
    a = min(attack, n)
    if a > 0:
        e[:a] *= np.linspace(0.0, 1.0, a)
    return e


def _kick(sr: int) -> np.ndarray:
    n = int(0.28 * sr)
    t = np.arange(n) / sr
    # Pitch sweep 110 Hz -> 45 Hz gives a body with real low-band content, which
    # is what the bass-swap transition needs in order to be audible.
    f = 45.0 + 65.0 * np.exp(-t / 0.03)
    phase = 2 * np.pi * np.cumsum(f) / sr
    return (np.sin(phase) * _env(n, 8, 0.09, sr) * 0.95).astype(np.float64)


def _clap(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.16 * sr)
    noise = rng.standard_normal(n)
    # Crude band-pass by differencing (highs) and smoothing (rolls off the top).
    noise = np.diff(noise, prepend=0.0)
    k = np.ones(6) / 6
    noise = np.convolve(noise, k, mode="same")
    return (noise * _env(n, 4, 0.045, sr) * 0.42).astype(np.float64)


def _hat(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.05 * sr)
    noise = np.diff(rng.standard_normal(n), prepend=0.0)
    return (noise * _env(n, 2, 0.012, sr) * 0.16).astype(np.float64)


def _add(dst: np.ndarray, src: np.ndarray, at: int) -> None:
    if at >= dst.size:
        return
    end = min(at + src.size, dst.size)
    dst[at:end] += src[: end - at]


def render_track(
    path: Path,
    bpm: float,
    root: str = "A",
    minor: bool = True,
    bars: int = 128,
    sr: int = 44100,
    stereo: bool = True,
    energy: float = 1.0,
    seed: int = 0,
    drift: float = 0.0,
) -> Path:
    """Render one synthetic track to ``path``. Returns the path.

    ``drift`` is a fractional tempo wander (0.02 = +-2% sinusoidal) used to give
    ``grid_confidence`` something to actually discriminate; 0 renders a
    machine-tight grid.
    """
    rng = np.random.default_rng(seed)
    spb = 60.0 / bpm
    total = int(math.ceil(bars * BEATS_PER_BAR * spb * sr * (1.0 + 2 * drift))) + sr
    buf = np.zeros(total, dtype=np.float64)

    def beat_time(n: float) -> float:
        """Time of beat ``n``, integrating a slow sinusoidal tempo wander."""
        if drift == 0.0:
            return n * spb
        # Integral of spb * (1 + drift*sin(2*pi*n/period)) over beats 0..n.
        period = 32.0
        return spb * (
            n + drift * period / (2 * math.pi) * (1 - math.cos(2 * math.pi * n / period))
        )

    kick, clap, hat = _kick(sr), _clap(sr, rng), _hat(sr, rng)

    # _NOTE_OFFSET is relative to C, so the base must be a C. MIDI 36 = C2.
    root_midi = 36 + _NOTE_OFFSET[root]
    # One chord per bar, as (semitones above tonic, third interval). The triad
    # quality has to vary per degree or the result is not in any key at all:
    # i-VI-III-VII in A minor is Am-F-C-G, not Am-Fm-Cm-Gm.
    if minor:
        degrees = [(0, 3), (8, 4), (3, 4), (10, 4)]  # i  VI III VII
    else:
        degrees = [(0, 4), (9, 3), (5, 4), (7, 4)]  # I  vi IV  V

    for bar in range(bars):
        bar_t = beat_time(bar * BEATS_PER_BAR)
        bar_end = beat_time((bar + 1) * BEATS_PER_BAR)
        degree, third = degrees[bar % len(degrees)]
        chord_root = root_midi + degree

        # Pad: a sustained triad for the whole bar. This is the harmonic signal
        # the beat-synchronous chroma downbeat estimator keys off.
        pad_n = int((bar_end - bar_t) * sr)
        t = np.arange(pad_n) / sr
        pad = np.zeros(pad_n)
        for semi in (0, third, 7):
            f = _midi_hz(chord_root + semi + 24)
            pad += np.sin(2 * np.pi * f * t) * 0.09
        # Soft edges so bar boundaries do not click.
        ramp = int(0.02 * sr)
        pad[:ramp] *= np.linspace(0, 1, ramp)
        pad[-ramp:] *= np.linspace(1, 0, ramp)
        _add(buf, pad * energy, int(bar_t * sr))

        for beat in range(BEATS_PER_BAR):
            n = bar * BEATS_PER_BAR + beat
            bt = beat_time(n)
            i = int(bt * sr)
            _add(buf, kick, i)
            if beat in (1, 3):
                _add(buf, clap * energy, i)
            _add(buf, hat * energy, int(beat_time(n + 0.5) * sr))

            # Bassline: root on 1 and 3, fifth on the and-of-4.
            bass_midi = chord_root + (0 if beat != 3 else 7)
            bn = int(spb * 0.45 * sr)
            tb = np.arange(bn) / sr
            f = _midi_hz(bass_midi)
            bass = np.sin(2 * np.pi * f * tb) * _env(bn, 32, 0.12, sr) * 0.5
            _add(buf, bass, i)

    peak = float(np.max(np.abs(buf)))
    if peak > 0:
        buf *= 0.89 / peak

    if stereo:
        # Tiny inter-channel decorrelation so stereo handling is exercised.
        out = np.stack([buf, np.roll(buf, 13) * 0.98], axis=1)
    else:
        out = buf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), out.astype(np.float32), sr, subtype="PCM_16")
    return path


#: A crate that spans a usable BPM range and a run of adjacent Camelot keys, so
#: the selector has both compatible and incompatible candidates to reject.
CRATE_SPEC: list[dict] = [
    {"name": "01_deep_start",   "bpm": 122.0, "root": "A",  "minor": True,  "energy": 0.75},
    {"name": "02_rolling",      "bpm": 124.0, "root": "E",  "minor": True,  "energy": 0.85},
    {"name": "03_driver",       "bpm": 125.0, "root": "B",  "minor": True,  "energy": 1.00},
    {"name": "04_peak",         "bpm": 126.0, "root": "F#", "minor": True,  "energy": 1.15},
    {"name": "05_hypnotic",     "bpm": 124.0, "root": "C#", "minor": True,  "energy": 0.95},
    {"name": "06_warm",         "bpm": 123.0, "root": "G",  "minor": False, "energy": 0.70},
    {"name": "07_late",         "bpm": 127.0, "root": "D",  "minor": False, "energy": 1.10},
    {"name": "08_dubby",        "bpm": 121.0, "root": "D",  "minor": True,  "energy": 0.65},
    {"name": "09_stepper",      "bpm": 128.0, "root": "A",  "minor": False, "energy": 1.20},
    {"name": "10_closer",       "bpm": 120.0, "root": "G",  "minor": True,  "energy": 0.60},
    # Deliberately sloppy: 2.5% tempo wander, so grid_confidence has a genuine
    # low-confidence case to separate from the machine-tight tracks.
    {"name": "11_offgrid",      "bpm": 130.0, "root": "F",  "minor": True,  "energy": 1.05,
     "drift": 0.025},
    {"name": "12_outlier",      "bpm": 145.0, "root": "C",  "minor": False, "energy": 1.30},
]


def build_crate(folder: Path, bars: int = 64, sr: int = 44100) -> list[Path]:
    """Render the whole synthetic crate into ``folder``, skipping existing files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for i, spec in enumerate(CRATE_SPEC):
        p = folder / f"{spec['name']}.wav"
        if not p.exists():
            render_track(
                p,
                bpm=spec["bpm"],
                root=spec["root"],
                minor=spec["minor"],
                bars=bars,
                sr=sr,
                # Mix in a couple of mono / 48k files to exercise the loader.
                stereo=(i % 5 != 3),
                energy=spec["energy"],
                seed=i,
                drift=spec.get("drift", 0.0),
            )
        out.append(p)
    return out


# --- structured tracks: known sections and known vocals ----------------------
#
# The ground truth for Phase 3's structure and vocal detection. Each section is
# built from a different instrumentation and level, the way a dance track's
# sections differ, and a crude sung line is laid over the sections listed as
# vocal. Every bar's true label and vocal flag is returned alongside the file.

#: Level of each part per section: kick, clap, hat, bass, pad, riser.
SECTION_PARTS = {
    "intro":     (0.8, 0.0, 0.5, 0.0, 0.35, 0.0),
    "build":     (0.6, 0.6, 0.7, 0.0, 0.5, 1.0),
    "drop":      (1.0, 1.0, 1.0, 1.0, 0.9, 0.0),
    "breakdown": (0.0, 0.0, 0.2, 0.0, 0.8, 0.0),
    "outro":     (0.8, 0.0, 0.4, 0.3, 0.3, 0.0),
}

#: Ten layouts ``([(label, bars), ...], vocal_section_indices)`` and tempos.
STRUCTURED_LAYOUTS: list[tuple[list[tuple[str, int]], tuple[int, ...]]] = [
    ([("intro", 16), ("build", 8), ("drop", 16), ("breakdown", 16), ("build", 8), ("drop", 16), ("outro", 16)], (3,)),
    ([("intro", 8), ("drop", 16), ("breakdown", 8), ("build", 8), ("drop", 24), ("outro", 16)], (2, 4)),
    ([("intro", 16), ("build", 8), ("drop", 32), ("outro", 16)], (2,)),
    ([("intro", 16), ("drop", 16), ("breakdown", 16), ("drop", 16), ("outro", 8)], ()),
    ([("intro", 8), ("build", 8), ("drop", 16), ("breakdown", 16), ("build", 8), ("drop", 16), ("breakdown", 8), ("outro", 16)], (1, 3, 5)),
    ([("intro", 32), ("build", 8), ("drop", 16), ("outro", 16)], (0,)),
    ([("intro", 16), ("drop", 32), ("breakdown", 8), ("drop", 16), ("outro", 16)], (2,)),
    ([("intro", 8), ("build", 16), ("drop", 16), ("breakdown", 16), ("build", 8), ("drop", 16), ("outro", 8)], (3, 4)),
    ([("intro", 16), ("build", 8), ("drop", 16), ("drop", 16), ("outro", 16)], (2,)),
    ([("intro", 16), ("breakdown", 16), ("build", 8), ("drop", 24), ("outro", 16)], (1,)),
]
STRUCTURED_BPMS: list[float] = [124, 126, 128, 122, 130, 125, 127, 123, 128, 126]


def _burst(rng: np.random.Generator, seconds: float, decay: float, gain: float,
           sr: int, bright: bool = False) -> np.ndarray:
    n = int(seconds * sr)
    x = rng.standard_normal(n)
    if bright:
        x = np.diff(x, prepend=0.0)
    return x * _env(n, 2, decay, sr) * gain


def _sung_line(rng: np.random.Generator, seconds: float, root_hz: float, sr: int) -> np.ndarray:
    """A crude voice: a vibrato sawtooth through three formants, in syllables."""
    from scipy.signal import butter, sosfilt

    n = int(seconds * sr)
    t = np.arange(n) / sr
    raw = np.zeros(n)
    pos = 0
    scale = (0, 2, 3, 5, 7, 8, 10, 12)
    while pos < n:
        syllable = int(rng.uniform(0.18, 0.45) * sr)
        gap = int(rng.uniform(0.03, 0.12) * sr)
        f0 = root_hz * 2 ** (scale[rng.integers(len(scale))] / 12)
        tt = t[: min(syllable, n - pos)]
        if tt.size:
            phase = np.cumsum(f0 * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * tt))) / sr
            shape = np.minimum(1, np.minimum(tt / 0.02, (tt[-1] - tt + 1e-3) / 0.04))
            raw[pos:pos + tt.size] = (2 * (phase % 1.0) - 1) * shape
        pos += syllable + gap
    voice = np.zeros(n)
    for centre, width, gain in ((700, 300, 1.0), (1200, 400, 0.6), (2600, 600, 0.35)):
        sos = butter(2, [centre - width / 2, centre + width / 2], btype="band", fs=sr, output="sos")
        voice += gain * sosfilt(sos, raw)
    return voice / (np.max(np.abs(voice)) + 1e-9)


def structured_track(
    path: Path,
    bpm: float,
    layout: list[tuple[str, int]],
    vocal_sections: tuple[int, ...] = (),
    seed: int = 0,
    root_midi: int = 45,
    sr: int = 44100,
) -> tuple[list[str], list[bool]]:
    """Render a track with a known layout. Returns per-bar labels and vocal flags."""
    rng = np.random.default_rng(seed)
    spb = 60.0 / bpm
    bars = sum(b for _, b in layout)
    buf = np.zeros(int(bars * BEATS_PER_BAR * spb * sr) + sr)
    kick = _kick(sr)
    labels: list[str] = []
    vocals: list[bool] = []
    bar = 0
    for index, (label, n_bars) in enumerate(layout):
        k, c, h, bass, pad, riser = SECTION_PARTS[label]
        start = bar * BEATS_PER_BAR * spb
        for b in range(n_bars):
            bt = (bar + b) * BEATS_PER_BAR * spb
            progress = b / max(1, n_bars - 1)
            degree = (0, 8, 3, 10)[(bar + b) % 4]
            pad_n = int(BEATS_PER_BAR * spb * sr)
            tp = np.arange(pad_n) / sr
            chord = sum(
                np.sin(2 * np.pi * _midi_hz(root_midi + degree + s + 12) * tp) for s in (0, 3, 7)
            )
            ramp = int(0.02 * sr)
            chord[:ramp] *= np.linspace(0, 1, ramp)
            chord[-ramp:] *= np.linspace(1, 0, ramp)
            _add(buf, chord * 0.07 * pad, int(bt * sr))
            for beat in range(BEATS_PER_BAR):
                i = int((bt + beat * spb) * sr)
                if k:
                    _add(buf, kick * k, i)
                if c and beat in (1, 3):
                    _add(buf, _burst(rng, 0.16, 0.045, 0.4 * c, sr), i)
                if h:
                    _add(buf, _burst(rng, 0.05, 0.012, 0.16 * h, sr, bright=True),
                         int((bt + (beat + 0.5) * spb) * sr))
                if label == "build":
                    div = 1 if progress < 0.5 else 2 if progress < 0.85 else 4
                    for d in range(div):
                        _add(buf, _burst(rng, 0.08, 0.03, 0.25 * (0.4 + 0.6 * progress), sr),
                             int((bt + (beat + d / div) * spb) * sr))
                if bass:
                    bn = int(spb * 0.45 * sr)
                    tb = np.arange(bn) / sr
                    tone = np.sin(2 * np.pi * _midi_hz(root_midi - 12 + degree) * tb)
                    _add(buf, tone * _env(bn, 32, 0.12, sr) * 0.5 * bass, i)
        seg_n = int(n_bars * BEATS_PER_BAR * spb * sr)
        if riser:
            _add(buf, rng.standard_normal(seg_n) * np.linspace(0, 1, seg_n) ** 2 * 0.12 * riser,
                 int(start * sr))
        if index in vocal_sections:
            line = _sung_line(rng, n_bars * BEATS_PER_BAR * spb, _midi_hz(root_midi + 12), sr)
            _add(buf, line * 0.22, int(start * sr))
        labels += [label] * n_bars
        vocals += [index in vocal_sections] * n_bars
        bar += n_bars
    buf *= 0.89 / np.max(np.abs(buf))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.stack([buf, buf], axis=1).astype(np.float32), sr, subtype="PCM_16")
    return labels, vocals


if __name__ == "__main__":  # pragma: no cover - manual fixture generation
    import sys

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tracks")
    n_bars = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    for p in build_crate(dest, bars=n_bars):
        print(p)
