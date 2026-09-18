"""Phase 5: importing grids, cues and playlists from Rekordbox and Serato.

THREADING CONTEXT: main thread (pytest). Real files throughout: synthetic
tracks rendered to disk, a Rekordbox XML export written the way Rekordbox
writes one, and a Serato crate plus an MP3 carrying Serato's own ID3 tags.

The source files are built here with known grids and cues, so every imported
value is checked against the number that was written.
"""

from __future__ import annotations

import base64
import json
import struct
import urllib.parse
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from djai import analysis as an
from djai import cli
from djai import library_import as li
from tests import synth

BPM = 128.0
BEAT = 60.0 / BPM
#: The Rekordbox marker sits on beat 2 of its bar at 0.1 s, so bar 1 is three
#: beats later: a grid analysis would not have found by itself.
INIZIO = 0.1
BATTITO = 2
FIRST_DOWNBEAT = INIZIO + 3 * BEAT


# --- a Rekordbox export --------------------------------------------------------


def _file_url(path: Path) -> str:
    posix = path.resolve().as_posix()
    if not posix.startswith("/"):
        posix = "/" + posix
    return "file://localhost" + urllib.parse.quote(posix, safe="/:")


def _write_rekordbox_xml(xml_path: Path, audio: Path, missing: Path) -> Path:
    xml_path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="rekordbox" Version="6.8.5" Company="AlphaTheta"/>
  <COLLECTION Entries="2">
    <TRACK TrackID="11" Name="Test Track" Artist="Tester" TotalTime="60" AverageBpm="{BPM:.2f}"
           Location="{_file_url(audio)}">
      <TEMPO Inizio="{INIZIO:.3f}" Bpm="{BPM:.2f}" Metro="4/4" Battito="{BATTITO}"/>
      <POSITION_MARK Name="Intro" Type="0" Start="{FIRST_DOWNBEAT:.3f}" Num="0" Red="40" Green="226" Blue="20"/>
      <POSITION_MARK Name="Drop" Type="0" Start="16.106" Num="2"/>
      <POSITION_MARK Name="Roll" Type="4" Start="20.000" End="21.875" Num="3"/>
      <POSITION_MARK Name="" Type="0" Start="30.000" Num="-1"/>
    </TRACK>
    <TRACK TrackID="12" Name="Gone" Artist="Tester" TotalTime="60" AverageBpm="126.00"
           Location="{_file_url(missing)}">
      <TEMPO Inizio="0.000" Bpm="126.00" Metro="4/4" Battito="1"/>
      <TEMPO Inizio="30.000" Bpm="127.50" Metro="4/4" Battito="1"/>
    </TRACK>
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT" Count="1">
      <NODE Type="0" Name="Sets" Count="1">
        <NODE Name="Peak Hour" Type="1" KeyType="0" Entries="2">
          <TRACK Key="11"/>
          <TRACK Key="12"/>
        </NODE>
      </NODE>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>
""", encoding="utf-8")
    return xml_path


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """A folder with a space in its name, one real track, one missing one."""
    root = tmp_path_factory.mktemp("rb") / "My Music"
    root.mkdir()
    audio = synth.render_track(root / "test track.wav", bpm=BPM, bars=32, seed=21)
    xml = _write_rekordbox_xml(root / "rekordbox.xml", audio, root / "gone.wav")
    return root, audio, xml


def test_the_rekordbox_xml_is_read_as_written(library):
    root, audio, xml = library
    tracks = {t.title: t for t in li.parse_rekordbox_xml(xml)}
    track = tracks["Test Track"]
    assert Path(track.path).resolve() == audio.resolve(), "file:// URL with %20 and a drive"
    assert track.bpm == pytest.approx(BPM)
    assert track.first_downbeat_s == pytest.approx(FIRST_DOWNBEAT, abs=1e-9)
    assert track.hot_cues == [
        (1, pytest.approx(round(FIRST_DOWNBEAT, 3)), "Intro"),
        (3, pytest.approx(16.106), "drop"),
        (4, pytest.approx(20.0), "Roll"),
    ]
    assert track.memory_cues_skipped == 1
    assert track.playlists == ["Sets / Peak Hour"]
    assert tracks["Gone"].variable_tempo


@pytest.fixture(scope="module")
def imported(library, tmp_path_factory):
    root, audio, xml = library
    cache = tmp_path_factory.mktemp("rb_cache")
    summaries = li.apply_import(li.parse_rekordbox_xml(xml), cache)
    return cache, summaries


def test_a_rekordbox_import_writes_the_source_grid_and_cues(library, imported):
    root, audio, xml = library
    cache, summaries = imported
    ta = an.load_cached(an.track_hash(audio), cache)
    assert ta is not None
    assert ta.bpm == pytest.approx(BPM, abs=1e-4)
    assert ta.first_downbeat == pytest.approx(FIRST_DOWNBEAT, abs=1e-3)
    # Every beat of the stored grid is on the source grid.
    beats = np.asarray(ta.beats)
    offsets = (beats - FIRST_DOWNBEAT) / BEAT
    assert np.allclose(offsets, np.round(offsets), atol=1e-3)
    assert ta.grid_manually_corrected, "an imported grid is a person's grid"
    assert ta.grid_confidence == an.IMPORTED_GRID_CONFIDENCE, "and it is ground truth"

    cues = {c["index"]: c for c in ta.hot_cues}
    assert cues[1]["label"] == "Intro"
    assert cues[1]["sample_position"] == round(round(FIRST_DOWNBEAT, 3) * an.HOT_CUE_SR)
    assert cues[3] == an.make_hot_cue(3, 16.106, "drop")
    assert cues[4] == an.make_hot_cue(4, 20.0, "Roll")


def test_the_import_summary_covers_every_file(imported):
    cache, summaries = imported
    by_name = {Path(s.path).name: s for s in summaries}
    ok = by_name["test track.wav"]
    assert ok.status == "imported"
    assert ok.cues == 3 and ok.memory_cues_skipped == 1
    assert ok.playlists == ["Sets / Peak Hour"]
    assert "128.00 BPM" in ok.line()
    gone = by_name["gone.wav"]
    assert gone.status == "missing"
    assert any("not found" in n for n in gone.notes)
    assert any("variable tempo" in n for n in gone.notes)


def test_playlists_are_stored_where_the_crate_loader_will_not_read_them(library, imported):
    root, audio, xml = library
    cache, _ = imported
    playlists = li.load_playlists(cache)
    assert playlists == {"Sets / Peak Hour": [an.track_hash(audio)]}
    assert (cache / "library" / "playlists.json").exists()
    assert len(an.load_crate(cache)) == 1, "only the track is a crate entry"


def test_imported_grids_survive_re_analysis(library, imported):
    root, audio, xml = library
    cache, _ = imported
    an.analyze_folder(root, cache, force=True)
    ta = an.load_cached(an.track_hash(audio), cache)
    assert ta.grid_manually_corrected
    assert ta.bpm == pytest.approx(BPM, abs=1e-4)
    assert ta.first_downbeat == pytest.approx(FIRST_DOWNBEAT, abs=0.002)
    assert {c["index"] for c in ta.hot_cues} >= {1, 3, 4}, "and the cues with it"
    assert ta.grid_confidence == an.IMPORTED_GRID_CONFIDENCE, "still ground truth"


def test_a_grid_corrected_in_this_program_keeps_its_measured_confidence(library, tmp_path):
    """Only an import is ground truth. A hand correction here -- which can be a
    wrong one, like a grid doubled once too often -- is re-measured."""
    root, audio, xml = library
    cache = tmp_path / "cache"
    an.analyze_folder(root, cache)
    ta = an.load_cached(an.track_hash(audio), cache)
    an.regrid(ta, ta.bpm * 2.0, ta.first_downbeat)
    an.write_sidecar(ta, cache)

    an.analyze_folder(root, cache, force=True)
    again = an.load_cached(an.track_hash(audio), cache)
    assert again.grid_manually_corrected
    assert again.grid_confidence < an.IMPORTED_GRID_CONFIDENCE


def test_an_imported_grid_turns_a_low_confidence_cut_into_a_blend(library, imported):
    import dataclasses
    from types import SimpleNamespace

    from djai import config, transition as tr

    root, audio, xml = library
    cache, _ = imported
    ta = an.load_cached(an.track_hash(audio), cache)
    weak = config.TRANSITION_MIN_GRID_CONFIDENCE / 2

    def choose(conf_a, conf_b):
        a = dataclasses.replace(ta, grid_confidence=conf_a)
        b = dataclasses.replace(ta, track_id="incoming", grid_confidence=conf_b)
        deck = SimpleNamespace(track=SimpleNamespace(analysis=a), rate=1.0, position=0.0)
        return tr.choose_transition(deck, b)

    detected = choose(weak, weak)
    assert detected.style == "cut" and "grid confidence" in detected.rule
    imported_pair = choose(ta.grid_confidence, ta.grid_confidence)
    assert imported_pair.style != "cut", imported_pair.rule


def test_importing_again_updates_a_cached_track(library, tmp_path):
    root, audio, xml = library
    cache = tmp_path / "cache"
    li.apply_import(li.parse_rekordbox_xml(xml), cache)
    moved = li.parse_rekordbox_xml(xml)
    for t in moved:
        if t.bpm:
            t.first_downbeat_s = FIRST_DOWNBEAT + BEAT      # a beat later
    summaries = li.apply_import(moved, cache)
    assert next(s for s in summaries if s.path.endswith("test track.wav")).status == "updated"
    ta = an.load_cached(an.track_hash(audio), cache)
    assert ta.first_downbeat % (4 * BEAT) == pytest.approx((FIRST_DOWNBEAT + BEAT) % (4 * BEAT), abs=1e-3)


def test_a_moved_library_is_found_with_a_relocate_rule(library, tmp_path):
    root, audio, xml = library
    tracks = li.parse_rekordbox_xml(xml)
    for t in tracks:
        t.path = Path("Z:/old library") / Path(t.path).name
    summaries = li.apply_import(tracks, tmp_path / "cache", analyze_missing=False,
                                relocate_rules=[("Z:/old library", str(root))])
    ok = next(s for s in summaries if s.path.endswith("test track.wav"))
    assert ok.status == "not analysed", "found at the new path; --no-analyze honoured"


def test_the_import_command(library, tmp_path, capsys):
    root, audio, xml = library
    cache = tmp_path / "cli_cache"
    code = cli.main(["import", "rekordbox", str(xml), "--cache", str(cache)])
    out = capsys.readouterr().out
    assert code == 0
    assert "2 track(s)" in out
    assert "imported" in out and "missing" in out
    assert "2 file(s): 1 imported, 1 missing" in out


# --- Serato ------------------------------------------------------------------


def _syncsafe(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def _geob(description: str, data: bytes) -> bytes:
    payload = (b"\x00" + b"application/octet-stream\x00" + b"\x00"
               + description.encode("latin-1") + b"\x00" + data)
    return b"GEOB" + _syncsafe(len(payload)) + b"\x00\x00" + payload


def _serato_beatgrid(first_beat: float, bpm: float) -> bytes:
    return b"\x01\x00" + struct.pack(">I", 1) + struct.pack(">ff", first_beat, bpm) + b"\x00"


def _serato_markers2(cues: list[tuple[int, int, str]]) -> bytes:
    inner = b"\x01\x01"
    for index, ms, name in cues:
        body = (b"\x00" + bytes([index]) + struct.pack(">I", ms) + b"\x00"
                + bytes([0xCC, 0x00, 0x00]) + b"\x00\x00" + name.encode() + b"\x00")
        inner += b"CUE\x00" + struct.pack(">I", len(body)) + body
    inner += b"\x00"
    return b"\x01\x01" + base64.b64encode(inner) + b"\x00"


def _tlv(tag: str, payload: bytes) -> bytes:
    return tag.encode("latin-1") + struct.pack(">I", len(payload)) + payload


SERATO_FIRST_BEAT = 0.25
SERATO_CUES = [(0, 1500, "Drop"), (1, 4250, "Verse")]


@pytest.fixture(scope="module")
def serato(tmp_path_factory):
    root = tmp_path_factory.mktemp("serato_root")
    wav = synth.render_track(root / "src.wav", bpm=BPM, bars=16, seed=5)
    audio, sr = sf.read(str(wav), dtype="float32")
    music = root / "Music"
    music.mkdir()
    mp3 = music / "serato track.mp3"
    sf.write(str(mp3), audio, sr, format="MP3", subtype="MPEG_LAYER_III")
    frames = (_geob("Serato BeatGrid", _serato_beatgrid(SERATO_FIRST_BEAT, BPM))
              + _geob("Serato Markers2", _serato_markers2(SERATO_CUES)))
    tag = b"ID3\x04\x00\x00" + _syncsafe(len(frames)) + frames
    mp3.write_bytes(tag + mp3.read_bytes())

    crates = root / "_Serato_" / "Subcrates"
    crates.mkdir(parents=True)
    rel = "Music/serato track.mp3"
    crate = (_tlv("vrsn", "1.0/Serato ScratchLive Crate".encode("utf-16-be"))
             + _tlv("otrk", _tlv("ptrk", rel.encode("utf-16-be"))))
    (crates / "House%%Warmup.crate").write_bytes(crate)
    return root, mp3


def test_serato_tags_are_read_from_the_mp3(serato):
    root, mp3 = serato
    frames = li.read_id3_geob(mp3)
    bpm, first, variable = li.parse_serato_beatgrid(frames["Serato BeatGrid"])
    assert bpm == pytest.approx(BPM, abs=1e-3)
    assert first == pytest.approx(SERATO_FIRST_BEAT, abs=1e-6)
    assert not variable
    assert li.parse_serato_markers2(frames["Serato Markers2"]) == [
        (1, 1.5, "drop"), (2, 4.25, "Verse"),
    ]


def test_a_serato_import_writes_crates_grids_and_cues(serato, tmp_path):
    root, mp3 = serato
    tracks = li.parse_serato(root / "_Serato_", root=root)
    assert len(tracks) == 1
    assert tracks[0].playlists == ["House / Warmup"]
    cache = tmp_path / "cache"
    summaries = li.apply_import(tracks, cache)
    assert summaries[0].status == "imported", summaries[0].notes
    ta = an.load_cached(an.track_hash(mp3), cache)
    assert ta.bpm == pytest.approx(BPM, abs=1e-3)
    assert ta.first_downbeat % (4 * BEAT) == pytest.approx(SERATO_FIRST_BEAT, abs=1e-3)
    assert ta.grid_manually_corrected
    assert ta.grid_confidence == an.IMPORTED_GRID_CONFIDENCE
    cues = {c["index"]: c for c in ta.hot_cues}
    assert cues[1] == an.make_hot_cue(1, 1.5, "drop")
    assert cues[2] == an.make_hot_cue(2, 4.25, "Verse")
    assert li.load_playlists(cache) == {"House / Warmup": [ta.track_id]}


def test_a_file_without_an_id3_tag_has_no_serato_frames(tmp_path):
    plain = tmp_path / "plain.wav"
    sf.write(str(plain), np.zeros((4410, 2), dtype=np.float32), 44100)
    assert li.read_id3_geob(plain) == {}
