"""Phase 0A.5: a grid too weak to trust is quarantined, not deleted.

Quarantine is a derived state, so there is nothing to migrate and nothing to
get out of step with the confidence it is derived from. The contract:

* the system never chooses a quarantined track itself, including on the
  last-resort path;
* a person can still play and cue it;
* preflight says so as a warning and still passes;
* correcting the grid clears it, with no confidence value invented.

The track this came from is "Stereo Love X Where Have You Been", a mashup whose
tempo genuinely changes: no constant grid fits it, so even a grid fitted to the
beat tracker's own detections scores 0.00 (measured 2026-09-21).
"""

from __future__ import annotations

from djai import config, preflight, selector
from tests.test_placement import make_analysis


def weak(title: str, bpm: float = 120.0) -> object:
    """A track whose measured grid confidence is under the preflight bar."""
    ta = make_analysis(bpm=bpm, title=title)
    ta.grid_confidence = 0.0
    return ta


def strong(title: str, bpm: float = 120.0) -> object:
    ta = make_analysis(bpm=bpm, title=title)
    ta.grid_confidence = 0.9
    return ta


# --- the derived state --------------------------------------------------------


def test_a_weak_grid_is_quarantined():
    assert weak("mashup").quarantined


def test_a_grid_at_the_bar_is_not_quarantined():
    ta = make_analysis(title="edge")
    ta.grid_confidence = config.PREFLIGHT_MIN_GRID_CONFIDENCE
    assert not ta.quarantined


def test_correcting_the_grid_clears_quarantine_without_inventing_a_confidence():
    ta = weak("mashup")
    assert ta.quarantined
    ta.grid_manually_corrected = True
    assert not ta.quarantined
    # the measured value is left exactly as measured -- regrid has no audio and
    # so cannot compute one, and a fabricated number is worse than a low one
    assert ta.grid_confidence == 0.0


# --- autonomous selection -----------------------------------------------------


def test_autonomous_selection_never_offers_a_quarantined_track():
    current = strong("current")
    crate = [current, strong("good"), weak("mashup")]
    ranked = selector.rank_candidates(crate, current)
    titles = {c.track.title for c in ranked}
    assert "good" in titles
    assert "mashup" not in titles


def test_the_last_resort_also_refuses_a_quarantined_track():
    """A track with no tempo partner still must not be a quarantined one."""
    current = strong("current", bpm=120.0)
    # far enough away that the normal ranking finds nothing blendable
    crate = [current, weak("mashup", bpm=180.0)]
    assert selector.nearest_tempo(crate, current) is None


def test_a_corrected_track_becomes_selectable_again():
    current = strong("current")
    mashup = weak("mashup")
    crate = [current, mashup]
    assert not selector.rank_candidates(crate, current)
    mashup.grid_manually_corrected = True
    titles = {c.track.title for c in selector.rank_candidates(crate, current)}
    assert "mashup" in titles


# --- preflight ----------------------------------------------------------------


def test_preflight_passes_with_a_quarantined_track_and_warns_about_it():
    crate = [strong("good"), weak("mashup")]
    check = preflight.check_grid(crate)
    assert check.ok
    warned = " ".join(check.warnings)
    assert "quarantined" in warned and "mashup" in warned


def test_preflight_fails_when_every_track_is_quarantined():
    """Passing with nothing left to select would be a lie."""
    check = preflight.check_grid([weak("a"), weak("b")])
    assert not check.ok


# --- the operator's path is not gated ------------------------------------------


def test_a_quarantined_track_is_still_findable_by_hand():
    """Quarantine gates the system's choice, never the operator's."""
    from djai.cli import _resolve_track

    mashup = weak("Stereo Love X Where Have You Been")
    crate = [strong("good"), mashup]
    assert _resolve_track(crate, "Stereo Love X Where Have You Been") is mashup
    assert _resolve_track(crate, "Stereo Love") is mashup
    assert mashup.quarantined  # still quarantined, still reachable
