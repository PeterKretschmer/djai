"""Suite-wide fixtures.

THREADING CONTEXT: main thread (pytest).
"""

from __future__ import annotations

import functools
import sys

import pytest


@pytest.fixture(autouse=True)
def _release_fastapi_handler_caches():
    """Let each test's UI server, and everything it holds, be freed.

    FastAPI keeps module-level ``functools.lru_cache`` wrappers in
    ``fastapi.dependencies.models`` that remember every route handler they have
    inspected, up to 4,096 each. :class:`djai.ui_server.UIServer` defines its
    handlers as closures over ``self``, so those caches pinned every server a
    test built -- and through it the Session, the engine and both decks' audio.
    Measured at 56 live sessions after 60 tests in ``test_ui_server.py``, and a
    6 GB peak that took the whole suite past the machine's memory.

    Production builds one server per process, so this only matters here. The
    caches are found by type rather than by name, so a FastAPI upgrade that
    renames them still gets cleared, and nothing is imported that the test did
    not already import.
    """
    yield
    models = sys.modules.get("fastapi.dependencies.models")
    if models is None:
        return
    for value in list(vars(models).values()):
        if isinstance(value, functools._lru_cache_wrapper):
            value.cache_clear()


@pytest.fixture(autouse=True, scope="session")
def _librosa_beat_tracker_by_default():
    """Analysis in tests uses the librosa detector, installed or not.

    Beat This! would otherwise load a PyTorch model in every process that
    analyses a synthetic track, and make detection results depend on whether it
    happens to be installed. ``tests/test_beat_tracker.py`` opts back in per
    test, with monkeypatch, which restores this default afterwards.

    Session-scoped deliberately: a function-scoped pin is set up *after* the
    module- and session-scoped fixtures that analyse audio, so the nine test
    modules that analyse in a module fixture would still have used Beat This!.
    That is exactly what happened once PyTorch was installed -- the preview's
    synthetic pair came out with a different grid, and its measured loudness
    dip went from over the 3 dB limit to 2.3 dB under it.
    """
    from djai import analysis

    previous = analysis.BEAT_TRACKER
    analysis.BEAT_TRACKER = "librosa"
    yield
    analysis.BEAT_TRACKER = previous


@pytest.fixture(autouse=True)
def _transition_preview_off_by_default(request, monkeypatch):
    """No live preview in tests that are not about the preview.

    A Session built with the preview on renders every transition it arms on a
    worker thread and may swap the queued commands a second later -- which is
    right in a set, and a race in a test asserting on exactly those commands.
    The preview's own tests opt back in by module name.
    """
    if request.module.__name__.rsplit(".", 1)[-1] in ("test_preview", "test_preview_session"):
        yield
        return
    from djai import config

    monkeypatch.setattr(config, "PREVIEW_ENABLED", False)
    yield
