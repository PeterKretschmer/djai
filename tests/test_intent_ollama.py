"""The Ollama transport, the fallback chain, and the local-inference config.

THREADING CONTEXT: main thread (pytest). No network and no Ollama: every
request is served by an httpx.MockTransport, so the failure modes that matter
(timeout, connection refused, bad HTTP, malformed body, invalid action) can each
be produced on demand rather than waited for.
"""

from __future__ import annotations

import importlib
import json

import httpx
import pytest

from djai import config
from djai.intent import (
    ACTIONS,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    Intent,
    IntentEngine,
    keyword_intent,
)

STATE = {"bpm": 124.0, "key": "8A", "bars_in": 48.0, "transition": False, "played": 3}


def engine_with(handler, **kwargs) -> IntentEngine:
    """An IntentEngine whose HTTP client is backed by a mock transport."""
    engine = IntentEngine(**kwargs)
    engine._client = httpx.Client(
        transport=httpx.MockTransport(handler), timeout=engine.timeout_s
    )
    return engine


def reply_with(action="next_track", params=None, reply="ok"):
    body = {"action": action, "params": params or {}, "reply": reply}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": json.dumps(body)}})

    return handler


# --- request shape -----------------------------------------------------------


def test_request_matches_the_required_ollama_shape():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": '{"action":"none","params":{},"reply":"x"}'}},
        )

    engine = engine_with(handler)
    engine.interpret("hello", STATE)
    body = captured["body"]

    assert captured["url"].endswith("/api/chat")
    assert body["model"] == config.OLLAMA_MODEL
    assert body["stream"] is False
    assert body["keep_alive"] == "30m"
    assert body["format"] == RESPONSE_SCHEMA
    assert body["options"] == {
        "num_thread": 6,
        "temperature": 0.1,
        "num_predict": 200,
    }
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == SYSTEM_PROMPT
    assert "hello" in body["messages"][1]["content"]
    assert '"bpm":124' in body["messages"][1]["content"].replace(" ", "")


def test_schema_pins_the_six_actions_as_an_enum():
    """Decoding is constrained, so an invalid action is unrepresentable."""
    assert set(RESPONSE_SCHEMA["properties"]["action"]["enum"]) == set(ACTIONS)
    assert RESPONSE_SCHEMA["required"] == ["action", "params", "reply"]
    assert set(RESPONSE_SCHEMA["properties"]) == {"action", "params", "reply"}


def test_system_prompt_stays_under_400_words():
    words = len(SYSTEM_PROMPT.split())
    assert words < 400, f"system prompt is {words} words"
    assert "<" not in SYSTEM_PROMPT, "no XML tags: the 8B follows flat text better"
    for action in ACTIONS:
        assert action in SYSTEM_PROMPT, f"{action} missing from the prompt"


def test_prompt_carries_worked_examples_for_every_shape_of_answer():
    """One worked example per answer shape, each a complete JSON reply."""
    count = SYSTEM_PROMPT.count("Example ")
    assert count >= 3
    assert SYSTEM_PROMPT.count('{"action"') == count
    # The two style examples are what teach it to name a style rather than
    # inventing timing, so they are not optional.
    assert '"set_transition_style"' in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.count('"set_transition_style"') >= 2


def test_a_good_reply_is_returned_unchanged():
    engine = engine_with(reply_with("next_track", {"energy": 0.8}, "Cueing harder."))
    intent = engine.interpret("give me something harder", STATE)
    assert intent.ok and intent.fallback is None
    assert intent.action == "next_track"
    assert intent.energy_direction == pytest.approx(0.8)
    assert intent.reply == "Cueing harder."


# --- fallback chain ----------------------------------------------------------


def raises(exc):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


FAILURES = [
    ("timeout", raises(httpx.ReadTimeout("too slow"))),
    ("connect", raises(httpx.ConnectError("refused"))),
    ("http_500", lambda r: httpx.Response(500, text="boom")),
    ("not_json", lambda r: httpx.Response(200, text="<html>nope</html>")),
    ("bad_action", reply_with("launch_missiles")),
    ("prose", lambda r: httpx.Response(200, json={"message": {"content": "sure!"}})),
    ("empty", lambda r: httpx.Response(200, json={"message": {"content": ""}})),
    ("no_message", lambda r: httpx.Response(200, json={"done": True})),
]


@pytest.mark.parametrize("label,handler", FAILURES, ids=[f[0] for f in FAILURES])
def test_every_failure_mode_falls_back_to_keywords(label, handler):
    engine = engine_with(handler)
    intent = engine.interpret("give me something harder", STATE)

    assert intent.action in ACTIONS
    assert intent.action in ("next_track", "set_energy"), (
        f"{label}: keyword chain should have caught 'harder'"
    )
    assert intent.ok, "a fallback is not an error the REPL should stop on"
    assert intent.fallback, f"{label}: the reason must be recorded for the log"
    assert intent.energy_direction > 0


@pytest.mark.parametrize("label,handler", FAILURES, ids=[f[0] for f in FAILURES])
def test_unmatchable_text_degrades_to_none_carrying_the_raw_text(label, handler):
    engine = engine_with(handler)
    intent = engine.interpret("zxcvbnm qwerty", STATE)
    assert intent.action == "none"
    assert intent.ok
    assert intent.fallback
    # The reply carries what failed, so the user sees something real.
    assert intent.reply


def test_unavailable_engine_skips_the_network_entirely():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json={"message": {"content": "{}"}})

    engine = engine_with(handler)
    engine.available = False
    intent = engine.interpret("next track", STATE)
    assert not calls, "fallback-only mode must not make a request"
    assert intent.action == "next_track"
    assert "unavailable" in (intent.fallback or "")


def test_fallback_never_invents_an_action_outside_the_six():
    engine = engine_with(reply_with("definitely_not_valid"))
    for text in ["do a barrel roll", "", "harder", "?????", "skip", "what"]:
        assert engine.interpret(text, STATE).action in ACTIONS


# --- keyword matcher ---------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("next track please", "next_track"),
        ("give me another one", "next_track"),
        ("switch it up", "next_track"),
        ("more energy please", "set_energy"),
        ("something harder", "set_energy"),
        ("chill it out", "set_energy"),
        ("keep them blended longer", "hold_blend"),
        ("hold the blend", "hold_blend"),
        ("cancel that", "skip_queued"),
        ("nevermind", "skip_queued"),
        ("what is playing", "describe_state"),
        ("status", "describe_state"),
    ],
)
def test_keyword_matcher_maps_common_phrasings(text, expected):
    intent = keyword_intent(text)
    assert intent is not None and intent.action == expected


def test_keyword_matcher_returns_none_when_nothing_matches():
    assert keyword_intent("the quick brown fox") is None
    assert keyword_intent("") is None


def test_keyword_matcher_reads_direction_from_the_words():
    assert keyword_intent("something harder").energy_direction > 0
    assert keyword_intent("make it calmer").energy_direction < 0
    assert keyword_intent("change the vibe").energy_direction == pytest.approx(0.0)


# --- warmup ------------------------------------------------------------------


def test_warmup_success_marks_available():
    engine = engine_with(reply_with("none"))
    ok, detail = engine.warmup(timeout_s=5)
    assert ok and engine.available
    assert config.OLLAMA_MODEL in detail


def test_warmup_connection_failure_is_not_fatal_and_disables_the_model():
    engine = engine_with(raises(httpx.ConnectError("refused")))
    ok, detail = engine.warmup(timeout_s=5)
    assert not ok
    assert engine.available is False, "unreachable Ollama means fallback-only"
    assert engine.base_url in detail, "the message must name the base URL"


def test_warmup_timeout_keeps_the_model_available():
    """A slow cold load is not an unreachable server.

    Loading an 8B off disk measured ~49 s here and can exceed the warmup
    budget; marking it unavailable would wrongly downgrade the whole session.
    """
    engine = engine_with(raises(httpx.ReadTimeout("still loading")))
    ok, _detail = engine.warmup(timeout_s=1)
    assert not ok
    assert engine.available is True


def test_warmup_404_names_the_pull_command():
    engine = engine_with(lambda r: httpx.Response(404, text="not found"))
    ok, detail = engine.warmup(timeout_s=5)
    assert not ok and engine.available is False
    assert "ollama pull" in detail


# --- config ------------------------------------------------------------------


def test_config_defaults():
    assert config.OLLAMA_MODEL == "llama3.1:8b"
    assert config.OLLAMA_BASE_URL == "http://localhost:11434"
    assert config.OLLAMA_TIMEOUT_S == 8.0
    assert config.AUDIO_BLOCKSIZE == 2048


def test_config_reads_environment_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://box:1234/")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("OLLAMA_TIMEOUT_S", "3.5")
    monkeypatch.setenv("AUDIO_BLOCKSIZE", "4096")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.OLLAMA_BASE_URL == "http://box:1234"  # trailing / stripped
        assert reloaded.OLLAMA_MODEL == "qwen2.5:7b"
        assert reloaded.OLLAMA_TIMEOUT_S == 3.5
        assert reloaded.AUDIO_BLOCKSIZE == 4096
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_config_ignores_malformed_overrides(monkeypatch):
    monkeypatch.setenv("AUDIO_BLOCKSIZE", "not-a-number")
    monkeypatch.setenv("OLLAMA_TIMEOUT_S", "")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.AUDIO_BLOCKSIZE == 2048
        assert reloaded.OLLAMA_TIMEOUT_S == 8.0
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_timeout_is_configured_on_the_client():
    engine = IntentEngine(timeout_s=2.5)
    assert engine._client.timeout.read == pytest.approx(2.5)
    engine.close()


# --- the audio-side change ---------------------------------------------------


def test_engine_blocksize_default_is_2048_and_within_deck_limits():
    from djai.deck import MAX_BLOCK
    from djai.engine import DEFAULT_BLOCKSIZE

    assert DEFAULT_BLOCKSIZE == 2048
    assert DEFAULT_BLOCKSIZE <= MAX_BLOCK


def test_a_2048_frame_block_still_leaves_cut_under_100ms():
    from djai.deck import SAMPLE_RATE
    from djai.engine import DEFAULT_BLOCKSIZE

    ms = DEFAULT_BLOCKSIZE / SAMPLE_RATE * 1000
    assert ms < 100.0, f"one block is {ms:.1f} ms, over the panic budget"


def test_intent_dataclass_still_carries_what_the_repl_reads():
    intent = Intent(action="next_track", params={"energy": 1.0})
    assert intent.energy_direction == pytest.approx(1.0)
    assert intent.hold_bars == 16
    assert intent.fallback is None
