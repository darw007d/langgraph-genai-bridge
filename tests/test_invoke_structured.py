"""Unit tests for GenAIBridge.invoke_structured (v0.2+).

These tests don't hit Google's API — they mock the genai.Client and verify
that response_schema is plumbed through to GenerateContentConfig and that
caching gates correctly.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# =====================================================================
# Stub google.genai so the bridge can import without a real API
# =====================================================================

@pytest.fixture(autouse=True)
def stub_google_genai(monkeypatch):
    """Insert a fake `google.genai` + `google.genai.types` into sys.modules."""

    class FakeGenerateContentConfig:
        def __init__(self, **kw):
            self.kwargs = kw
            for k, v in kw.items():
                setattr(self, k, v)
            # Mirror SDK: tools/system_instruction can be set after construction
            if "tools" not in kw:
                self.tools = None
            if "system_instruction" not in kw:
                self.system_instruction = None

    class FakeTool:
        def __init__(self, function_declarations=None):
            self.function_declarations = function_declarations

    class FakeCreateCachedContentConfig:
        def __init__(self, **kw):
            self.kwargs = kw

    fake_types = types.ModuleType("google.genai.types")
    fake_types.GenerateContentConfig = FakeGenerateContentConfig
    fake_types.Tool = FakeTool
    fake_types.CreateCachedContentConfig = FakeCreateCachedContentConfig

    fake_genai = types.ModuleType("google.genai")
    fake_genai.types = fake_types
    fake_genai.Client = MagicMock

    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    # Drop cached bridge so imports re-resolve under stubs
    for mod in list(sys.modules):
        if mod.startswith("langgraph_genai_bridge"):
            sys.modules.pop(mod, None)
    yield


# =====================================================================
# Tests
# =====================================================================

def _make_bridge(model="gemini-2.5-pro"):
    from langgraph_genai_bridge import GenAIBridge
    bridge = GenAIBridge(api_key="fake", model=model, temperature=0.0)
    # The `Client` MagicMock auto-creates `.models.generate_content` on access
    return bridge


def test_invoke_structured_requires_schema():
    bridge = _make_bridge()
    with pytest.raises(ValueError, match="requires response_schema"):
        bridge.invoke_structured([], response_schema=None)


def test_invoke_structured_sets_response_schema_and_mime_type(monkeypatch):
    """The schema + mime type must reach GenerateContentConfig."""
    from langchain_core.messages import HumanMessage

    class Verdict:
        action: str
        confidence: float

    bridge = _make_bridge()

    captured = {}

    def fake_generate(*, model, contents, config):
        captured["model"] = model
        captured["config"] = config
        # Build a fake response with .parsed populated
        resp = MagicMock()
        resp.parsed = "PARSED_VALUE"
        return resp

    bridge.client.models.generate_content = fake_generate

    out = bridge.invoke_structured(
        messages=[HumanMessage("score this ticker")],
        response_schema=Verdict,
        system_prompt="You are a judge.",
    )

    assert out == "PARSED_VALUE"
    cfg = captured["config"]
    assert cfg.kwargs["response_schema"] is Verdict
    assert cfg.kwargs["response_mime_type"] == "application/json"
    # System instruction was passed as instruction (no cache manager attached)
    assert cfg.system_instruction == "You are a judge."


def test_invoke_structured_does_not_attach_tools_when_schema_present():
    """Gemini API: tools and response_schema are mutually exclusive."""
    from langchain_core.messages import HumanMessage

    class Out:
        x: str

    bridge = _make_bridge()
    # Bypass set_tools' real conversion by injecting declarations directly
    bridge._tool_declarations = [{"name": "fake_tool"}]

    captured = {}
    bridge.client.models.generate_content = lambda *, model, contents, config: (
        captured.update(config=config) or _mock_parsed("ok")
    )

    bridge.invoke_structured(
        messages=[HumanMessage("hi")],
        response_schema=Out,
    )
    cfg = captured["config"]
    # tools must NOT be set when response_schema is present
    assert cfg.tools is None


def test_invoke_structured_uses_cache_when_enabled():
    """When cache_manager has a cache for the system prompt, model_to_use is
    swapped to the cache id."""
    from langchain_core.messages import HumanMessage

    class Out:
        x: str

    bridge = _make_bridge(model="gemini-2.5-pro")
    bridge._cache_manager = MagicMock()
    bridge._cache_manager.get_or_create.return_value = "cachedContents/abc123"

    captured = {}
    bridge.client.models.generate_content = lambda *, model, contents, config: (
        captured.update(model=model, config=config) or _mock_parsed("ok")
    )

    bridge.invoke_structured(
        messages=[HumanMessage("hi")],
        response_schema=Out,
        system_prompt="long stable playbook",
    )

    assert captured["model"] == "cachedContents/abc123"
    # When cache fires, system_instruction is NOT also set (cache carries it)
    assert captured["config"].system_instruction is None


def test_invoke_structured_falls_back_to_instruction_when_cache_fails():
    """If the cache manager returns None (cache creation failed), the system
    prompt must still be passed inline as system_instruction."""
    from langchain_core.messages import HumanMessage

    class Out:
        x: str

    bridge = _make_bridge()
    bridge._cache_manager = MagicMock()
    bridge._cache_manager.get_or_create.return_value = None

    captured = {}
    bridge.client.models.generate_content = lambda *, model, contents, config: (
        captured.update(model=model, config=config) or _mock_parsed("ok")
    )

    bridge.invoke_structured(
        messages=[HumanMessage("hi")],
        response_schema=Out,
        system_prompt="playbook",
    )
    assert captured["config"].system_instruction == "playbook"
    assert captured["model"] == bridge.model  # not swapped


def test_invoke_structured_raises_on_missing_parsed():
    """Defensive check: if .parsed is None, raise rather than silently return None."""
    from langchain_core.messages import HumanMessage

    class Out:
        x: str

    bridge = _make_bridge()

    def fake_generate(*, model, contents, config):
        resp = MagicMock()
        resp.parsed = None  # simulate non-JSON output
        return resp

    bridge.client.models.generate_content = fake_generate

    with pytest.raises(RuntimeError, match="response.parsed is None"):
        bridge.invoke_structured([HumanMessage("hi")], response_schema=Out)


def test_invoke_structured_return_raw_returns_dict():
    from langchain_core.messages import HumanMessage

    class Out:
        x: str

    bridge = _make_bridge()
    sentinel_resp = MagicMock()
    sentinel_resp.parsed = {"x": "hello"}
    bridge.client.models.generate_content = lambda *, model, contents, config: sentinel_resp

    out = bridge.invoke_structured(
        [HumanMessage("hi")],
        response_schema=Out,
        return_raw=True,
    )
    assert out["parsed"] == {"x": "hello"}
    assert out["raw"] is sentinel_resp


def test_invoke_structured_extracts_system_prompt_from_first_message():
    """Convenience: SystemMessage as first message acts as the system prompt."""
    from langchain_core.messages import HumanMessage, SystemMessage

    class Out:
        x: str

    bridge = _make_bridge()
    captured = {}
    bridge.client.models.generate_content = lambda *, model, contents, config: (
        captured.update(config=config) or _mock_parsed("ok")
    )

    bridge.invoke_structured(
        messages=[
            SystemMessage("auto-extracted prompt"),
            HumanMessage("hi"),
        ],
        response_schema=Out,
    )
    assert captured["config"].system_instruction == "auto-extracted prompt"


def test_invoke_structured_falls_back_to_langchain_on_sdk_failure():
    from langchain_core.messages import HumanMessage

    class Out:
        x: str

    bridge = _make_bridge()
    bridge.client.models.generate_content = MagicMock(
        side_effect=RuntimeError("503 rate limit"),
    )

    fallback = MagicMock()
    fallback_so = MagicMock()
    fallback_so.invoke.return_value = "fallback-result"
    fallback.with_structured_output.return_value = fallback_so
    bridge.set_langchain_fallback(fallback)

    out = bridge.invoke_structured(
        [HumanMessage("hi")],
        response_schema=Out,
    )
    assert out == "fallback-result"
    fallback.with_structured_output.assert_called_once_with(Out)


def test_invoke_structured_raises_when_no_fallback():
    """Without a fallback set, SDK errors must raise (not return None)."""
    from langchain_core.messages import HumanMessage

    class Out:
        x: str

    bridge = _make_bridge()
    bridge.client.models.generate_content = MagicMock(
        side_effect=RuntimeError("boom"),
    )

    with pytest.raises(RuntimeError, match="boom"):
        bridge.invoke_structured([HumanMessage("hi")], response_schema=Out)


# ---------- regression: invoke() unchanged by v0.2 changes ----------

def test_invoke_does_not_set_response_schema():
    """Ordinary invoke() must NOT set response_schema/response_mime_type."""
    from langchain_core.messages import HumanMessage

    bridge = _make_bridge()

    captured = {}

    def fake_generate(*, model, contents, config):
        captured["config"] = config
        # Mock a normal AIMessage-shaped response
        resp = MagicMock()
        resp.candidates = []
        resp.text = "hi"
        return resp

    bridge.client.models.generate_content = fake_generate

    # Mock the genai_to_langchain converter so we don't need to
    # round-trip a real google response shape
    import langgraph_genai_bridge.bridge as bridge_mod
    bridge_mod.genai_to_langchain = lambda r: "ai-msg"

    bridge.invoke([HumanMessage("hi")], system_prompt="sp")

    cfg = captured["config"]
    assert "response_schema" not in cfg.kwargs
    assert "response_mime_type" not in cfg.kwargs


# =====================================================================
# Helpers
# =====================================================================

def _mock_parsed(value):
    resp = MagicMock()
    resp.parsed = value
    return resp
