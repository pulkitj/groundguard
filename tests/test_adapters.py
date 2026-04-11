"""Tests for adapters/registry.py — get_adapter routing + post_process behaviour."""

import pytest
from unittest.mock import MagicMock

from agentic_verifier.adapters.registry import (
    get_adapter,
    ModelAdapter,
    OLLAMA_ADAPTER,
    NIM_THINKING_ADAPTER,
    NEMOTRON_NIM_ADAPTER,
    JSON_OBJECT_ADAPTER,
    GOOGLE_ADAPTER,
    ANTHROPIC_ADAPTER,
    OPENAI_REASONING_ADAPTER,
    DEFAULT_ADAPTER,
)
from agentic_verifier.exceptions import VerificationFailedError


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_response(content, reasoning_content=None):
    msg = MagicMock()
    msg.content = content
    msg.reasoning_content = reasoning_content
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = msg
    return response


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------

def test_get_adapter_ollama_prefix():
    assert get_adapter("ollama/qwen3:30b") is OLLAMA_ADAPTER


def test_get_adapter_ollama_chat_prefix():
    assert get_adapter("ollama_chat/deepseek-r1") is OLLAMA_ADAPTER


def test_get_adapter_gemini_prefix():
    assert get_adapter("gemini/gemini-3-flash") is GOOGLE_ADAPTER


def test_get_adapter_vertex_ai_gemini_prefix():
    assert get_adapter("vertex_ai/gemini-3.1-pro-preview") is GOOGLE_ADAPTER


def test_get_adapter_anthropic_prefix():
    assert get_adapter("anthropic/claude-opus-4-6") is ANTHROPIC_ADAPTER


def test_get_adapter_claude_prefix():
    assert get_adapter("claude-sonnet-4-6") is ANTHROPIC_ADAPTER


def test_get_adapter_o3():
    assert get_adapter("o3") is OPENAI_REASONING_ADAPTER


def test_get_adapter_gpt5():
    assert get_adapter("gpt-5.4") is OPENAI_REASONING_ADAPTER


def test_get_adapter_gpt4o_mini_default():
    assert get_adapter("gpt-4o-mini") is DEFAULT_ADAPTER


def test_get_adapter_custom_model_default():
    assert get_adapter("my-custom-model") is DEFAULT_ADAPTER


# ---------------------------------------------------------------------------
# OLLAMA_ADAPTER.post_process tests
# ---------------------------------------------------------------------------

def test_ollama_post_process_strips_think_tags():
    response = _make_response('<think>reasoning</think>\n{"valid": "json"}')
    result = OLLAMA_ADAPTER.post_process(response)
    assert result == '{"valid": "json"}'


def test_ollama_post_process_none_content_json_fallback():
    response = _make_response(None, reasoning_content='{"valid": "json"}')
    result = OLLAMA_ADAPTER.post_process(response)
    assert result == '{"valid": "json"}'


def test_ollama_post_process_none_content_non_json_reasoning_raises():
    response = _make_response(None, reasoning_content="Let me think...")
    with pytest.raises(VerificationFailedError):
        OLLAMA_ADAPTER.post_process(response)


def test_ollama_post_process_no_closing_think_tag_raises():
    response = _make_response("<think>incomplete reasoning with no closing tag")
    with pytest.raises(VerificationFailedError):
        OLLAMA_ADAPTER.post_process(response)


# ---------------------------------------------------------------------------
# GOOGLE_ADAPTER.post_process tests
# ---------------------------------------------------------------------------

def test_google_post_process_none_content_raises():
    response = _make_response(None)
    with pytest.raises(VerificationFailedError):
        GOOGLE_ADAPTER.post_process(response)


# ---------------------------------------------------------------------------
# OPENAI_REASONING_ADAPTER.build_kwargs test
# ---------------------------------------------------------------------------

def test_openai_reasoning_build_kwargs_removes_temperature():
    kwargs = {"temperature": 0.0, "model": "o3"}
    result = OPENAI_REASONING_ADAPTER.build_kwargs(kwargs)
    assert "temperature" not in result


# ---------------------------------------------------------------------------
# DEFAULT_ADAPTER.post_process test
# ---------------------------------------------------------------------------

def test_default_post_process_strips_fences():
    response = _make_response('```json\n{"key": "value"}\n```')
    result = DEFAULT_ADAPTER.post_process(response)
    assert result == '{"key": "value"}'


def test_nvidia_nim_deepseek_routes_to_nim_thinking_adapter():
    """nvidia_nim/deepseek-ai/* must use NIM_THINKING_ADAPTER for reasoning_content fallback."""
    adapter = get_adapter("nvidia_nim/deepseek-ai/deepseek-r1")
    assert adapter is NIM_THINKING_ADAPTER, (
        f"DeepSeek on NIM routes to NIM_THINKING_ADAPTER, got: {adapter.name!r}"
    )


def test_nvidia_nim_nemotron_super_routes_to_nemotron_adapter():
    """nemotron-3-super requires chat_template_kwargs — must use NEMOTRON_NIM_ADAPTER."""
    adapter = get_adapter("nvidia_nim/nvidia/nemotron-3-super-120b-a12b")
    assert adapter is NEMOTRON_NIM_ADAPTER, (
        f"Nemotron Super must route to NEMOTRON_NIM_ADAPTER, got: {adapter.name!r}"
    )


def test_nvidia_nim_nemotron_nano_routes_to_nemotron_adapter():
    """nemotron-3-nano requires same chat_template_kwargs pattern — must use NEMOTRON_NIM_ADAPTER."""
    adapter = get_adapter("nvidia_nim/nvidia/nemotron-3-nano-30b-a3b")
    assert adapter is NEMOTRON_NIM_ADAPTER, (
        f"Nemotron Nano must route to NEMOTRON_NIM_ADAPTER, got: {adapter.name!r}"
    )


def test_nvidia_nim_nemotron_adapter_injects_chat_template_kwargs():
    """NEMOTRON_NIM_ADAPTER.build_kwargs must add chat_template_kwargs + reasoning_budget."""
    base = {"model": "nvidia_nim/nvidia/nemotron-3-super-120b-a12b", "messages": []}
    result = NEMOTRON_NIM_ADAPTER.build_kwargs(base)
    assert result["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert result["extra_body"]["reasoning_budget"] == 16384


def test_nvidia_nim_phi4_routes_to_json_object_adapter():
    """phi-4-mini only supports json_object — must use JSON_OBJECT_ADAPTER."""
    adapter = get_adapter("nvidia_nim/microsoft/phi-4-mini-instruct")
    assert adapter is JSON_OBJECT_ADAPTER, (
        f"phi-4-mini must route to JSON_OBJECT_ADAPTER, got: {adapter.name!r}"
    )


def test_json_object_adapter_sets_response_format():
    """JSON_OBJECT_ADAPTER must replace response_format with {type: json_object}."""
    base = {"model": "nvidia_nim/microsoft/phi-4-mini-instruct", "response_format": {"type": "json_schema"}}
    result = JSON_OBJECT_ADAPTER.build_kwargs(base)
    assert result["response_format"] == {"type": "json_object"}
