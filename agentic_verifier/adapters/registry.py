"""Model adapter registry — provider-specific pre/post-processing for litellm calls."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Callable

from agentic_verifier.exceptions import VerificationFailedError

_THINK_TAG_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_MD_FENCE_RE = re.compile(r'^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$', re.DOTALL)


def _strip_fences(content: str) -> str:
    """Strip markdown code fences and surrounding whitespace."""
    m = _MD_FENCE_RE.match(content)
    return m.group(1).strip() if m else content.strip()


def _strip_think_tags(content: str) -> str:
    """
    Strip chain-of-thought <think> blocks from Ollama thinking-capable models.

    Uses rfind('</think>') split rather than regex-only, because quantized/local
    LLMs frequently hallucinate malformed closing tags (</thinking>, <\\think>, or
    omit the closing tag entirely). rfind on the last occurrence is resilient to
    all of these — it discards everything up to and including the last </think>
    variant if present, then falls through to regex for well-formed tags.

    Edge case — max_tokens exhaustion mid-thought: if the model emits <think> but
    hits the token limit before writing </think>, there is no closing tag. In this
    case rfind returns -1 AND the regex matches nothing, so stripped == content.
    Detecting a leading <think> opener here returns "" to signal "no usable JSON".
    """
    lower = content.lower()
    # Find the last occurrence of any </think...> closing tag variant
    think_end = lower.rfind('</think')
    if think_end != -1:
        # Advance past the tag's closing >
        close_bracket = content.find('>', think_end)
        if close_bracket != -1:
            return content[close_bracket + 1:].strip()
    # Fallback: regex for well-formed <think>...</think> blocks
    stripped = _THINK_TAG_RE.sub('', content).strip()
    # If regex changed nothing and content opens with <think>, the model hit
    # max_tokens mid-thought — entire content is reasoning, no JSON present.
    if stripped == content.strip() and lower.lstrip().startswith('<think'):
        return ""
    return stripped


@dataclass
class ModelAdapter:
    """
    Protocol for provider-specific LLM quirk handling.

    build_kwargs(base_kwargs: dict) -> dict
        Takes the base litellm.completion() kwargs dict and returns the final kwargs dict.
        Adapter is free to add, remove, or modify any key (e.g. OPENAI_REASONING_ADAPTER
        pops 'temperature' to avoid API errors on o1/o3/o4/gpt-5 models).
        Default: return base_kwargs unchanged.

    post_process(response, model) -> str
        Extract normalized content string from a raw LiteLLM response object.
        Raises VerificationFailedError on unrecoverable content.
        Content returned here is fed directly into Tier3ResponseModel.model_validate_json().
    """
    name: str
    build_kwargs: Callable[[dict], dict]
    post_process: Callable[[Any, str], str]


# ---------------------------------------------------------------------------
# DEFAULT_ADAPTER — used for all unrecognized models
# ---------------------------------------------------------------------------
def _default_post_process(response: Any, model: str = "") -> str:
    content = response.choices[0].message.content or ""
    return _strip_fences(content)


DEFAULT_ADAPTER = ModelAdapter(
    name="default",
    build_kwargs=lambda base: dict(base),
    post_process=_default_post_process,
)


# ---------------------------------------------------------------------------
# OLLAMA_ADAPTER — ollama/ and ollama_chat/ prefixes
# ---------------------------------------------------------------------------
def _ollama_build_kwargs(base: dict) -> dict:
    """Force ollama/ → ollama_chat/ and ensure sufficient context for structured output.

    Two issues this fixes:

    1. litellm routes 'ollama/' to /api/generate, which mishandles structured-output
       responses from thinking-capable models (qwen3, DeepSeek-R1, etc.): the JSON
       schema output lands in the 'thinking' field while 'response' is empty.
       /api/chat correctly splits 'content' (JSON) from 'thinking' (reasoning).

    2. Models with a small default num_ctx (e.g. 4K) exhaust their token budget
       during the thinking phase, leaving nothing for the JSON output. We override
       num_ctx to 8192 so thinking-capable models have room to reason AND output
       the full structured response. This override can be raised further if needed.
    """
    base = dict(base)
    model = base.get("model", "")
    if model.startswith("ollama/"):
        base["model"] = "ollama_chat/" + model[len("ollama/"):]
    # Ensure enough context for thinking + structured JSON output (16K covers full Tier3 prompts)
    options = base.get("extra_body", {}).get("options", {})
    options.setdefault("num_ctx", 16384)
    base.setdefault("extra_body", {})["options"] = options
    # keep_alive=300 holds the model in memory for 5 min so sequential calls don't reload
    base.setdefault("extra_body", {}).setdefault("keep_alive", 300)
    return base


def _ollama_post_process(response: Any, model: str = "") -> str:
    msg = response.choices[0].message
    content = msg.content
    if content:
        content = _strip_think_tags(content)
    if not content:
        # litellm may drop content if reasoning_content is present — try fallback
        fallback = getattr(msg, 'reasoning_content', None) or ""
        fallback = fallback.strip()
        if fallback.startswith('{'):
            content = fallback
        else:
            raise VerificationFailedError(
                "Ollama returned empty content and reasoning_content does not contain JSON. "
                "The model may have failed to generate structured output."
            )
    return _strip_fences(content)


OLLAMA_ADAPTER = ModelAdapter(
    name="ollama",
    build_kwargs=_ollama_build_kwargs,
    post_process=_ollama_post_process,
)


# ---------------------------------------------------------------------------
# NIM_THINKING_ADAPTER — NIM-hosted thinking models (kimi-k2, gpt-oss, etc.)
# Uses reasoning_content fallback like OLLAMA_ADAPTER but sends no
# Ollama-specific extra_body fields (options/keep_alive) to the NIM endpoint.
# ---------------------------------------------------------------------------
NIM_THINKING_ADAPTER = ModelAdapter(
    name="nim_thinking",
    build_kwargs=lambda base: dict(base),
    post_process=_ollama_post_process,
)


# ---------------------------------------------------------------------------
# NEMOTRON_NIM_ADAPTER — nvidia/nemotron-3-super-120b-a12b
# Requires chat_template_kwargs + reasoning_budget in extra_body, otherwise
# the server hangs without returning a response.
# ---------------------------------------------------------------------------
def _nemotron_build_kwargs(base: dict) -> dict:
    base = dict(base)
    extra = base.setdefault("extra_body", {})
    extra.setdefault("chat_template_kwargs", {"enable_thinking": True})
    extra.setdefault("reasoning_budget", 16384)
    return base


NEMOTRON_NIM_ADAPTER = ModelAdapter(
    name="nemotron_nim",
    build_kwargs=_nemotron_build_kwargs,
    post_process=_ollama_post_process,
)


# ---------------------------------------------------------------------------
# OPENAI_REASONING_ADAPTER — o1, o3, o4, gpt-5 series
# ---------------------------------------------------------------------------
def _openai_reasoning_build_kwargs(base: dict) -> dict:
    base = dict(base)
    base.pop("temperature", None)
    return base


OPENAI_REASONING_ADAPTER = ModelAdapter(
    name="openai_reasoning",
    build_kwargs=_openai_reasoning_build_kwargs,
    post_process=_default_post_process,
)


# ---------------------------------------------------------------------------
# ANTHROPIC_ADAPTER — anthropic/ prefix and claude- prefix models
# ---------------------------------------------------------------------------
def _anthropic_post_process(response: Any, model: str = "") -> str:
    content = response.choices[0].message.content or ""
    # Never use message.parsed — force raw content path to avoid litellm #20533
    return _strip_fences(content)


ANTHROPIC_ADAPTER = ModelAdapter(
    name="anthropic",
    build_kwargs=lambda base: dict(base),
    post_process=_anthropic_post_process,
)


# ---------------------------------------------------------------------------
# GOOGLE_ADAPTER — gemini/ and vertex_ai/gemini prefixes
# ---------------------------------------------------------------------------
def _google_post_process(response: Any, model: str = "") -> str:
    content = response.choices[0].message.content
    if not content:
        raise VerificationFailedError(
            "Gemini returned empty content — safety filter may have blocked this response."
        )
    return _strip_fences(content)


GOOGLE_ADAPTER = ModelAdapter(
    name="google",
    build_kwargs=lambda base: dict(base),
    post_process=_google_post_process,
)


# ---------------------------------------------------------------------------
# JSON_OBJECT_ADAPTER — models that support only json_object (not json_schema)
# e.g. nvidia_nim/microsoft/phi-4-mini-instruct
# ---------------------------------------------------------------------------
def _json_object_build_kwargs(base: dict) -> dict:
    base = dict(base)
    base["response_format"] = {"type": "json_object"}
    return base


JSON_OBJECT_ADAPTER = ModelAdapter(
    name="json_object",
    build_kwargs=_json_object_build_kwargs,
    post_process=_default_post_process,
)


# ---------------------------------------------------------------------------
# Registry & Lookup — ordered most-specific to least-specific prefix
# ---------------------------------------------------------------------------
_REGISTRY: list[tuple[str, ModelAdapter]] = [
    ("ollama_chat/", OLLAMA_ADAPTER),
    ("ollama/", OLLAMA_ADAPTER),
    # NIM thinking models — emit reasoning_content
    ("nvidia_nim/deepseek", NIM_THINKING_ADAPTER),           # DeepSeek-R1/V3 on NIM
    ("nvidia_nim/nvidia/nemotron-3-super", NEMOTRON_NIM_ADAPTER),  # requires chat_template_kwargs
    ("nvidia_nim/moonshotai/kimi-k2", NIM_THINKING_ADAPTER),      # Kimi K2 thinking
    ("nvidia_nim/openai/gpt-oss", NIM_THINKING_ADAPTER),          # GPT-OSS thinking
    # NIM json_object-only models
    ("nvidia_nim/microsoft/phi-4-mini", JSON_OBJECT_ADAPTER),
    ("vertex_ai/gemini", GOOGLE_ADAPTER),
    ("gemini/", GOOGLE_ADAPTER),
    ("anthropic/", ANTHROPIC_ADAPTER),
    ("claude-", ANTHROPIC_ADAPTER),
    ("o1", OPENAI_REASONING_ADAPTER),
    ("o3", OPENAI_REASONING_ADAPTER),
    ("o4", OPENAI_REASONING_ADAPTER),
    ("gpt-5", OPENAI_REASONING_ADAPTER),
]


def get_adapter(model: str) -> ModelAdapter:
    """
    Longest-prefix match against _REGISTRY. Returns DEFAULT_ADAPTER for unrecognized models.

    The registry is ordered by prefix length (longest first) to ensure that
    'ollama_chat/' matches before 'ollama/' for models like 'ollama_chat/deepseek-r1'.
    """
    for prefix, adapter in _REGISTRY:
        if model.startswith(prefix):
            return adapter
    return DEFAULT_ADAPTER
