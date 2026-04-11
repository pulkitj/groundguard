# tests/integration/compat_models.py
"""
Compat model registry — defines all target models for multi-model compatibility testing.

Each CompatModel entry specifies:
  model_str       litellm model string passed directly to verify()
  description     short human label used as pytest parameter ID
  required_env    env var that must be non-empty (empty string = local, no key required)
  adapter         which adapter path this model exercises (for documentation)

To skip a model in a run: ensure its required_env is unset.
To add a new model: append a CompatModel entry here — no other file needs changing.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CompatModel:
    model_str: str
    description: str
    required_env: str   # empty string means local (no key required)
    adapter: str
    api_base: str | None = None  # override endpoint (e.g. local Ollama Anthropic endpoint)


# ---------------------------------------------------------------------------
# Ollama — local, no API key required
# ---------------------------------------------------------------------------
OLLAMA_MODELS = [
    CompatModel(
        model_str="ollama/qwen3:30b",
        description="ollama-qwen3-30b",
        required_env="",
        adapter="ollama",
    ),
    CompatModel(
        model_str="ollama/qwen3:14b",
        description="ollama-qwen3-14b",
        required_env="",
        adapter="ollama",
    ),
    CompatModel(
        model_str="ollama/qwen3.5:9b",
        description="ollama-qwen35-9b",
        required_env="",
        adapter="ollama",
    ),
    CompatModel(
        model_str="ollama/phi4-mini",
        description="ollama-phi4-mini",
        required_env="",
        adapter="ollama",
    ),
    CompatModel(
        model_str="ollama/granite3.3:8b",
        description="ollama-granite3.3-8b",
        required_env="",
        adapter="ollama",
    ),
]

# ---------------------------------------------------------------------------
# NVIDIA NIM — requires NVIDIA_NIM_API_KEY
# All use OpenAI-compatible /v1/chat/completions; litellm routes via nvidia_nim/ prefix.
# DeepSeek-R1 uses OLLAMA_ADAPTER (think-tag stripping); all others use DEFAULT_ADAPTER.
# ---------------------------------------------------------------------------
NIM_MODELS = [
    CompatModel(
        model_str="nvidia_nim/meta/llama-3.3-70b-instruct",
        description="nim-llama33-70b",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="default",
    ),
    CompatModel(
        model_str="nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct",
        description="nim-nemotron-70b",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="default",
    ),
    CompatModel(
        model_str="nvidia_nim/qwen/qwen2.5-72b-instruct",
        description="nim-qwen25-72b",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="default",
    ),
    CompatModel(
        model_str="nvidia_nim/deepseek-ai/deepseek-r1",
        description="nim-deepseek-r1",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="ollama",   # think-tag stripping
    ),
    CompatModel(
        model_str="nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1",
        description="nim-mixtral-8x22b",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="default",
    ),
    CompatModel(
        model_str="nvidia_nim/nvidia/nemotron-4-340b-instruct",
        description="nim-nemotron-340b",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="default",
    ),
    CompatModel(
        model_str="nvidia_nim/mistralai/mistral-small-24b-instruct",
        description="nim-mistral-small-24b",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="default",
    ),
    # --- newly added 2026-04-10 ---
    CompatModel(
        model_str="nvidia_nim/deepseek-ai/deepseek-v3.2",
        description="nim-deepseek-v3.2",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="nim_thinking",
    ),
    CompatModel(
        model_str="nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        description="nim-nemotron-super-120b",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="nemotron_nim",  # requires chat_template_kwargs + reasoning_budget
    ),
    CompatModel(
        model_str="nvidia_nim/nvidia/nemotron-3-nano-30b-a3b",
        description="nim-nemotron-nano-30b",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="nemotron_nim",  # same chat_template_kwargs + reasoning_budget pattern
    ),
    CompatModel(
        model_str="nvidia_nim/google/gemma-4-31b-it",
        description="nim-gemma4-31b",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="default",
    ),
    CompatModel(
        model_str="nvidia_nim/moonshotai/kimi-k2-thinking",
        description="nim-kimi-k2",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="nim_thinking",
    ),
    CompatModel(
        model_str="nvidia_nim/mistralai/mistral-small-4-119b-2603",
        description="nim-mistral-small-4",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="default",
    ),
    CompatModel(
        model_str="nvidia_nim/openai/gpt-oss-120b",
        description="nim-gpt-oss-120b",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="nim_thinking",
    ),
    CompatModel(
        model_str="nvidia_nim/minimaxai/minimax-m2.5",
        description="nim-minimax-m2.5",
        required_env="NVIDIA_NIM_API_KEY",
        adapter="default",
    ),
]

# ---------------------------------------------------------------------------
# Gemini — requires GEMINI_API_KEY (free tier at Google AI Studio)
# ---------------------------------------------------------------------------
GEMINI_MODELS = [
    CompatModel(
        model_str="gemini/gemini-2.0-flash-lite",
        description="gemini-2-flash-lite",
        required_env="GEMINI_API_KEY",
        adapter="google",
    ),
]

# ---------------------------------------------------------------------------
# Anthropic via Ollama — tests ANTHROPIC_ADAPTER path without real API keys.
# Uses Ollama's /api/messages Anthropic-compatible endpoint.
# Requires: ollama serve + the model pulled locally.
# model_str uses anthropic/ prefix so litellm uses Anthropic wire format;
# api_base redirects to localhost instead of Anthropic's servers.
# ---------------------------------------------------------------------------
ANTHROPIC_VIA_OLLAMA_MODELS = [
    CompatModel(
        model_str="anthropic/qwen3:14b",
        description="anthropic-via-ollama-qwen3-14b",
        required_env="",
        adapter="anthropic",
        api_base="http://localhost:11434",
    ),
]

# ---------------------------------------------------------------------------
# Full list — order determines test parametrize order
# ---------------------------------------------------------------------------
ALL_COMPAT_MODELS: list[CompatModel] = OLLAMA_MODELS + NIM_MODELS + GEMINI_MODELS + ANTHROPIC_VIA_OLLAMA_MODELS
