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
# Full list — order determines test parametrize order
# ---------------------------------------------------------------------------
ALL_COMPAT_MODELS: list[CompatModel] = OLLAMA_MODELS + NIM_MODELS + GEMINI_MODELS
