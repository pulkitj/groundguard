"""Pytest configuration and shared fixtures for the agentic_verifier test suite."""
from __future__ import annotations

import os

import litellm
import pytest


# ---------------------------------------------------------------------------
# LLM model selection for the Real Suite (pytest -m llm)
# Default: local Ollama qwen3:30b (no API key required)
# Override via CLI:  pytest -m llm --llm-model ollama/qwen3:30b
# Override via env:  LLM_TEST_MODEL=ollama/qwen3:30b pytest -m llm
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--llm-model",
        default="ollama/qwen3:30b",
        help="LiteLLM model string for real LLM integration tests (default: ollama/qwen3:30b)",
    )


@pytest.fixture(scope="session")
def llm_model(request: pytest.FixtureRequest) -> str:
    """LiteLLM model string used by @pytest.mark.llm integration tests."""
    return os.environ.get("LLM_TEST_MODEL") or request.config.getoption("--llm-model")


# ---------------------------------------------------------------------------
# Patch litellm transient exception constructors so tests can instantiate them
# with just a message string (the real classes require llm_provider and model
# as additional positional arguments, which is inconvenient for unit tests).
# ---------------------------------------------------------------------------

class _EasyAPIConnectionError(litellm.exceptions.APIConnectionError):
    """APIConnectionError that accepts a bare message with no extra required args."""

    def __init__(self, message: str = "", llm_provider: str = "test", model: str = "test", **kwargs):  # type: ignore[override]
        super().__init__(message, llm_provider=llm_provider, model=model, **kwargs)


@pytest.fixture(autouse=True)
def _patch_litellm_exceptions(monkeypatch):
    """Make litellm transient exceptions constructable with a single string argument."""
    monkeypatch.setattr(litellm.exceptions, "APIConnectionError", _EasyAPIConnectionError)
    monkeypatch.setattr(litellm, "APIConnectionError", _EasyAPIConnectionError, raising=False)
    yield
