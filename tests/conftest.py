"""Pytest configuration and shared fixtures for the groundguard test suite."""
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


# ---------------------------------------------------------------------------
# Loader fixtures — generates sample.pdf and sample.docx on-demand
# ---------------------------------------------------------------------------

import pathlib

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=False)
def loader_fixtures():
    """
    Generate sample.pdf and sample.docx in tests/fixtures/ for loaders tests.
    Generated on first run; skipped on subsequent runs if files already exist.
    Requires [loaders] extras: fpdf2 and python-docx.
    """
    FIXTURES_DIR.mkdir(exist_ok=True)
    pdf_path = FIXTURES_DIR / "sample.pdf"
    docx_path = FIXTURES_DIR / "sample.docx"

    if not pdf_path.exists():
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Sample fixture content for agentic-verifier loaders tests.")
            pdf.output(str(pdf_path))
        except ImportError:
            pytest.skip("fpdf2 not installed — run: pip install fpdf2")

    if not docx_path.exists():
        try:
            from docx import Document  # type: ignore[import-not-found]
            doc = Document()
            doc.add_paragraph("Sample fixture content for agentic-verifier loaders tests.")
            doc.save(str(docx_path))
        except ImportError:
            pytest.skip("python-docx not installed — run: pip install python-docx")


# ---------------------------------------------------------------------------
# Compat model fixture — parametrized across ALL_COMPAT_MODELS
# Each model auto-skips if its required_env is not set in the environment.
# Run: pytest -m compat -v
# Filter to one model: pytest -m compat -k "nim-llama33"
# ---------------------------------------------------------------------------

from tests.integration.compat_models import ALL_COMPAT_MODELS, CompatModel


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize compat_model fixture across all registered compat models."""
    if "compat_model" in metafunc.fixturenames:
        metafunc.parametrize(
            "compat_model",
            ALL_COMPAT_MODELS,
            ids=[m.description for m in ALL_COMPAT_MODELS],
        )


def _ollama_pulled_models() -> set[str]:
    """Return set of model names currently pulled in Ollama (tags endpoint)."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            import json
            data = json.loads(r.read())
            return {m["name"] for m in data.get("models", [])}
    except Exception:
        return set()


def _ollama_unload(tag: str) -> None:
    """
    Unload a model from Ollama VRAM by setting keep_alive=0.

    BUG-04: Uses /api/generate (not /api/chat). /api/chat requires a
    'messages' field and returns 400 without it, silently failing the
    unload. /api/generate accepts a bare {model, prompt, keep_alive}
    payload.
    """
    try:
        import urllib.request
        import json
        payload = json.dumps({
            "model": tag,
            "prompt": "",
            "keep_alive": 0,
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # best-effort — don't fail tests if unload fails


_current_ollama_model: list[str] = []   # mutable sentinel; holds at most one tag


@pytest.fixture
def compat_model(request: pytest.FixtureRequest) -> CompatModel:
    """
    Yields one CompatModel per parametrize iteration.
    Skips automatically when:
    - model's required_env is not set (cloud models)
    - ollama/ model string is not found in locally pulled models

    For local Ollama models: unloads the previously loaded model from VRAM when the
    active model changes, so only one model occupies GPU/RAM at a time.
    """
    model: CompatModel = request.param
    if model.required_env and not os.environ.get(model.required_env):
        pytest.skip(
            f"Skipped — {model.required_env} not set "
            f"(required for {model.description})"
        )
    if model.model_str.startswith(("ollama/", "ollama_chat/")):
        tag = model.model_str.split("/", 1)[1]
        pulled = _ollama_pulled_models()
        if not pulled:
            pytest.skip(f"Skipped — Ollama not running or unreachable (start with: ollama serve)")
        if tag not in pulled:
            pytest.skip(f"Skipped — {tag} not pulled in local Ollama (run: ollama pull {tag})")
        # Unload previous model if switching to a different one
        if _current_ollama_model and _current_ollama_model[0] != tag:
            _ollama_unload(_current_ollama_model[0])
        if not _current_ollama_model:
            _current_ollama_model.append(tag)
        else:
            _current_ollama_model[0] = tag
    return model
