# tests/integration/test_compat_suite.py
"""
Multi-model compatibility smoke suite — @pytest.mark.compat

These tests run against every model in compat_models.ALL_COMPAT_MODELS.
Assertions are structural: they verify the pipeline completes correctly and the
adapter extracts valid JSON — they do NOT assume GPT-4o-level reasoning quality.

Run all models:
    pytest -m compat -v --timeout=300

Run one model:
    pytest -m compat -v -k "nim-llama33" --timeout=300
"""
from __future__ import annotations

import pytest

from agentic_verifier.core.verifier import verify
from agentic_verifier.models.result import Source
from tests.integration.compat_models import CompatModel

VALID_STATUSES = {"VERIFIED", "CONTRADICTED", "UNVERIFIABLE", "PARSE_ERROR", "ERROR", "SKIPPED_DUE_TO_COST"}


# ---------------------------------------------------------------------------
# CS-01: Pipeline completes — any claim returns a valid status (no uncaught exception)
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_cs01_pipeline_completes(compat_model: CompatModel):
    """Any model must complete without uncaught exceptions and return a valid status."""
    result = verify(
        claim="The company reported revenue of $5 million in Q3.",
        sources=[
            Source(content="The company reported revenue of $5 million in Q3.", source_id="report.pdf"),
        ],
        model=compat_model.model_str,
        max_spend=1.0,
    )
    assert result.status in VALID_STATUSES, (
        f"[{compat_model.description}] Unexpected status: {result.status!r}"
    )
    assert isinstance(result.factual_consistency_score, float), (
        f"[{compat_model.description}] factual_consistency_score must be float"
    )


# ---------------------------------------------------------------------------
# CS-02: Lexical pass is model-agnostic — BM25 handles perfect match, no LLM call
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_cs02_lexical_pass_model_agnostic(compat_model: CompatModel):
    """Perfect verbatim match must be VERIFIED via tier2_lexical regardless of model."""
    # 5 sources ensures positive BM25 IDF (see T-34 for explanation)
    sources = [
        Source(content="The Q3 revenue was $5 million.", source_id="report.pdf"),
        Source(content="Engineering specifications and design documents.", source_id="eng.pdf"),
        Source(content="Legal terms governing the contractual agreement.", source_id="legal.pdf"),
        Source(content="Marketing strategy and promotional campaigns.", source_id="mkt.pdf"),
        Source(content="Human resources policy and staffing guidelines.", source_id="hr.pdf"),
    ]
    result = verify(
        claim="The Q3 revenue was $5 million.",
        sources=sources,
        model=compat_model.model_str,
        max_spend=1.0,
    )
    assert result.status == "VERIFIED", (
        f"[{compat_model.description}] Perfect match must be VERIFIED, got: {result.status!r}"
    )
    assert result.verification_method == "tier2_lexical", (
        f"[{compat_model.description}] Perfect match must skip LLM, got method: {result.verification_method!r}"
    )


# ---------------------------------------------------------------------------
# CS-03: Clear numeric contradiction — 10× magnitude error, unambiguous
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_cs03_clear_contradiction_detected(compat_model: CompatModel):
    """A 10× numeric error (300% vs 30%) must be detected as CONTRADICTED by any model."""
    result = verify(
        claim="Revenue grew by 300% year-over-year.",
        sources=[Source(content="Revenue grew by 30% year-over-year.", source_id="report.pdf")],
        model=compat_model.model_str,
        max_spend=1.0,
    )
    assert result.status == "CONTRADICTED", (
        f"[{compat_model.description}] 10× numeric error must be CONTRADICTED, got: {result.status!r}"
    )
    assert result.verification_method == "tier3_llm", (
        f"[{compat_model.description}] Must go through LLM for contradiction detection"
    )


# ---------------------------------------------------------------------------
# CS-04: Out-of-domain hallucination guard — claim not supported by any source
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_cs04_out_of_domain_not_verified(compat_model: CompatModel):
    """Claim about a product launch absent from financial source must NOT be VERIFIED."""
    result = verify(
        claim="The company launched a new AI product in 2023.",
        sources=[Source(
            content="Q3 revenue was $5 million. Net income was $1 million.",
            source_id="report.pdf",
        )],
        model=compat_model.model_str,
        max_spend=1.0,
    )
    assert result.status in ("CONTRADICTED", "UNVERIFIABLE"), (
        f"[{compat_model.description}] Out-of-domain hallucination must not be VERIFIED, got: {result.status!r}"
    )
