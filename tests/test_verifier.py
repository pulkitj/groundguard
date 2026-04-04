"""Tests for core/verifier.py — TDD items T-26 (#6c, #9, #19a, #19b, gate-only, averify e2e)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_verifier.exceptions import (
    ParseError,
    VerificationCostExceededError,
    VerificationFailedError,
)
from agentic_verifier.loaders.chunker import Chunk
from agentic_verifier.models.internal import RoutingDecision, Tier2Result
from agentic_verifier.models.result import Source, VerificationResult
from agentic_verifier.models.tier3 import (
    AtomicVerification,
    ConceptualCoverage,
    SourceAttribution,
    TextualEntailment,
    Tier3ResponseModel,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sources() -> list[Source]:
    return [Source(content="Revenue was $5M.", source_id="doc.pdf")]


def _chunks() -> list[Chunk]:
    return [Chunk(parent_source_id="doc.pdf", text_content="Revenue was $5M.", char_start=0, char_end=16)]


def _valid_t3() -> Tier3ResponseModel:
    return Tier3ResponseModel(
        textual_entailment=TextualEntailment(label="Entailment", probability=0.95),
        conceptual_coverage=ConceptualCoverage(
            percentage=90.0, covered_concepts=["revenue"], missing_concepts=[]
        ),
        factual_consistency_score=90.0,
        verifications=[
            AtomicVerification(
                claim_text="Revenue was $5M.",
                status="VERIFIED",
                source_id="doc.pdf",
            )
        ],
        source_attributions=[SourceAttribution(source_id="doc.pdf", role="Supporting")],
        overall_verdict="The source fully supports the claim.",
    )


def _verified_result() -> VerificationResult:
    return VerificationResult(
        is_valid=True,
        overall_verdict="Verified",
        verification_method="tier3_llm",
        atomic_claims=[],
        factual_consistency_score=0.9,
        sources_used=["doc.pdf"],
        rationale="Supported",
        offending_claim=None,
        status="VERIFIED",
        total_cost_usd=0.001,
    )


def _escalate_t2() -> Tier2Result:
    return Tier2Result(
        decision=RoutingDecision.ESCALATE_TO_LLM,
        top_k_chunks=_chunks(),
        highest_score=0.5,
    )


# ---------------------------------------------------------------------------
# Convenience: common pipeline patches for synchronous verify()
# Every test patches classifier, chunker, tier1 so the module-level imports
# resolve even in RED state, and each test patches only the specific
# dependency it cares about.
# ---------------------------------------------------------------------------

_BASE_PATCHES = {
    "classifier": "agentic_verifier.core.verifier.classifier.parse_and_classify",
    "chunker": "agentic_verifier.core.verifier.chunker.chunk_sources",
    "tier1": "agentic_verifier.core.verifier.tier1_authenticity.check_fuzzy",
    "tier2": "agentic_verifier.core.verifier.tier2_semantic.route_claim",
    "tier3": "agentic_verifier.core.verifier.tier3_evaluation.evaluate",
    "builder_llm": "agentic_verifier.core.verifier.ResultBuilder.build_llm_result",
}


def _apply_base_patches(mocker, *, tier3_side_effect=None, tier3_return=None):
    """Apply standard no-op patches; returns dict of mock objects."""
    mocks = {}

    mocks["classifier"] = mocker.patch(
        _BASE_PATCHES["classifier"], return_value=[]
    )
    mocks["chunker"] = mocker.patch(
        _BASE_PATCHES["chunker"], return_value=_chunks()
    )
    mocks["tier1"] = mocker.patch(
        _BASE_PATCHES["tier1"], return_value=None
    )
    mocks["tier2"] = mocker.patch(
        _BASE_PATCHES["tier2"], return_value=_escalate_t2()
    )
    if tier3_side_effect is not None:
        mocks["tier3"] = mocker.patch(
            _BASE_PATCHES["tier3"], side_effect=tier3_side_effect
        )
    else:
        mocks["tier3"] = mocker.patch(
            _BASE_PATCHES["tier3"],
            return_value=(_valid_t3() if tier3_return is None else tier3_return),
        )
    mocks["builder_llm"] = mocker.patch(
        _BASE_PATCHES["builder_llm"], return_value=_verified_result()
    )
    return mocks


# ===========================================================================
# TDD #6c — ParseError caught by orchestrator → status="PARSE_ERROR"
# ===========================================================================

def test_parse_error_returns_parse_error_status(mocker):
    """#6c: ParseError from evaluate() must NOT propagate; result has status='PARSE_ERROR'."""
    from agentic_verifier.core.verifier import verify

    _apply_base_patches(mocker, tier3_side_effect=ParseError("bad json"))

    result = verify(claim="Revenue was $5M.", sources=_sources())

    assert result.status == "PARSE_ERROR"


def test_parse_error_sets_verification_method_skipped(mocker):
    """#6c: ParseError result must have verification_method='skipped'."""
    from agentic_verifier.core.verifier import verify

    _apply_base_patches(mocker, tier3_side_effect=ParseError("bad json"))

    result = verify(claim="Revenue was $5M.", sources=_sources())

    assert result.verification_method == "skipped"


# ===========================================================================
# TDD #19a — VerificationCostExceededError propagates (fail-loud)
# ===========================================================================

def test_cost_exceeded_propagates(mocker):
    """#19a: VerificationCostExceededError raised by evaluate() must propagate out of verify()."""
    from agentic_verifier.core.verifier import verify

    _apply_base_patches(mocker, tier3_side_effect=VerificationCostExceededError("cap hit"))

    with pytest.raises(VerificationCostExceededError):
        verify(claim="Revenue was $5M.", sources=_sources())


# ===========================================================================
# TDD #19b — litellm.APIConnectionError wrapped as VerificationFailedError
# ===========================================================================

def test_api_connection_error_wrapped_as_verification_failed(mocker):
    """#19b: litellm.exceptions.APIConnectionError from evaluate() → VerificationFailedError."""
    import litellm

    from agentic_verifier.core.verifier import verify

    _apply_base_patches(
        mocker,
        tier3_side_effect=litellm.exceptions.APIConnectionError("connection refused"),
    )

    with pytest.raises(VerificationFailedError):
        verify(claim="Revenue was $5M.", sources=_sources())


# ===========================================================================
# Tier 1 gate-only assertion — route_claim still called after check_fuzzy
# ===========================================================================

def test_tier1_gate_only_does_not_short_circuit(mocker):
    """Tier 1 returning without raising must NOT prevent tier2 route_claim from being called."""
    from agentic_verifier.core.verifier import verify

    mocks = _apply_base_patches(mocker)
    # tier1 returns a chunk (simulating a passing gate) without raising
    mocks["tier1"].return_value = _chunks()[0]

    verify(claim="Revenue was $5M.", sources=_sources())

    mocks["tier2"].assert_called_once()


# ===========================================================================
# TDD #9 — averify() dispatches BM25 (route_claim) to run_in_executor
# ===========================================================================

async def test_averify_dispatches_bm25_to_executor(mocker):
    """#9: averify() must offload tier2 route_claim via loop.run_in_executor (not blocking event loop)."""
    from agentic_verifier.tiers import tier2_semantic

    from agentic_verifier.core.verifier import averify

    # Patch classifier, chunker, tier1 so the pipeline can get to tier2
    mocker.patch(
        "agentic_verifier.core.verifier.classifier.parse_and_classify",
        return_value=[],
    )
    mocker.patch(
        "agentic_verifier.core.verifier.chunker.chunk_sources",
        return_value=_chunks(),
    )
    mocker.patch(
        "agentic_verifier.core.verifier.tier1_authenticity.check_fuzzy",
        return_value=None,
    )
    mocker.patch(
        "agentic_verifier.core.verifier.tier3_evaluation.evaluate_async",
        new_callable=AsyncMock,
        return_value=_valid_t3(),
    )
    mocker.patch(
        "agentic_verifier.core.verifier.ResultBuilder.build_llm_result",
        return_value=_verified_result(),
    )

    # Build a mock loop whose run_in_executor is an AsyncMock that returns the Tier2Result
    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=_escalate_t2())

    mocker.patch(
        "agentic_verifier.core.verifier.asyncio.get_running_loop",
        return_value=mock_loop,
    )

    await averify(claim="Revenue was $5M.", sources=_sources())

    # The second positional arg to run_in_executor must be tier2_semantic.route_claim
    call_args = mock_loop.run_in_executor.call_args
    assert call_args is not None, "run_in_executor was never called"
    positional = call_args[0]  # (executor, fn, *args)
    assert positional[1] is tier2_semantic.route_claim, (
        f"Expected tier2_semantic.route_claim to be dispatched to executor, got {positional[1]!r}"
    )


# ===========================================================================
# averify() end-to-end async path
# ===========================================================================

async def test_averify_end_to_end_async_returns_verified(mocker):
    """averify() wired end-to-end: fully mocked async path must return status='VERIFIED'."""
    from agentic_verifier.core.verifier import averify

    mocker.patch(
        "agentic_verifier.core.verifier.classifier.parse_and_classify",
        return_value=[],
    )
    mocker.patch(
        "agentic_verifier.core.verifier.chunker.chunk_sources",
        return_value=_chunks(),
    )
    mocker.patch(
        "agentic_verifier.core.verifier.tier1_authenticity.check_fuzzy",
        return_value=None,
    )
    mocker.patch(
        "agentic_verifier.core.verifier.tier2_semantic.route_claim",
        return_value=_escalate_t2(),
    )
    mocker.patch(
        "agentic_verifier.core.verifier.tier3_evaluation.evaluate_async",
        new_callable=AsyncMock,
        return_value=_valid_t3(),
    )
    mocker.patch(
        "agentic_verifier.core.verifier.ResultBuilder.build_llm_result",
        return_value=_verified_result(),
    )

    # Provide a real loop so run_in_executor just calls the fn synchronously via executor=None
    # but tier2 is already mocked so the executor won't actually reach route_claim.
    # We still need run_in_executor to return the Tier2Result.
    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=_escalate_t2())
    mocker.patch(
        "agentic_verifier.core.verifier.asyncio.get_running_loop",
        return_value=mock_loop,
    )

    result = await averify(claim="Revenue was $5M.", sources=_sources())

    assert result.status == "VERIFIED"
