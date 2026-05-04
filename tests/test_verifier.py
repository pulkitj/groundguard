"""Tests for core/verifier.py — TDD items T-26 (#6c, #9, #19a, #19b, gate-only, averify e2e)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from groundguard.exceptions import (
    ParseError,
    VerificationCostExceededError,
    VerificationFailedError,
)
from groundguard.loaders.chunker import Chunk
from groundguard.models.internal import RoutingDecision, Tier2Result
from groundguard.models.result import Source, VerificationResult
from groundguard.models.tier3 import (
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
    return [Chunk(source_id="doc.pdf", text_content="Revenue was $5M.", char_start=0, char_end=16)]


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
    "classifier": "groundguard.core.verifier.classifier.parse_and_classify",
    "chunker": "groundguard.core.verifier.chunker.chunk_sources",
    "tier1": "groundguard.core.verifier.tier1_authenticity.check_fuzzy",
    "tier2": "groundguard.core.verifier.tier2_semantic.route_claim",
    "tier3": "groundguard.core.verifier.tier3_evaluation.evaluate",
    "builder_llm": "groundguard.core.verifier.ResultBuilder.build_llm_result",
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
    from groundguard.core.verifier import verify

    _apply_base_patches(mocker, tier3_side_effect=ParseError("bad json"))

    result = verify(claim="Revenue was $5M.", sources=_sources())

    assert result.status == "PARSE_ERROR"


def test_parse_error_sets_verification_method_skipped(mocker):
    """#6c: ParseError result must have verification_method='skipped'."""
    from groundguard.core.verifier import verify

    _apply_base_patches(mocker, tier3_side_effect=ParseError("bad json"))

    result = verify(claim="Revenue was $5M.", sources=_sources())

    assert result.verification_method == "skipped"


# ===========================================================================
# TDD #19a — VerificationCostExceededError propagates (fail-loud)
# ===========================================================================

def test_cost_exceeded_propagates(mocker):
    """#19a: VerificationCostExceededError raised by evaluate() must propagate out of verify()."""
    from groundguard.core.verifier import verify

    _apply_base_patches(mocker, tier3_side_effect=VerificationCostExceededError("cap hit"))

    with pytest.raises(VerificationCostExceededError):
        verify(claim="Revenue was $5M.", sources=_sources())


# ===========================================================================
# TDD #19b — litellm.APIConnectionError wrapped as VerificationFailedError
# ===========================================================================

def test_api_connection_error_wrapped_as_verification_failed(mocker):
    """#19b: litellm.exceptions.APIConnectionError from evaluate() → VerificationFailedError."""
    import litellm

    from groundguard.core.verifier import verify

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
    from groundguard.core.verifier import verify

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
    from groundguard.tiers import tier2_semantic

    from groundguard.core.verifier import averify

    # Patch classifier, chunker, tier1 so the pipeline can get to tier2
    mocker.patch(
        "groundguard.core.verifier.classifier.parse_and_classify",
        return_value=[],
    )
    mocker.patch(
        "groundguard.core.verifier.chunker.chunk_sources",
        return_value=_chunks(),
    )
    mocker.patch(
        "groundguard.core.verifier.tier1_authenticity.check_fuzzy",
        return_value=None,
    )
    mocker.patch(
        "groundguard.core.verifier.tier3_evaluation.evaluate_async",
        new_callable=AsyncMock,
        return_value=_valid_t3(),
    )
    mocker.patch(
        "groundguard.core.verifier.ResultBuilder.build_llm_result",
        return_value=_verified_result(),
    )

    # Build a mock loop whose run_in_executor is an AsyncMock that returns the Tier2Result
    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=_escalate_t2())

    mocker.patch(
        "groundguard.core.verifier.asyncio.get_running_loop",
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
    from groundguard.core.verifier import averify

    mocker.patch(
        "groundguard.core.verifier.classifier.parse_and_classify",
        return_value=[],
    )
    mocker.patch(
        "groundguard.core.verifier.chunker.chunk_sources",
        return_value=_chunks(),
    )
    mocker.patch(
        "groundguard.core.verifier.tier1_authenticity.check_fuzzy",
        return_value=None,
    )
    mocker.patch(
        "groundguard.core.verifier.tier2_semantic.route_claim",
        return_value=_escalate_t2(),
    )
    mocker.patch(
        "groundguard.core.verifier.tier3_evaluation.evaluate_async",
        new_callable=AsyncMock,
        return_value=_valid_t3(),
    )
    mocker.patch(
        "groundguard.core.verifier.ResultBuilder.build_llm_result",
        return_value=_verified_result(),
    )

    # Provide a real loop so run_in_executor just calls the fn synchronously via executor=None
    # but tier2 is already mocked so the executor won't actually reach route_claim.
    # We still need run_in_executor to return the Tier2Result.
    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=_escalate_t2())
    mocker.patch(
        "groundguard.core.verifier.asyncio.get_running_loop",
        return_value=mock_loop,
    )

    result = await averify(claim="Revenue was $5M.", sources=_sources())

    assert result.status == "VERIFIED"


# ===========================================================================
# T-52 — Empty / missing data guards raise ValueError
# ===========================================================================

def test_verify_empty_claim_raises_value_error():
    """T-52a: verify(claim='', sources=[...]) raises ValueError before pipeline runs."""
    from groundguard.core.verifier import verify
    with pytest.raises(ValueError):
        verify(claim="", sources=_sources())


def test_verify_none_claim_raises_value_error():
    """T-52b: verify(claim=None, sources=[...]) raises ValueError."""
    from groundguard.core.verifier import verify
    with pytest.raises(ValueError):
        verify(claim=None, sources=_sources())  # type: ignore[arg-type]


def test_verify_empty_sources_raises_value_error():
    """T-52c: verify(claim='valid', sources=[]) raises ValueError."""
    from groundguard.core.verifier import verify
    with pytest.raises(ValueError):
        verify(claim="Revenue was $5M.", sources=[])


# ===========================================================================
# T-53 — Tiny budget: single-item batch returns SKIPPED_DUE_TO_COST
# ===========================================================================

async def test_tiny_budget_single_item_batch_returns_skipped_due_to_cost(mocker):
    """T-53: max_spend=0.000001 with LLM path → status='SKIPPED_DUE_TO_COST', no network call.

    verify() is fail-loud (VerificationCostExceededError propagates).
    verify_batch_async absorbs it → SKIPPED_DUE_TO_COST. Test via 1-item batch.
    """
    from unittest.mock import AsyncMock, MagicMock
    from groundguard.core.verifier import verify_batch_async
    from groundguard.exceptions import VerificationCostExceededError
    from groundguard.models.internal import ClaimInput, VerificationContext

    mocker.patch("groundguard.core.verifier.classifier.parse_and_classify", return_value=[])
    mocker.patch("groundguard.core.verifier.chunker.chunk_sources", return_value=_chunks())
    mocker.patch("groundguard.core.verifier.tier1_authenticity.check_fuzzy", return_value=None)

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=_escalate_t2())
    mocker.patch("groundguard.core.verifier.asyncio.get_running_loop", return_value=mock_loop)

    async def _raise_cost_exceeded(ctx: VerificationContext, _chunks):
        ctx.cost_tracker.add_cost(1.0)  # always blows the cap
        return _valid_t3()

    mocker.patch(
        "groundguard.core.verifier.tier3_evaluation.evaluate_async",
        side_effect=_raise_cost_exceeded,
    )

    inputs = [ClaimInput(claim="Revenue was $5M.", sources=_sources())]
    results = await verify_batch_async(inputs=inputs, max_spend=0.000001)

    assert len(results) == 1
    assert results[0].status == "SKIPPED_DUE_TO_COST"
    assert results[0].total_cost_usd == 0.0


# ===========================================================================
# T-30 — verify_structured: schema failure raises ValueError
# ===========================================================================

def test_verify_structured_schema_failure():
    from pydantic import BaseModel
    from groundguard.core.verifier import verify_structured
    from groundguard.models.result import Source

    class MySchema(BaseModel):
        revenue: float  # expects a float

    sources = [Source(content="Revenue was $5M.", source_id="doc.pdf")]
    # Pass a string for a float field -> Pydantic ValidationError -> ValueError
    with pytest.raises(ValueError):
        verify_structured(
            claim_dict={"revenue": "not-a-number"},
            schema=MySchema,
            sources=sources,
        )


# ===========================================================================
# T-30 — verify_structured: schema success calls verify() with flattened string
# ===========================================================================

async def test_averify_cost_exceeded_error_propagates(mocker):
    """averify() must propagate VerificationCostExceededError (fail-loud contract)."""
    from groundguard.core.verifier import averify
    from groundguard.exceptions import VerificationCostExceededError
    from unittest.mock import AsyncMock, MagicMock

    mocker.patch("groundguard.core.verifier.classifier.parse_and_classify", return_value=[])
    mocker.patch("groundguard.core.verifier.chunker.chunk_sources", return_value=_chunks())
    mocker.patch("groundguard.core.verifier.tier1_authenticity.check_fuzzy", return_value=None)

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=_escalate_t2())
    mocker.patch("groundguard.core.verifier.asyncio.get_running_loop", return_value=mock_loop)

    mocker.patch(
        "groundguard.core.verifier.tier3_evaluation.evaluate_async",
        new_callable=AsyncMock,
        side_effect=VerificationCostExceededError("cap hit"),
    )

    with pytest.raises(VerificationCostExceededError):
        await averify(claim="Revenue was $5M.", sources=_sources())


def test_auto_chunk_false_pipeline_path(mocker):
    """auto_chunk=False: source forwarded as single chunk, not split by chunker."""
    from groundguard.core.verifier import verify

    chunker_mock = mocker.patch("groundguard.core.verifier.chunker.chunk_sources")
    mocker.patch("groundguard.core.verifier.classifier.parse_and_classify", return_value=[])
    mocker.patch("groundguard.core.verifier.tier1_authenticity.check_fuzzy", return_value=None)
    mocker.patch("groundguard.core.verifier.tier2_semantic.route_claim", return_value=_escalate_t2())
    mocker.patch("groundguard.core.verifier.tier3_evaluation.evaluate", return_value=_valid_t3())
    mocker.patch("groundguard.core.verifier.ResultBuilder.build_llm_result", return_value=_verified_result())

    verify(claim="Revenue was $5M.", sources=_sources(), auto_chunk=False)

    # chunker.chunk_sources should NOT be called when auto_chunk=False
    chunker_mock.assert_not_called()


def test_verify_structured_schema_success(mocker):
    from pydantic import BaseModel
    from groundguard.core.verifier import verify_structured
    from groundguard.models.result import Source

    class RevenueSchema(BaseModel):
        revenue: float
        period: str

    # Patch verify() itself to capture what flattened string it receives
    mock_verify = mocker.patch(
        "groundguard.core.verifier.verify",
        return_value=_verified_result(),
    )

    sources = [Source(content="Revenue was $5M.", source_id="doc.pdf")]
    result = verify_structured(
        claim_dict={"revenue": 5.0, "period": "Q3"},
        schema=RevenueSchema,
        sources=sources,
    )

    assert result.status == "VERIFIED"
    # verify() must have been called with a flattened string (not the raw dict)
    call_args = mock_verify.call_args
    flattened_claim = call_args[1]["claim"] if "claim" in call_args[1] else call_args[0][0]
    assert isinstance(flattened_claim, str)
    assert "revenue" in flattened_claim
    assert "period" in flattened_claim


# ---------------------------------------------------------------------------
# Phase 24 — Profile param + Tier 2.5 wiring (T-86)
# ---------------------------------------------------------------------------

def _mock_tier3_result(status: str):
    from groundguard.models.tier3 import (
        Tier3ResponseModel, TextualEntailment, ConceptualCoverage,
        AtomicVerification, SourceAttribution,
    )
    return Tier3ResponseModel(
        textual_entailment=TextualEntailment(label="Entailment", probability=0.95),
        conceptual_coverage=ConceptualCoverage(
            percentage=90.0, covered_concepts=["x"], missing_concepts=[]
        ),
        factual_consistency_score=90.0,
        verifications=[AtomicVerification(claim_text="x", status=status, source_id="s1")],
        source_attributions=[SourceAttribution(source_id="s1", role="Supporting")],
        overall_verdict="Supported.",
    )


def test_verify_profile_strict_disables_bm25_fast_path(mocker):
    from groundguard.core.verifier import verify
    from groundguard.models.result import Source
    from groundguard.profiles import STRICT_PROFILE
    # STRICT_PROFILE.tier2_lexical_threshold=2.0 -> BM25 score never reaches 2.0 -> always escalates to LLM
    mock_llm = mocker.patch("groundguard.tiers.tier3_evaluation.evaluate")
    mock_llm.return_value = _mock_tier3_result("VERIFIED")
    src = Source(source_id="s1", content="Revenue was $4.2M.")
    verify("Revenue was $4.2M.", [src], profile=STRICT_PROFILE, model="gpt-4o-mini")
    mock_llm.assert_called_once()


def test_verify_profile_general_default(mocker):
    from groundguard.core.verifier import verify
    from groundguard.models.result import Source
    from groundguard.profiles import GENERAL_PROFILE
    mocker.patch("groundguard.tiers.tier3_evaluation.evaluate",
                 return_value=_mock_tier3_result("VERIFIED"))
    src = Source(source_id="s1", content="x")
    result = verify("x", [src], model="gpt-4o-mini")
    # no error — GENERAL_PROFILE is default
    assert result is not None


def test_verify_tier25_triggers_on_numerical_conflict():
    from groundguard.core.verifier import verify
    from groundguard.models.result import Source
    src = Source(source_id="s1", content="The fee shall not exceed 30% of revenue.")
    result = verify("The fee shall not exceed 300% of revenue.", [src], model="gpt-4o-mini")
    assert result.status == "CONTRADICTED"
    assert result.verification_method == "tier25_numerical"


def test_averify_profile_param(mocker):
    import asyncio
    from groundguard.core.verifier import averify
    from groundguard.models.result import Source
    from groundguard.profiles import GENERAL_PROFILE
    mocker.patch("groundguard.tiers.tier3_evaluation.evaluate",
                 return_value=_mock_tier3_result("VERIFIED"))
    src = Source(source_id="s1", content="x")
    result = asyncio.get_event_loop().run_until_complete(
        averify("x", [src], profile=GENERAL_PROFILE, model="gpt-4o-mini")
    )
    assert result is not None
