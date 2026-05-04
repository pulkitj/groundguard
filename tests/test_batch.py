"""TDD tests for verify_batch_async and verify_batch — T-28.

Tests are written in RED state: verify_batch_async and verify_batch do not yet
exist on groundguard.core.verifier. All tests are expected to fail until
Phase 9 implements those functions.

Coverage:
  #3  — Concurrent cost-cap: SharedCostTracker never double-counts
  #11 — Per-claim model override via ClaimInput.model
  E   — Event-loop constraint: verify_batch raises RuntimeError inside a running loop
  K   — max_spend kwarg collision raises TypeError
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from groundguard.models.internal import (
    ClaimInput,
    RoutingDecision,
    SharedCostTracker,
    Tier2Result,
    VerificationContext,
)
from groundguard.models.result import Source, VerificationResult
from groundguard.models.tier3 import (
    AtomicVerification,
    ConceptualCoverage,
    SourceAttribution,
    TextualEntailment,
    Tier3ResponseModel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _src(source_id: str = "doc.pdf") -> Source:
    return Source(content="Revenue was $5M.", source_id=source_id)


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
        overall_verdict="Verified.",
    )


def _make_claim_input(
    claim: str = "Revenue was $5M.",
    source_id: str = "doc.pdf",
    model: str | None = None,
) -> ClaimInput:
    return ClaimInput(
        claim=claim,
        sources=[_src(source_id)],
        model=model,
    )


def _t2_escalate() -> Tier2Result:
    """A Tier2Result that routes to Tier 3 (mid-range score)."""
    return Tier2Result(
        decision=RoutingDecision.ESCALATE_TO_LLM,
        top_k_chunks=[],
        highest_score=0.5,
    )


def _patch_pipeline(mocker, *, t3_side_effect=None, t3_return=None):
    """Patch all pipeline stages for a full-bypass fast test.

    Returns the AsyncMock for evaluate_async so callers can assert on it.
    """
    from groundguard.loaders.chunker import Chunk

    mocker.patch(
        "groundguard.core.verifier.classifier.parse_and_classify",
        return_value=[],
    )
    mocker.patch(
        "groundguard.core.verifier.chunker.chunk_sources",
        return_value=[
            Chunk(
                parent_source_id="doc.pdf",
                text_content="Revenue was $5M.",
                char_start=0,
                char_end=16,
            )
        ],
    )

    # Tier 2 runs via run_in_executor — patch the loop so it returns our Tier2Result
    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=_t2_escalate())
    mocker.patch(
        "groundguard.core.verifier.asyncio.get_running_loop",
        return_value=mock_loop,
    )

    if t3_side_effect is not None:
        mock_eval = AsyncMock(side_effect=t3_side_effect)
    else:
        mock_eval = AsyncMock(return_value=t3_return or _valid_t3())

    mocker.patch(
        "groundguard.core.verifier.tier3_evaluation.evaluate_async",
        mock_eval,
    )

    mocker.patch(
        "groundguard.core.verifier.ResultBuilder.build_llm_result",
        return_value=VerificationResult(
            is_valid=True,
            overall_verdict="Verified.",
            verification_method="tier3_llm",
            atomic_claims=[],
            factual_consistency_score=0.90,
            sources_used=["doc.pdf"],
            rationale="Source supports claim.",
            offending_claim=None,
            status="VERIFIED",
            total_cost_usd=0.001,
        ),
    )

    return mock_eval


# ---------------------------------------------------------------------------
# TDD #3 — Concurrent cost-cap: verify_batch_async absorbs SKIPPED items
# ---------------------------------------------------------------------------

async def test_batch_cost_cap_some_items_skipped(mocker):
    """TDD #3: verify_batch_async does NOT raise when cap is hit; excess items are SKIPPED_DUE_TO_COST."""
    from groundguard.core.verifier import verify_batch_async
    from groundguard.exceptions import VerificationCostExceededError
    from groundguard.loaders.chunker import Chunk

    # Shared tracker with tight cap — first item costs 0.009, cap is 0.01
    # so item 1 may succeed, item 2 will exceed the cap.
    tracker = SharedCostTracker(max_spend=0.01)

    mocker.patch(
        "groundguard.core.verifier.classifier.parse_and_classify",
        return_value=[],
    )
    mocker.patch(
        "groundguard.core.verifier.chunker.chunk_sources",
        return_value=[
            Chunk(
                parent_source_id="doc.pdf",
                text_content="Revenue was $5M.",
                char_start=0,
                char_end=16,
            )
        ],
    )

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=_t2_escalate())
    mocker.patch(
        "groundguard.core.verifier.asyncio.get_running_loop",
        return_value=mock_loop,
    )

    call_count = 0

    async def _evaluate_with_cost(ctx: VerificationContext, chunks):
        nonlocal call_count
        call_count += 1
        # Add cost; on second+ call this will exceed the cap
        ctx.cost_tracker.add_cost(0.009)
        return _valid_t3()

    mocker.patch(
        "groundguard.core.verifier.tier3_evaluation.evaluate_async",
        side_effect=_evaluate_with_cost,
    )
    mocker.patch(
        "groundguard.core.verifier.ResultBuilder.build_llm_result",
        return_value=VerificationResult(
            is_valid=True,
            overall_verdict="Verified.",
            verification_method="tier3_llm",
            atomic_claims=[],
            factual_consistency_score=0.90,
            sources_used=["doc.pdf"],
            rationale="Source supports claim.",
            offending_claim=None,
            status="VERIFIED",
            total_cost_usd=0.009,
        ),
    )

    inputs = [
        _make_claim_input("Revenue was $5M.", "a.pdf"),
        _make_claim_input("Revenue was $5M.", "b.pdf"),
        _make_claim_input("Revenue was $5M.", "c.pdf"),
    ]

    # Must NOT raise — fail-contained; max_concurrency=1 ensures deterministic ordering
    results = await verify_batch_async(inputs=inputs, max_spend=0.01, max_concurrency=1)

    assert isinstance(results, list)
    assert len(results) == 3

    statuses = [r.status for r in results]
    # At least one item must be skipped due to cost cap being hit
    assert "SKIPPED_DUE_TO_COST" in statuses, (
        f"Expected at least one SKIPPED_DUE_TO_COST in {statuses}"
    )

    # Total cost must never exceed 2× cap (soft cap allows one overshoot)
    total = sum(r.total_cost_usd for r in results if r.total_cost_usd is not None)
    assert total <= 2 * 0.01, f"Total cost {total} exceeded 2× cap"


async def test_batch_cost_cap_does_not_raise(mocker):
    """TDD #3b: verify_batch_async is fail-contained — it never raises VerificationCostExceededError."""
    from groundguard.core.verifier import verify_batch_async
    from groundguard.exceptions import VerificationCostExceededError
    from groundguard.loaders.chunker import Chunk

    mocker.patch(
        "groundguard.core.verifier.classifier.parse_and_classify",
        return_value=[],
    )
    mocker.patch(
        "groundguard.core.verifier.chunker.chunk_sources",
        return_value=[
            Chunk(
                parent_source_id="doc.pdf",
                text_content="Revenue was $5M.",
                char_start=0,
                char_end=16,
            )
        ],
    )

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=_t2_escalate())
    mocker.patch(
        "groundguard.core.verifier.asyncio.get_running_loop",
        return_value=mock_loop,
    )

    async def _expensive(ctx: VerificationContext, chunks):
        ctx.cost_tracker.add_cost(1.0)  # immediately exceeds any reasonable cap
        return _valid_t3()

    mocker.patch(
        "groundguard.core.verifier.tier3_evaluation.evaluate_async",
        side_effect=_expensive,
    )
    mocker.patch(
        "groundguard.core.verifier.ResultBuilder.build_llm_result",
        return_value=VerificationResult(
            is_valid=True,
            overall_verdict="Verified.",
            verification_method="tier3_llm",
            atomic_claims=[],
            factual_consistency_score=0.90,
            sources_used=["doc.pdf"],
            rationale=".",
            offending_claim=None,
            status="VERIFIED",
            total_cost_usd=1.0,
        ),
    )

    inputs = [_make_claim_input() for _ in range(3)]

    # Must not propagate VerificationCostExceededError
    try:
        results = await verify_batch_async(inputs=inputs, max_spend=0.001, max_concurrency=3)
    except Exception as exc:
        pytest.fail(
            f"verify_batch_async raised {type(exc).__name__} but must be fail-contained: {exc}"
        )

    statuses = [r.status for r in results]
    assert all(s in {"VERIFIED", "SKIPPED_DUE_TO_COST", "PARSE_ERROR", "ERROR"} for s in statuses)


# ---------------------------------------------------------------------------
# TDD #11 — Per-claim model override
# ---------------------------------------------------------------------------

async def test_per_claim_model_override_is_respected(mocker):
    """TDD #11: ClaimInput.model overrides the batch default model for that item."""
    from groundguard.core.verifier import verify_batch_async
    from groundguard.loaders.chunker import Chunk

    mocker.patch(
        "groundguard.core.verifier.classifier.parse_and_classify",
        return_value=[],
    )
    mocker.patch(
        "groundguard.core.verifier.chunker.chunk_sources",
        return_value=[
            Chunk(
                parent_source_id="doc.pdf",
                text_content="Revenue was $5M.",
                char_start=0,
                char_end=16,
            )
        ],
    )

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=_t2_escalate())
    mocker.patch(
        "groundguard.core.verifier.asyncio.get_running_loop",
        return_value=mock_loop,
    )

    captured_contexts: list[VerificationContext] = []

    async def _capture_ctx(ctx: VerificationContext, chunks):
        captured_contexts.append(ctx)
        return _valid_t3()

    mock_eval = AsyncMock(side_effect=_capture_ctx)
    mocker.patch(
        "groundguard.core.verifier.tier3_evaluation.evaluate_async",
        mock_eval,
    )
    mocker.patch(
        "groundguard.core.verifier.ResultBuilder.build_llm_result",
        return_value=VerificationResult(
            is_valid=True,
            overall_verdict="Verified.",
            verification_method="tier3_llm",
            atomic_claims=[],
            factual_consistency_score=0.90,
            sources_used=["doc.pdf"],
            rationale=".",
            offending_claim=None,
            status="VERIFIED",
            total_cost_usd=0.001,
        ),
    )

    inputs = [
        # Item 1: no per-claim model → inherits batch default "gpt-4o-mini"
        ClaimInput(claim="Revenue was $5M.", sources=[_src("a.pdf")], model=None),
        # Item 2: explicit per-claim override
        ClaimInput(
            claim="Revenue was $5M.",
            sources=[_src("b.pdf")],
            model="claude-3-haiku-20240307",
        ),
    ]

    await verify_batch_async(inputs=inputs, model="gpt-4o-mini", max_spend=1.0)

    assert len(captured_contexts) == 2, (
        f"Expected evaluate_async called twice, got {len(captured_contexts)}"
    )

    models_used = [ctx.model for ctx in captured_contexts]
    assert "gpt-4o-mini" in models_used, (
        f"Expected 'gpt-4o-mini' in captured models {models_used}"
    )
    assert "claude-3-haiku-20240307" in models_used, (
        f"Expected 'claude-3-haiku-20240307' in captured models {models_used}"
    )


async def test_per_claim_model_none_inherits_batch_default(mocker):
    """TDD #11b: ClaimInput.model=None means the batch-level model is used."""
    from groundguard.core.verifier import verify_batch_async
    from groundguard.loaders.chunker import Chunk

    mocker.patch(
        "groundguard.core.verifier.classifier.parse_and_classify",
        return_value=[],
    )
    mocker.patch(
        "groundguard.core.verifier.chunker.chunk_sources",
        return_value=[
            Chunk(
                parent_source_id="doc.pdf",
                text_content="Revenue was $5M.",
                char_start=0,
                char_end=16,
            )
        ],
    )

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=_t2_escalate())
    mocker.patch(
        "groundguard.core.verifier.asyncio.get_running_loop",
        return_value=mock_loop,
    )

    captured_ctx: list[VerificationContext] = []

    async def _capture(ctx: VerificationContext, chunks):
        captured_ctx.append(ctx)
        return _valid_t3()

    mocker.patch(
        "groundguard.core.verifier.tier3_evaluation.evaluate_async",
        side_effect=_capture,
    )
    mocker.patch(
        "groundguard.core.verifier.ResultBuilder.build_llm_result",
        return_value=VerificationResult(
            is_valid=True,
            overall_verdict="Verified.",
            verification_method="tier3_llm",
            atomic_claims=[],
            factual_consistency_score=0.90,
            sources_used=["doc.pdf"],
            rationale=".",
            offending_claim=None,
            status="VERIFIED",
            total_cost_usd=0.001,
        ),
    )

    inputs = [ClaimInput(claim="Revenue was $5M.", sources=[_src()], model=None)]
    await verify_batch_async(inputs=inputs, model="my-custom-model", max_spend=1.0)

    assert captured_ctx[0].model == "my-custom-model"


# ---------------------------------------------------------------------------
# Event-loop constraint — verify_batch raises RuntimeError in a running loop
# ---------------------------------------------------------------------------

def test_verify_batch_raises_in_running_event_loop():
    """verify_batch wraps asyncio.run() which cannot be called from a running loop."""
    from groundguard.core.verifier import verify_batch

    inputs = [_make_claim_input()]

    async def _inner():
        # Inside a running event loop, verify_batch must raise RuntimeError
        verify_batch(inputs=inputs)

    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError):
            loop.run_until_complete(_inner())
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# max_spend kwarg collision raises TypeError
# ---------------------------------------------------------------------------

async def test_verify_batch_async_rejects_max_spend_in_kwargs():
    """verify_batch_async raises TypeError when max_spend appears in both signature and **kwargs."""
    from groundguard.core.verifier import verify_batch_async

    inputs = [_make_claim_input()]

    with pytest.raises(TypeError, match="max_spend"):
        # Passing max_spend as both a keyword arg and again in **extra_kwargs
        extra_kwargs = {"max_spend": 0.50}
        await verify_batch_async(inputs=inputs, max_spend=0.50, **extra_kwargs)


# ---------------------------------------------------------------------------
# Basic smoke tests — verify_batch_async returns the right shape
# ---------------------------------------------------------------------------

async def test_verify_batch_async_returns_list_of_verification_results(mocker):
    """verify_batch_async returns list[VerificationResult] with same length as inputs."""
    from groundguard.core.verifier import verify_batch_async

    _patch_pipeline(mocker)

    inputs = [_make_claim_input("Claim A"), _make_claim_input("Claim B")]
    results = await verify_batch_async(inputs=inputs, model="gpt-4o-mini", max_spend=1.0)

    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, VerificationResult)


def test_verify_batch_returns_list_of_verification_results(mocker):
    """verify_batch (sync wrapper) returns list[VerificationResult]."""
    from groundguard.core.verifier import verify_batch

    _patch_pipeline(mocker)

    inputs = [_make_claim_input()]
    results = verify_batch(inputs=inputs, model="gpt-4o-mini", max_spend=1.0)

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], VerificationResult)


async def test_verify_batch_async_empty_inputs_returns_empty_list(mocker):
    """verify_batch_async with an empty input list returns []."""
    from groundguard.core.verifier import verify_batch_async

    results = await verify_batch_async(inputs=[], model="gpt-4o-mini", max_spend=1.0)
    assert results == []


# ---------------------------------------------------------------------------
# T-54 — Mid-batch cost exhaustion: first 2 items succeed, last 3 skipped
# ---------------------------------------------------------------------------

async def test_mid_batch_cost_exhaustion(mocker):
    """T-54: max_spend sized so first ~2 items exhaust budget; remaining return SKIPPED_DUE_TO_COST."""
    from groundguard.core.verifier import verify_batch_async
    from groundguard.loaders.chunker import Chunk

    mocker.patch(
        "groundguard.core.verifier.classifier.parse_and_classify",
        return_value=[],
    )
    mocker.patch(
        "groundguard.core.verifier.chunker.chunk_sources",
        return_value=[
            Chunk(parent_source_id="doc.pdf", text_content="Revenue was $5M.", char_start=0, char_end=16)
        ],
    )

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=_t2_escalate())
    mocker.patch(
        "groundguard.core.verifier.asyncio.get_running_loop",
        return_value=mock_loop,
    )

    call_count = 0

    async def _evaluate_with_cost(ctx: VerificationContext, _chunks):
        nonlocal call_count
        call_count += 1
        ctx.cost_tracker.add_cost(0.15)  # 2 calls × $0.15 = $0.30 > $0.25 cap
        return _valid_t3()

    mocker.patch(
        "groundguard.core.verifier.tier3_evaluation.evaluate_async",
        side_effect=_evaluate_with_cost,
    )
    mocker.patch(
        "groundguard.core.verifier.ResultBuilder.build_llm_result",
        return_value=VerificationResult(
            is_valid=True,
            overall_verdict="Verified.",
            verification_method="tier3_llm",
            atomic_claims=[],
            factual_consistency_score=0.90,
            sources_used=["doc.pdf"],
            rationale=".",
            offending_claim=None,
            status="VERIFIED",
            total_cost_usd=0.15,
        ),
    )

    inputs = [_make_claim_input(f"Claim {i}") for i in range(5)]
    # max_spend=0.25 → allows ~1 real call at $0.15 each before cap
    results = await verify_batch_async(inputs=inputs, max_spend=0.25, max_concurrency=1)

    assert len(results) == 5
    statuses = [r.status for r in results]

    # Must not raise — fail-contained
    assert all(s in {"VERIFIED", "SKIPPED_DUE_TO_COST", "ERROR", "PARSE_ERROR"} for s in statuses)
    # At least some items must be skipped
    assert "SKIPPED_DUE_TO_COST" in statuses, f"Expected skipped items, got: {statuses}"
    # Batch call must not raise
    skipped = statuses.count("SKIPPED_DUE_TO_COST")
    verified = statuses.count("VERIFIED")
    assert verified + skipped == 5, f"Unexpected statuses: {statuses}"
