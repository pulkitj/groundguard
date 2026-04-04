"""Core verifier orchestrator."""
from __future__ import annotations

import asyncio

import litellm

from agentic_verifier._log import logger
from agentic_verifier.core import classifier
from agentic_verifier.exceptions import ParseError, VerificationFailedError
from agentic_verifier.loaders import chunker
from agentic_verifier.models.builder import ResultBuilder
from agentic_verifier.models.internal import (
    RoutingDecision,
    SharedCostTracker,
    VerificationContext,
)
from agentic_verifier.models.result import VerificationResult
from agentic_verifier.tiers import tier1_authenticity, tier2_semantic, tier3_evaluation

TRANSIENT_LITELLM_ERRORS = (
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
)


def verify(
    claim,
    sources,
    model="gpt-4o-mini",
    auto_chunk=True,
    chunk_size=500,
    chunk_overlap=50,
    max_source_tokens=8000,
    tier1_min_similarity=0.90,
    max_spend=0.50,
    agent_provided_evidence=None,
) -> VerificationResult:
    """Synchronous verification entry point."""
    if not claim or not isinstance(claim, str):
        raise ValueError("claim must be a non-empty string")
    if not sources:
        raise ValueError("sources must be a non-empty list")

    tracker = SharedCostTracker(max_spend=max_spend)
    ctx = VerificationContext(
        claim=claim,
        original_sources=sources,
        model=model,
        auto_chunk=auto_chunk,
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=chunk_overlap,
        max_source_tokens=max_source_tokens,
        tier1_min_similarity=tier1_min_similarity,
        agent_provided_evidence=agent_provided_evidence,
        cost_tracker=tracker,
    )

    ctx.tier0_atoms = classifier.parse_and_classify(claim)
    chunks = chunker.chunk_sources(ctx)

    if ctx.agent_provided_evidence:
        tier1_authenticity.check_fuzzy(
            ctx.agent_provided_evidence, chunks, ctx.tier1_min_similarity
        )

    t2_res = tier2_semantic.route_claim(ctx, chunks)

    if t2_res.decision == RoutingDecision.SKIP_LLM_HIGH_CONFIDENCE:
        result = ResultBuilder.build_lexical_pass(ctx, t2_res.top_k_chunks)
        logger.info(
            "verify(): status=%s method=%s cost=$%.4f [boundary=%s]",
            result.status,
            result.verification_method,
            ctx.cost_tracker.total_cost_usd,
            ctx._boundary_id,
        )
        return result

    try:
        t3_model = tier3_evaluation.evaluate(ctx, t2_res.top_k_chunks)
    except ParseError:
        result = VerificationResult(
            is_valid=False,
            overall_verdict="Verification failed — LLM returned unparseable output after retry.",
            verification_method="skipped",
            atomic_claims=[],
            factual_consistency_score=0.0,
            sources_used=[],
            rationale="Tier 3 LLM returned malformed JSON after 2 attempts. See logs for details.",
            offending_claim=None,
            status="PARSE_ERROR",
            total_cost_usd=ctx.cost_tracker.total_cost_usd,
        )
        logger.error(
            "verify(): ParseError — status=%s method=%s cost=$%.4f [boundary=%s]",
            result.status,
            result.verification_method,
            ctx.cost_tracker.total_cost_usd,
            ctx._boundary_id,
        )
        return result
    except TRANSIENT_LITELLM_ERRORS as e:
        raise VerificationFailedError(f"Upstream LLM transient failure: {type(e).__name__}.")

    result = ResultBuilder.build_llm_result(ctx, t3_model, "tier3_llm")
    logger.info(
        "verify(): status=%s method=%s cost=$%.4f [boundary=%s]",
        result.status,
        result.verification_method,
        ctx.cost_tracker.total_cost_usd,
        ctx._boundary_id,
    )
    return result


async def averify(
    claim,
    sources,
    model="gpt-4o-mini",
    auto_chunk=True,
    chunk_size=500,
    chunk_overlap=50,
    max_source_tokens=8000,
    tier1_min_similarity=0.90,
    max_spend=0.50,
    agent_provided_evidence=None,
    cost_tracker: SharedCostTracker | None = None,
) -> VerificationResult:
    """Asynchronous verification entry point."""
    if not claim or not isinstance(claim, str):
        raise ValueError("claim must be a non-empty string")
    if not sources:
        raise ValueError("sources must be a non-empty list")

    tracker = cost_tracker or SharedCostTracker(max_spend=max_spend)
    ctx = VerificationContext(
        claim=claim,
        original_sources=sources,
        model=model,
        auto_chunk=auto_chunk,
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=chunk_overlap,
        max_source_tokens=max_source_tokens,
        tier1_min_similarity=tier1_min_similarity,
        agent_provided_evidence=agent_provided_evidence,
        cost_tracker=tracker,
    )

    ctx.tier0_atoms = classifier.parse_and_classify(claim)
    chunks = chunker.chunk_sources(ctx)

    if ctx.agent_provided_evidence:
        tier1_authenticity.check_fuzzy(
            ctx.agent_provided_evidence, chunks, ctx.tier1_min_similarity
        )

    # BM25 is CPU-bound — dispatch to thread-pool executor to avoid blocking event loop
    loop = asyncio.get_running_loop()
    t2_res = await loop.run_in_executor(
        None, tier2_semantic.route_claim, ctx, chunks
    )

    if t2_res.decision == RoutingDecision.SKIP_LLM_HIGH_CONFIDENCE:
        result = ResultBuilder.build_lexical_pass(ctx, t2_res.top_k_chunks)
        logger.info(
            "averify(): status=%s method=%s cost=$%.4f [boundary=%s]",
            result.status,
            result.verification_method,
            ctx.cost_tracker.total_cost_usd,
            ctx._boundary_id,
        )
        return result

    try:
        t3_model = await tier3_evaluation.evaluate_async(ctx, t2_res.top_k_chunks)
    except ParseError:
        result = VerificationResult(
            is_valid=False,
            overall_verdict="Verification failed — LLM returned unparseable output after retry.",
            verification_method="skipped",
            atomic_claims=[],
            factual_consistency_score=0.0,
            sources_used=[],
            rationale="Tier 3 LLM returned malformed JSON after 2 attempts. See logs for details.",
            offending_claim=None,
            status="PARSE_ERROR",
            total_cost_usd=ctx.cost_tracker.total_cost_usd,
        )
        logger.error(
            "averify(): ParseError — status=%s method=%s cost=$%.4f [boundary=%s]",
            result.status,
            result.verification_method,
            ctx.cost_tracker.total_cost_usd,
            ctx._boundary_id,
        )
        return result
    except TRANSIENT_LITELLM_ERRORS as e:
        raise VerificationFailedError(f"Upstream LLM transient failure: {type(e).__name__}.")

    result = ResultBuilder.build_llm_result(ctx, t3_model, "tier3_llm")
    logger.info(
        "averify(): status=%s method=%s cost=$%.4f [boundary=%s]",
        result.status,
        result.verification_method,
        ctx.cost_tracker.total_cost_usd,
        ctx._boundary_id,
    )
    return result
