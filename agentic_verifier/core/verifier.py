"""Core verifier orchestrator."""
from __future__ import annotations

import asyncio

import litellm
import pydantic

from agentic_verifier._log import logger
from agentic_verifier.core import classifier
from agentic_verifier.exceptions import (
    ParseError,
    VerificationCostExceededError,
    VerificationFailedError,
)
from agentic_verifier.loaders import chunker
from agentic_verifier.models.builder import ResultBuilder
from agentic_verifier.models.internal import (
    ClaimInput,
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
    api_base=None,
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
        api_base=api_base,
        cost_tracker=tracker,
    )

    ctx.tier0_atoms = classifier.parse_and_classify(claim)
    if ctx.auto_chunk:
        chunks = chunker.chunk_sources(ctx)
    else:
        chunks = chunker.wrap_as_chunks(ctx.original_sources)

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
    api_base=None,
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
        api_base=api_base,
        cost_tracker=tracker,
    )

    ctx.tier0_atoms = classifier.parse_and_classify(claim)
    if ctx.auto_chunk:
        chunks = chunker.chunk_sources(ctx)
    else:
        chunks = chunker.wrap_as_chunks(ctx.original_sources)

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


async def verify_batch_async(
    inputs: list[ClaimInput],
    model: str = "gpt-4o-mini",
    max_concurrency: int = 5,
    max_spend: float = 0.50,
    **kwargs,
) -> list[VerificationResult]:
    """
    Runs all inputs concurrently under a semaphore. Each input gets its own
    fresh VerificationContext but all share one SharedCostTracker for global
    budget enforcement across the batch.

    On budget exhaustion: items that triggered VerificationCostExceededError
    are returned as SKIPPED_DUE_TO_COST results rather than raising.
    The batch always returns a result per input.

    Constraint: max_spend cannot be passed via **kwargs — use the named parameter.
    """
    if "max_spend" in kwargs:
        raise TypeError(
            "max_spend cannot be passed as a **kwargs to verify_batch_async. "
            "Use the max_spend parameter directly."
        )

    tracker = SharedCostTracker(max_spend=max_spend)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _sem_bounded_verify(inp: ClaimInput) -> VerificationResult:
        async with semaphore:
            return await averify(
                claim=inp.claim,
                sources=inp.sources,
                model=inp.model or model,
                agent_provided_evidence=inp.agent_provided_evidence,
                cost_tracker=tracker,
                **kwargs,
            )

    raw_results = await asyncio.gather(
        *(_sem_bounded_verify(i) for i in inputs),
        return_exceptions=True,
    )

    results: list[VerificationResult] = []
    for r in raw_results:
        if isinstance(r, VerificationCostExceededError):
            results.append(VerificationResult(
                is_valid=False,
                overall_verdict="Verification skipped — batch spend cap exceeded.",
                verification_method="skipped",
                atomic_claims=[],
                factual_consistency_score=0.0,
                sources_used=[],
                rationale=str(r),
                offending_claim=None,
                status="SKIPPED_DUE_TO_COST",
                total_cost_usd=0.0,  # item was not billed; incremental cost is zero
            ))
        elif isinstance(r, Exception):
            logger.error(
                "verify_batch: item failed with %s — returning ERROR result",
                type(r).__name__,
            )
            results.append(VerificationResult(
                is_valid=False,
                overall_verdict="Verification failed.",
                verification_method="skipped",
                atomic_claims=[],
                factual_consistency_score=0.0,
                sources_used=[],
                rationale=str(r),
                offending_claim=None,
                status="ERROR",
                total_cost_usd=0.0,  # item's incremental cost unknown on failure
            ))
        else:
            results.append(r)

    return results


def verify_batch(
    inputs: list[ClaimInput],
    model: str = "gpt-4o-mini",
    max_concurrency: int = 5,
    max_spend: float = 0.50,
    **kwargs,
) -> list[VerificationResult]:
    """
    Sync wrapper around verify_batch_async. Creates a new event loop via asyncio.run().

    Constraint: asyncio.run() cannot be called from within an already-running event
    loop (e.g., Jupyter notebooks, FastAPI request handlers). In those contexts,
    use verify_batch_async() directly with await.
    """
    return asyncio.run(
        verify_batch_async(
            inputs=inputs,
            model=model,
            max_concurrency=max_concurrency,
            max_spend=max_spend,
            **kwargs,
        )
    )


def dict_to_string_flattener(obj, prefix: str = "") -> str:
    """
    Recursively flattens nested dicts and lists into dot-notation key: value lines.

    Examples:
        {"revenue": "5M"}                          -> "revenue: 5M"
        {"company": {"q3": {"revenue": "5M"}}}     -> "company.q3.revenue: 5M"
        {"risks": ["regulatory", "market"]}        -> "risks[0]: regulatory\\nrisks[1]: market"
        {"items": [{"name": "A"}, {"name": "B"}]}  -> "items[0].name: A\\nitems[1].name: B"
    """
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            lines.append(dict_to_string_flattener(v, full_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            lines.append(dict_to_string_flattener(v, f"{prefix}[{i}]"))
    else:
        lines.append(f"{prefix}: {obj}")
    return "\n".join(filter(None, lines))


def verify_structured(
    claim_dict: dict,
    schema: type[pydantic.BaseModel],
    sources: list,
    **kwargs,
) -> VerificationResult:
    """
    Validates claim_dict against the provided Pydantic schema, flattens it to a
    string, then delegates to verify(). Raises ValueError on schema mismatch.
    """
    try:
        validated = schema.model_validate(claim_dict)
        normalised = validated.model_dump()
    except pydantic.ValidationError as e:
        raise ValueError(f"claim_dict does not conform to the provided schema: {e}")

    flattened = dict_to_string_flattener(normalised)
    return verify(claim=flattened, sources=sources, **kwargs)
