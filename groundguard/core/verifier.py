"""Core verifier orchestrator."""
from __future__ import annotations

import asyncio
import dataclasses

import litellm
import pydantic

from groundguard._log import logger
from groundguard.core import classifier
from groundguard.exceptions import (
    InvariantError,
    ParseError,
    VerificationCostExceededError,
    VerificationFailedError,
)
from groundguard.loaders import chunker
from groundguard.models.builder import ResultBuilder
from groundguard.models.internal import (
    ClaimInput,
    RoutingDecision,
    SharedCostTracker,
    VerificationContext,
)
from groundguard.core import claim_extractor
from groundguard.models.result import GroundingResult, VerificationResult
from groundguard.profiles import GENERAL_PROFILE, VerificationProfile
from groundguard.tiers import tier1_authenticity, tier2_semantic, tier3_evaluation, tier25_preprocessing
from groundguard.core.result_builder import ResultBuilder as CoreResultBuilder

from groundguard._constants import TRANSIENT_LITELLM_ERRORS  # FIX-02: unified tuple
from groundguard.loaders.legal import decompose_clause


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
    profile: VerificationProfile | None = None,
    context: str | None = None,
) -> VerificationResult:
    """Synchronous verification entry point.
    
    Raises:
        InvariantError: verification pipeline produced an internally inconsistent result
                        (e.g. VERIFIED claim with no citation). The claim cannot be trusted.
                        Caller should treat this as unverified and decide whether to retry.
        VerificationCostExceededError: max_spend cap hit
        ParseError: returned as status="PARSE_ERROR" in the result, not raised
    """
    if not claim or not isinstance(claim, str):
        raise ValueError("claim must be a non-empty string")
    if not sources:
        raise ValueError("sources must be a non-empty list")

    profile = profile or GENERAL_PROFILE
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
        context=context,
        cost_tracker=tracker,
        profile=profile,
        tier2_lexical_threshold=profile.tier2_lexical_threshold,
        top_k_chunks=profile.bm25_top_k,
    )

    logger.debug(
        "verify() entry [boundary=%s]",
        ctx._boundary_id,
        extra={
            "boundary_id": ctx._boundary_id,
            "model": model,
            "source_count": len(sources),
            "claim_length_chars": len(claim),
        },
    )

    ctx.tier0_atoms = classifier.parse_and_classify(claim)
    if ctx.auto_chunk:
        chunks = chunker.chunk_sources(ctx)
    else:
        chunks = chunker.wrap_as_chunks(ctx.original_sources)

    tier25_result = tier25_preprocessing.run(ctx, chunks)
    if tier25_result.has_conflict:
        atomic = CoreResultBuilder.build_numerical_fast_exit(
            claim, tier25_result, ctx.original_sources[0]
        )
        logger.info(
            "verify() completion tier25 fast exit [boundary=%s]",
            ctx._boundary_id,
            extra={
                "boundary_id": ctx._boundary_id,
                "verdict": "CONTRADICTED",
                "verification_method": "tier25_numerical",
                "cost_usd": ctx.cost_tracker.total_cost_usd,
                "numerical_fast_exit": True,
            },
        )
        return VerificationResult(
            is_valid=False,
            overall_verdict="Numerical conflict detected in source.",
            verification_method="tier25_numerical",
            atomic_claims=[atomic],
            factual_consistency_score=0.0,
            sources_used=[ctx.original_sources[0].source_id],
            rationale=f"Claim number conflicts with source: {atomic.citation}",
            offending_claim=claim,
            status="CONTRADICTED",
            total_cost_usd=ctx.cost_tracker.total_cost_usd,
        )

    tier1_chunk = None
    if isinstance(ctx.agent_provided_evidence, str) and ctx.agent_provided_evidence:
        tier1_chunk = tier1_authenticity.check_fuzzy(
            ctx.agent_provided_evidence, chunks, ctx.tier1_min_similarity
        )

    t2_res = tier2_semantic.route_claim(ctx, chunks)

    if tier1_chunk is not None:
        existing_ids = {c.chunk_id for c in t2_res.top_k_chunks if c.chunk_id}
        is_duplicate = False
        if tier1_chunk.chunk_id:
            is_duplicate = tier1_chunk.chunk_id in existing_ids
        else:
            is_duplicate = any(
                c.source_id == tier1_chunk.source_id and
                c.char_start == tier1_chunk.char_start and
                c.char_end == tier1_chunk.char_end and
                c.text_content == tier1_chunk.text_content
                for c in t2_res.top_k_chunks
            )
        if not is_duplicate:
            t2_res.top_k_chunks.append(tier1_chunk)

    if t2_res.decision == RoutingDecision.SKIP_LLM_HIGH_CONFIDENCE:
        result = ResultBuilder.build_lexical_pass(ctx, t2_res.top_k_chunks)
        logger.info(
            "verify() completion [boundary=%s]",
            ctx._boundary_id,
            extra={
                "boundary_id": ctx._boundary_id,
                "verdict": result.status,
                "verification_method": result.verification_method,
                "cost_usd": ctx.cost_tracker.total_cost_usd,
                "numerical_fast_exit": False,
            },
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

    try:
        result = ResultBuilder.build_llm_result(ctx, t3_model, "tier3_llm")
    except InvariantError as e:
        e.cost_usd = ctx.cost_tracker.total_cost_usd
        raise
    logger.info(
        "verify() completion [boundary=%s]",
        ctx._boundary_id,
        extra={
            "boundary_id": ctx._boundary_id,
            "verdict": result.status,
            "verification_method": result.verification_method,
            "cost_usd": ctx.cost_tracker.total_cost_usd,
            "numerical_fast_exit": False,
        },
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
    profile: VerificationProfile | None = None,
    context: str | None = None,
) -> VerificationResult:
    """Asynchronous verification entry point.
    
    Raises:
        InvariantError: verification pipeline produced an internally inconsistent result
                        (e.g. VERIFIED claim with no citation). The claim cannot be trusted.
                        Caller should treat this as unverified and decide whether to retry.
        VerificationCostExceededError: max_spend cap hit
        ParseError: returned as status="PARSE_ERROR" in the result, not raised
    """
    if not claim or not isinstance(claim, str):
        raise ValueError("claim must be a non-empty string")
    if not sources:
        raise ValueError("sources must be a non-empty list")

    profile = profile or GENERAL_PROFILE
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
        context=context,
        cost_tracker=tracker,
        profile=profile,
        tier2_lexical_threshold=profile.tier2_lexical_threshold,
        top_k_chunks=profile.bm25_top_k,
    )

    ctx.tier0_atoms = classifier.parse_and_classify(claim)
    if ctx.auto_chunk:
        chunks = chunker.chunk_sources(ctx)
    else:
        chunks = chunker.wrap_as_chunks(ctx.original_sources)

    tier25_result = tier25_preprocessing.run(ctx, chunks)
    if tier25_result.has_conflict:
        atomic = CoreResultBuilder.build_numerical_fast_exit(
            claim, tier25_result, ctx.original_sources[0]
        )
        return VerificationResult(
            is_valid=False,
            overall_verdict="Numerical conflict detected in source.",
            verification_method="tier25_numerical",
            atomic_claims=[atomic],
            factual_consistency_score=0.0,
            sources_used=[ctx.original_sources[0].source_id],
            rationale=f"Claim number conflicts with source: {atomic.citation}",
            offending_claim=claim,
            status="CONTRADICTED",
            total_cost_usd=ctx.cost_tracker.total_cost_usd,
        )

    tier1_chunk = None
    if isinstance(ctx.agent_provided_evidence, str) and ctx.agent_provided_evidence:
        tier1_chunk = tier1_authenticity.check_fuzzy(
            ctx.agent_provided_evidence, chunks, ctx.tier1_min_similarity
        )

    # BM25 is CPU-bound — dispatch to thread-pool executor to avoid blocking event loop
    loop = asyncio.get_running_loop()
    t2_res = await loop.run_in_executor(
        None, tier2_semantic.route_claim, ctx, chunks
    )

    if tier1_chunk is not None:
        existing_ids = {c.chunk_id for c in t2_res.top_k_chunks if c.chunk_id}
        is_duplicate = False
        if tier1_chunk.chunk_id:
            is_duplicate = tier1_chunk.chunk_id in existing_ids
        else:
            is_duplicate = any(
                c.source_id == tier1_chunk.source_id and
                c.char_start == tier1_chunk.char_start and
                c.char_end == tier1_chunk.char_end and
                c.text_content == tier1_chunk.text_content
                for c in t2_res.top_k_chunks
            )
        if not is_duplicate:
            t2_res.top_k_chunks.append(tier1_chunk)

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

    try:
        result = ResultBuilder.build_llm_result(ctx, t3_model, "tier3_llm")
    except InvariantError as e:
        e.cost_usd = ctx.cost_tracker.total_cost_usd
        raise
    logger.info(
        "averify(): status=%s method=%s cost=$%.4f [boundary=%s]",
        result.status,
        result.verification_method,
        ctx.cost_tracker.total_cost_usd,
        ctx._boundary_id,
    )
    return result


async def averify_batch(
    inputs: list[ClaimInput],
    model: str = "gpt-4o-mini",
    max_concurrency: int = 5,
    max_spend: float = 0.50,
    auto_chunk: bool = True,
    cost_tracker = None,
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

    tracker = cost_tracker or SharedCostTracker(max_spend=max_spend)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _sem_bounded_verify(inp: ClaimInput) -> VerificationResult:
        async with semaphore:
            item_auto_chunk = auto_chunk if inp.auto_chunk is None else inp.auto_chunk
            return await averify(
                claim=inp.claim,
                sources=inp.sources,
                model=inp.model or model,
                agent_provided_evidence=inp.agent_provided_evidence,
                cost_tracker=tracker,
                auto_chunk=item_auto_chunk,
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
        elif isinstance(r, InvariantError):
            logger.error(
                "averify_batch: item failed with InvariantError — returning INVARIANT_ERROR result"
            )
            results.append(VerificationResult(
                is_valid=False,
                overall_verdict="Verification failed — internal integrity violation.",
                verification_method="skipped",
                atomic_claims=[],
                factual_consistency_score=0.0,
                sources_used=[],
                rationale=str(r),
                offending_claim=None,
                status="INVARIANT_ERROR",
                total_cost_usd=r.cost_usd,
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
                total_cost_usd=0.0,
            ))
        else:
            results.append(r)

    return results


verify_batch_async = averify_batch  # backward-compat alias


def verify_batch(
    inputs: list[ClaimInput],
    model: str = "gpt-4o-mini",
    max_concurrency: int = 5,
    max_spend: float = 0.50,
    auto_chunk: bool = True,
    cost_tracker = None,
    **kwargs,
) -> list[VerificationResult]:
    """
    Sync wrapper around averify_batch. Creates a new event loop via asyncio.run().

    Constraint: asyncio.run() cannot be called from within an already-running event
    loop (e.g., Jupyter notebooks, FastAPI request handlers). In those contexts,
    use averify_batch() directly with await.
    """
    return asyncio.run(
        averify_batch(
            inputs=inputs,
            model=model,
            max_concurrency=max_concurrency,
            max_spend=max_spend,
            auto_chunk=auto_chunk,
            cost_tracker=cost_tracker,
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


def _verify_single_claim(claim: str, sources: list, **kwargs) -> VerificationResult:
    """Single-claim verification — used internally by verify_analysis for per-claim calls."""
    return verify(claim=claim, sources=sources, **kwargs)


def _aggregate_analysis_results(results: list, profile=None) -> GroundingResult:
    """Aggregate a list of per-claim results into a GroundingResult."""
    profile = profile or GENERAL_PROFILE

    supported = sum(1 for r in results if r.status == "VERIFIED")
    contradicted = sum(1 for r in results if r.status == "CONTRADICTED")
    errored = sum(1 for r in results if r.status not in ("VERIFIED", "CONTRADICTED", "UNVERIFIABLE"))
    denom = supported + contradicted + errored
    score = supported / denom if denom > 0 else 0.0

    all_audit_records = []
    for r in results:
        if hasattr(r, "audit_records") and r.audit_records:
            all_audit_records.extend(r.audit_records)
    audit_records = all_audit_records if all_audit_records else None

    # Special case: all UNVERIFIABLE (denom == 0 but results not empty)
    if results and all(r.status == "UNVERIFIABLE" for r in results):
        return GroundingResult(
            is_grounded=False,
            score=0.0,
            status="NOT_GROUNDED",
            evaluation_method="claim_extraction",
            total_units=len(results),
            unverifiable_units=len(results),
            audit_records=audit_records,
        )

    if score >= profile.faithfulness_threshold:
        status = "GROUNDED"
    elif score > 0:
        status = "PARTIALLY_GROUNDED"
    else:
        status = "NOT_GROUNDED"

    return GroundingResult(
        is_grounded=(status == "GROUNDED"),
        score=score,
        status=status,
        evaluation_method="claim_extraction",
        total_units=len(results),
        grounded_units=supported,
        ungrounded_units=contradicted,
        unverifiable_units=sum(1 for r in results if r.status == "UNVERIFIABLE"),
        audit_records=audit_records,
    )


def verify_analysis(
    analysis_text: str,
    sources: list,
    context: str | None = None,
    model: str = "gpt-4o-mini",
    profile=None,
    max_spend: float = float("inf"),
    api_base: str | None = None,
    audit: bool = False,
    auto_chunk: bool = True,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    max_source_tokens: int = 8000,
    tier1_min_similarity: float = 0.90,
) -> GroundingResult:
    """Verify that analysis_text is grounded in sources by extracting and checking each claim.

    InvariantError from per-claim verification is absorbed into the GroundingResult
    (affected item returns status="INVARIANT_ERROR") rather than propagating to the caller.

    Raises:
        VerificationCostExceededError: max_spend cap hit
        ParseError: from claim_extractor — returned as status="ERROR" GroundingResult, not raised


    auto_chunk: Pass False when using large-context models (Gemini 1.5 Pro, Claude 3.5+) to
        send each source as a single unit without BM25 sliding-window chunking. Avoids the
        Lost Context Problem where low-scoring chunks containing negating context are dropped.
        Applied uniformly to all extracted claims in the batch.
    """
    profile = profile or GENERAL_PROFILE
    if audit and not profile.audit:
        profile = dataclasses.replace(profile, audit=True)

    tracker = SharedCostTracker(max_spend=max_spend)
    try:
        claims = claim_extractor.extract_claims(
            analysis_text, sources, model, context=context, max_spend=max_spend, api_base=api_base, cost_tracker=tracker, audit=audit
        )
    except ParseError:
        return GroundingResult(
            is_grounded=False,
            score=0.0,
            status="ERROR",
            evaluation_method="claim_extraction",
        )

    inputs = [
        ClaimInput(claim=c, sources=sources, model=model)
        for c in claims
    ]
    results = verify_batch(
        inputs,
        model=model,
        max_spend=max_spend,
        auto_chunk=auto_chunk,
        api_base=api_base,
        profile=profile,
        cost_tracker=tracker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_source_tokens=max_source_tokens,
        tier1_min_similarity=tier1_min_similarity,
    )

    return _aggregate_analysis_results(results, profile)


async def averify_analysis(
    analysis_text: str,
    sources: list,
    context: str | None = None,
    model: str = "gpt-4o-mini",
    profile=None,
    max_spend: float = float("inf"),
    api_base: str | None = None,
    audit: bool = False,
    auto_chunk: bool = True,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    max_source_tokens: int = 8000,
    tier1_min_similarity: float = 0.90,
) -> GroundingResult:
    """Pure-async implementation: no thread pool, no secondary event loop.

    InvariantError from per-claim verification is absorbed into the GroundingResult
    (affected item returns status="INVARIANT_ERROR") rather than propagating to the caller.

    Raises:
        VerificationCostExceededError: max_spend cap hit
        ParseError: from claim_extractor — returned as status="ERROR" GroundingResult, not raised


    auto_chunk: Pass False when using large-context models (Gemini 1.5 Pro, Claude 3.5+) to
        send each source as a single unit without BM25 sliding-window chunking. Avoids the
        Lost Context Problem where low-scoring chunks containing negating context are dropped.
        Applied uniformly to all extracted claims in the batch.
    """
    profile = profile or GENERAL_PROFILE
    if audit and not profile.audit:
        profile = dataclasses.replace(profile, audit=True)

    tracker = SharedCostTracker(max_spend=max_spend)
    try:
        claims = await claim_extractor.extract_claims_async(
            analysis_text, sources, model, context=context, max_spend=max_spend, api_base=api_base, cost_tracker=tracker, audit=audit
        )
    except ParseError:
        return GroundingResult(
            is_grounded=False,
            score=0.0,
            status="ERROR",
            evaluation_method="claim_extraction",
        )
    inputs = [ClaimInput(claim=c, sources=sources, model=model) for c in claims]
    results = await averify_batch(
        inputs,
        model=model,
        max_spend=max_spend,
        auto_chunk=auto_chunk,
        api_base=api_base,
        profile=profile,
        cost_tracker=tracker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_source_tokens=max_source_tokens,
        tier1_min_similarity=tier1_min_similarity,
    )
    return _aggregate_analysis_results(results, profile)


def verify_answer(
    answer: str,
    sources: list,
    context: str | None = None,
    *,
    profile: VerificationProfile = None,
    faithfulness_threshold: float = None,
    model: str = "gpt-4o-mini",
    max_spend: float = float("inf"),
    auto_chunk: bool = True,
    api_base: str | None = None,
    audit: bool = False,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    max_source_tokens: int = 8000,
    tier1_min_similarity: float = 0.90,
) -> GroundingResult:
    """Verify that an answer is grounded in the provided sources.

    Raises:
        VerificationCostExceededError: max_spend cap hit
        ParseError: returned as status="PARSE_ERROR" in the result, not raised


    auto_chunk: Pass False when using large-context models (Gemini 1.5 Pro, Claude 3.5+) to
        send each source as a single unit without BM25 sliding-window chunking. Avoids the
        Lost Context Problem where low-scoring chunks containing negating context are dropped.
        Note: verify_answer uses evaluate_faithfulness, not the Tier 2/3 pipeline.
    """
    from groundguard.models.result import VerificationAuditRecord

    profile = profile or GENERAL_PROFILE
    if audit and not profile.audit:
        profile = dataclasses.replace(profile, audit=True)

    threshold = faithfulness_threshold if faithfulness_threshold is not None else profile.faithfulness_threshold

    ctx = VerificationContext(
        claim=answer,
        sources=sources,
        model=model,
        cost_tracker=SharedCostTracker(max_spend=max_spend),
        profile=profile,
        auto_chunk=auto_chunk,
        api_base=api_base,
        context=context,
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=chunk_overlap,
        max_source_tokens=max_source_tokens,
        tier1_min_similarity=tier1_min_similarity,
    )
    chunks = chunker.chunk_sources(ctx)

    result = tier3_evaluation.evaluate_faithfulness(ctx, chunks)
    audit_records = getattr(result, "audit_records", None)

    if profile.majority_vote:
        results = [result]
        for _ in range(2):
            r = tier3_evaluation.evaluate_faithfulness(ctx, chunks)
            results.append(r)
        
        # Apply threshold to each result for voting
        vote_statuses = []
        for r in results:
            if r.status == "NOT_GROUNDED":
                vote_statuses.append("NOT_GROUNDED")
            elif r.score >= threshold:
                vote_statuses.append("GROUNDED")
            else:
                vote_statuses.append("PARTIALLY_GROUNDED")
                
        from collections import Counter
        counts = Counter(vote_statuses)
        winner, top_count = counts.most_common(1)[0]
        is_tie = top_count == 1
        audit_records = []
        for i, r in enumerate(results):
            rec = VerificationAuditRecord(
                boundary_id=ctx._boundary_id,
                claim_text=answer,
                verdict=vote_statuses[i],
                tier_path=["tier3_faithfulness"],
                model=model,
                cost_usd=0.0,
                timestamp_utc="",
                profile_name=profile.name if hasattr(profile, "name") else "unknown",
                majority_vote_triggered=True,
            )
            audit_records.append(rec)
        if is_tie:
            audit_records[0].tie_broken = True
            gr = GroundingResult(
                is_grounded=False,
                score=min(r.score for r in results),
                status="NOT_GROUNDED",
                evaluation_method="sentence_entailment",
                audit_records=audit_records,
                total_cost_usd=ctx.cost_tracker.total_cost_usd,
            )
            logger.info(
                "verify_answer() completion [boundary=%s]",
                ctx._boundary_id,
                extra={"boundary_id": ctx._boundary_id, "score": gr.score},
            )
            return gr
        winning_result = next(r for i, r in enumerate(results) if vote_statuses[i] == winner)
        gr = GroundingResult(
            is_grounded=winner == "GROUNDED",
            score=winning_result.score,
            status=winner,
            evaluation_method="sentence_entailment",
            audit_records=audit_records,
            total_cost_usd=ctx.cost_tracker.total_cost_usd,
        )
        logger.info(
            "verify_answer() completion [boundary=%s]",
            ctx._boundary_id,
            extra={"boundary_id": ctx._boundary_id, "score": gr.score},
        )
        return gr

    is_grounded = result.status != "NOT_GROUNDED" and result.score >= threshold
    gr = GroundingResult(
        is_grounded=is_grounded,
        score=result.score,
        status=result.status if not is_grounded else "GROUNDED",
        evaluation_method=result.evaluation_method,
        audit_records=audit_records,
        total_cost_usd=ctx.cost_tracker.total_cost_usd,
    )
    logger.info(
        "verify_answer() completion [boundary=%s]",
        ctx._boundary_id,
        extra={"boundary_id": ctx._boundary_id, "score": gr.score},
    )
    return gr


async def averify_answer(
    answer: str,
    sources: list,
    context: str | None = None,
    *,
    profile: VerificationProfile = None,
    faithfulness_threshold: float = None,
    model: str = "gpt-4o-mini",
    max_spend: float = float("inf"),
    auto_chunk: bool = True,
    api_base: str | None = None,
    audit: bool = False,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    max_source_tokens: int = 8000,
    tier1_min_similarity: float = 0.90,
) -> GroundingResult:
    """Async version of verify_answer.

    Raises:
        VerificationCostExceededError: max_spend cap hit
        ParseError: returned as status="PARSE_ERROR" in the result, not raised


    auto_chunk: Pass False when using large-context models (Gemini 1.5 Pro, Claude 3.5+) to
        send each source as a single unit without BM25 sliding-window chunking. Avoids the
        Lost Context Problem where low-scoring chunks containing negating context are dropped.
        Note: verify_answer uses evaluate_faithfulness, not the Tier 2/3 pipeline.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: verify_answer(
            answer, sources, context,
            profile=profile,
            faithfulness_threshold=faithfulness_threshold,
            model=model,
            max_spend=max_spend,
            auto_chunk=auto_chunk,
            api_base=api_base,
            audit=audit,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_source_tokens=max_source_tokens,
            tier1_min_similarity=tier1_min_similarity,
        ),
    )


def verify_structured(
    claim_dict: dict,
    schema: type[pydantic.BaseModel],
    sources: list,
    auto_chunk: bool = True,
    **kwargs,
) -> VerificationResult:
    """
    Validates claim_dict against the provided Pydantic schema, flattens it to a
    string, then delegates to verify(). Raises ValueError on schema mismatch.
    
    Raises:
        InvariantError: verification pipeline produced an internally inconsistent result
                        (e.g. VERIFIED claim with no citation). The claim cannot be trusted.
                        Caller should treat this as unverified and decide whether to retry.
        VerificationCostExceededError: max_spend cap hit
        ParseError: returned as status="PARSE_ERROR" in the result, not raised
    """
    try:
        validated = schema.model_validate(claim_dict)
        normalised = validated.model_dump()
    except pydantic.ValidationError as e:
        raise ValueError(f"claim_dict does not conform to the provided schema: {e}")

    flattened = dict_to_string_flattener(normalised)
    return verify(claim=flattened, sources=sources, auto_chunk=auto_chunk, **kwargs)


def verify_clause(
    clause_text: str,
    sources: list,
    *,
    term_registry=None,
    profile=None,
    model: str = "gpt-4o-mini",
    max_spend: float = float("inf"),
    api_base: str | None = None,
    auto_chunk: bool = True,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    max_source_tokens: int = 8000,
    tier1_min_similarity: float = 0.90,
    audit: bool = False,
):
    """Decompose and verify a legal clause against source documents.
    
    Raises:
        InvariantError: verification pipeline produced an internally inconsistent result
                        (e.g. VERIFIED claim with no citation). The claim cannot be trusted.
                        Caller should treat this as unverified and decide whether to retry.
        VerificationCostExceededError: max_spend cap hit
        ParseError: returned as status="PARSE_ERROR" in the result, not raised


    auto_chunk: Pass False when using large-context models (Gemini 1.5 Pro, Claude 3.5+) to
        send each source as a single unit without BM25 sliding-window chunking. Avoids the
        Lost Context Problem where low-scoring chunks containing negating context are dropped.
        Particularly relevant for long contract documents where modal operators and definitions
        may appear far from the main proposition.
    """
    from groundguard.profiles import STRICT_PROFILE
    profile = profile or STRICT_PROFILE
    if audit and not profile.audit:
        profile = dataclasses.replace(profile, audit=True)
    unit = decompose_clause(clause_text)
    context_parts = [
        f"Clause modifiers: {unit.subordinate_modifiers}",
        f"Obligation type: {unit.modal_operator or 'unknown'}",
    ]
    context = "\n".join(context_parts)
    extra_sources = []
    if term_registry is not None:
        for term in unit.defined_terms_referenced:
            src = term_registry.resolve(term)
            if src:
                extra_sources.append(src)
    return verify(
        unit.main_proposition,
        sources + extra_sources,
        model=model,
        max_spend=max_spend,
        api_base=api_base,
        profile=profile,
        context=context,
        auto_chunk=auto_chunk,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_source_tokens=max_source_tokens,
        tier1_min_similarity=tier1_min_similarity,
    )


async def averify_clause(
    clause_text: str,
    sources: list,
    *,
    term_registry=None,
    profile=None,
    model: str = "gpt-4o-mini",
    max_spend: float = float("inf"),
    api_base: str | None = None,
    auto_chunk: bool = True,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    max_source_tokens: int = 8000,
    tier1_min_similarity: float = 0.90,
    audit: bool = False,
):
    """Async version of verify_clause.
    
    Raises:
        InvariantError: verification pipeline produced an internally inconsistent result
                        (e.g. VERIFIED claim with no citation). The claim cannot be trusted.
                        Caller should treat this as unverified and decide whether to retry.
        VerificationCostExceededError: max_spend cap hit
        ParseError: returned as status="PARSE_ERROR" in the result, not raised


    auto_chunk: Pass False when using large-context models (Gemini 1.5 Pro, Claude 3.5+) to
        send each source as a single unit without BM25 sliding-window chunking. Avoids the
        Lost Context Problem where low-scoring chunks containing negating context are dropped.
        Particularly relevant for long contract documents where modal operators and definitions
        may appear far from the main proposition.
    """
    from groundguard.profiles import STRICT_PROFILE
    profile = profile or STRICT_PROFILE
    if audit and not profile.audit:
        profile = dataclasses.replace(profile, audit=True)
    unit = decompose_clause(clause_text)
    context_parts = [
        f"Clause modifiers: {unit.subordinate_modifiers}",
        f"Obligation type: {unit.modal_operator or 'unknown'}",
    ]
    context = "\n".join(context_parts)
    extra_sources = []
    if term_registry is not None:
        for term in unit.defined_terms_referenced:
            src = term_registry.resolve(term)
            if src:
                extra_sources.append(src)
    return await averify(
        unit.main_proposition,
        sources + extra_sources,
        model=model,
        max_spend=max_spend,
        api_base=api_base,
        profile=profile,
        context=context,
        auto_chunk=auto_chunk,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_source_tokens=max_source_tokens,
        tier1_min_similarity=tier1_min_similarity,
    )
