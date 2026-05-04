"""Tier 2 semantic routing — BM25-based claim-to-source routing."""
from __future__ import annotations
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from groundguard.models.internal import RoutingDecision, Tier2Result
from groundguard._log import logger

if TYPE_CHECKING:
    from groundguard.loaders.chunker import Chunk
    from groundguard.models.internal import VerificationContext


def route_claim(ctx: VerificationContext, all_chunks: list[Chunk]) -> Tier2Result:
    """
    Scores ctx.claim against all chunks using BM25 (rank_bm25.BM25Okapi).

    Three-branch routing:
    1. highest_score >= tier2_lexical_threshold (0.85):
       → SKIP_LLM_HIGH_CONFIDENCE (strong lexical match, skip LLM)
    2. highest_score <= tier2_low_score_floor (0.01):
       → ESCALATE_ALL_LOW_SCORE (all chunks score near zero)
       Caps at top_k_chunks * 3 in document order
    3. Otherwise:
       → ESCALATE_TO_LLM with top-k chunks by BM25 score
    """
    if not all_chunks:
        return Tier2Result(
            decision=RoutingDecision.ESCALATE_ALL_LOW_SCORE,
            top_k_chunks=[],
            highest_score=0.0,
        )

    # Tokenize all chunks for BM25
    tokenized_corpus = [chunk.text_content.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Score the claim
    claim_tokens = ctx.claim.lower().split()
    scores = bm25.get_scores(claim_tokens)

    # Sort by score descending, keeping original index for document order
    scored_chunks = sorted(zip(scores, range(len(all_chunks))), reverse=True)
    raw_highest = float(scored_chunks[0][0]) if scored_chunks else 0.0
    # BM25Okapi can return negative scores when the corpus is very small (e.g. single document)
    # due to IDF computation. A negative score still signals vocabulary overlap, so we clamp
    # to 0 for the public highest_score field. Routing uses a vocabulary-overlap check to
    # distinguish "genuinely no match" (floor branch) from "small corpus IDF artefact".
    highest_score = max(raw_highest, 0.0)

    logger.debug(
        "Tier 2 routing: raw_highest=%.4f, highest_score=%.4f, threshold=%.2f, floor=%.2f",
        raw_highest, highest_score, ctx.tier2_lexical_threshold, ctx.tier2_low_score_floor,
    )

    if highest_score >= ctx.tier2_lexical_threshold:
        # Branch A: high confidence match — skip LLM
        top_k = [all_chunks[i] for _, i in scored_chunks[:ctx.top_k_chunks]]
        return Tier2Result(
            decision=RoutingDecision.SKIP_LLM_HIGH_CONFIDENCE,
            top_k_chunks=top_k,
            highest_score=highest_score,
        )
    elif highest_score <= ctx.tier2_low_score_floor and raw_highest >= 0.0:
        # Branch C: all scores near zero — escalate all chunks (capped, document order)
        # raw_highest >= 0 guards against BM25Okapi IDF artefacts (negative scores in tiny
        # corpora still indicate vocabulary overlap; those fall through to Branch B instead)
        ALL_LOW_MAX_CHUNKS = ctx.top_k_chunks * 3
        escalate_chunks = all_chunks[:ALL_LOW_MAX_CHUNKS]  # document order, not BM25 rank
        return Tier2Result(
            decision=RoutingDecision.ESCALATE_ALL_LOW_SCORE,
            top_k_chunks=escalate_chunks,
            highest_score=highest_score,
        )
    else:
        # Branch B: partial match — escalate top-k to LLM
        top_k = [all_chunks[i] for _, i in scored_chunks[:ctx.top_k_chunks]]
        return Tier2Result(
            decision=RoutingDecision.ESCALATE_TO_LLM,
            top_k_chunks=top_k,
            highest_score=highest_score,
        )
