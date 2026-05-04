"""Result builder — maps pipeline outputs to public VerificationResult."""
from __future__ import annotations
from typing import TYPE_CHECKING

from groundguard.models.result import VerificationResult, AtomicClaimResult

if TYPE_CHECKING:
    from groundguard.models.internal import VerificationContext
    from groundguard.models.tier3 import Tier3ResponseModel
    from groundguard.loaders.chunker import Chunk


class ResultBuilder:
    """Builds VerificationResult from pipeline tier outputs."""

    @staticmethod
    def build_lexical_pass(
        ctx: VerificationContext,
        matched_chunks: list[Chunk],
    ) -> VerificationResult:
        """
        Called when Tier 2 BM25 score exceeds the lexical threshold.
        No LLM call — result assembled from BM25 match data only.
        """
        sources_used = list(dict.fromkeys(c.source_id for c in matched_chunks))
        primary_claim_type = (
            ctx.tier0_atoms[0].claim_type if ctx.tier0_atoms else "Extractive"
        )

        return VerificationResult(
            is_valid=True,
            overall_verdict=(
                "The source material fully supports the claim via "
                "high-confidence lexical retrieval."
            ),
            verification_method="tier2_lexical",
            atomic_claims=[
                AtomicClaimResult(
                    claim_text=ctx.claim,
                    claim_type=primary_claim_type,
                    status="VERIFIED",
                    source_id=sources_used[0] if sources_used else None,
                    source_excerpt=None,
                    reasoning_basis=None,
                    page_hint=None,
                )
            ],
            factual_consistency_score=1.0,
            sources_used=sources_used,
            rationale=(
                "High-confidence BM25 match found in source material. "
                "LLM evaluation skipped for cost efficiency."
            ),
            offending_claim=None,
            status="VERIFIED",
            total_cost_usd=ctx.cost_tracker.total_cost_usd,
        )

    @staticmethod
    def build_llm_result(
        ctx: VerificationContext,
        t3_model: Tier3ResponseModel,
        method: str,
    ) -> VerificationResult:
        """
        Maps Tier3ResponseModel → VerificationResult.

        Key mapping rules:
        - Entailment → VERIFIED, is_valid=True
        - Contradiction → CONTRADICTED, is_valid=False
        - Neutral → UNVERIFIABLE always (never promoted, coverage is informational only)
        - factual_consistency_score: divide by 100 (t3 is 0–100; public API is 0.0–1.0)
        - sources_used: only source_ids from ctx.original_sources (scrubs hallucinated IDs)
        - page_hint: transplanted from ctx.original_sources via source_id lookup
        """
        label = t3_model.textual_entailment.label
        if label == "Entailment":
            status = "VERIFIED"
        elif label == "Contradiction":
            status = "CONTRADICTED"
        else:  # Neutral
            status = "UNVERIFIABLE"

        is_valid = (status == "VERIFIED")
        norm_score = t3_model.factual_consistency_score / 100.0

        # Build source_id → page_hint lookup from original sources
        page_hints: dict[str, str | None] = {
            s.source_id: s.page_hint for s in ctx.original_sources
        }
        # Set of valid source IDs to scrub hallucinated ones
        valid_source_ids = {s.source_id for s in ctx.original_sources}

        # Build claim_text -> claim_type lookup from Tier 0 classification
        atom_types: dict[str, str] = {
            a.claim_text: a.claim_type for a in ctx.tier0_atoms
        }

        atomic_claims = [
            AtomicClaimResult(
                claim_text=v.claim_text,
                claim_type=atom_types.get(v.claim_text, "Extractive"),  # fallback to Extractive
                status=v.status,
                source_id=v.source_id,
                source_excerpt=v.source_excerpt,
                reasoning_basis=v.reasoning_basis,
                page_hint=page_hints.get(v.source_id) if v.source_id else None,
            )
            for v in t3_model.verifications
        ]

        offending_claim = next(
            (a.claim_text for a in atomic_claims if a.status == "CONTRADICTED"), None
        )

        # Only include source_ids that actually exist in ctx.original_sources
        sources_used = [
            sa.source_id for sa in t3_model.source_attributions
            if sa.role != "Not Used" and sa.source_id in valid_source_ids
        ]

        return VerificationResult(
            is_valid=is_valid,
            overall_verdict=t3_model.overall_verdict,
            verification_method=method,
            atomic_claims=atomic_claims,
            factual_consistency_score=norm_score,
            sources_used=sources_used,
            rationale=t3_model.overall_verdict,
            offending_claim=offending_claim,
            status=status,
            total_cost_usd=ctx.cost_tracker.total_cost_usd,
        )
