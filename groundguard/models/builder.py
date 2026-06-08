"""Result builder — maps pipeline outputs to public VerificationResult."""
from __future__ import annotations
from typing import TYPE_CHECKING

from groundguard._log import logger
from groundguard.models.result import VerificationResult, AtomicClaimResult, Citation

if TYPE_CHECKING:
    from groundguard.models.internal import VerificationContext
    from groundguard.models.tier3 import Tier3ResponseModel
    from groundguard.loaders.chunker import Chunk


class ResultBuilder:
    """Builds VerificationResult from pipeline tier outputs."""

    @staticmethod
    def _safe_citation_status(
        status: str,
        citation: object,
        evaluation_method: str = "extractive",
    ) -> str:
        """Return a safe atom status — downgrades VERIFIED→UNVERIFIABLE when citation is absent.

        Inferential claims are exempt: they use reasoning_basis instead of source_excerpt,
        so citation=None is expected and correct for those.
        """
        if status == "VERIFIED" and citation is None and evaluation_method != "inferential":
            logger.warning(
                "VERIFIED %s atom has no citation — downgrading to UNVERIFIABLE",
                evaluation_method,
            )
            return "UNVERIFIABLE"
        return status

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

        lexical_citation: Citation | None = None
        if matched_chunks:
            top = matched_chunks[0]
            lexical_citation = Citation(
                source_id=top.source_id,
                excerpt=top.text_content,
                excerpt_char_start=top.char_start,
                excerpt_char_end=top.char_end,
                citation_confidence=1.0,
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
                    citation=lexical_citation,
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
        evidence_bundle: list[Chunk] | None = None,
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
        - VERIFIED extractive atom with no source_excerpt → downgraded to UNVERIFIABLE
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

        page_hints: dict[str, str | None] = {
            s.source_id: s.page_hint for s in ctx.original_sources
        }
        valid_source_ids = {s.source_id for s in ctx.original_sources}

        atom_types: dict[str, str] = {
            a.claim_text: a.claim_type for a in ctx.tier0_atoms
        }

        atomic_claims = []
        for v in t3_model.verifications:
            claim_type = atom_types.get(v.claim_text, "Extractive")
            atom_status = ResultBuilder._safe_citation_status(
                status=v.status,
                citation=v.source_excerpt,
                evaluation_method=claim_type.lower(),
            )
            citation_obj: Citation | None = None
            if v.source_id and v.source_excerpt and atom_status == "VERIFIED":
                confidence: float | None = None
                if evidence_bundle:
                    from groundguard.tiers.tier25_preprocessing import _bm25_score_single
                    matching = next(
                        (c for c in evidence_bundle if c.source_id == v.source_id), None
                    )
                    if matching:
                        confidence = _bm25_score_single(ctx.claim.split(), matching)
                citation_obj = Citation(
                    source_id=v.source_id,
                    excerpt=v.source_excerpt,
                    page_hint=page_hints.get(v.source_id),
                    citation_confidence=confidence,
                )
            atomic_claims.append(AtomicClaimResult(
                claim_text=v.claim_text,
                claim_type=claim_type,
                status=atom_status,
                verification_method=method,
                source_id=v.source_id,
                source_excerpt=v.source_excerpt,
                reasoning_basis=v.reasoning_basis,
                page_hint=page_hints.get(v.source_id) if v.source_id else None,
                citation=citation_obj,
            ))

        downgraded = any(
            v.status == "VERIFIED" and a.status == "UNVERIFIABLE"
            for v, a in zip(t3_model.verifications, atomic_claims)
        )
        if status == "VERIFIED" and downgraded:
            status = "UNVERIFIABLE"
            is_valid = False
            norm_score = 0.0

        offending_claim = next(
            (a.claim_text for a in atomic_claims if a.status == "CONTRADICTED"), None
        )

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
