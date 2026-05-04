"""Phase 23 ResultBuilder — citation extraction and invariant enforcement."""
from __future__ import annotations
from typing import TYPE_CHECKING

from groundguard.models.result import AtomicClaimResult, Citation
from groundguard.exceptions import InvariantError

if TYPE_CHECKING:
    from groundguard.models.result import Source
    from groundguard.tiers.tier25_preprocessing import Tier25Result


class ResultBuilder:

    @staticmethod
    def build_numerical_fast_exit(claim: str, tier25: Tier25Result, source: Source) -> AtomicClaimResult:
        citation = tier25.conflict_citation
        result = AtomicClaimResult(
            claim_text=claim, claim_type="Extractive", status="CONTRADICTED",
            source_id=source.source_id, verification_method="tier25_numerical", citation=citation,
        )
        ResultBuilder._assert_citation_invariant("CONTRADICTED", citation)
        return result

    @staticmethod
    def build_lexical_pass(claim: str, top_chunks: list, score: float, source: Source) -> AtomicClaimResult:
        if top_chunks:
            chunk = top_chunks[0]
            excerpt_text = chunk.text_content
            char_start = chunk.char_start
            char_end = chunk.char_end
        else:
            excerpt_text = source.content[:100] if source.content else ""
            char_start, char_end = 0, len(excerpt_text)

        citation = Citation(
            source_id=source.source_id, excerpt=excerpt_text,
            excerpt_char_start=char_start, excerpt_char_end=char_end, citation_confidence=1.0,
        )
        result = AtomicClaimResult(
            claim_text=claim, claim_type="Extractive", status="VERIFIED",
            source_id=source.source_id, verification_method="tier2_lexical", citation=citation,
        )
        ResultBuilder._assert_citation_invariant("VERIFIED", citation)
        return result

    @staticmethod
    def build_llm_result(claim: str, verdict: str, citation: Citation | None = None) -> AtomicClaimResult:
        effective_citation = None if verdict == "UNVERIFIABLE" else citation
        result = AtomicClaimResult(
            claim_text=claim, claim_type="Extractive", status=verdict,
            verification_method="tier3_llm", citation=effective_citation,
        )
        ResultBuilder._assert_citation_invariant(verdict, result.citation)
        return result

    @staticmethod
    def _assert_citation_invariant(verdict: str, citation: Citation | None) -> None:
        if verdict == "VERIFIED" and citation is None:
            raise InvariantError("citation must be non-null for VERIFIED results")
