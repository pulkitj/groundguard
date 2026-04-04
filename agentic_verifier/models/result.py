"""Public output models."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass
class Source:
    """
    Developer-provided ground truth document.

    Attributes:
        content: The full text of the source document.
        source_id: A unique identifier, e.g. "contract_v3.pdf".
        source_type: Optional hint for future source-type-aware routing.
        page_hint: Optional location hint, e.g. "page 4" or "section 3.2".
    """

    content: str
    source_id: str
    source_type: str | None = None
    page_hint: str | None = None


@dataclass
class AtomicClaimResult:
    """
    Result for a single atomic claim extracted from the full claim.

    Attributes:
        claim_text: The atomic claim sentence.
        claim_type: Whether it is Extractive or Inferential.
        status: The verdict for this specific atom.
        source_id: Which source document supported or contradicted this atom.
        source_excerpt: Verbatim text excerpt from source (Extractive claims only).
        reasoning_basis: LLM reasoning chain (Inferential claims only).
        page_hint: Location hint from the source, if available.
    """

    claim_text: str
    claim_type: Literal["Extractive", "Inferential"]
    status: Literal["VERIFIED", "CONTRADICTED", "UNVERIFIABLE"]
    source_id: str | None = None
    source_excerpt: str | None = None
    reasoning_basis: str | None = None
    page_hint: str | None = None


@dataclass
class VerificationResult:
    """
    The public output of verify(), averify(), verify_batch(), and verify_structured().

    Attributes:
        is_valid: True if the claim is verified; False otherwise.
        overall_verdict: Human-readable summary verdict.
        verification_method: How the result was determined.
        atomic_claims: Per-atom results.
        factual_consistency_score: 0.0–1.0 score.
        sources_used: Deduplicated list of source_ids that contributed to verdict.
        rationale: LLM rationale or brief explanation.
        offending_claim: The first claim atom that caused a non-VERIFIED result, if any.
        status: Machine-readable status code.
        total_cost_usd: Total LLM cost for this verification call.
    """

    is_valid: bool
    overall_verdict: str
    verification_method: Literal["tier2_lexical", "tier3_llm", "skipped"]
    atomic_claims: list[AtomicClaimResult]
    factual_consistency_score: float
    sources_used: list[str]
    rationale: str
    offending_claim: str | None
    status: Literal[
        "VERIFIED",
        "CONTRADICTED",
        "UNVERIFIABLE",
        "SKIPPED_DUE_TO_COST",
        "PARSE_ERROR",
        "ERROR",
    ]
    total_cost_usd: float
