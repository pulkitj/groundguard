"""Tier 3 LLM response models."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class TextualEntailment(BaseModel):
    """
    The LLM's textual entailment judgment.

    Attributes:
        label: The entailment decision.
        probability: Confidence score 0.0–1.0.
    """
    label: Literal["Entailment", "Contradiction", "Neutral"]
    probability: float = Field(ge=0.0, le=1.0)


class ConceptualCoverage(BaseModel):
    """
    How much of the claim's concepts are covered by the source.

    Attributes:
        percentage: 0–100 coverage percentage.
        covered_concepts: List of concepts found in source.
        missing_concepts: List of concepts not found in source.
    """
    percentage: float = Field(ge=0.0, le=100.0)
    covered_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)


class AtomicVerification(BaseModel):
    """
    Verification result for a single atomic claim.

    Attributes:
        claim_text: The atomic sentence being verified.
        status: The verdict for this atom.
        source_id: The source document that supports/contradicts this atom.
        source_excerpt: Verbatim text from source (Extractive claims).
        reasoning_basis: LLM reasoning chain (Inferential claims).
    """
    claim_text: str
    status: Literal["VERIFIED", "CONTRADICTED", "UNVERIFIABLE"]
    source_id: str | None = None
    source_excerpt: str | None = None
    reasoning_basis: list[str] | None = None  # FIX-04: PRD specifies list[str], not str


class SourceAttribution(BaseModel):
    """
    How a source document contributed to the overall verdict.

    Attributes:
        source_id: The source document identifier.
        role: How this source relates to the claim.
    """
    source_id: str
    role: Literal["Supporting", "Contradicting", "Partially Relevant", "Not Used"]


class Tier3ResponseModel(BaseModel):
    """
    The complete structured response from the Tier 3 LLM evaluation.

    Note: factual_consistency_score is 0–100 (not 0.0–1.0).
    ResultBuilder divides by 100 when mapping to VerificationResult.
    """
    textual_entailment: TextualEntailment
    conceptual_coverage: ConceptualCoverage
    factual_consistency_score: float = Field(ge=0.0, le=100.0)
    verifications: list[AtomicVerification]
    source_attributions: list[SourceAttribution]
    overall_verdict: str
