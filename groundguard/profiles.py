from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationProfile:
    name: str
    faithfulness_threshold: float
    tier2_lexical_threshold: float
    bm25_top_k: int
    majority_vote: bool
    audit: bool


STRICT_PROFILE = VerificationProfile(
    name="strict",
    faithfulness_threshold=0.97,
    tier2_lexical_threshold=2.0,
    bm25_top_k=6,
    majority_vote=True,
    audit=True,
)

GENERAL_PROFILE = VerificationProfile(
    name="general",
    faithfulness_threshold=0.80,
    tier2_lexical_threshold=0.85,
    bm25_top_k=3,
    majority_vote=False,
    audit=False,
)

RESEARCH_PROFILE = VerificationProfile(
    name="research",
    faithfulness_threshold=0.70,
    tier2_lexical_threshold=0.85,
    bm25_top_k=4,
    majority_vote=False,
    audit=False,
)
