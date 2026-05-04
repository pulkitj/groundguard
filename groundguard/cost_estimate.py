"""Cheap, side-effect-free cost estimates for public verification helpers."""
from __future__ import annotations

from dataclasses import dataclass

from groundguard.models.result import Source
from groundguard.profiles import VerificationProfile


@dataclass
class CostEstimate:
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_usd: float
    majority_vote_multiplier: int = 1


_PRICING: dict[str, tuple[float, float]] = {
    # USD per 1K tokens. Unknown models intentionally estimate to zero.
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
    "claude-3-5-sonnet": (0.00300, 0.01500),
    "gemini-1.5-pro": (0.00350, 0.01050),
}

_PROMPT_SKELETON_TOKENS = 350
_JSON_RESPONSE_TOKENS = 200


def _approx_tokens(text: str) -> int:
    return int(round(len(text.split()) * 1.3))


def _source_text(source: str | Source) -> str:
    if isinstance(source, Source):
        return source.content
    return source


def _top_source_tokens(sources: list[str | Source]) -> int:
    return sum(_approx_tokens(_source_text(source)) for source in sources[:3])


def _pricing_for(model: str) -> tuple[float, float]:
    for prefix, prices in _PRICING.items():
        if model.startswith(prefix):
            return prices
    return (0.0, 0.0)


def _majority_vote_multiplier(profile: VerificationProfile | None) -> int:
    if profile is None:
        return 1

    majority_vote = profile.majority_vote
    if majority_vote is True:
        return 3
    if majority_vote is False or majority_vote is None:
        return 1
    if isinstance(majority_vote, int) and majority_vote > 1:
        return majority_vote
    return 1


def _build_estimate(
    input_tokens: int,
    output_tokens: int,
    model: str,
    multiplier: int,
    return_breakdown: bool,
) -> float | CostEstimate:
    input_tokens *= multiplier
    output_tokens *= multiplier

    input_usd_per_1k, output_usd_per_1k = _pricing_for(model)
    input_cost_usd = (input_tokens / 1000) * input_usd_per_1k
    output_cost_usd = (output_tokens / 1000) * output_usd_per_1k
    total_usd = input_cost_usd + output_cost_usd

    if not return_breakdown:
        return total_usd

    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_cost_usd,
        output_cost_usd=output_cost_usd,
        total_usd=total_usd,
        majority_vote_multiplier=multiplier,
    )


def estimate_verify_faithfulness_cost(
    claim: str,
    sources: list[str | Source],
    model: str,
    profile: VerificationProfile | None = None,
    return_breakdown: bool = False,
) -> float | CostEstimate:
    input_tokens = _PROMPT_SKELETON_TOKENS + _approx_tokens(claim) + _top_source_tokens(sources)
    output_tokens = _JSON_RESPONSE_TOKENS
    multiplier = _majority_vote_multiplier(profile)

    return _build_estimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        multiplier=multiplier,
        return_breakdown=return_breakdown,
    )


def estimate_verify_analysis_cost(
    output: str,
    sources: list[str | Source],
    model: str,
    profile: VerificationProfile | None = None,
    return_breakdown: bool = False,
) -> float | CostEstimate:
    sentences = [sentence.strip() for sentence in output.split(".") if sentence.strip()]
    sentence_count = max(1, len(sentences))
    per_sentence_context_tokens = _PROMPT_SKELETON_TOKENS + _top_source_tokens(sources)
    input_tokens = (per_sentence_context_tokens * sentence_count) + sum(
        _approx_tokens(sentence) for sentence in sentences
    )
    output_tokens = _JSON_RESPONSE_TOKENS * sentence_count
    multiplier = _majority_vote_multiplier(profile)

    return _build_estimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        multiplier=multiplier,
        return_breakdown=return_breakdown,
    )
