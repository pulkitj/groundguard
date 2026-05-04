"""Tier 1 authenticity check — fuzzy evidence-in-source verification."""
from __future__ import annotations
from typing import TYPE_CHECKING

from rapidfuzz import fuzz
from groundguard.exceptions import HallucinatedEvidenceError
from groundguard._log import logger

if TYPE_CHECKING:
    from groundguard.loaders.chunker import Chunk


def check_fuzzy(
    evidence: str,
    chunks: list[Chunk],
    min_similarity: float,
) -> Chunk:
    """
    Searches all chunks for the best fuzzy substring match to `evidence`.

    Uses rapidfuzz.fuzz.partial_token_set_ratio — tokenises both strings, takes the
    set-intersection approach on substrings, and returns int 0–100. This handles
    dropped filler words, minor punctuation drift, and word-order variation.

    Args:
        evidence: The agent-provided evidence string to validate.
        chunks: All chunks from source documents.
        min_similarity: Float 0.0–1.0; scaled to int 0–100 for comparison.

    Returns:
        The Chunk with the highest similarity score.

    Raises:
        HallucinatedEvidenceError: If best score < min_similarity threshold.
    """
    threshold_int = int(min_similarity * 100)
    best_score = 0
    best_chunk = None

    for chunk in chunks:
        score = fuzz.partial_token_set_ratio(evidence, chunk.text_content)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_score >= threshold_int and best_chunk is not None:
        logger.debug(
            "Tier 1 pass: evidence similarity %.2f >= %.2f threshold",
            best_score / 100,
            min_similarity,
        )
        return best_chunk

    logger.warning(
        "Tier 1 FAIL: evidence similarity %.2f < %.2f threshold — HallucinatedEvidenceError raised",
        best_score / 100,
        min_similarity,
    )
    raise HallucinatedEvidenceError(
        f"agent_provided_evidence not found in any source "
        f"(best similarity: {best_score / 100:.2f}, required: {min_similarity:.2f}). "
        "Note: threshold is calibrated for rapidfuzz partial_token_set_ratio."
    )
