"""Tests for Tier 1 authenticity check — TDD item #10."""
import pytest
from agentic_verifier.loaders.chunker import Chunk
from agentic_verifier.exceptions import HallucinatedEvidenceError
from agentic_verifier.tiers.tier1_authenticity import check_fuzzy


def _make_chunks(*texts) -> list[Chunk]:
    return [Chunk(parent_source_id="doc.pdf", text_content=t, char_start=0, char_end=len(t)) for t in texts]


def test_exact_match_passes_threshold():
    """Exact match should score 100 and return the matching Chunk."""
    evidence = "The contract was signed on March 1st."
    chunks = _make_chunks("The contract was signed on March 1st.")
    result = check_fuzzy(evidence, chunks, min_similarity=0.90)
    assert isinstance(result, Chunk)


def test_punctuation_drift_passes_threshold():
    """Minor punctuation changes (score ~95) should pass 0.90 threshold."""
    evidence = "Contract signed March 1st, 2025"
    chunks = _make_chunks("The contract was signed on March 1st, 2025.")
    result = check_fuzzy(evidence, chunks, min_similarity=0.90)
    assert isinstance(result, Chunk)


def test_missing_fillers_passes_threshold():
    """Dropped filler words (score ~92) should pass 0.90 threshold."""
    evidence = "revenue grew 30 percent"
    chunks = _make_chunks("The Q3 revenue grew by 30% year-over-year.")
    result = check_fuzzy(evidence, chunks, min_similarity=0.90)
    assert isinstance(result, Chunk)


def test_paraphrase_below_threshold_raises():
    """Paraphrase with score ~35 should raise HallucinatedEvidenceError at 0.90."""
    evidence = "Deal closed in early 2024 worth approximately ten million"
    chunks = _make_chunks("The contract was signed on March 1st, 2025 for $2.5M.")
    with pytest.raises(HallucinatedEvidenceError):
        check_fuzzy(evidence, chunks, min_similarity=0.90)


def test_threshold_scaling_0_90_means_90_int():
    """min_similarity=0.50 maps to integer threshold 50 (not float 0.50).

    Uses an evidence/chunk pair that scores ~42 on rapidfuzz (0-100 scale).
    - Correct impl: threshold_int = int(0.50 * 100) = 50 → 42 < 50 → raises
    - Buggy impl:   threshold     = 0.50 (float)     → 42 >= 0.50 → passes (WRONG)

    This boundary test confirms the int scale conversion is applied.
    """
    from agentic_verifier.tiers.tier1_authenticity import check_fuzzy
    # Score ~42: dissimilar enough that threshold 50 rejects it
    evidence = "the company reported gains"
    chunks = _make_chunks("revenue grew thirty percent quarterly")

    # threshold=50 (int) → 42 < 50 → must raise
    with pytest.raises(HallucinatedEvidenceError):
        check_fuzzy(evidence, chunks, min_similarity=0.50)

    # threshold=30 (int) → 42 >= 30 → must pass
    result = check_fuzzy(evidence, chunks, min_similarity=0.30)
    assert isinstance(result, Chunk)


def test_returns_best_matching_chunk():
    """Returns the chunk with the highest similarity score."""
    evidence = "revenue 30%"
    chunks = _make_chunks(
        "Unrelated content about weather",
        "The Q3 revenue grew by 30%",  # best match
        "More unrelated content",
    )
    result = check_fuzzy(evidence, chunks, min_similarity=0.50)
    assert result.text_content == "The Q3 revenue grew by 30%"


def test_error_message_contains_score_and_threshold():
    """HallucinatedEvidenceError message must include score and threshold."""
    evidence = "completely unrelated xyz123"
    chunks = _make_chunks("The contract was signed on March 1st.")
    with pytest.raises(HallucinatedEvidenceError, match=r"similarity"):
        check_fuzzy(evidence, chunks, min_similarity=0.90)


# ---------------------------------------------------------------------------
# Open Issue #3 — check_fuzzy() with empty chunks list
# ---------------------------------------------------------------------------

def test_check_fuzzy_with_empty_chunks_raises_hallucinated_evidence_error():
    """Open Issue #3: check_fuzzy(evidence, [], ...) raises HallucinatedEvidenceError (score=0, no match)."""
    evidence = "Revenue was $5M."
    with pytest.raises(HallucinatedEvidenceError):
        check_fuzzy(evidence, [], min_similarity=0.90)
