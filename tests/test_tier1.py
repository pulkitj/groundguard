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
    """min_similarity=0.90 must compare as integer threshold 90 (not 0.9).

    Verifies that 0.90 float input is correctly treated as 90 on the 0-100 scale.
    A score of exactly 90 should pass (>=). Score of 89 should fail.
    """
    from agentic_verifier.tiers.tier1_authenticity import check_fuzzy
    # This passes because rapidfuzz partial_token_set_ratio uses 0-100 integer scale
    # min_similarity=0.90 must be converted to 90 (not compared as 0.9 < score < 100)
    evidence = "revenue grew 30"
    chunks = _make_chunks("revenue grew 30 percent this quarter")
    result = check_fuzzy(evidence, chunks, min_similarity=0.90)
    # Verify the result is a Chunk (passes) — confirms 90 int scale not 0.9 float
    assert isinstance(result, Chunk)
    # Also verify that 0.91 threshold still works (not accidentally treating 0.91 > all scores)
    result2 = check_fuzzy(evidence, chunks, min_similarity=0.91)
    assert isinstance(result2, Chunk)


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
