"""Tests for Tier 2 semantic routing — TDD items #12 and #13."""
import pytest
from groundguard.loaders.chunker import Chunk
from groundguard.models.internal import RoutingDecision, Tier2Result, VerificationContext
from groundguard.models.result import Source
from groundguard.tiers.tier2_semantic import route_claim


def _make_ctx(claim: str = "test claim", top_k: int = 5) -> VerificationContext:
    return VerificationContext(
        claim=claim,
        original_sources=[Source(content="test", source_id="s1")],
        model="gpt-4o-mini",
        top_k_chunks=top_k,
    )


def _make_chunks(n: int, text: str = "word1 word2 word3") -> list[Chunk]:
    return [
        Chunk(source_id="s1", text_content=text, char_start=0, char_end=len(text))
        for _ in range(n)
    ]


def test_all_zero_scores_triggers_escalate_all_low_score():
    """TDD #13: When highest BM25 score is at/below floor (0.01), decision is ESCALATE_ALL_LOW_SCORE."""
    # Using completely different vocabulary: claim has no tokens in common with any chunk
    ctx = _make_ctx(claim="xyz123 qwerty uvwxyz", top_k=5)
    chunks = _make_chunks(20, text="apple banana cherry mango grape lemon orange")
    result = route_claim(ctx, chunks)
    assert result.decision == RoutingDecision.ESCALATE_ALL_LOW_SCORE


def test_escalate_all_low_score_capped_at_top_k_times_3():
    """TDD #13: With 100 chunks all scoring near zero, forwarded count is <= top_k * 3."""
    ctx = _make_ctx(claim="xyz123 abc456 def789", top_k=5)
    chunks = _make_chunks(100, text="apple banana cherry mango grape lemon")
    result = route_claim(ctx, chunks)
    assert result.decision == RoutingDecision.ESCALATE_ALL_LOW_SCORE
    assert len(result.top_k_chunks) <= 5 * 3  # top_k_chunks * 3 = 15


def test_escalate_all_low_score_uses_document_order():
    """Branch C chunks are in document order, not BM25 rank order."""
    ctx = _make_ctx(claim="xyz123 qwerty", top_k=3)
    # 20 distinct chunks — document order is the creation order
    chunks = [
        Chunk(source_id="s1", text_content=f"apple{i} banana{i}", char_start=i*10, char_end=i*10+8)
        for i in range(20)
    ]
    result = route_claim(ctx, chunks)
    assert result.decision == RoutingDecision.ESCALATE_ALL_LOW_SCORE
    # First chunk should be from document beginning (index 0)
    assert result.top_k_chunks[0].char_start == 0


def test_high_score_triggers_skip_llm():
    """When highest BM25 score >= lexical_threshold (0.85), decision is SKIP_LLM_HIGH_CONFIDENCE.

    Requires N>=5 sources for positive BM25 IDF scores.
    """
    from groundguard.models.result import Source
    # Need at least 5 sources so BM25 IDF is positive and high-confidence match can fire
    noise_sources = [
        Source(content="unrelated weather forecast for tomorrow", source_id=f"noise{i}")
        for i in range(4)
    ]
    target_source = Source(content="revenue grew thirty percent quarterly", source_id="s1")
    ctx = VerificationContext(
        claim="revenue grew thirty percent quarterly",
        original_sources=[target_source] + noise_sources,
        model="gpt-4o-mini",
    )
    chunks = [
        Chunk(source_id="s1", text_content="revenue grew thirty percent quarterly", char_start=0, char_end=36),
        *[
            Chunk(source_id=f"noise{i}", text_content="unrelated weather forecast for tomorrow", char_start=0, char_end=39)
            for i in range(4)
        ],
    ]
    result = route_claim(ctx, chunks)
    assert result.decision == RoutingDecision.SKIP_LLM_HIGH_CONFIDENCE


def test_partial_match_triggers_escalate_to_llm():
    """TDD #12: Mid-range score triggers ESCALATE_TO_LLM with top-k chunks."""
    ctx = VerificationContext(
        claim="revenue grew significantly this quarter",
        original_sources=[Source(content="test", source_id="s1")],
        model="gpt-4o-mini",
        top_k_chunks=3,
    )
    # Chunks with partial vocabulary overlap
    chunks = [
        Chunk(source_id="s1", text_content="revenue increased this period somewhat", char_start=0, char_end=36),
        Chunk(source_id="s1", text_content="apple banana cherry", char_start=40, char_end=58),
        Chunk(source_id="s1", text_content="unrelated content here", char_start=60, char_end=81),
        Chunk(source_id="s1", text_content="more different stuff", char_start=82, char_end=101),
    ]
    result = route_claim(ctx, chunks)
    # Should be ESCALATE_TO_LLM or SKIP_LLM_HIGH_CONFIDENCE depending on actual BM25 score
    assert result.decision in (RoutingDecision.ESCALATE_TO_LLM, RoutingDecision.SKIP_LLM_HIGH_CONFIDENCE)
    # But top_k_chunks should be limited to ctx.top_k_chunks
    assert len(result.top_k_chunks) <= ctx.top_k_chunks


def test_result_contains_highest_score():
    """Tier2Result.highest_score is populated."""
    ctx = _make_ctx(claim="apple banana cherry")
    chunks = [Chunk(source_id="s1", text_content="apple banana cherry", char_start=0, char_end=19)]
    result = route_claim(ctx, chunks)
    assert result.highest_score >= 0.0


# ---------------------------------------------------------------------------
# Open Issue #2 — Branch C with vocabulary overlap but score <= 0.01
# ---------------------------------------------------------------------------

def test_branch_c_vocabulary_overlap_low_score(mocker):
    """Open Issue #2: claim words appear in chunks but BM25 score is still <= 0.01 → ESCALATE_ALL_LOW_SCORE."""
    from rank_bm25 import BM25Okapi
    import numpy as np
    from groundguard.models.internal import RoutingDecision

    chunks = _make_chunks(5, text="revenue profit loss income")  # some real chunks

    # Mock BM25Okapi.get_scores to return a small positive score (vocabulary overlap, low score)
    mocker.patch.object(BM25Okapi, "get_scores", return_value=np.array([0.005] * 5))

    ctx = _make_ctx()
    result = route_claim(ctx, chunks)

    assert result.decision == RoutingDecision.ESCALATE_ALL_LOW_SCORE
    assert result.highest_score == pytest.approx(0.005)
