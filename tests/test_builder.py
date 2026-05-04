"""Tests for ResultBuilder — TDD items #7, #16, #18, #20."""
import pytest
from groundguard.models.result import Source, VerificationResult, AtomicClaimResult
from groundguard.models.internal import VerificationContext, ClassifiedAtom
from groundguard.models.tier3 import (
    Tier3ResponseModel, TextualEntailment, ConceptualCoverage,
    AtomicVerification, SourceAttribution,
)
from groundguard.loaders.chunker import Chunk
from groundguard.models.builder import ResultBuilder


def _make_ctx(sources=None, atoms=None) -> VerificationContext:
    sources = sources or [Source(content="Revenue was $5M.", source_id="doc.pdf")]
    ctx = VerificationContext(
        claim="Revenue was $5M.",
        original_sources=sources,
        model="gpt-4o-mini",
    )
    if atoms:
        ctx.tier0_atoms = atoms
    return ctx


def _make_t3_model(label="Entailment", score=75.0, verifications=None, attributions=None):
    verifications = verifications or [
        AtomicVerification(
            claim_text="Revenue was $5M.",
            status="VERIFIED",
            source_id="doc.pdf",
            source_excerpt="Revenue was $5M.",
            reasoning_basis=None,
        )
    ]
    attributions = attributions or [SourceAttribution(source_id="doc.pdf", role="Supporting")]
    return Tier3ResponseModel(
        textual_entailment=TextualEntailment(label=label, probability=0.95),
        conceptual_coverage=ConceptualCoverage(percentage=90.0, covered_concepts=["revenue"], missing_concepts=[]),
        factual_consistency_score=score,
        verifications=verifications,
        source_attributions=attributions,
        overall_verdict="The source supports the claim.",
    )


# TDD #7a: factual_consistency_score division
def test_build_llm_result_divides_score_by_100():
    """TDD #7a: t3_model.factual_consistency_score=75 → result.factual_consistency_score=0.75."""
    ctx = _make_ctx()
    t3 = _make_t3_model(score=75.0)
    result = ResultBuilder.build_llm_result(ctx, t3, "tier3_llm")
    assert result.factual_consistency_score == pytest.approx(0.75)


# TDD #7b: Neutral entailment always UNVERIFIABLE
def test_neutral_entailment_always_unverifiable():
    """TDD #7b: Neutral label → status=UNVERIFIABLE, is_valid=False regardless of coverage."""
    ctx = _make_ctx()
    t3 = _make_t3_model(label="Neutral", score=85.0)
    result = ResultBuilder.build_llm_result(ctx, t3, "tier3_llm")
    assert result.status == "UNVERIFIABLE"
    assert result.is_valid is False


def test_entailment_maps_to_verified():
    """Entailment label → status=VERIFIED, is_valid=True."""
    ctx = _make_ctx()
    t3 = _make_t3_model(label="Entailment")
    result = ResultBuilder.build_llm_result(ctx, t3, "tier3_llm")
    assert result.status == "VERIFIED"
    assert result.is_valid is True


def test_contradiction_maps_to_contradicted():
    """Contradiction label → status=CONTRADICTED, is_valid=False."""
    ctx = _make_ctx()
    t3 = _make_t3_model(label="Contradiction", verifications=[
        AtomicVerification(claim_text="Revenue was $5M.", status="CONTRADICTED",
                           source_id="doc.pdf", source_excerpt=None, reasoning_basis=None)
    ])
    result = ResultBuilder.build_llm_result(ctx, t3, "tier3_llm")
    assert result.status == "CONTRADICTED"
    assert result.is_valid is False


# TDD #16: build_lexical_pass
def test_build_lexical_pass_deduplicates_sources():
    """TDD #16: sources_used is deduplicated (encounter order), factual_consistency_score==1.0."""
    ctx = _make_ctx(sources=[
        Source(content="part A", source_id="contract.pdf"),
        Source(content="part B", source_id="addendum.pdf"),
    ])
    chunks = [
        Chunk(source_id="contract.pdf", text_content="part A", char_start=0, char_end=6),
        Chunk(source_id="contract.pdf", text_content="part A2", char_start=0, char_end=7),
        Chunk(source_id="contract.pdf", text_content="part A3", char_start=0, char_end=7),
        Chunk(source_id="addendum.pdf", text_content="part B", char_start=0, char_end=6),
        Chunk(source_id="addendum.pdf", text_content="part B2", char_start=0, char_end=7),
    ]
    result = ResultBuilder.build_lexical_pass(ctx, chunks)
    assert result.factual_consistency_score == 1.0
    assert result.sources_used == ["contract.pdf", "addendum.pdf"]  # dedup, encounter order
    assert result.verification_method == "tier2_lexical"
    assert result.status == "VERIFIED"
    assert result.is_valid is True


def test_build_lexical_pass_no_source_excerpts():
    """TDD #16: No AtomicClaimResult has non-None source_excerpt (no LLM to extract quotes)."""
    ctx = _make_ctx()
    chunks = [Chunk(source_id="doc.pdf", text_content="Revenue was $5M.", char_start=0, char_end=16)]
    result = ResultBuilder.build_lexical_pass(ctx, chunks)
    for atom in result.atomic_claims:
        assert atom.source_excerpt is None


# TDD #18: hallucinated source_id scrubbing
def test_build_llm_result_scrubs_hallucinated_source_ids():
    """TDD #18: source_id not in ctx.original_sources is excluded from sources_used."""
    ctx = _make_ctx(sources=[Source(content="real source", source_id="real.pdf")])
    t3 = _make_t3_model(
        attributions=[
            SourceAttribution(source_id="real.pdf", role="Supporting"),
            SourceAttribution(source_id="wikipedia.org", role="Supporting"),  # hallucinated
        ]
    )
    result = ResultBuilder.build_llm_result(ctx, t3, "tier3_llm")
    assert "wikipedia.org" not in result.sources_used
    assert "real.pdf" in result.sources_used


# TDD #20: page_hint transplantation
def test_build_llm_result_transplants_page_hint():
    """TDD #20: page_hint from Source is copied to AtomicClaimResult via source_id lookup."""
    ctx = _make_ctx(sources=[Source(content="text", source_id="D1", page_hint="pg:3")])
    t3 = _make_t3_model(verifications=[
        AtomicVerification(
            claim_text="text claim",
            status="VERIFIED",
            source_id="D1",
            source_excerpt="text",
            reasoning_basis=None,
        )
    ])
    result = ResultBuilder.build_llm_result(ctx, t3, "tier3_llm")
    assert result.atomic_claims[0].page_hint == "pg:3"


# ---------------------------------------------------------------------------
# Open Issue #4 — build_lexical_pass() with empty matched_chunks
# ---------------------------------------------------------------------------

def test_build_lexical_pass_empty_matched_chunks():
    """Open Issue #4: build_lexical_pass([]) returns valid result with sources_used=[] and source_id=None."""
    from groundguard.models.builder import ResultBuilder
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source

    ctx = VerificationContext(
        claim="Revenue was $5M.",
        original_sources=[Source(content="Revenue was $5M.", source_id="doc.pdf")],
        model="gpt-4o-mini",
    )
    result = ResultBuilder.build_lexical_pass(ctx, matched_chunks=[])

    assert result.sources_used == []
    assert result.status == "VERIFIED"
    assert result.verification_method == "tier2_lexical"
    assert result.factual_consistency_score == 1.0
    assert len(result.atomic_claims) == 1
    assert result.atomic_claims[0].source_id is None


# ---------------------------------------------------------------------------
# Phase 23 — T-84: ResultBuilder v2 tests (new core/result_builder module)
# ---------------------------------------------------------------------------

def test_build_numerical_fast_exit_returns_contradicted():
    from groundguard.core.result_builder import ResultBuilder
    from groundguard.models.result import Source, Citation
    from groundguard.tiers.tier25_preprocessing import Tier25Result
    from groundguard.loaders.chunker import Chunk
    conflict_citation = Citation(source_id="s1", excerpt="30%", excerpt_char_start=12,
                                 excerpt_char_end=15, citation_confidence=1.0)
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="rate is 30%.",
                  char_start=0, char_end=12, token_count=3)
    tier25 = Tier25Result(has_conflict=True, evidence_bundle=[chunk],
                          conflict_citation=conflict_citation)
    result = ResultBuilder.build_numerical_fast_exit(
        claim="rate is 300%.", tier25=tier25,
        source=Source(source_id="s1", content="rate is 30%.")
    )
    assert result.status == "CONTRADICTED"
    assert result.verification_method == "tier25_numerical"
    assert result.citation is not None
    assert result.citation.excerpt == "30%"


def test_build_lexical_pass_citation_non_null():
    from groundguard.core.result_builder import ResultBuilder
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    src = Source(source_id="s1", content="The contract value is one million dollars.")
    chunk = Chunk(chunk_id="c1", source_id="s1",
                  text_content="The contract value is one million dollars.",
                  char_start=0, char_end=41, token_count=8)
    result = ResultBuilder.build_lexical_pass(
        claim="contract value is one million dollars",
        top_chunks=[chunk], score=1.0, source=src
    )
    assert result.citation is not None
    assert result.citation.excerpt is not None
    assert result.citation.excerpt in src.content


def test_build_llm_result_verified_citation_non_null():
    from groundguard.core.result_builder import ResultBuilder
    from groundguard.models.result import Source
    from groundguard.models.tier3 import Tier3ResponseModel
    # mock Tier3ResponseModel with VERIFIED
    # citation must be non-null
    pass  # stub — implement with mocker in real test


def test_build_llm_result_unverifiable_citation_is_none():
    from groundguard.core.result_builder import ResultBuilder
    pass  # stub


def test_citation_invariant_raises_on_verified_null_citation():
    from groundguard.core.result_builder import ResultBuilder
    from groundguard.models.result import AtomicClaimResult, Citation
    from groundguard.exceptions import InvariantError
    with pytest.raises(InvariantError, match="citation"):
        ResultBuilder._assert_citation_invariant(
            verdict="VERIFIED", citation=None
        )


def test_citation_invariant_passes_for_unverifiable():
    from groundguard.core.result_builder import ResultBuilder
    ResultBuilder._assert_citation_invariant(verdict="UNVERIFIABLE", citation=None)
