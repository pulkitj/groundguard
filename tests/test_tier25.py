import re

import pytest


def _make_chunk(source_id: str, text: str, chunk_id: str = "c1") -> "Chunk":
    from groundguard.loaders.chunker import Chunk
    return Chunk(chunk_id=chunk_id, source_id=source_id,
                 text_content=text, char_start=0, char_end=len(text), token_count=len(text.split()))

def _make_ctx(claim: str, source_content: str = "", source_id: str = "s1") -> "VerificationContext":
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    src = Source(source_id=source_id, content=source_content)
    return VerificationContext(claim=claim, sources=[src])


def test_number_pattern_matches_percentage():
    from groundguard.tiers.tier25_preprocessing import _NUMBER_PATTERN
    assert re.search(_NUMBER_PATTERN, "30%")
    assert re.search(_NUMBER_PATTERN, "300%")


def test_number_pattern_matches_currency():
    from groundguard.tiers.tier25_preprocessing import _NUMBER_PATTERN
    assert re.search(_NUMBER_PATTERN, "$4.2M")
    assert re.search(_NUMBER_PATTERN, "$300")


def test_normalise_number_strips_commas():
    from groundguard.tiers.tier25_preprocessing import _normalise_number
    assert _normalise_number("1,000,000") == 1000000.0


def test_normalise_number_strips_currency_suffix():
    from groundguard.tiers.tier25_preprocessing import _normalise_number
    assert _normalise_number("$4.2") == 4.2


def test_tier25_detects_numerical_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    src = Source(source_id="s1", content="The fee shall not exceed 30% of revenue.")
    ctx = VerificationContext(claim="The fee shall not exceed 300% of revenue.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="The fee shall not exceed 30% of revenue.",
                  char_start=0, char_end=40, token_count=8)
    result = run(ctx, [chunk])
    assert result.has_conflict is True
    assert result.verification_method == "tier25_numerical"


def test_tier25_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    src = Source(source_id="s1", content="Revenue was $4.2M in Q1.")
    ctx = VerificationContext(claim="Revenue was $4.2M in Q1.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="Revenue was $4.2M in Q1.",
                  char_start=0, char_end=24, token_count=6)
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_tier25_result_has_verification_method():
    from groundguard.tiers.tier25_preprocessing import run, Tier25Result
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    src = Source(source_id="s1", content="X is 30%.")
    ctx = VerificationContext(claim="X is 300%.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="X is 30%.",
                  char_start=0, char_end=9, token_count=3)
    result = run(ctx, [chunk])
    assert result.verification_method == "tier25_numerical"


def test_tier25_evidence_bundle_populated():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    src = Source(source_id="s1", content="The rate is 30%.")
    ctx = VerificationContext(claim="The rate is 300%.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="The rate is 30%.",
                  char_start=0, char_end=16, token_count=4)
    result = run(ctx, [chunk])
    assert len(result.evidence_bundle) > 0


def test_tier25_citation_excerpt_points_to_conflicting_value():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    src = Source(source_id="s1", content="The rate is 30%.")
    ctx = VerificationContext(claim="The rate is 300%.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="The rate is 30%.",
                  char_start=0, char_end=16, token_count=4)
    result = run(ctx, [chunk])
    assert result.conflict_citation is not None
    assert "30" in result.conflict_citation.excerpt


def test_tier25_no_conflict_for_range_containing_source_value():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    src = Source(source_id="s1", content="Revenue grew 40%.")
    ctx = VerificationContext(claim="Revenue grew 30% to 50%.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="Revenue grew 40%.",
                  char_start=0, char_end=17, token_count=3)
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_tier25_no_conflict_for_section_reference_insufficient_context():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    src = Source(source_id="s1", content="Reference Section 4.3.")
    ctx = VerificationContext(claim="See Section 4.2 for details.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="Reference Section 4.3.",
                  char_start=0, char_end=22, token_count=3)
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_tier25_year_conflict_in_temporal_context():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    src = Source(source_id="s1", content="In 2023, revenue grew 30%.")
    ctx = VerificationContext(claim="In 2024, revenue grew 30%.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="In 2023, revenue grew 30%.",
                  char_start=0, char_end=26, token_count=5)
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_number_pattern_matches_negative_percentage():
    from groundguard.tiers.tier25_preprocessing import _NUMBER_PATTERN
    import re
    assert re.search(_NUMBER_PATTERN, "-5%")
    assert re.search(_NUMBER_PATTERN, "-300%")


def test_number_pattern_matches_negative_currency():
    from groundguard.tiers.tier25_preprocessing import _NUMBER_PATTERN
    import re
    assert re.search(_NUMBER_PATTERN, "-$4.2M")


def test_number_pattern_no_false_positive_on_hyphen_word():
    from groundguard.tiers.tier25_preprocessing import _NUMBER_PATTERN
    import re
    # Hyphen in compound word must NOT produce a negative number match
    m = re.search(_NUMBER_PATTERN, "non-5 year contract")
    # If it matches, it must not start with '-' (i.e. must not capture -5)
    assert m is None or not m.group(0).startswith("-")


@pytest.mark.skip(reason="T-P1 will reactivate: version strings become Gate 1 structural blocks, so partial _NUMBER_PATTERN matching of 4.2.1 is no longer valid")
def test_number_pattern_no_match_on_version_string():
    from groundguard.tiers.tier25_preprocessing import _NUMBER_PATTERN
    import re
    # "4.2.1" — second dot makes it a version, not a decimal number
    # Pattern should match "4.2" (stopping before the second dot) rather than "4.2.1"
    m = re.search(_NUMBER_PATTERN, "4.2.1")
    assert m is not None
    assert m.group(0) == "4.2"  # stops before second dot


def test_normalise_number_negative_percentage():
    from groundguard.tiers.tier25_preprocessing import _normalise_number
    assert _normalise_number("-5%") == -5.0


def test_normalise_number_negative_currency():
    from groundguard.tiers.tier25_preprocessing import _normalise_number
    assert _normalise_number("-$4.2") == -4.2


def test_tier25_detects_conflict_for_negative_vs_positive():
    """Claim says profit 5%, source says -5% — must detect conflict."""
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    src = Source(source_id="s1", content="Profit margin was -5% in Q3.")
    ctx = VerificationContext(claim="Profit margin was 5% in Q3.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1",
                  text_content="Profit margin was -5% in Q3.",
                  char_start=0, char_end=28, token_count=6)
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_tier25_arithmetic_sum_not_flagged_as_conflict():
    """Tier 2.5 must not flag arithmetic derivations as conflicts.

    When the claim value equals the sum of source values ($5M + $10M = $15M),
    it is a valid arithmetic derivation — Tier 2.5 must pass it through to Tier 3.
    """
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    src = Source(source_id="r", content="Q1 revenue was $5 million. Q2 revenue was $10 million.")
    ctx = VerificationContext(
        claim="Total H1 revenue was $15 million.",
        original_sources=[src],
        model="gpt-4o-mini",
    )
    chunk = Chunk(chunk_id="c1", source_id="r",
                  text_content="Q1 revenue was $5 million. Q2 revenue was $10 million.",
                  char_start=0, char_end=53, token_count=10)
    result = run(ctx, [chunk])
    assert result.has_conflict is False, (
        "Tier 2.5 must not flag $5M+$10M=$15M as a conflict — "
        "arithmetic sum of source values equals claim value"
    )


# ---------------------------------------------------------------------------
# Tier 2.5 Year Subset and Premature Break Fixes
# ---------------------------------------------------------------------------

def test_year_subset_no_false_positive():
    """chunk has years {2022, 2023}, claim has year {2023} -> no conflict (subset is fine)."""
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    src = Source(source_id="s1", content="In 2022 and in 2023, growth was steady.")
    ctx = VerificationContext(claim="In 2023, growth was steady.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="In 2022 and in 2023, growth was steady.",
                  char_start=0, char_end=39, token_count=9)

    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_year_mismatch_is_conflict():
    """chunk has years {2022}, claim has year {2023} -> conflict (disjoint years)."""
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    src = Source(source_id="s1", content="In 2022, growth was steady.")
    ctx = VerificationContext(claim="In 2023, growth was steady.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="In 2022, growth was steady.",
                  char_start=0, char_end=27, token_count=6)

    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_later_chunk_clears_year_conflict():
    """Chunk 1 has year {2022}, Chunk 2 has year {2023}, claim has year {2023} -> no conflict."""
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    src = Source(source_id="s1", content="In 2022 growth was high. In 2023 growth was high.")
    ctx = VerificationContext(claim="In 2023 growth was high.", sources=[src])
    chunk1 = Chunk(chunk_id="c1", source_id="s1", text_content="In 2022 growth was high.",
                   char_start=0, char_end=24, token_count=5)
    chunk2 = Chunk(chunk_id="c2", source_id="s1", text_content="In 2023 growth was high.",
                   char_start=25, char_end=49, token_count=5)

    result = run(ctx, [chunk1, chunk2])
    assert result.has_conflict is False


# T25-A
def test_t25a_range_conflict_outside_range():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("Revenue grew 30% to 50%.", "Revenue grew 60%.", "s1")
    chunk = _make_chunk("s1", "Revenue grew 60%.")
    result = run(ctx, [chunk])
    assert result.has_conflict is True

# T25-B — first chunk has no numbers; conflict found only in second chunk
def test_t25b_multi_chunk_conflict_in_second_chunk():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("The rate is 300%.", "irrelevant", "s1")
    chunk1 = _make_chunk("s1", "Revenue grew significantly.", chunk_id="c1")  # no numbers
    chunk2 = _make_chunk("s2", "The fee is 30%.", chunk_id="c2")   # 300 != 30 — conflict
    result = run(ctx, [chunk1, chunk2])
    assert result.has_conflict is True
    assert result.conflict_citation is not None
    assert result.conflict_citation.source_id == "s2"

# T25-C
def test_t25c_stops_at_first_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("The rate is 300%.", "irrelevant", "s1")
    chunk1 = _make_chunk("s1", "The fee is 30%.",   chunk_id="c1")
    chunk2 = _make_chunk("s2", "Revenue was $5M.",  chunk_id="c2")
    chunk3 = _make_chunk("s3", "Costs were $1M.",   chunk_id="c3")
    result = run(ctx, [chunk1, chunk2, chunk3])
    assert result.has_conflict is True
    assert result.conflict_citation is not None
    assert result.conflict_citation.source_id == "s1"
    assert "30" in result.conflict_citation.excerpt

# T25-D
def test_t25d_year_conflict_citation_populated():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("In 2024, revenue grew 30%.", "In 2023, revenue grew 30%.", "s1")
    chunk = _make_chunk("s1", "In 2023, revenue grew 30%.")
    result = run(ctx, [chunk])
    assert result.has_conflict is True
    assert result.conflict_citation is not None
    assert "2023" in result.conflict_citation.excerpt
    assert result.conflict_citation.source_id == "s1"

# T25-E
def test_t25e_build_evidence_bundle_top_k_cap():
    from groundguard.tiers.tier25_preprocessing import build_evidence_bundle
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    src = Source(source_id="s1", content="x")
    ctx = VerificationContext(claim="fee is 10%.", sources=[src])
    chunks = [_make_chunk("s1", f"value is {i}%.", chunk_id=f"c{i}") for i in range(5)]
    result = build_evidence_bundle(ctx, chunks, top_k=3)
    assert len(result) == 3

# T25-F
def test_t25f_has_sufficient_metric_context_boundary_cases():
    from groundguard.tiers.tier25_preprocessing import _has_sufficient_metric_context
    assert _has_sufficient_metric_context("300%") is False
    assert _has_sufficient_metric_context("300% margin") is True
    assert _has_sufficient_metric_context("the 300% in") is False
def test_later_chunk_clears_number_conflict():
    """Chunk 1 has number 4.5, Chunk 2 has number 4.2, claim asserts 4.2 -> no conflict."""
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    src = Source(source_id="s1", content="Revenue was $4.5M in Q1. Revenue was $4.2M in Q2.")
    ctx = VerificationContext(claim="Revenue was $4.2M.", sources=[src])
    chunk1 = Chunk(chunk_id="c1", source_id="s1", text_content="Revenue was $4.5M in Q1.",
                   char_start=0, char_end=24, token_count=6)
    chunk2 = Chunk(chunk_id="c2", source_id="s1", text_content="Revenue was $4.2M in Q2.",
                   char_start=25, char_end=49, token_count=6)

    result = run(ctx, [chunk1, chunk2])
    assert result.has_conflict is False


# ---------------------------------------------------------------------------
# T-P1 Gate 1 Blocklist and Composite Pre-Pass
# ---------------------------------------------------------------------------

def test_t25p1_mask_structural_replaces_section_label_with_spaces_and_preserves_length():
    from groundguard.tiers.tier25_preprocessing import mask_structural

    text = "Section 3 revenue was $1M"
    masked = mask_structural(text)

    assert len(masked) == len(text)
    assert masked.startswith(" " * len("Section 3"))
    assert masked.endswith(" revenue was $1M")


def test_t25p1_gate1_blocklist_is_module_level_compiled_pattern_list():
    from groundguard.tiers.tier25_preprocessing import _GATE1_BLOCKLIST

    assert isinstance(_GATE1_BLOCKLIST, list)
    assert _GATE1_BLOCKLIST
    assert all(isinstance(pattern, re.Pattern) for pattern in _GATE1_BLOCKLIST)


def test_t25p1_mask_structural_replaces_citation_bracket_with_spaces_and_preserves_length():
    from groundguard.tiers.tier25_preprocessing import mask_structural

    text = "See [12] for details"
    masked = mask_structural(text)

    assert len(masked) == len(text)
    assert masked == "See      for details"


def test_t25p1_mask_structural_leaves_factual_patient_count_unchanged():
    from groundguard.tiers.tier25_preprocessing import mask_structural

    text = "42 patients enrolled"

    assert mask_structural(text) == text


def test_t25p1_extract_composite_numbers_extracts_three_million_and_removes_span():
    from groundguard.tiers.tier25_preprocessing import extract_composite_numbers

    values, remaining = extract_composite_numbers("3 million users enrolled")

    assert values == [(3000000.0, "3 million")]
    assert "3 million" not in remaining


def test_t25p1_extract_composite_numbers_extracts_decimal_billion():
    from groundguard.tiers.tier25_preprocessing import extract_composite_numbers

    values, remaining = extract_composite_numbers("2.5 billion revenue")

    assert values == [(2500000000.0, "2.5 billion")]
    assert "2.5 billion" not in remaining


def test_t25p1_passage_label_is_masked_and_currency_value_remains_extractable():
    from groundguard.tiers.tier25_preprocessing import mask_structural, run

    claim = "Passage 1 states the revenue was $400M"
    masked = mask_structural(claim)
    ctx = _make_ctx(claim, "The revenue was $400M.", "s1")
    chunk = _make_chunk("s1", "The revenue was $400M.")
    result = run(ctx, [chunk])

    assert "Passage 1" not in masked
    assert "$400M" in masked
    assert result.has_conflict is False
    assert any(
        check.match and (check.claim_number == "$400M" or check.source_number in {"$400M", "400.0"})
        for check in result.numerical_checks
    )


def test_t25p1_composite_prepass_prevents_bare_digit_conflict_for_three_million():
    from groundguard.tiers.tier25_preprocessing import extract_composite_numbers, run

    values, remaining = extract_composite_numbers("3 million users")
    ctx = _make_ctx("3 million users", "There were 3000000 users.", "s1")
    chunk = _make_chunk("s1", "There were 3000000 users.")
    result = run(ctx, [chunk])

    assert values == [(3000000.0, "3 million")]
    assert "3" not in remaining
    assert result.has_conflict is False


def test_t25p1_accounting_negative_normalizes_parenthesized_suffix_number():
    from groundguard.tiers.tier25_preprocessing import normalize_accounting_negatives

    assert normalize_accounting_negatives("revenue was ($4.2M)") == "revenue was -$4.2M"


def test_t25p1_accounting_negative_normalizes_comma_decimal_number():
    from groundguard.tiers.tier25_preprocessing import normalize_accounting_negatives

    assert normalize_accounting_negatives("(1,234.56) loss") == "-1234.56 loss"


def test_t25p1_accounting_negative_leaves_parenthesized_percentage_unchanged():
    from groundguard.tiers.tier25_preprocessing import normalize_accounting_negatives

    assert normalize_accounting_negatives("(5%) decline") == "(5%) decline"


def test_t25p1_normalize_eu_numbers_converts_comma_decimal_and_dot_thousands():
    from groundguard.tiers.tier25_preprocessing import normalize_eu_numbers

    assert normalize_eu_numbers("1.234,56 revenue") == "1234.56 revenue"


def test_t25p1_normalize_eu_numbers_leaves_non_eu_text_unchanged():
    from groundguard.tiers.tier25_preprocessing import normalize_eu_numbers

    assert normalize_eu_numbers("price is 1.5 million") == "price is 1.5 million"


def test_t25p1_prepasses_apply_to_source_structural_accounting_and_composite_values():
    from groundguard.tiers.tier25_preprocessing import run

    claim = "Revenue was -1234.56 and enrollment was 3000000."
    source = "Section 3 reports revenue of (1,234.56) and enrollment of 3 million."
    ctx = _make_ctx(claim, source, "s1")
    chunk = _make_chunk("s1", source)
    result = run(ctx, [chunk])

    assert result.has_conflict is False


def test_t25p1_fig_label_is_masked_patient_count_fast_accepts_without_conflict():
    from groundguard.tiers.tier25_preprocessing import mask_structural, run

    claim = "Fig. 3 shows 42 patients"
    masked = mask_structural(claim)
    ctx = _make_ctx(claim, "The study shows 42 patients.", "s1")
    chunk = _make_chunk("s1", "The study shows 42 patients.")
    result = run(ctx, [chunk])

    assert "Fig. 3" not in masked
    assert "42 patients" in masked
    assert result.has_conflict is False


def test_t25p1_step_progress_numbers_are_masked_and_tier25_skips():
    from groundguard.tiers.tier25_preprocessing import mask_structural, run

    claim = "Step 2 of 5 completed"
    masked = mask_structural(claim)
    ctx = _make_ctx(claim, "The task is completed.", "s1")
    chunk = _make_chunk("s1", "The task is completed.")
    result = run(ctx, [chunk])

    assert "2" not in masked
    assert "5" not in masked
    assert result.has_conflict is False
    assert result.evidence_bundle == []


def test_t25p1_roman_phase_label_is_masked_and_patient_count_remains_extractable():
    from groundguard.tiers.tier25_preprocessing import mask_structural, run

    claim = "Phase II trial enrolled 200 patients"
    masked = mask_structural(claim)
    ctx = _make_ctx(claim, "The trial enrolled 200 patients.", "s1")
    chunk = _make_chunk("s1", "The trial enrolled 200 patients.")
    result = run(ctx, [chunk])

    assert "Phase II" not in masked
    assert "200 patients" in masked
    assert result.has_conflict is False


def test_t25p1_product_version_is_masked_and_percentage_remains_extractable():
    from groundguard.tiers.tier25_preprocessing import mask_structural, run

    claim = "Windows 11 reached 20% market share"
    masked = mask_structural(claim)
    ctx = _make_ctx(claim, "Windows reached 20% market share.", "s1")
    chunk = _make_chunk("s1", "Windows reached 20% market share.")
    result = run(ctx, [chunk])

    assert "11" not in masked
    assert "20%" in masked
    assert result.has_conflict is False


# T-P2 Tests


def test_extract_and_normalize_euro_revenue():
    from groundguard.tiers.tier25_preprocessing import _NUMBER_PATTERN, _normalise_number
    import re
    m = re.search(_NUMBER_PATTERN, "€1,234.56 revenue")
    assert _normalise_number(m.group(0) if m else "") == 1234.56


def test_extract_and_normalize_positive_growth():
    from groundguard.tiers.tier25_preprocessing import _NUMBER_PATTERN, _normalise_number
    import re
    m = re.search(_NUMBER_PATTERN, "+5% growth")
    assert _normalise_number(m.group(0) if m else "") == 5.0


def test_extract_and_normalize_negative_decline():
    from groundguard.tiers.tier25_preprocessing import _NUMBER_PATTERN, _normalise_number
    import re
    m = re.search(_NUMBER_PATTERN, "-3% decline")
    assert _normalise_number(m.group(0) if m else "") == -3.0


def test_extract_and_normalize_scientific_notation():
    from groundguard.tiers.tier25_preprocessing import _NUMBER_PATTERN, _normalise_number
    import re
    m = re.search(_NUMBER_PATTERN, "1.5e6 infections")
    assert _normalise_number(m.group(0) if m else "") == 1500000.0


def test_extract_and_normalize_basis_points():
    from groundguard.tiers.tier25_preprocessing import _NUMBER_PATTERN, _normalise_number
    import re
    m = re.search(_NUMBER_PATTERN, "50bps rate increase")
    assert _normalise_number(m.group(0) if m else "") == 0.50


def test_normalize_european_format_normalise_number():
    from groundguard.tiers.tier25_preprocessing import _normalise_number
    assert _normalise_number("1.234,56") == 1234.56


def test_extract_and_normalize_usd_magnitude():
    from groundguard.tiers.tier25_preprocessing import _NUMBER_PATTERN, _normalise_number
    import re
    m = re.search(_NUMBER_PATTERN, "USD 4.2M")
    assert _normalise_number(m.group(0) if m else "") == 4200000.0


def test_normalise_number_empty_string_raises_value_error():
    from groundguard.tiers.tier25_preprocessing import _normalise_number
    with pytest.raises(ValueError):
        _normalise_number("")


def test_normalise_number_none_raises_type_error():
    from groundguard.tiers.tier25_preprocessing import _normalise_number
    with pytest.raises(TypeError):
        _normalise_number(None)


def test_normalise_number_signed_zero():
    from groundguard.tiers.tier25_preprocessing import _normalise_number
    assert _normalise_number("-0.0") == 0.0


def test_extract_and_normalize_basis_points_excess_whitespace():
    from groundguard.tiers.tier25_preprocessing import _NUMBER_PATTERN, _normalise_number
    import re
    m = re.search(_NUMBER_PATTERN, "GBP 50 basis  points")
    assert _normalise_number(m.group(0) if m else "") == 0.50


def test_normalize_european_format_with_thousands_separators():
    from groundguard.tiers.tier25_preprocessing import _normalise_number
    assert _normalise_number("12.345.678,90") == 12345678.90


# T-P3 Tests

def test_t25p3_rhetorical_nouns_set_constant():
    from groundguard.tiers.tier25_preprocessing import _RHETORICAL_NOUNS
    assert isinstance(_RHETORICAL_NOUNS, set)
    assert "reason" in _RHETORICAL_NOUNS
    assert "ways" in _RHETORICAL_NOUNS


def test_t25p3_measurable_units_and_entity_nouns_split():
    from groundguard.tiers.tier25_preprocessing import _MEASURABLE_UNITS, _ENTITY_NOUNS
    assert isinstance(_MEASURABLE_UNITS, set)
    assert isinstance(_ENTITY_NOUNS, set)
    assert "kg" in _MEASURABLE_UNITS
    assert "%" in _MEASURABLE_UNITS
    assert "patients" in _ENTITY_NOUNS
    assert "locations" in _ENTITY_NOUNS


def test_t25p3_numerical_value_named_tuple():
    from groundguard.tiers.tier25_preprocessing import NumericalValue
    val = NumericalValue(20.0, "kg")
    assert val.value == 20.0
    assert val.unit == "kg"


def test_t25p3_rhetorical_noun_reasons_discarded_in_run():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    # Claim has "3 reasons", which should be discarded (0 numbers extracted).
    # Source has "5 reasons", which is NOT discarded (since Gate 2 only applies to claim).
    # Because claim number is discarded, no conflict should be found.
    src = Source(source_id="s1", content="There are 5 reasons.")
    ctx = VerificationContext(claim="There are 3 reasons.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="There are 5 reasons.",
                  char_start=0, char_end=20, token_count=4)
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_t25p3_entity_noun_locations_unit_none():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    # "3 locations confirmed" has the entity noun "locations", so its unit should be None.
    # When compared with a unitless "5" in the source, it should be a conflict.
    src = Source(source_id="s1", content="There were 5 confirmed.")
    ctx = VerificationContext(claim="3 locations confirmed", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="There were 5 confirmed.",
                  char_start=0, char_end=23, token_count=4)
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p3_entity_noun_amino_acids_unit_none():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    # "20 amino acids found" has the entity noun "amino acids" (2-gram), so its unit should be None.
    # When compared with a unitless "30" in the source, it should be a conflict.
    src = Source(source_id="s1", content="There were 30 found.")
    ctx = VerificationContext(claim="20 amino acids found", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="There were 30 found.",
                  char_start=0, char_end=20, token_count=4)
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p3_year_old_anchor_fast_accept():
    from groundguard.tiers.tier25_preprocessing import _extract_unit_anchor
    # "year-old" is an entity noun, so _extract_unit_anchor should return '_entity'.
    anchor = _extract_unit_anchor("year-old patient")
    assert anchor == '_entity'


def test_t25p3_slash_rate_anchor_fast_accept():
    from groundguard.tiers.tier25_preprocessing import _extract_unit_anchor
    # "/" is a measurable unit rate anchor.
    # We expect _extract_unit_anchor to return "/share" to preserve the denominator for comparison.
    assert _extract_unit_anchor("/share dividend") == "/share"


def test_t25p3_rhetorical_noun_ways_discarded_in_run():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    # Claim has "5 ways", which should be discarded.
    # Source has "10 ways", which is NOT discarded.
    # Because claim number is discarded, no conflict should be found.
    src = Source(source_id="s1", content="There are 10 ways.")
    ctx = VerificationContext(claim="There are 5 ways.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="There are 10 ways.",
                  char_start=0, char_end=18, token_count=4)
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_t25p3_unit_label_mismatch_escalates():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    # For Phase 3, this should assert has_conflict is False, and check escalate_reason.
    src = Source(source_id="s1", content="The weight is 20 lbs.")
    ctx = VerificationContext(claim="The weight is 20 kg.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="The weight is 20 lbs.",
                  char_start=0, char_end=21, token_count=5)
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert getattr(result, "escalate_reason", None) == "unit_label_mismatch"


def test_t25p3_unit_unitless_mismatch_escalation():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    # For Phase 3, it should assert escalate_reason and assert has_conflict is False.
    src = Source(source_id="s1", content="The weight is 10.")
    ctx = VerificationContext(claim="The weight is 20 kg.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="The weight is 10.",
                  char_start=0, char_end=17, token_count=4)
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert getattr(result, "escalate_reason", None) == "unit_unitless_mismatch"


def test_t25p3_numerical_value_boundary_values():
    from groundguard.tiers.tier25_preprocessing import NumericalValue
    # Cover zero, negative zero, inf, empty unit, None unit
    nv1 = NumericalValue(0.0, None)
    nv2 = NumericalValue(-0.0, "")
    nv3 = NumericalValue(float('inf'), "kg")
    assert nv1.value == 0.0
    assert nv1.unit is None
    assert nv2.value == -0.0
    assert nv2.unit == ""
    assert nv3.value == float('inf')
    assert nv3.unit == "kg"


def test_t25p3_extract_unit_anchor_boundary_empty():
    from groundguard.tiers.tier25_preprocessing import _extract_unit_anchor
    # Empty string should return None.
    assert _extract_unit_anchor("") is None


def test_t25p3_extract_unit_anchor_boundary_whitespace():
    from groundguard.tiers.tier25_preprocessing import _extract_unit_anchor
    # Whitespace only should return None.
    assert _extract_unit_anchor("   \t\n   ") is None


def test_t25p3_extract_unit_anchor_adversarial_embedded_word():
    from groundguard.tiers.tier25_preprocessing import _extract_unit_anchor
    # A unit anchor embedded inside another word (e.g. "kg" in "background") should not match.
    assert _extract_unit_anchor("background noise") is None


def test_t25p3_extract_unit_anchor_punctuation_stripping():
    from groundguard.tiers.tier25_preprocessing import _extract_unit_anchor
    # Check that punctuation is stripped correctly before checking against anchors.
    assert _extract_unit_anchor("patients...") == "_entity"


def test_t25p3_extract_unit_anchor_two_token_boundary():
    from groundguard.tiers.tier25_preprocessing import _extract_unit_anchor
    assert _extract_unit_anchor("about kg") == "kg"
    assert _extract_unit_anchor("roughly patients") == "_entity"


def test_t25p3_has_rhetorical_head_boundary_empty():
    from groundguard.tiers.tier25_preprocessing import _has_rhetorical_head
    assert _has_rhetorical_head("") is False


def test_t25p3_has_rhetorical_head_boundary_whitespace():
    from groundguard.tiers.tier25_preprocessing import _has_rhetorical_head
    assert _has_rhetorical_head("   ") is False


def test_t25p3_has_rhetorical_head_adversarial_substring():
    from groundguard.tiers.tier25_preprocessing import _has_rhetorical_head
    # Substring matches should not be flagged as rhetorical head.
    assert _has_rhetorical_head("reasonable debates") is False


def test_t25p3_has_rhetorical_head_capitalization():
    from groundguard.tiers.tier25_preprocessing import _has_rhetorical_head
    # Case insensitivity check.
    assert _has_rhetorical_head("REASONS to support") is True


def test_t25p3_has_rhetorical_head_three_token_boundary():
    from groundguard.tiers.tier25_preprocessing import _has_rhetorical_head
    assert _has_rhetorical_head("few different reasons") is True


def test_t25p3_aged_pattern_and_verbal_fractions():
    from groundguard.tiers.tier25_preprocessing import _AGED_PATTERN, _VERBAL_FRACTIONS
    import re
    # Check _AGED_PATTERN functionality
    assert isinstance(_AGED_PATTERN, (re.Pattern, str))
    pattern = re.compile(_AGED_PATTERN) if isinstance(_AGED_PATTERN, str) else _AGED_PATTERN
    assert pattern.search("aged 42") is not None
    assert pattern.search("age 42") is not None
    assert pattern.search("ageing 42") is None

    # Check _VERBAL_FRACTIONS functionality
    assert isinstance(_VERBAL_FRACTIONS, dict)
    assert _VERBAL_FRACTIONS["half"] == 0.5
    assert _VERBAL_FRACTIONS["two-thirds"] == 0.667


def test_t25p3_year_old_patient_integration_fast_accept():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    src = Source(source_id="s1", content="He was 50 years old.")
    ctx = VerificationContext(claim="a 42-year-old patient", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="He was 50 years old.",
                  char_start=0, char_end=20, token_count=5)
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p3_slash_share_dividend_integration_fast_accept():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    src = Source(source_id="s1", content="The dividend was $10/share.")
    ctx = VerificationContext(claim="$5/share dividend", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="The dividend was $10/share.",
                  char_start=0, char_end=26, token_count=4)
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p3_measurable_and_entity_disjoint():
    from groundguard.tiers.tier25_preprocessing import _MEASURABLE_UNITS, _ENTITY_NOUNS
    assert _MEASURABLE_UNITS.isdisjoint(_ENTITY_NOUNS) is True


def test_t25p3_entity_noun_value_mismatch_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    src = Source(source_id="s1", content="There were 5 locations.")
    ctx = VerificationContext(claim="There were 3 locations.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="There were 5 locations.",
                  char_start=0, char_end=23, token_count=4)
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p3_measurable_unit_value_mismatch_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    src = Source(source_id="s1", content="The weight is 30 kg.")
    ctx = VerificationContext(claim="The weight is 20 kg.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="The weight is 30 kg.",
                  char_start=0, char_end=20, token_count=5)
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p3_extract_unit_anchor_window_boundary():
    from groundguard.tiers.tier25_preprocessing import _extract_unit_anchor
    # anchor "kg" is 3 tokens away from the start (exceeding window limit 2)
    assert _extract_unit_anchor("word1 word2 kg") is None


def test_t25p3_has_rhetorical_head_window_boundary():
    from groundguard.tiers.tier25_preprocessing import _has_rhetorical_head
    # rhetorical head "reasons" is 4 tokens away from start (exceeding window limit 3)
    assert _has_rhetorical_head("word1 word2 word3 reasons") is False


def test_t25p3_unitless_claim_vs_unit_source():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    src = Source(source_id="s1", content="The weight is 10 kg.")
    ctx = VerificationContext(claim="The weight is 20.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="The weight is 10 kg.",
                  char_start=0, char_end=20, token_count=5)
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert getattr(result, "escalate_reason", None) == "unit_unitless_mismatch"


def test_t25p3_verbal_fractions_integration():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    # Claim "two-thirds of the patients" vs source "66.7% of the patients"
    # Claim verbal fraction should escalate
    src = Source(source_id="s1", content="66.7% of the patients")
    ctx = VerificationContext(claim="two-thirds of the patients", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="66.7% of the patients",
                  char_start=0, char_end=21, token_count=4)
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert getattr(result, "escalate_reason", None) == "fraction"


def test_t25p3_aged_pattern_integration_fast_accept():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    # Claim has "aged 42", which should match _AGED_PATTERN and be fast-accepted.
    # Source has "50", which is unitless. They should conflict (value mismatch).
    src = Source(source_id="s1", content="The patient was 50.")
    ctx = VerificationContext(claim="The patient was aged 42", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="The patient was 50.",
                  char_start=0, char_end=19, token_count=4)
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p3_claim_fraction_escalation():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    # Claim has standard fraction "1/3", which should escalate
    src = Source(source_id="s1", content="0.333 of the patients")
    ctx = VerificationContext(claim="1/3 of the patients", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="0.333 of the patients",
                  char_start=0, char_end=21, token_count=4)
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert getattr(result, "escalate_reason", None) == "fraction"


def test_t25p3_source_fraction_normalization():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    # Claim "0.333 of the patients" vs source "1/3 of the patients" should not conflict (normalizes to 0.333)
    src = Source(source_id="s1", content="1/3 of the patients")
    ctx = VerificationContext(claim="0.333 of the patients", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="1/3 of the patients",
                  char_start=0, char_end=19, token_count=4)
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert getattr(result, "escalate_reason", None) is None


def test_t25p3_extract_unit_anchor_measurable_units():
    from groundguard.tiers.tier25_preprocessing import _extract_unit_anchor
    assert _extract_unit_anchor("kg remaining") == "kg"
    assert _extract_unit_anchor("g of sugar") == "g"
    assert _extract_unit_anchor("% increase") == "%"


def test_t25p3_extract_unit_anchor_multi_word_entity():
    from groundguard.tiers.tier25_preprocessing import _extract_unit_anchor
    assert _extract_unit_anchor("amino acids found") == "_entity"


def test_t25p3_differing_rate_denominators_conflict_or_escalate():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    # Claim $5/share vs source $5/hour should not verify because of differing denominators.
    # It must either conflict (has_conflict=True) or escalate (has_conflict=False, escalate_reason="unit_label_mismatch").
    src = Source(source_id="s1", content="The price is $5/hour.")
    ctx = VerificationContext(claim="The price is $5/share.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="The price is $5/hour.",
                  char_start=0, char_end=21, token_count=5)
    result = run(ctx, [chunk])
    # The result must either be a conflict or escalate
    assert (result.has_conflict is True) or (result.has_conflict is False and getattr(result, "escalate_reason", None) == "unit_label_mismatch")


def test_t25p3_rhetorical_noun_in_source_not_discarded():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    # Claim has "3 locations" (factual, kept).
    # Source has "5 reasons" (rhetorical, but source Gate 2 is not applied, so kept).
    # Since both are kept, and values mismatch (3 vs 5), it must result in a conflict.
    src = Source(source_id="s1", content="There were 5 reasons.")
    ctx = VerificationContext(claim="There were 3 locations.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="There were 5 reasons.",
                  char_start=0, char_end=21, token_count=4)
    result = run(ctx, [chunk])
    assert result.has_conflict is True


# T-P4 Tests

def test_t25p4_hedge_lower_set_contains_required_elements():
    from groundguard.tiers.tier25_preprocessing import _HEDGE_LOWER
    assert "at least" in _HEDGE_LOWER


def test_t25p4_hedge_upper_set_contains_required_elements():
    from groundguard.tiers.tier25_preprocessing import _HEDGE_UPPER
    assert "fewer than" in _HEDGE_UPPER


def test_t25p4_hedge_approx_set_contains_required_elements():
    from groundguard.tiers.tier25_preprocessing import _HEDGE_APPROX
    assert "approximately" in _HEDGE_APPROX


def test_t25p4_detect_hedge_almost_percentage():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("almost 32% of patients", start_offset=7) == "approx"


def test_t25p4_detect_hedge_at_least_employees():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("at least 100 employees hired", start_offset=9) == "lower"


def test_t25p4_detect_hedge_fewer_than_cases():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("fewer than 5 cases reported", start_offset=11) == "upper"


def test_t25p4_detect_hedge_approximately_currency():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("approximately $5M revenue earned", start_offset=14) == "approx"


def test_t25p4_detect_hedge_no_hedge_present():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("the company had 100 employees", start_offset=16) is None


def test_t25p4_detect_hedge_sentence_boundary_stops_scan():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("price is under. 50 patients treated", start_offset=16) is None


def test_t25p4_detect_hedge_empty_claim():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("", 0) is None


def test_t25p4_detect_hedge_offset_out_of_bounds():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("almost 32%", 100) is None


def test_t25p4_detect_hedge_offset_negative():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("almost 32%", -5) is None


def test_t25p4_detect_hedge_offset_zero():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("32% of patients", 0) is None


def test_t25p4_detect_hedge_no_index_error_near_start():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("at 10", 3) is None


def test_t25p4_detect_hedge_adversarial_whitespace():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("almost \n\t 32%", 10) == "approx"


def test_t25p4_detect_hedge_adversarial_mixed_case():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("ALMOST 32%", 7) == "approx"


def test_t25p4_detect_hedge_adversarial_non_word_boundary():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    assert detect_hedge("almostly 32%", 9) is None


def test_t25p4_run_approx_within_ten_percent_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("almost 32% of patients", "31.5% of patients")
    chunk = _make_chunk("s1", "31.5% of patients")
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_t25p4_run_lower_bound_greater_than_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("at least 100 employees", "150 employees")
    chunk = _make_chunk("s1", "150 employees")
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_t25p4_run_lower_bound_less_than_has_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("at least 100 employees", "80 employees")
    chunk = _make_chunk("s1", "80 employees")
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p4_run_upper_bound_less_than_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("fewer than 5 cases", "3 cases")
    chunk = _make_chunk("s1", "3 cases")
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_t25p4_run_approx_currency_within_ten_percent_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("approximately $5M revenue", "$4.6M")
    chunk = _make_chunk("s1", "$4.6M")
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_t25p4_run_approx_currency_outside_ten_percent_has_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("approximately $5M revenue", "$3M")
    chunk = _make_chunk("s1", "$3M")
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p4_run_approx_zero_claim_within_tolerance_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("about 0% growth", "0.08% growth")
    chunk = _make_chunk("s1", "0.08% growth")
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_t25p4_run_approx_zero_claim_outside_tolerance_has_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("about 0% growth", "0.15% growth")
    chunk = _make_chunk("s1", "0.15% growth")
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p4_run_upper_bound_exactly_equal_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("fewer than 5 cases", "5 cases")
    chunk = _make_chunk("s1", "5 cases")
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_t25p4_run_upper_bound_greater_than_has_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("fewer than 5 cases", "6 cases")
    chunk = _make_chunk("s1", "6 cases")
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p4_run_lower_bound_exactly_equal_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("at least 100 employees", "100 employees")
    chunk = _make_chunk("s1", "100 employees")
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_t25p4_run_multiple_numbers_one_hedged_exact_still_required_for_unhedged():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("100 employees and at least 20 managers", "80 employees and 25 managers")
    chunk = _make_chunk("s1", "80 employees and 25 managers")
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p4_run_multiple_numbers_both_hedged_evaluated_independently():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("at least 100 employees and fewer than 20 managers", "120 employees and 15 managers")
    chunk = _make_chunk("s1", "120 employees and 15 managers")
    result = run(ctx, [chunk])
    assert result.has_conflict is False




