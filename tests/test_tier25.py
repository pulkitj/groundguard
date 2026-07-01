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
    # We expect _extract_unit_anchor to return "/" (bare form, never compound like "/share").
    assert _extract_unit_anchor("/share dividend") == "/"


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


def test_t25p3_differing_rate_denominators_now_same_unit():
    from groundguard.tiers.tier25_preprocessing import run
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

    # After the fix, $5/share and $5/hour both extract unit "/" (bare form, never compound).
    # Both have value 5 and unit "/", so they now match (same number, same rate anchor unit).
    # Previously /share and /hour were different units, but now they normalize to "/" for comparison.
    src = Source(source_id="s1", content="The price is $5/hour.")
    ctx = VerificationContext(claim="The price is $5/share.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="The price is $5/hour.",
                  char_start=0, char_end=21, token_count=5)
    result = run(ctx, [chunk])
    # Both extract "/" as unit anchor and 5 as value, so should match (no conflict)
    assert result.has_conflict is False


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
    assert _HEDGE_LOWER == {
        "at least", "more than", "over", "above", "exceeding",
        "greater than", "no fewer than", "no less than",
    }


def test_t25p4_hedge_upper_set_contains_required_elements():
    from groundguard.tiers.tier25_preprocessing import _HEDGE_UPPER
    assert _HEDGE_UPPER == {
        "at most", "fewer than", "less than", "under", "below",
        "no more than", "no greater than",
    }


def test_t25p4_hedge_approx_set_contains_required_elements():
    from groundguard.tiers.tier25_preprocessing import _HEDGE_APPROX
    assert _HEDGE_APPROX == {
        "about", "around", "approximately", "roughly", "nearly",
        "almost", "close to", "some", "up to", "as many as",
    }


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


def test_t25p4_detect_hedge_lookback_limit_approx():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    # "100" starts at index 32. 5 tokens preceding are "almost", "the", "company", "had", "enrolled".
    # Since HEDGE_SCAN_TOKENS = 5, "almost" should be detected.
    assert detect_hedge("almost the company had enrolled 100 employees", start_offset=32) == "approx"


def test_t25p4_detect_hedge_lookback_exceeded_none():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    # "100" starts at index 35. 6 tokens preceding. "almost" is 6th token back, exceeding limit of 5.
    assert detect_hedge("almost in the company had enrolled 100 employees", start_offset=35) is None


def test_t25p4_detect_hedge_exclamation_stops_scan():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    # "50" starts at index 16. The exclamation mark is a sentence boundary, stopping the scan.
    assert detect_hedge("price is under! 50 patients", start_offset=16) is None


def test_t25p4_detect_hedge_question_stops_scan():
    from groundguard.tiers.tier25_preprocessing import detect_hedge
    # "50" starts at index 16. The question mark is a sentence boundary, stopping the scan.
    assert detect_hedge("price is under? 50 patients", start_offset=16) is None


def test_t25p4_run_approx_exact_ten_percent_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    # abs(90 - 100) / 100 = 0.10 <= 0.10. Exact boundary.
    ctx = _make_ctx("about 100", "90")
    chunk = _make_chunk("s1", "90")
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_t25p4_run_approx_exceed_ten_percent_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    # abs(89.9 - 100) / 100 = 0.101 > 0.10. Slightly exceeding.
    ctx = _make_ctx("about 100", "89.9")
    chunk = _make_chunk("s1", "89.9")
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p4_run_approx_zero_exact_boundary_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    # abs(0.10 - 0.0) = 0.10 <= 0.10. Exact absolute boundary.
    ctx = _make_ctx("about 0", "0.10")
    chunk = _make_chunk("s1", "0.10")
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_t25p4_run_approx_zero_exceed_boundary_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    # abs(0.101 - 0.0) = 0.101 > 0.10. Slightly exceeding absolute boundary.
    ctx = _make_ctx("about 0", "0.101")
    chunk = _make_chunk("s1", "0.101")
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p4_run_approx_negative_value_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    # abs(-95 - -100) / abs(-100) = 5 / 100 = 0.05 <= 0.10.
    ctx = _make_ctx("about -100", "-95")
    chunk = _make_chunk("s1", "-95")
    result = run(ctx, [chunk])
    assert result.has_conflict is False


def test_t25p4_run_approx_negative_value_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    # abs(-50 - -100) / abs(-100) = 50 / 100 = 0.50 > 0.10.
    ctx = _make_ctx("about -100", "-50")
    chunk = _make_chunk("s1", "-50")
    result = run(ctx, [chunk])
    assert result.has_conflict is True


def test_t25p4_run_repeated_identical_values_one_hedged_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    # Claim: "100" (first: exact check, fails against 80), "at least 100" (second: lower bound, passes against 120)
    # The first mismatch makes the whole run a conflict.
    ctx = _make_ctx("100 employees and at least 100 managers", "80 employees and 120 managers")
    chunk = _make_chunk("s1", "80 employees and 120 managers")
    result = run(ctx, [chunk])
    assert result.has_conflict is True






# T-P5 Tests


def test_extract_ranges_multiple_ordered_by_position():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("between 20 and 30% and 50–60 patients") == [
        (20.0, 30.0, "between 20 and 30%"),
        (50.0, 60.0, "50–60")
    ]


def test_extract_ranges_skips_malformed_normalization_value_error():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("between 20 and 30.0.0%") == []


def test_extract_ranges_suffix_distribution_percent():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("10–20%") == [(10.0, 20.0, "10–20%")]


def test_extract_ranges_currency_prefix_distribution_upper():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("10 to $20") == [(10.0, 20.0, "10 to $20")]

def test_extract_ranges_between_numeric_percent():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("between 20 and 30%") == [(20.0, 30.0, "between 20 and 30%")]

def test_extract_ranges_hyphenated_with_en_dash():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("50–60 patients enrolled") == [(50.0, 60.0, "50–60")]

def test_extract_ranges_hyphenated_ascii():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("ages 18-65") == [(18.0, 65.0, "18-65")]

def test_extract_ranges_currency_magnitudes():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("$10M–$20M revenue") == [(10000000.0, 20000000.0, "$10M–$20M")]

def test_extract_ranges_no_ranges_present():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("no ranges here") == []

def test_extract_ranges_empty_string():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("") == []

def test_extract_ranges_none_raises_type_error():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    with pytest.raises(TypeError):
        extract_ranges(None)

def test_extract_ranges_open_lower_bound_at_least():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("at least 20") == []

def test_extract_ranges_open_upper_bound_at_most():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("at most 30") == []

def test_extract_ranges_suffix_distribution_magnitude():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("10–20M") == [(10000000.0, 20000000.0, "10–20M")]

def test_range_claim_and_source_value_in_range_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30%", "25%")
    chunk = _make_chunk("s1", "25%")
    assert run(ctx, [chunk]).has_conflict is False

def test_range_claim_and_source_value_below_range_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30%", "15%")
    chunk = _make_chunk("s1", "15%")
    assert run(ctx, [chunk]).has_conflict is True

def test_range_claim_and_source_value_above_range_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30%", "35%")
    chunk = _make_chunk("s1", "35%")
    assert run(ctx, [chunk]).has_conflict is True

def test_range_claim_and_source_value_at_lower_boundary_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30%", "20%")
    chunk = _make_chunk("s1", "20%")
    assert run(ctx, [chunk]).has_conflict is False

def test_range_claim_and_source_value_at_upper_boundary_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30%", "30%")
    chunk = _make_chunk("s1", "30%")
    assert run(ctx, [chunk]).has_conflict is False

def test_range_claim_with_patients_and_source_value_in_range_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("50–60 patients enrolled", "55 patients")
    chunk = _make_chunk("s1", "55 patients")
    assert run(ctx, [chunk]).has_conflict is False

def test_range_containment_source_inside_claim_no_conflict():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30%", "between 22 and 28%")
    chunk = _make_chunk("s1", "between 22 and 28%")
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert result.escalate_reason is None

def test_range_overlap_with_strict_containment_true_conflicts():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30%", "between 25 and 35%")
    chunk = _make_chunk("s1", "between 25 and 35%")
    result = run(ctx, [chunk])
    assert result.has_conflict is True
    assert result.escalate_reason is None


def test_range_overlap_with_strict_containment_false_escalates(monkeypatch):
    import groundguard.tiers.tier25_preprocessing
    monkeypatch.setattr(groundguard.tiers.tier25_preprocessing, "RANGE_CONTAINMENT_STRICT", False)
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30%", "between 25 and 35%")
    chunk = _make_chunk("s1", "between 25 and 35%")
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert result.escalate_reason is not None

def test_range_disjoint_source_outside_claim_conflicts():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30%", "between 35 and 45%")
    chunk = _make_chunk("s1", "between 35 and 45%")
    result = run(ctx, [chunk])
    assert result.has_conflict is True
    assert result.escalate_reason is None

def test_range_claim_does_not_produce_separate_single_numbers():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30% revenue", "25% revenue")
    chunk = _make_chunk("s1", "25% revenue")
    result = run(ctx, [chunk])
    assert len(result.numerical_checks) == 1
    assert result.numerical_checks[0].claim_number in ("between 20 and 30%", "20 and 30%")


def test_extract_ranges_suffix_distribution_word_form_magnitude():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("5 to 10 million") == [(5000000.0, 10000000.0, "5 to 10 million")]

def test_range_containment_source_superset_of_claim_escalates():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 22 and 28%", "between 20 and 30%")
    chunk = _make_chunk("s1", "between 20 and 30%")
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert result.escalate_reason is not None

def test_range_claim_vs_source_no_numbers_escalates():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30%", "no numbers here")
    chunk = _make_chunk("s1", "no numbers here")
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert result.escalate_reason is not None

def test_range_claim_unit_mismatch_escalates():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30 kg", "25 lbs")
    chunk = _make_chunk("s1", "25 lbs")
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert result.escalate_reason == "unit_label_mismatch"

def test_range_claim_unit_unitless_mismatch_escalates():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30%", "25")
    chunk = _make_chunk("s1", "25")
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert result.escalate_reason == "unit_unitless_mismatch"

def test_extract_ranges_em_dash():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("50—60") == [(50.0, 60.0, "50—60")]

def test_extract_ranges_currency_prefix_distribution_lo():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    # Should distribute prefix from lower to upper bound if upper lacks it but has magnitude
    assert extract_ranges("$10–20M") == [(10000000.0, 20000000.0, "$10–20M")]

def test_extract_ranges_signed_bounds():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("-10 to -5") == [(-10.0, -5.0, "-10 to -5")]
    assert extract_ranges("-5% to +5%") == [(-5.0, 5.0, "-5% to +5%")]

def test_extract_ranges_scientific_notation():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    assert extract_ranges("1e6 to 2e6") == [(1000000.0, 2000000.0, "1e6 to 2e6")]

def test_year_conflict_aggregate_across_chunks():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 2023 and 2024", "in 2023 and in 2024")
    chunk1 = _make_chunk("s1", "in 2023")
    chunk2 = _make_chunk("s2", "in 2024")
    result = run(ctx, [chunk1, chunk2])
    assert result.has_conflict is False


def test_range_to_range_unit_mismatch_escalates():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30 kg", "between 22 and 28 lbs")
    chunk = _make_chunk("s1", "between 22 and 28 lbs")
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert result.escalate_reason == "unit_label_mismatch"


def test_range_to_range_unit_unitless_mismatch_escalates():
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30%", "between 22 and 28")
    chunk = _make_chunk("s1", "between 22 and 28")
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert result.escalate_reason == "unit_unitless_mismatch"


def test_range_currency_prefix_distribution_unit_check():
    from groundguard.tiers.tier25_preprocessing import run
    # Claim: "10 to $20" -> both bounds should have currency unit USD
    # Source: "15" (unitless) -> should trigger unit_unitless_mismatch
    ctx = _make_ctx("10 to $20", "15")
    chunk = _make_chunk("s1", "15")
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert result.escalate_reason == "unit_unitless_mismatch"


def test_range_disjoint_with_strict_containment_false_conflicts(monkeypatch):
    import groundguard.tiers.tier25_preprocessing
    monkeypatch.setattr(groundguard.tiers.tier25_preprocessing, "RANGE_CONTAINMENT_STRICT", False)
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("between 20 and 30%", "between 35 and 45%")
    chunk = _make_chunk("s1", "between 35 and 45%")
    result = run(ctx, [chunk])
    assert result.has_conflict is True
    assert result.escalate_reason is None

def test_run_multiple_range_claims_evaluated():
    from groundguard.tiers.tier25_preprocessing import run
    # Two ranges in claim: "between 20 and 30%" and "50–60 patients"
    # Source has "35%" (violates first range) and "55 patients" (satisfies second range)
    # The violation in the first range should trigger conflict.
    ctx = _make_ctx("between 20 and 30% and 50–60 patients", "35% and 55 patients")
    chunk = _make_chunk("s1", "35% and 55 patients")
    result = run(ctx, [chunk])
    assert result.has_conflict is True

def test_range_to_range_comparison_uses_normalized_bounds():
    from groundguard.tiers.tier25_preprocessing import run
    # Both bounds in source "22–28%" should be normalized before comparing.
    # Claim: "between 20 and 30%" vs source: "22–28%"
    ctx = _make_ctx("between 20 and 30%", "22–28%")
    chunk = _make_chunk("s1", "22–28%")
    result = run(ctx, [chunk])
    assert result.has_conflict is False
    assert result.escalate_reason is None

# T-P6 Tests


def test_tier25_result_defaults_to_none_escalate_reason():
    """Verify that Tier25Result has field escalate_reason that defaults to None."""
    from groundguard.tiers.tier25_preprocessing import Tier25Result
    result = Tier25Result(has_conflict=False)
    assert result.escalate_reason is None


def test_tier25_result_custom_escalate_reason_is_preserved():
    """Verify that Tier25Result preserves a custom escalate_reason string."""
    from groundguard.tiers.tier25_preprocessing import Tier25Result
    result = Tier25Result(has_conflict=False, escalate_reason="ratio_times")
    assert result.escalate_reason == "ratio_times"


def test_ratio_times_escalates_on_x_suffix():
    """Verify that a claim with an 'x' ratio suffix (e.g. '3x') escalates with 'ratio_times'."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("revenue grew 3x", "revenue grew 2x")
    chunk = _make_chunk("s1", "revenue grew 2x")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "ratio_times"


def test_ratio_times_escalates_on_multiplication_sign():
    """Verify that a claim with a multiplication sign (e.g. '5×') escalates with 'ratio_times'."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("grew 5× faster", "grew 2x faster")
    chunk = _make_chunk("s1", "grew 2x faster")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "ratio_times"


def test_ratio_notation_escalates_on_colon_format():
    """Verify that a claim with ratio notation (e.g. '2:1') escalates with 'ratio_notation'."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("a 2:1 ratio", "a 3:1 ratio")
    chunk = _make_chunk("s1", "a 3:1 ratio")
    result = run(ctx, [chunk])
    assert result.escalate_reason in ("ratio_notation", "ratio")


def test_decade_reference_four_digits_escalates():
    """Verify that a claim referencing a decade using four digits (e.g. '1980s') escalates."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("in the 1980s", "in the 1970s")
    chunk = _make_chunk("s1", "in the 1970s")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "decade_reference"


def test_decade_reference_two_digits_escalates():
    """Verify that a claim referencing a decade using two digits (e.g. '90s') escalates."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("the 90s", "the 80s")
    chunk = _make_chunk("s1", "the 80s")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "decade_reference"


def test_score_expression_out_of_escalates():
    """Verify that a score expression using 'out of' escalates with 'score_expression'."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("an 8 out of 10 rating", "a 9 out of 10 rating")
    chunk = _make_chunk("s1", "a 9 out of 10 rating")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "score_expression"


def test_score_expression_score_of_escalates():
    """Verify that a score expression using 'score of' escalates with 'score_expression'."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("a score of 74", "a score of 80")
    chunk = _make_chunk("s1", "a score of 80")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "score_expression"


def test_percentage_points_words_escalates():
    """Verify that a percentage points verbal reference escalates with 'percentage_points'."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("rose 3 percentage points", "rose 3%")
    chunk = _make_chunk("s1", "rose 3%")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "percentage_points"


def test_percentage_points_abbreviation_escalates():
    """Verify that a percentage points abbreviation 'pp' reference escalates with 'percentage_points'."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("rose 3pp", "rose 3%")
    chunk = _make_chunk("s1", "rose 3%")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "percentage_points"


def test_abbreviated_year_range_hyphen_escalates():
    """Verify that an abbreviated year range with a hyphen (e.g. '2025-26') escalates."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("revenue in 2025-26", "revenue was $5M")
    chunk = _make_chunk("s1", "revenue was $5M")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "abbreviated_year_range"


def test_abbreviated_year_range_fy_prefix_escalates():
    """Verify that an abbreviated year range with a 'FY' prefix (e.g. 'FY2024-25') escalates."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("FY2024-25 results", "results for FY2023-24")
    chunk = _make_chunk("s1", "results for FY2023-24")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "abbreviated_year_range"


def test_bare_fraction_without_entity_noun_escalates():
    """Verify that a bare fraction without an adjacent entity noun escalates."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("1/3 completed", "0.33 completed")
    chunk = _make_chunk("s1", "0.33 completed")
    result = run(ctx, [chunk])
    assert result.escalate_reason in ("fraction", "bare_fraction")


def test_fraction_with_entity_noun_does_not_escalate():
    """Verify that a fraction with an adjacent entity noun (e.g. '1/3 of patients') does not escalate."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("1/3 of patients recovered", "1/3 of patients recovered")
    chunk = _make_chunk("s1", "1/3 of patients recovered")
    result = run(ctx, [chunk])
    assert result.escalate_reason is None


def test_combined_multiple_escalation_conditions_escalates():
    """Test combined escalation conditions: multiple patterns in one claim must escalate."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("revenue grew 3x in the 1980s", "revenue grew in 1985")
    chunk = _make_chunk("s1", "revenue grew in 1985")
    result = run(ctx, [chunk])
    assert result.escalate_reason in ("ratio_times", "decade_reference")


def test_non_escalating_plain_number_returns_none():
    """Verify that a plain number claim without escalation triggers does not escalate."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("revenue grew 3 percent", "revenue grew 3 percent")
    chunk = _make_chunk("s1", "revenue grew 3 percent")
    result = run(ctx, [chunk])
    assert result.escalate_reason is None


def test_tier25_preprocessing_run_with_empty_claim_returns_no_conflict_and_none_escalate():
    """Verify that running pre-processing with an empty claim returns no conflict and no escalation."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("", "revenue grew 3 percent")
    chunk = _make_chunk("s1", "revenue grew 3 percent")
    result = run(ctx, [chunk])
    assert result.escalate_reason is None


def test_t2_routing_with_escalate_reason_forces_llm_route():
    """Verify that a non-None escalate_reason in Tier25Result forces LLM routing in route_claim."""
    from groundguard.tiers.tier2_semantic import route_claim
    from groundguard.tiers.tier25_preprocessing import Tier25Result
    from groundguard.models.internal import RoutingDecision, VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk

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

    t25_res = Tier25Result(has_conflict=False, escalate_reason="ratio_times")
    res_escalated = route_claim(ctx, chunks, tier25_result=t25_res)
    assert res_escalated.decision == RoutingDecision.ESCALATE_TO_LLM


def test_extract_unit_anchor_slash_always_returns_bare_slash():
    from groundguard.tiers.tier25_preprocessing import _extract_unit_anchor
    assert _extract_unit_anchor("/ share") == "/"
    assert _extract_unit_anchor("/share") == "/"
    assert _extract_unit_anchor("/ day") == "/"
    assert _extract_unit_anchor("/") == "/"


def test_extract_unit_anchor_per_always_returns_bare_per():
    from groundguard.tiers.tier25_preprocessing import _extract_unit_anchor
    assert _extract_unit_anchor("per share") == "per"
    assert _extract_unit_anchor("per day") == "per"
    assert _extract_unit_anchor("per") == "per"


def test_verbal_fraction_with_entity_noun_does_not_escalate():
    """Claim 'two-thirds of patients recovered' should NOT escalate — entity noun present."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("two-thirds of patients recovered")
    chunk = _make_chunk("s1", "0.667 of patients recovered")
    result = run(ctx, [chunk])
    assert result.escalate_reason != "fraction", (
        "Verbal fraction followed by entity noun must fast-accept, not escalate"
    )


def test_verbal_fraction_without_entity_noun_escalates():
    """Claim 'one-third completed' should escalate — no entity noun."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("one-third completed")
    chunk = _make_chunk("s1", "some tasks completed")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "fraction"


def test_extract_verbal_fractions_source_basic():
    """Source text verbal fractions are normalised to floats."""
    from groundguard.tiers.tier25_preprocessing import _extract_verbal_fractions_source
    results = _extract_verbal_fractions_source("two-thirds of the budget was spent")
    assert any(abs(v - 0.667) < 0.001 for v, _ in results)


def test_verbal_compound_split_escalates():
    """'one hundred and fifty thousand jobs' must escalate, not compare as two values."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("one hundred and fifty thousand jobs were created")
    chunk = _make_chunk("s1", "150,000 jobs were created")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "verbal_compound_split"
    assert not result.has_conflict


def test_verbal_compound_split_two_million_three_hundred_thousand():
    """'two million three hundred thousand' — another multi-scale split."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("two million three hundred thousand customers")
    chunk = _make_chunk("s1", "2,300,000 customers")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "verbal_compound_split"


def test_two_separate_verbal_numbers_not_adjacent_do_not_escalate():
    """Two verbal numbers with substantial text between are independent, not compound."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("one million employees in Asia and two million in Europe")
    chunk = _make_chunk("s1", "1,000,000 employees in Asia and 2,000,000 in Europe")
    result = run(ctx, [chunk])
    assert result.escalate_reason != "verbal_compound_split"


def test_source_verbal_fraction_contributes_to_comparison():
    """Claim '0.667 of patients' vs source 'two-thirds of patients' — no conflict."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("0.667 of patients recovered")
    chunk = _make_chunk("s1", "two-thirds of patients recovered")
    result = run(ctx, [chunk])
    assert not result.has_conflict


def test_eu_integer_ambiguous_helper_true():
    from groundguard.tiers.tier25_preprocessing import _is_eu_integer_ambiguous
    assert _is_eu_integer_ambiguous("1.234") is True
    assert _is_eu_integer_ambiguous("12.345") is True
    assert _is_eu_integer_ambiguous("123.456") is True


def test_eu_integer_ambiguous_helper_false_for_clear_decimals():
    from groundguard.tiers.tier25_preprocessing import _is_eu_integer_ambiguous
    assert _is_eu_integer_ambiguous("0.234") is False   # leading zero → unambiguously US
    assert _is_eu_integer_ambiguous("1.2345") is False  # 4 decimal digits → not EU integer
    assert _is_eu_integer_ambiguous("1.23") is False    # 2 decimal digits → not EU integer
    assert _is_eu_integer_ambiguous("1,234") is False   # comma → already-normalised US


def test_eu_integer_ambiguous_claim_escalates():
    """Claim containing bare ambiguous '1.234' must escalate."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("the ratio was 1.234")
    chunk = _make_chunk("s1", "the ratio was 1.234")
    result = run(ctx, [chunk])
    assert result.escalate_reason == "eu_integer_ambiguous"


def test_non_ambiguous_decimal_does_not_escalate():
    """0.234 is unambiguously a US decimal; must not trigger eu_integer_ambiguous."""
    from groundguard.tiers.tier25_preprocessing import run
    ctx = _make_ctx("the share was 0.234 of revenue")
    chunk = _make_chunk("s1", "the share was 0.234 of revenue")
    result = run(ctx, [chunk])
    assert result.escalate_reason != "eu_integer_ambiguous"


def test_extract_ranges_ignores_abbreviated_year_range():
    """extract_ranges('2025-26') must not return (2025.0, 26.0)."""
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    result = extract_ranges("revenue in 2025-26")
    assert result == [], f"Expected [], got {result}"


def test_extract_ranges_ignores_fy_abbreviated_year_range():
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    result = extract_ranges("FY2024-25 results")
    assert result == [], f"Expected [], got {result}"


def test_extract_ranges_still_works_for_normal_ranges_after_masking():
    """Normal numeric ranges are unaffected by the masking."""
    from groundguard.tiers.tier25_preprocessing import extract_ranges
    result = extract_ranges("between 10 and 20 percent")
    assert len(result) == 1
    lo, hi, raw = result[0]
    assert lo == 10.0
    assert hi == 20.0
