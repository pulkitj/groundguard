import re


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
    assert _normalise_number("1,000,000") == "1000000"


def test_normalise_number_strips_currency_suffix():
    from groundguard.tiers.tier25_preprocessing import _normalise_number
    assert _normalise_number("$4.2M") == "4.2"


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
    assert _normalise_number("-5%") == "-5"


def test_normalise_number_negative_currency():
    from groundguard.tiers.tier25_preprocessing import _normalise_number
    assert _normalise_number("-$4.2M") == "-4.2"


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
