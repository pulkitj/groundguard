"""Tests for groundguard public API exports — TDD T-105 (RED state)."""


def test_verify_answer_importable():
    from groundguard import verify_answer
    assert callable(verify_answer)


def test_averify_answer_importable():
    from groundguard import averify_answer
    assert callable(averify_answer)


def test_verify_analysis_importable():
    from groundguard import verify_analysis
    assert callable(verify_analysis)


def test_averify_analysis_importable():
    from groundguard import averify_analysis
    assert callable(averify_analysis)


def test_averify_batch_importable():
    from groundguard import averify_batch
    assert callable(averify_batch)


def test_verify_clause_importable():
    from groundguard import verify_clause
    assert callable(verify_clause)


def test_averify_clause_importable():
    from groundguard import averify_clause
    assert callable(averify_clause)


def test_grounding_result_importable():
    from groundguard import GroundingResult
    assert GroundingResult is not None


def test_verification_profile_importable():
    from groundguard import VerificationProfile, STRICT_PROFILE, GENERAL_PROFILE, RESEARCH_PROFILE
    assert STRICT_PROFILE.name == "strict"


def test_circuit_breaker_importable():
    from groundguard import assert_faithful, assert_grounded, verify_or_retry, GroundingError
    assert callable(assert_faithful)
