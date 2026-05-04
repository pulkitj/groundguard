"""Tests for verify_analysis / averify_analysis — TDD T-94 (RED state)."""
import pytest


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _mock_atomic(status: str):
    from groundguard.models.result import VerificationResult
    return VerificationResult(
        is_valid=(status == "VERIFIED"),
        overall_verdict="mock",
        verification_method="tier3_llm",
        atomic_claims=[],
        factual_consistency_score=0.9,
        sources_used=[],
        rationale="mock",
        offending_claim=None,
        status=status,
        total_cost_usd=0.0,
    )


def _mock_batch_results(pairs):
    from groundguard.models.result import VerificationResult
    results = []
    for status, score in pairs:
        results.append(VerificationResult(
            is_valid=(status == "VERIFIED"),
            overall_verdict="mock",
            verification_method="tier3_llm",
            atomic_claims=[],
            factual_consistency_score=score,
            sources_used=[],
            rationale="mock",
            offending_claim=None,
            status=status,
            total_cost_usd=0.0,
        ))
    return results


def _mock_single_result(status: str):
    return _mock_atomic(status)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_verify_analysis_returns_grounding_result(mocker):
    from groundguard.core.verifier import verify_analysis
    from groundguard.models.result import Source, GroundingResult
    mocker.patch("groundguard.core.claim_extractor.extract_claims",
                 return_value=["Claim A.", "Claim B."])
    mocker.patch("groundguard.core.verifier.verify_batch",
                 return_value=_mock_batch_results([("VERIFIED", 0.9), ("VERIFIED", 0.85)]))
    src = Source(source_id="s1", content="Claim A. Claim B.")
    result = verify_analysis("Claim A. Claim B.", [src], model="gpt-4o-mini")
    assert isinstance(result, GroundingResult)
    assert result.evaluation_method == "claim_extraction"


def test_verify_analysis_score_excludes_unverifiable():
    from groundguard.core.verifier import _aggregate_analysis_results
    results = [
        _mock_atomic("VERIFIED"),
        _mock_atomic("VERIFIED"),
        _mock_atomic("UNVERIFIABLE"),
        _mock_atomic("CONTRADICTED"),
    ]
    gr = _aggregate_analysis_results(results)
    assert abs(gr.score - 2/3) < 0.01


def test_verify_analysis_all_unverifiable():
    from groundguard.core.verifier import _aggregate_analysis_results
    results = [_mock_atomic("UNVERIFIABLE"), _mock_atomic("UNVERIFIABLE")]
    gr = _aggregate_analysis_results(results)
    assert gr.status == "NOT_GROUNDED"
    assert gr.score == 0.0


def test_verify_analysis_fully_grounded():
    from groundguard.core.verifier import _aggregate_analysis_results
    results = [_mock_atomic("VERIFIED"), _mock_atomic("VERIFIED")]
    gr = _aggregate_analysis_results(results)
    assert gr.status == "GROUNDED"
    assert gr.is_grounded is True


def test_verify_analysis_fail_contained(mocker):
    from groundguard.core.verifier import verify_analysis
    from groundguard.models.result import Source
    mocker.patch("groundguard.core.claim_extractor.extract_claims",
                 return_value=["Claim A.", "Claim B."])
    mocker.patch("groundguard.core.verifier.verify_batch",
                 return_value=_mock_batch_results([("VERIFIED", 0.9), ("UNVERIFIABLE", 0.0)]))
    src = Source(source_id="s1", content="x")
    result = verify_analysis("Claim A. Claim B.", [src], model="gpt-4o-mini")
    assert result.status in ("GROUNDED", "PARTIALLY_GROUNDED", "NOT_GROUNDED")
    assert 0.0 <= result.score <= 1.0
    assert result.evaluation_method == "claim_extraction"
    # 1 VERIFIED + 1 UNVERIFIABLE → denom=1 → score=1.0 → GROUNDED
    assert result.score == 1.0
    assert result.status == "GROUNDED"


def test_verify_analysis_parse_errors_count_against_score():
    from groundguard.core.verifier import _aggregate_analysis_results
    results = [
        _mock_atomic("VERIFIED"),
        _mock_atomic("PARSE_ERROR"),
        _mock_atomic("PARSE_ERROR"),
    ]
    gr = _aggregate_analysis_results(results)
    # 1 VERIFIED out of 3 total (2 PARSE_ERROR in denom) → score = 1/3
    assert abs(gr.score - 1/3) < 0.01
    assert gr.status in ("PARTIALLY_GROUNDED", "NOT_GROUNDED")


def test_averify_analysis_returns_coroutine(mocker):
    import asyncio
    from unittest.mock import AsyncMock
    from groundguard.core.verifier import averify_analysis
    from groundguard.models.result import Source
    mocker.patch(
        "groundguard.core.claim_extractor.extract_claims_async",
        new_callable=AsyncMock,
        return_value=["x"],
    )
    mocker.patch(
        "groundguard.core.verifier.averify_batch",
        new_callable=AsyncMock,
        return_value=_mock_batch_results([("VERIFIED", 0.9)]),
    )
    src = Source(source_id="s1", content="x")
    result = asyncio.get_event_loop().run_until_complete(
        averify_analysis("x", [src], model="gpt-4o-mini")
    )
    assert result is not None
    assert result.score == 1.0
