"""Tests for groundguard/circuit_breaker.py — TDD T-103 (RED state)."""
import pytest
from groundguard.models.result import GroundingResult


def _mock_grounding_result(status: str, score: float = 0.9):
    return GroundingResult(
        is_grounded=(status == "GROUNDED"),
        score=score,
        status=status,
        evaluation_method="sentence_entailment",
    )


def test_assert_faithful_passes_on_grounded(mocker):
    from groundguard.circuit_breaker import assert_faithful
    from groundguard.models.result import Source
    mocker.patch("groundguard.circuit_breaker.verify_answer",
                 return_value=_mock_grounding_result("GROUNDED", score=0.95))
    src = Source(source_id="s1", content="x")
    assert_faithful("x", [src])  # must not raise


def test_assert_faithful_raises_on_not_grounded(mocker):
    from groundguard.circuit_breaker import assert_faithful, GroundingError
    from groundguard.models.result import Source
    mocker.patch("groundguard.circuit_breaker.verify_answer",
                 return_value=_mock_grounding_result("NOT_GROUNDED", score=0.3))
    src = Source(source_id="s1", content="x")
    with pytest.raises(GroundingError):
        assert_faithful("x", [src])


def test_assert_grounded_passes(mocker):
    from groundguard.circuit_breaker import assert_grounded
    from groundguard.models.result import Source
    mocker.patch("groundguard.circuit_breaker.verify_analysis",
                 return_value=_mock_grounding_result("GROUNDED", score=0.9))
    src = Source(source_id="s1", content="x")
    assert_grounded("x", [src])


def test_assert_grounded_raises_on_not_grounded(mocker):
    from groundguard.circuit_breaker import assert_grounded, GroundingError
    from groundguard.models.result import Source
    mocker.patch("groundguard.circuit_breaker.verify_analysis",
                 return_value=_mock_grounding_result("NOT_GROUNDED", score=0.1))
    src = Source(source_id="s1", content="x")
    with pytest.raises(GroundingError):
        assert_grounded("x", [src])


def test_verify_or_retry_returns_on_first_pass(mocker):
    from groundguard.circuit_breaker import verify_or_retry
    from groundguard.models.result import Source
    call_count = [0]
    def generator():
        call_count[0] += 1
        return "output text"
    mocker.patch("groundguard.circuit_breaker.verify_answer",
                 return_value=_mock_grounding_result("GROUNDED", score=0.95))
    src = Source(source_id="s1", content="x")
    result = verify_or_retry(generator, [src], max_retries=3)
    assert result == "output text"
    assert call_count[0] == 1


def test_verify_or_retry_retries_on_not_grounded(mocker):
    from groundguard.circuit_breaker import verify_or_retry, GroundingError
    from groundguard.models.result import Source
    call_count = [0]
    def generator():
        call_count[0] += 1
        return "output text"
    mocker.patch("groundguard.circuit_breaker.verify_answer",
                 return_value=_mock_grounding_result("NOT_GROUNDED", score=0.2))
    src = Source(source_id="s1", content="x")
    with pytest.raises(GroundingError):
        verify_or_retry(generator, [src], max_retries=2)
    assert call_count[0] == 2


def test_grounding_error_is_exception():
    from groundguard.circuit_breaker import GroundingError
    e = GroundingError("test")
    assert isinstance(e, Exception)
