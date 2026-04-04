import pytest
from agentic_verifier.exceptions import (
    HallucinatedEvidenceError,
    VerificationCostExceededError,
    VerificationFailedError,
    ParseError,
)


def test_hallucinated_evidence_error_is_exception():
    exc = HallucinatedEvidenceError("evidence not found in sources")
    assert isinstance(exc, Exception)
    assert "evidence" in str(exc)


def test_verification_cost_exceeded_error_is_exception():
    exc = VerificationCostExceededError("budget limit exceeded")
    assert isinstance(exc, Exception)
    assert "budget" in str(exc)


def test_verification_failed_error_is_exception():
    exc = VerificationFailedError("pipeline failed to complete")
    assert isinstance(exc, Exception)
    assert "pipeline" in str(exc)


def test_parse_error_is_exception():
    exc = ParseError("invalid JSON response")
    assert isinstance(exc, Exception)
    assert "JSON" in str(exc)
