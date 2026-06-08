import pytest
from groundguard.exceptions import (
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

def test_verify_handles_invariant_error_correctly(mocker):
    from groundguard.core.verifier import verify
    from groundguard.models.result import Source
    from groundguard.exceptions import InvariantError
    
    mock_tier1 = mocker.patch("groundguard.tiers.tier1_authenticity.check_fuzzy")
    mock_tier1.side_effect = InvariantError("Simulated InvariantError")
    
    src = Source(source_id="s1", content="The fee is 30%.")
    result = verify("The fee is 300%.", [src], model="gpt-4o-mini")
    
    assert result.status == "SYSTEM_ERROR"
    assert result.is_valid is False
    assert "InvariantError" in result.rationale or "Simulated" in result.rationale
    assert result.total_cost_usd >= 0.0

@pytest.mark.asyncio
async def test_averify_handles_invariant_error_correctly(mocker):
    from groundguard.core.verifier import averify
    from groundguard.models.result import Source
    from groundguard.exceptions import InvariantError
    
    mock_tier1 = mocker.patch("groundguard.tiers.tier1_authenticity.check_fuzzy")
    mock_tier1.side_effect = InvariantError("Simulated InvariantError Async")
    
    src = Source(source_id="s1", content="The fee is 30%.")
    result = await averify("The fee is 300%.", [src], model="gpt-4o-mini")
    
    assert result.status == "SYSTEM_ERROR"
    assert result.is_valid is False
    assert "InvariantError" in result.rationale or "Simulated" in result.rationale
    assert result.total_cost_usd >= 0.0
