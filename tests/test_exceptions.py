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
    from groundguard.models.internal import RoutingDecision, Tier2Result
    
    # Force Tier 2 to escalate
    dummy_t2 = Tier2Result(decision=RoutingDecision.ESCALATE_TO_LLM, highest_score=0.1, top_k_chunks=[])
    mocker.patch("groundguard.core.verifier.tier2_semantic.route_claim", return_value=dummy_t2)
    
    # Mock evaluate to return a dummy model so we don't hit the network
    def dummy_eval_sync(ctx, chunks):
        ctx.cost_tracker.total_cost_usd = 1.23
        return mocker.MagicMock()
    mocker.patch("groundguard.tiers.tier3_evaluation.evaluate", side_effect=dummy_eval_sync)
    mock_builder = mocker.patch("groundguard.core.verifier.ResultBuilder.build_llm_result")
    mock_builder.side_effect = InvariantError("Simulated InvariantError", cost_usd=1.23)
    
    src = Source(source_id="s1", content="The color is red.")
    with pytest.raises(InvariantError) as exc_info:
        verify("The color is blue.", [src], model="gpt-4o-mini")
    
    assert exc_info.value.cost_usd == 1.23

@pytest.mark.asyncio
async def test_averify_handles_invariant_error_correctly(mocker):
    from groundguard.core.verifier import averify
    from groundguard.models.result import Source
    from groundguard.exceptions import InvariantError
    from groundguard.models.internal import RoutingDecision, Tier2Result
    
    dummy_t2 = Tier2Result(decision=RoutingDecision.ESCALATE_TO_LLM, highest_score=0.1, top_k_chunks=[])
    mocker.patch("groundguard.core.verifier.tier2_semantic.route_claim", return_value=dummy_t2)
    
    # Mock aevaluate
    async def dummy_eval_async(ctx, chunks):
        ctx.cost_tracker.total_cost_usd = 4.56
        return mocker.MagicMock()
    mocker.patch("groundguard.tiers.tier3_evaluation.evaluate_async", side_effect=dummy_eval_async)
    
    mock_builder = mocker.patch("groundguard.core.verifier.ResultBuilder.build_llm_result")
    mock_builder.side_effect = InvariantError("Simulated InvariantError", cost_usd=4.56)
    
    src = Source(source_id="s1", content="The color is red.")
    
    with pytest.raises(InvariantError) as exc_info:
        await averify("The color is blue.", [src], model="gpt-4o-mini")
        
    assert exc_info.value.cost_usd == 4.56
