"""Tests for verify_clause() / averify_clause() — TDD T-101 (RED state)."""
import pytest
from groundguard.models.result import AtomicClaimResult, Source, VerificationResult


def _mock_atomic_result(status: str):
    return VerificationResult(
        is_valid=(status == "VERIFIED"),
        overall_verdict=status,
        verification_method="tier3_llm",
        atomic_claims=[],
        factual_consistency_score=0.9 if status == "VERIFIED" else 0.0,
        sources_used=["s1"],
        rationale="mock",
        offending_claim=None,
        status=status,
        total_cost_usd=0.0,
    )


@pytest.mark.legal
def test_verify_clause_with_term_registry(mocker):
    from groundguard.core.verifier import verify_clause
    from groundguard.loaders.legal import TermRegistry
    def_src = Source(source_id="def1", content='"Permitted Costs" means approved costs.',
                     source_type="legal_definition")
    registry = TermRegistry.from_sources([def_src])
    src = Source(source_id="s1", content="Permitted Costs are approved by the board.")
    mocker.patch("groundguard.core.verifier.verify",
                 return_value=_mock_atomic_result("VERIFIED"))
    result = verify_clause("Permitted Costs shall be approved.", [src],
                           term_registry=registry, model="gpt-4o-mini")
    assert result is not None


@pytest.mark.legal
def test_verify_clause_context_annotation_contains_modifiers(mocker):
    from groundguard.core.verifier import verify_clause
    calls = []
    def capture_verify(claim, sources, **kwargs):
        calls.append(kwargs.get("context", ""))
        return _mock_atomic_result("VERIFIED")
    mocker.patch("groundguard.core.verifier.verify", side_effect=capture_verify)
    src = Source(source_id="s1", content="x")
    verify_clause("The fee as defined in Schedule 1 shall not exceed 30%.", [src],
                  model="gpt-4o-mini")
    assert any("Clause modifiers:" in c for c in calls)


@pytest.mark.legal
def test_verify_clause_context_annotation_contains_obligation_type(mocker):
    from groundguard.core.verifier import verify_clause
    calls = []
    def capture_verify(claim, sources, **kwargs):
        calls.append(kwargs.get("context", ""))
        return _mock_atomic_result("VERIFIED")
    mocker.patch("groundguard.core.verifier.verify", side_effect=capture_verify)
    src = Source(source_id="s1", content="x")
    verify_clause("The party shall not breach.", [src], model="gpt-4o-mini")
    assert any("Obligation type:" in c for c in calls)


@pytest.mark.legal
def test_verify_clause_default_profile_strict(mocker):
    from groundguard.core.verifier import verify_clause
    from groundguard.profiles import STRICT_PROFILE
    captured = []
    def capture_verify(claim, sources, **kwargs):
        captured.append(kwargs.get("profile"))
        return _mock_atomic_result("VERIFIED")
    mocker.patch("groundguard.core.verifier.verify", side_effect=capture_verify)
    src = Source(source_id="s1", content="x")
    verify_clause("The party shall not breach.", [src], model="gpt-4o-mini")
    assert captured[0] is STRICT_PROFILE


@pytest.mark.legal
def test_averify_clause_preprocessing_synchronous(mocker):
    import asyncio
    from groundguard.core.verifier import averify_clause
    decompose_calls = []
    orig_decompose = __import__("groundguard.loaders.legal",
                                fromlist=["decompose_clause"]).decompose_clause
    def tracking_decompose(text):
        decompose_calls.append(text)
        return orig_decompose(text)
    mocker.patch("groundguard.core.verifier.decompose_clause", side_effect=tracking_decompose)
    mocker.patch("groundguard.core.verifier.averify",
                 return_value=_mock_atomic_result("VERIFIED"))
    src = Source(source_id="s1", content="x")
    asyncio.get_event_loop().run_until_complete(
        averify_clause("The party shall comply.", [src], model="gpt-4o-mini")
    )
    assert len(decompose_calls) == 1
