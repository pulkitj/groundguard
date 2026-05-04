"""Deterministic integration tests — zero real LLM calls.

Exercises the full verify() and verify_answer() pipeline using schema-aware
mocks that return correct JSON payloads.  Every mock helper also enforces
the same state invariants as the real code so that impossible states are
caught at fixture construction time, not silently propagated into assertions.
"""
from __future__ import annotations

import pytest

from groundguard.core.verifier import verify, verify_answer
from groundguard.models.result import GroundingResult, Source


# ---------------------------------------------------------------------------
# Schema-aware LLM mock helpers
# ---------------------------------------------------------------------------

_ENTAILMENT_RESPONSE = (
    '{"textual_entailment": {"label": "Entailment", "probability": 0.95},'
    ' "conceptual_coverage": {"percentage": 100.0, "covered_concepts": [], "missing_concepts": []},'
    ' "factual_consistency_score": 95.0,'
    ' "verifications": [{"claim_text": "Revenue grew.", "status": "VERIFIED",'
    '   "source_id": "s1", "source_excerpt": "Revenue grew."}],'
    ' "source_attributions": [{"source_id": "s1", "role": "Supporting"}],'
    ' "overall_verdict": "Verified."}'
)

_CONTRADICTION_RESPONSE = (
    '{"textual_entailment": {"label": "Contradiction", "probability": 0.95},'
    ' "conceptual_coverage": {"percentage": 100.0, "covered_concepts": [], "missing_concepts": []},'
    ' "factual_consistency_score": 5.0,'
    ' "verifications": [{"claim_text": "Revenue grew 300%.", "status": "CONTRADICTED",'
    '   "source_id": "s1", "source_excerpt": "Revenue grew 30%."}],'
    ' "source_attributions": [{"source_id": "s1", "role": "Contradicting"}],'
    ' "overall_verdict": "Contradicted."}'
)

_ENTAILMENT_NO_EXCERPT_RESPONSE = (
    '{"textual_entailment": {"label": "Entailment", "probability": 0.95},'
    ' "conceptual_coverage": {"percentage": 100.0, "covered_concepts": [], "missing_concepts": []},'
    ' "factual_consistency_score": 95.0,'
    ' "verifications": [{"claim_text": "Revenue grew.", "status": "VERIFIED",'
    '   "source_id": "s1", "source_excerpt": null}],'
    ' "source_attributions": [{"source_id": "s1", "role": "Supporting"}],'
    ' "overall_verdict": "Verified."}'
)

_INFERENTIAL_ENTAILMENT_RESPONSE = (
    '{"textual_entailment": {"label": "Entailment", "probability": 0.95},'
    ' "conceptual_coverage": {"percentage": 100.0, "covered_concepts": [], "missing_concepts": []},'
    ' "factual_consistency_score": 95.0,'
    ' "verifications": [{"claim_text": "Revenue trend suggests growth.", "status": "VERIFIED",'
    '   "source_id": "s1", "source_excerpt": null, "reasoning_basis": ["Revenue grew 20%."]}],'
    ' "source_attributions": [{"source_id": "s1", "role": "Supporting"}],'
    ' "overall_verdict": "Verified."}'
)

_FAITHFULNESS_ENTAILMENT = (
    '{"sentence_results": [{"sentence": "Revenue grew.", "verdict": "Entailment",'
    '   "confidence": 0.95, "grounding_source_id": "s1"}]}'
)

_FAITHFULNESS_CONTRADICTION = (
    '{"sentence_results": [{"sentence": "Revenue grew 300%.", "verdict": "Contradiction",'
    '   "confidence": 0.95, "grounding_source_id": "s1"}]}'
)


def _make_mock_response(content: str):
    """Build a minimal litellm response object with the given JSON content."""
    class _Msg:
        pass

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 10

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    _Resp.choices[0].message.content = content
    return _Resp()


def _schema_dispatch(kwargs: dict, entailment_content: str, faithfulness_content: str) -> str:
    """Return the right JSON content based on the requested response_format schema."""
    from groundguard.models.tier3 import FaithfulnessResponseModel
    schema = kwargs.get("response_format")
    if schema is FaithfulnessResponseModel:
        return faithfulness_content
    return entailment_content


# ---------------------------------------------------------------------------
# Mock invariant guard — enforces same rules as real code
# ---------------------------------------------------------------------------

def _assert_mock_grounding_invariants(status: str, score: float) -> None:
    """Raise ValueError on impossible status/score combinations.

    This is the fuzz guard for mock builders: any test helper that creates a
    GroundingResult must pass through this check so that impossible states are
    caught at fixture creation time rather than silently propagated into
    assertions that will trivially pass.

    Invariants enforced:
    - GROUNDED requires score == 1.0  (perfect entailment across all units)
    - NOT_GROUNDED requires score < 1.0
    - PARTIALLY_GROUNDED requires 0.0 < score < 1.0  (at least one hit AND one miss)
    """
    if status == "GROUNDED" and score < 1.0:
        raise ValueError(
            f"Mock invariant violated: GROUNDED requires score==1.0, got {score}. "
            "A GROUNDED result means every sentence was entailed."
        )
    if status == "NOT_GROUNDED" and score == 1.0:
        raise ValueError(
            f"Mock invariant violated: NOT_GROUNDED requires score<1.0, got {score}."
        )
    if status == "PARTIALLY_GROUNDED" and score in (0.0, 1.0):
        raise ValueError(
            f"Mock invariant violated: PARTIALLY_GROUNDED requires 0.0<score<1.0, got {score}."
        )


def _make_grounding_result(status: str, score: float, evaluation_method: str = "sentence_entailment") -> GroundingResult:
    """Construct a GroundingResult with invariant checking."""
    _assert_mock_grounding_invariants(status, score)
    return GroundingResult(
        is_grounded=(status == "GROUNDED"),
        score=score,
        status=status,
        evaluation_method=evaluation_method,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_entailment(mocker):
    """Mock both sync and async litellm calls to return Entailment."""
    def _sync(**kwargs):
        content = _schema_dispatch(kwargs, _ENTAILMENT_RESPONSE, _FAITHFULNESS_ENTAILMENT)
        return _make_mock_response(content)

    async def _async(**kwargs):
        content = _schema_dispatch(kwargs, _ENTAILMENT_RESPONSE, _FAITHFULNESS_ENTAILMENT)
        return _make_mock_response(content)

    mocker.patch("groundguard.tiers.tier3_evaluation.litellm.completion", side_effect=_sync)
    mocker.patch("groundguard.tiers.tier3_evaluation.litellm.acompletion", side_effect=_async)


@pytest.fixture
def mock_llm_contradiction(mocker):
    """Mock both sync and async litellm calls to return Contradiction."""
    def _sync(**kwargs):
        content = _schema_dispatch(kwargs, _CONTRADICTION_RESPONSE, _FAITHFULNESS_CONTRADICTION)
        return _make_mock_response(content)

    async def _async(**kwargs):
        content = _schema_dispatch(kwargs, _CONTRADICTION_RESPONSE, _FAITHFULNESS_CONTRADICTION)
        return _make_mock_response(content)

    mocker.patch("groundguard.tiers.tier3_evaluation.litellm.completion", side_effect=_sync)
    mocker.patch("groundguard.tiers.tier3_evaluation.litellm.acompletion", side_effect=_async)


@pytest.fixture
def mock_llm_no_excerpt(mocker):
    """Mock litellm to return VERIFIED with no source_excerpt — tests citation invariant."""
    def _sync(**kwargs):
        content = _schema_dispatch(kwargs, _ENTAILMENT_NO_EXCERPT_RESPONSE, _FAITHFULNESS_ENTAILMENT)
        return _make_mock_response(content)

    async def _async(**kwargs):
        content = _schema_dispatch(kwargs, _ENTAILMENT_NO_EXCERPT_RESPONSE, _FAITHFULNESS_ENTAILMENT)
        return _make_mock_response(content)

    mocker.patch("groundguard.tiers.tier3_evaluation.litellm.completion", side_effect=_sync)
    mocker.patch("groundguard.tiers.tier3_evaluation.litellm.acompletion", side_effect=_async)


@pytest.fixture
def mock_llm_inferential_no_excerpt(mocker):
    """Mock litellm to return VERIFIED inferential result with no source_excerpt."""
    def _sync(**kwargs):
        content = _schema_dispatch(kwargs, _INFERENTIAL_ENTAILMENT_RESPONSE, _FAITHFULNESS_ENTAILMENT)
        return _make_mock_response(content)

    async def _async(**kwargs):
        content = _schema_dispatch(kwargs, _INFERENTIAL_ENTAILMENT_RESPONSE, _FAITHFULNESS_ENTAILMENT)
        return _make_mock_response(content)

    mocker.patch("groundguard.tiers.tier3_evaluation.litellm.completion", side_effect=_sync)
    mocker.patch("groundguard.tiers.tier3_evaluation.litellm.acompletion", side_effect=_async)


@pytest.fixture
def bypass_tier2(mocker):
    """Force Tier 2 routing to escalate to LLM so every test hits Tier 3."""
    from groundguard.models.internal import Tier2Result, RoutingDecision
    mocker.patch(
        "groundguard.core.verifier.tier2_semantic.route_claim",
        return_value=Tier2Result(
            decision=RoutingDecision.ESCALATE_TO_LLM,
            top_k_chunks=[],
            highest_score=0.5,
        ),
    )


# ---------------------------------------------------------------------------
# Deterministic pipeline smoke tests
# ---------------------------------------------------------------------------

def test_deterministic_verify_entailment(mock_llm_entailment, bypass_tier2):
    """Happy path: verify() pipeline returns VERIFIED end-to-end."""
    src = Source(source_id="s1", content="Revenue grew.")
    result = verify("Revenue grew.", [src])
    assert result.is_valid is True
    assert result.status == "VERIFIED"


def test_deterministic_verify_answer_entailment(mock_llm_entailment, bypass_tier2):
    """Happy path: verify_answer() returns GROUNDED end-to-end."""
    src = Source(source_id="s1", content="Revenue grew.")
    result = verify_answer("Revenue grew.", [src])
    assert result.is_grounded is True
    assert result.status == "GROUNDED"


def test_deterministic_verify_contradiction(mock_llm_contradiction, bypass_tier2):
    """Contradiction path: verify() pipeline returns CONTRADICTED end-to-end."""
    src = Source(source_id="s1", content="Revenue grew 30%.")
    result = verify("Revenue grew 300%.", [src])
    assert result.is_valid is False
    assert result.status == "CONTRADICTED"


# ---------------------------------------------------------------------------
# Behavioural edge-case 1 — Fatal contradiction overrides score threshold
# ---------------------------------------------------------------------------

def test_fatal_contradiction_overrides_score_threshold(mocker):
    """A hard contradiction returned by the LLM must NOT be promoted to grounded,
    even when score >= faithfulness_threshold.

    Scenario: evaluate_faithfulness returns NOT_GROUNDED with score=0.9.
    GENERAL_PROFILE threshold is 0.80.  score (0.9) >= threshold (0.80), but the
    LLM explicitly flagged NOT_GROUNDED, so the result must remain NOT_GROUNDED.
    """
    result = _make_grounding_result("NOT_GROUNDED", score=0.9)
    mocker.patch(
        "groundguard.tiers.tier3_evaluation.evaluate_faithfulness",
        return_value=result,
    )
    src = Source(source_id="s1", content="Revenue grew 30%.")
    out = verify_answer("Revenue grew 300%.", [src], model="gpt-4o-mini")

    assert out.is_grounded is False, (
        "A hard LLM contradiction must not be promoted to grounded by score threshold comparison."
    )
    assert out.status == "NOT_GROUNDED"


# ---------------------------------------------------------------------------
# Behavioural edge-case 2 — Citation invariant enforced through the pipeline
# ---------------------------------------------------------------------------

def test_verify_raises_invariant_error_for_extractive_without_excerpt(
    mock_llm_no_excerpt, bypass_tier2
):
    """VERIFIED extractive claim with no source_excerpt must raise InvariantError
    from within the full verify() pipeline.

    The LLM returns Entailment / VERIFIED but omits source_excerpt.  ResultBuilder
    must detect this and raise before producing a VerificationResult, since a
    grounded extractive claim without an evidence excerpt is unprovable.
    """
    from groundguard.exceptions import InvariantError
    src = Source(source_id="s1", content="Revenue grew.")
    with pytest.raises(InvariantError, match="VERIFIED extractive claim has no citation"):
        verify("Revenue grew.", [src])


def test_verify_does_not_raise_for_inferential_without_excerpt(
    mock_llm_inferential_no_excerpt, bypass_tier2
):
    """VERIFIED inferential claim with no source_excerpt must NOT raise.

    Inferential claims (containing signals like 'suggests') produce reasoning_basis
    instead of direct quotes.  The pipeline must pass this through without error.
    Claim text contains 'suggests' so the Tier 0 classifier marks it Inferential.
    """
    src = Source(source_id="s1", content="Revenue grew 20%.")
    result = verify("Revenue trend suggests growth.", [src])
    assert result.is_valid is True


def test_verify_does_not_raise_for_extractive_with_excerpt(
    mock_llm_entailment, bypass_tier2
):
    """VERIFIED extractive with a source_excerpt present must not raise."""
    src = Source(source_id="s1", content="Revenue grew.")
    result = verify("Revenue grew.", [src])
    assert result.is_valid is True


def test_verify_does_not_raise_for_unverifiable_without_excerpt(
    mock_llm_contradiction, bypass_tier2
):
    """UNVERIFIABLE / CONTRADICTED results with no excerpt must never raise."""
    src = Source(source_id="s1", content="Revenue grew 30%.")
    result = verify("Revenue grew 300%.", [src])
    assert result.is_valid is False


# ---------------------------------------------------------------------------
# Behavioural edge-case 3 — Accumulator denominator (Issue 2)
# ---------------------------------------------------------------------------

def test_grounding_accumulator_unverifiable_excluded_from_denominator():
    """1 VERIFIED + 99 UNVERIFIABLE claims → GroundingAccumulator.overall_score must be 1.0.

    The denominator is grounded_units + ungrounded_units; unverifiable units are
    excluded.  Using total_units (100) as denominator would give 0.01 — wrong.
    """
    from groundguard.loaders.accumulator import GroundingAccumulator

    result = GroundingResult(
        is_grounded=True,
        score=1.0,
        status="GROUNDED",
        evaluation_method="sentence_entailment",
        grounded_units=1,
        ungrounded_units=0,
        unverifiable_units=99,
        total_units=100,
    )
    acc = GroundingAccumulator()
    acc.add(result)

    assert acc.overall_score == 1.0, (
        f"Expected 1.0 (1 grounded / 1 scorable), got {acc.overall_score}. "
        "Unverifiable units must not be included in the denominator."
    )


def test_grounding_accumulator_score_excludes_unverifiable_mixed():
    """2 VERIFIED + 1 CONTRADICTED + 1 UNVERIFIABLE → GroundingAccumulator.overall_score == 2/3."""
    from groundguard.loaders.accumulator import GroundingAccumulator

    result = GroundingResult(
        is_grounded=False,
        score=2 / 3,
        status="PARTIALLY_GROUNDED",
        evaluation_method="sentence_entailment",
        grounded_units=2,
        ungrounded_units=1,
        unverifiable_units=1,
        total_units=4,
    )
    acc = GroundingAccumulator()
    acc.add(result)

    assert acc.overall_score == pytest.approx(2 / 3), (
        f"Expected 2/3 (2 grounded / 3 scorable), got {acc.overall_score}. "
        "Unverifiable unit must be excluded from denominator."
    )


# ---------------------------------------------------------------------------
# Behavioural edge-case 4 — All-unverifiable returns NOT_GROUNDED (Issue 4)
# ---------------------------------------------------------------------------

def test_all_unverifiable_status_is_not_grounded():
    """When every per-claim result is UNVERIFIABLE, _aggregate_analysis_results
    must return status=NOT_GROUNDED, not PARTIALLY_GROUNDED.

    PARTIALLY_GROUNDED implies at least one claim was verified.  Zero verified
    claims means nothing was grounded, so NOT_GROUNDED is the correct status.
    """
    from groundguard.core.verifier import _aggregate_analysis_results
    from groundguard.models.result import VerificationResult

    unverifiable_results = [
        VerificationResult(
            is_valid=False,
            overall_verdict="Unverifiable.",
            verification_method="tier3_llm",
            atomic_claims=[],
            factual_consistency_score=0.0,
            sources_used=[],
            rationale="",
            offending_claim=None,
            status="UNVERIFIABLE",
            total_cost_usd=0.0,
        )
        for _ in range(5)
    ]

    gr = _aggregate_analysis_results(unverifiable_results)

    assert gr.status == "NOT_GROUNDED", (
        f"Expected NOT_GROUNDED for all-unverifiable input, got {gr.status!r}."
    )
    assert gr.score == 0.0


# ---------------------------------------------------------------------------
# Behavioural edge-case 5 — Mock invariant guard (fuzz guards) self-test
# ---------------------------------------------------------------------------

def test_mock_invariant_rejects_grounded_with_partial_score():
    """_assert_mock_grounding_invariants must catch GROUNDED + score<1.0."""
    with pytest.raises(ValueError, match="GROUNDED requires score==1.0"):
        _assert_mock_grounding_invariants("GROUNDED", 0.9)


def test_mock_invariant_rejects_not_grounded_with_perfect_score():
    """_assert_mock_grounding_invariants must catch NOT_GROUNDED + score==1.0."""
    with pytest.raises(ValueError, match="NOT_GROUNDED requires score<1.0"):
        _assert_mock_grounding_invariants("NOT_GROUNDED", 1.0)


def test_mock_invariant_rejects_partially_grounded_at_extremes():
    """PARTIALLY_GROUNDED at score==0.0 or 1.0 is logically impossible."""
    with pytest.raises(ValueError, match="PARTIALLY_GROUNDED requires 0.0<score<1.0"):
        _assert_mock_grounding_invariants("PARTIALLY_GROUNDED", 0.0)
    with pytest.raises(ValueError, match="PARTIALLY_GROUNDED requires 0.0<score<1.0"):
        _assert_mock_grounding_invariants("PARTIALLY_GROUNDED", 1.0)


def test_mock_invariant_accepts_valid_combinations():
    """Valid status/score combinations must not raise."""
    _assert_mock_grounding_invariants("GROUNDED", 1.0)
    _assert_mock_grounding_invariants("NOT_GROUNDED", 0.0)
    _assert_mock_grounding_invariants("NOT_GROUNDED", 0.9)
    _assert_mock_grounding_invariants("PARTIALLY_GROUNDED", 0.5)


# ---------------------------------------------------------------------------
# Behavioural edge-case 6 — Branch C cap math (Issue 6)
# ---------------------------------------------------------------------------

def test_branch_c_cap_uses_actual_profile_top_k():
    """Branch C chunk cap must be bm25_top_k * 3 from the active profile.

    GENERAL_PROFILE.bm25_top_k == 3, so cap == 9.
    The prior documentation incorrectly assumed top_k==5 (cap==15); no defined
    profile uses bm25_top_k=5.
    """
    from groundguard.profiles import GENERAL_PROFILE, STRICT_PROFILE, RESEARCH_PROFILE

    assert GENERAL_PROFILE.bm25_top_k * 3 == 9, "GENERAL cap must be 9"
    assert STRICT_PROFILE.bm25_top_k * 3 == 18, "STRICT cap must be 18"
    assert RESEARCH_PROFILE.bm25_top_k * 3 == 12, "RESEARCH cap must be 12"

    # No profile should have bm25_top_k==5 (the old assumption)
    for profile in (GENERAL_PROFILE, STRICT_PROFILE, RESEARCH_PROFILE):
        assert profile.bm25_top_k != 5, (
            f"Profile '{profile.name}' uses bm25_top_k=5 which contradicts all defined profiles."
        )
