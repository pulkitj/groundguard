"""Tests for verify_answer() / averify_answer() — TDD T-90 (RED state)."""
import pytest
from groundguard.profiles import GENERAL_PROFILE


def _mock_grounding_result(status: str, score: float = 0.9):
    from groundguard.models.result import GroundingResult
    if score < 1.0 and status == "GROUNDED":
        raise ValueError(f"Impossible state: GROUNDED requires score 1.0, got {score}")
    if status == "PARTIALLY_GROUNDED" and (score == 1.0 or score == 0.0):
        raise ValueError(f"Impossible state: PARTIALLY_GROUNDED requires 0.0 < score < 1.0, got {score}")
    if status == "NOT_GROUNDED" and score == 1.0:
        raise ValueError(f"Impossible state: NOT_GROUNDED requires score < 1.0, got {score}")
    
    return GroundingResult(
        is_grounded=(status == "GROUNDED"),
        score=score,
        status=status,
        evaluation_method="sentence_entailment",
    )


def test_verify_answer_returns_grounding_result(mocker):
    from groundguard.core.verifier import verify_answer
    from groundguard.models.result import Source, GroundingResult
    mocker.patch(
        "groundguard.tiers.tier3_evaluation.evaluate_faithfulness",
        return_value=_mock_grounding_result("GROUNDED", score=1.0),
    )
    src = Source(source_id="s1", content="Revenue was $4.2M.")
    result = verify_answer("Revenue was $4.2M.", [src], model="gpt-4o-mini")
    assert isinstance(result, GroundingResult)
    assert result.evaluation_method == "sentence_entailment"


def test_verify_answer_grounded_when_entailment(mocker):
    from groundguard.core.verifier import verify_answer
    from groundguard.models.result import Source
    mocker.patch(
        "groundguard.tiers.tier3_evaluation.evaluate_faithfulness",
        return_value=_mock_grounding_result("PARTIALLY_GROUNDED", score=0.95),
    )
    src = Source(source_id="s1", content="x")
    result = verify_answer("x", [src], model="gpt-4o-mini")
    assert result.is_grounded is True
    assert result.score >= GENERAL_PROFILE.faithfulness_threshold


def test_verify_answer_not_grounded_below_threshold(mocker):
    from groundguard.core.verifier import verify_answer
    from groundguard.models.result import Source
    from groundguard.profiles import GENERAL_PROFILE
    mocker.patch(
        "groundguard.tiers.tier3_evaluation.evaluate_faithfulness",
        return_value=_mock_grounding_result("NOT_GROUNDED", score=0.3),
    )
    src = Source(source_id="s1", content="x")
    result = verify_answer("x", [src], model="gpt-4o-mini")
    assert result.is_grounded is False


def test_averify_answer_returns_coroutine(mocker):
    import asyncio
    from groundguard.core.verifier import averify_answer
    from groundguard.models.result import Source
    mocker.patch(
        "groundguard.tiers.tier3_evaluation.evaluate_faithfulness",
        return_value=_mock_grounding_result("GROUNDED", score=1.0),
    )
    src = Source(source_id="s1", content="x")
    coro = averify_answer("x", [src], model="gpt-4o-mini")
    result = asyncio.get_event_loop().run_until_complete(coro)
    assert result is not None


def test_verify_answer_strict_profile_majority_vote(mocker):
    from groundguard.core.verifier import verify_answer
    from groundguard.models.result import Source
    from groundguard.profiles import STRICT_PROFILE
    mock_eval = mocker.patch(
        "groundguard.tiers.tier3_evaluation.evaluate_faithfulness",
        return_value=_mock_grounding_result("PARTIALLY_GROUNDED", score=0.80),  # below 0.85 -> triggers vote
    )
    src = Source(source_id="s1", content="x")
    verify_answer("x", [src], profile=STRICT_PROFILE, model="gpt-4o-mini")
    # majority_vote=True + confidence<0.85 -> 3 calls total
    assert mock_eval.call_count == 3


def test_verify_answer_majority_vote_three_way_split_returns_unverifiable(mocker):
    from groundguard.core.verifier import verify_answer
    from groundguard.models.result import Source
    from groundguard.profiles import STRICT_PROFILE
    call_n = [0]
    def rotating(*a, **kw):
        call_n[0] += 1
        verdicts = ["GROUNDED", "NOT_GROUNDED", "PARTIALLY_GROUNDED"]
        scores = [1.0, 0.0, 0.50]
        return _mock_grounding_result(verdicts[(call_n[0] - 1) % 3], score=scores[(call_n[0] - 1) % 3])
    mocker.patch("groundguard.tiers.tier3_evaluation.evaluate_faithfulness",
                 side_effect=rotating)
    src = Source(source_id="s1", content="x")
    result = verify_answer("x", [src], profile=STRICT_PROFILE, model="gpt-4o-mini")
    assert result.is_grounded is False  # tie -> conservative fallback -> not grounded
    assert result.audit_records is not None
    assert any(getattr(r, "tie_broken", False) for r in result.audit_records)


def test_verify_answer_majority_vote_two_one_returns_winner(mocker):
    from groundguard.core.verifier import verify_answer
    from groundguard.models.result import Source
    from groundguard.profiles import STRICT_PROFILE
    call_n = [0]
    def two_grounded(*a, **kw):
        call_n[0] += 1
        score = 1.0 if call_n[0] != 2 else 0.0
        status = "GROUNDED" if call_n[0] != 2 else "NOT_GROUNDED"
        return _mock_grounding_result(status, score=score)
    mocker.patch("groundguard.tiers.tier3_evaluation.evaluate_faithfulness",
                 side_effect=two_grounded)
    src = Source(source_id="s1", content="x")
    result = verify_answer("x", [src], profile=STRICT_PROFILE, model="gpt-4o-mini")
    assert result.is_grounded is True  # 2 GROUNDED vs 1 NOT_GROUNDED -> GROUNDED wins


def test_verify_answer_explicit_threshold_overrides_profile(mocker):
    from groundguard.core.verifier import verify_answer
    from groundguard.models.result import Source
    from groundguard.profiles import STRICT_PROFILE
    mocker.patch(
        "groundguard.tiers.tier3_evaluation.evaluate_faithfulness",
        return_value=_mock_grounding_result("PARTIALLY_GROUNDED", score=0.50),
    )
    src = Source(source_id="s1", content="x")
    # STRICT_PROFILE.faithfulness_threshold=0.97; score=0.50 < 0.97 -> not grounded
    # but explicit threshold=0.40 -> 0.50 >= 0.40 -> grounded
    result = verify_answer("x", [src], profile=STRICT_PROFILE,
                           faithfulness_threshold=0.40, model="gpt-4o-mini")
    assert result.is_grounded is True

def test_verify_answer_fatal_contradiction_overrides_score_threshold(mocker):
    from groundguard.core.verifier import verify_answer
    from groundguard.models.result import Source
    mocker.patch(
        'groundguard.tiers.tier3_evaluation.evaluate_faithfulness',
        return_value=_mock_grounding_result('NOT_GROUNDED', score=0.9),
    )
    src = Source(source_id='s1', content='x')
    result = verify_answer('x', [src], model='gpt-4o-mini')
    assert result.is_grounded is False
    assert result.status == 'NOT_GROUNDED'


def test_verify_answer_auto_chunk_false_wraps_source_without_splitting(mocker):
    from groundguard.core.verifier import verify_answer
    from groundguard.models.result import Source

    captured = {}

    def capture_faithfulness(ctx, chunks):
        captured["chunks"] = chunks
        return _mock_grounding_result("GROUNDED", score=1.0)

    mocker.patch(
        "groundguard.tiers.tier3_evaluation.evaluate_faithfulness",
        side_effect=capture_faithfulness,
    )

    src = Source(source_id="s1", content="Revenue was $4.2M.")
    verify_answer("Revenue was $4.2M.", [src], model="gpt-4o-mini", auto_chunk=False)

    assert len(captured["chunks"]) == 1
    assert captured["chunks"][0].text_content == src.content


def test_averify_answer_auto_chunk_false_forwarded_to_verify_answer(mocker):
    import asyncio
    from groundguard.core.verifier import averify_answer
    from groundguard.models.result import Source

    verify_mock = mocker.patch(
        "groundguard.core.verifier.verify_answer",
        return_value=_mock_grounding_result("GROUNDED", score=1.0),
    )

    src = Source(source_id="s1", content="Revenue was $4.2M.")
    asyncio.run(averify_answer("Revenue was $4.2M.", [src], auto_chunk=False))

    call_kwargs = verify_mock.call_args[1]
    assert call_kwargs.get("auto_chunk") is False


def test_verify_answer_auto_chunk_false_with_majority_vote_all_calls_receive_unwrapped_chunks(mocker):
    from groundguard.core.verifier import verify_answer
    from groundguard.models.result import Source
    from groundguard.profiles import VerificationProfile

    mv_profile = VerificationProfile(
        name="test_mv",
        faithfulness_threshold=0.8,
        tier2_lexical_threshold=0.85,
        bm25_top_k=3,
        majority_vote=True,
        audit=False,
    )

    call_chunk_counts = []

    def capture_faithfulness(ctx, chunks):
        call_chunk_counts.append(len(chunks))
        return _mock_grounding_result("GROUNDED", score=1.0)

    mocker.patch(
        "groundguard.tiers.tier3_evaluation.evaluate_faithfulness",
        side_effect=capture_faithfulness,
    )

    src = Source(source_id="s1", content="Revenue was $4.2M.")
    verify_answer(
        "Revenue was $4.2M.", [src],
        model="gpt-4o-mini",
        auto_chunk=False,
        profile=mv_profile,
    )

    assert len(call_chunk_counts) == 3  # majority vote makes exactly 3 calls
    assert all(n == 1 for n in call_chunk_counts)  # every call gets 1 unwrapped chunk

