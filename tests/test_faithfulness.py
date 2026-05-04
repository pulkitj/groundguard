"""Faithfulness evaluation tests — FaithfulnessResponseModel + evaluate_faithfulness (RED).

These tests will FAIL until:
  - groundguard/models/tier3.py gains SentenceResult + FaithfulnessResponseModel
  - groundguard/tiers/tier3_evaluation.py gains evaluate_faithfulness()
"""
import pytest


# ---------------------------------------------------------------------------
# Module-level helper (import of not-yet-existing symbols is intentional RED)
# ---------------------------------------------------------------------------

def _mock_faithfulness_response(verdict: str, count: int = 1):
    from unittest.mock import MagicMock
    from groundguard.models.tier3 import FaithfulnessResponseModel, SentenceResult
    msg = MagicMock()
    msg.content = FaithfulnessResponseModel(
        sentence_results=[
            SentenceResult(sentence=f"x{i}", verdict=verdict, confidence=0.90)
            for i in range(count)
        ]
    ).model_dump_json()
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Model field tests
# ---------------------------------------------------------------------------

def test_sentence_result_fields():
    from groundguard.models.tier3 import SentenceResult
    sr = SentenceResult(
        sentence="Revenue was $4.2M.",
        verdict="Entailment",
        confidence=0.95,
        grounding_source_id="s1",
        reasoning="Source states $4.2M.",
    )
    assert sr.verdict in ("Entailment", "Contradiction", "Neutral")


def test_faithfulness_response_model():
    from groundguard.models.tier3 import FaithfulnessResponseModel
    m = FaithfulnessResponseModel(sentence_results=[])
    assert m.sentence_results == []


# ---------------------------------------------------------------------------
# evaluate_faithfulness() core contract
# ---------------------------------------------------------------------------

def test_evaluate_faithfulness_returns_grounding_result(mocker):
    from groundguard.tiers.tier3_evaluation import evaluate_faithfulness
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    from groundguard.profiles import GENERAL_PROFILE
    mocker.patch(
        "groundguard.tiers.tier3_evaluation._completion_with_backoff",
        return_value=_mock_faithfulness_response("Entailment"),
    )
    src = Source(source_id="s1", content="Revenue was $4.2M.")
    ctx = VerificationContext(claim="Revenue was $4.2M.", sources=[src], profile=GENERAL_PROFILE)
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="Revenue was $4.2M.",
                  char_start=0, char_end=18, token_count=5)
    result = evaluate_faithfulness(ctx, [chunk])
    from groundguard.models.result import GroundingResult
    assert isinstance(result, GroundingResult)
    assert result.evaluation_method == "sentence_entailment"


def test_evaluate_faithfulness_entailment_grounded(mocker):
    from groundguard.tiers.tier3_evaluation import evaluate_faithfulness
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    from groundguard.profiles import GENERAL_PROFILE
    mocker.patch(
        "groundguard.tiers.tier3_evaluation._completion_with_backoff",
        return_value=_mock_faithfulness_response("Entailment"),
    )
    src = Source(source_id="s1", content="x")
    ctx = VerificationContext(claim="x", sources=[src], profile=GENERAL_PROFILE)
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="x",
                  char_start=0, char_end=1, token_count=1)
    result = evaluate_faithfulness(ctx, [chunk])
    assert result.is_grounded is True
    assert result.status == "GROUNDED"


def test_evaluate_faithfulness_contradiction_not_grounded(mocker):
    from groundguard.tiers.tier3_evaluation import evaluate_faithfulness
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    from groundguard.profiles import GENERAL_PROFILE
    mocker.patch(
        "groundguard.tiers.tier3_evaluation._completion_with_backoff",
        return_value=_mock_faithfulness_response("Contradiction"),
    )
    src = Source(source_id="s1", content="x")
    ctx = VerificationContext(claim="y", sources=[src], profile=GENERAL_PROFILE)
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="x",
                  char_start=0, char_end=1, token_count=1)
    result = evaluate_faithfulness(ctx, [chunk])
    assert result.is_grounded is False
    assert result.status == "NOT_GROUNDED"


def test_evaluate_faithfulness_unit_results_are_contextualized_claim_units(mocker):
    from groundguard.tiers.tier3_evaluation import evaluate_faithfulness
    from groundguard.models.result import ContextualizedClaimUnit, Source
    from groundguard.models.internal import VerificationContext
    from groundguard.loaders.chunker import Chunk
    from groundguard.profiles import GENERAL_PROFILE
    mocker.patch(
        "groundguard.tiers.tier3_evaluation._completion_with_backoff",
        return_value=_mock_faithfulness_response("Entailment"),
    )
    src = Source(source_id="s1", content="x")
    ctx = VerificationContext(claim="x", sources=[src], profile=GENERAL_PROFILE)
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="x",
                  char_start=0, char_end=1, token_count=1)
    result = evaluate_faithfulness(ctx, [chunk])
    assert all(isinstance(u, ContextualizedClaimUnit) for u in result.unit_results)


# ---------------------------------------------------------------------------
# Profile-driven audit records
# ---------------------------------------------------------------------------

def test_evaluate_faithfulness_audit_profile_populates_audit_records(mocker):
    from groundguard.tiers.tier3_evaluation import evaluate_faithfulness
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    from groundguard.profiles import STRICT_PROFILE
    mocker.patch(
        "groundguard.tiers.tier3_evaluation._completion_with_backoff",
        return_value=_mock_faithfulness_response("Entailment"),
    )
    src = Source(source_id="s1", content="x")
    ctx = VerificationContext(claim="x", sources=[src], profile=STRICT_PROFILE)
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="x",
                  char_start=0, char_end=1, token_count=1)
    result = evaluate_faithfulness(ctx, [chunk])
    assert result.audit_records is not None
    assert len(result.audit_records) > 0


# ---------------------------------------------------------------------------
# Context injection (prev_context / next_context)
# ---------------------------------------------------------------------------

def test_prev_context_injected_in_faithfulness_prompt(mocker):
    from groundguard.tiers.tier3_evaluation import evaluate_faithfulness
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    from groundguard.profiles import GENERAL_PROFILE
    captured = []
    def capture(*args, **kwargs):
        captured.append(str(args) + str(kwargs))
        return _mock_faithfulness_response("Entailment")
    mocker.patch("groundguard.tiers.tier3_evaluation._completion_with_backoff",
                 side_effect=capture)
    src = Source(source_id="s1", content="Revenue was $5M.",
                 prev_context="Q3 results follow.", next_context="Net income was $1M.")
    ctx = VerificationContext(claim="Revenue was $5M.", sources=[src], profile=GENERAL_PROFILE)
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="Revenue was $5M.",
                  char_start=0, char_end=16, token_count=3)
    evaluate_faithfulness(ctx, [chunk])
    assert any("Q3 results follow" in c for c in captured)
    assert any("Net income was $1M" in c for c in captured)


# ---------------------------------------------------------------------------
# Pronoun / coreference enrichment
# ---------------------------------------------------------------------------

def test_pronoun_sentence_gets_llm_coreference_enrichment(mocker):
    from groundguard.tiers.tier3_evaluation import evaluate_faithfulness
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    from groundguard.profiles import GENERAL_PROFILE
    mocker.patch(
        "groundguard.tiers.tier3_evaluation._completion_with_backoff",
        return_value=_mock_faithfulness_response("Entailment", count=2),
    )
    src = Source(source_id="s1", content="Adobe grew 30%. It reported record results.")
    ctx = VerificationContext(
        claim="Adobe reported strong results. It grew 30% in Q3.",
        sources=[src], profile=GENERAL_PROFILE,
    )
    chunk = Chunk(chunk_id="c1", source_id="s1",
                  text_content="Adobe grew 30%. It reported record results.",
                  char_start=0, char_end=42, token_count=7)
    result = evaluate_faithfulness(ctx, [chunk])
    pronoun_units = [u for u in result.unit_results
                     if u.display_text.startswith("It")]
    assert all(u.enrichment_method == "llm_coreference" for u in pronoun_units)
    assert all(u.preceding_sentence is not None for u in pronoun_units)


# ---------------------------------------------------------------------------
# Structural hints override
# ---------------------------------------------------------------------------

def test_structural_hints_override_automatic_sentence_inference(mocker):
    from groundguard.tiers.tier3_evaluation import evaluate_faithfulness
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    from groundguard.profiles import GENERAL_PROFILE
    mocker.patch(
        "groundguard.tiers.tier3_evaluation._completion_with_backoff",
        return_value=_mock_faithfulness_response("Entailment"),
    )
    hints = [{"display_text": "5.1", "claim_text": "Revenue in Q2 was $5.1M",
              "enrichment_method": "structural_code", "structural_type": "table_cell",
              "column_header": "Q2 2025", "row_label": "Revenue", "heading_path": []}]
    src = Source(source_id="s1", content="Revenue Q2=$5.1M")
    ctx = VerificationContext(claim="5.1", sources=[src], profile=GENERAL_PROFILE)
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="Revenue Q2=$5.1M",
                  char_start=0, char_end=16, token_count=3)
    result = evaluate_faithfulness(ctx, [chunk], structural_hints=hints)
    assert result.unit_results[0].structural_type == "table_cell"
    assert result.unit_results[0].claim_text == "Revenue in Q2 was $5.1M"


# ---------------------------------------------------------------------------
# Count mismatch validation
# ---------------------------------------------------------------------------

def test_evaluate_faithfulness_raises_on_count_mismatch(mocker):
    """LLM returns fewer sentence_results than units → ParseError."""
    from groundguard.tiers.tier3_evaluation import evaluate_faithfulness
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source
    from groundguard.loaders.chunker import Chunk
    from groundguard.exceptions import ParseError
    from unittest.mock import MagicMock
    from groundguard.models.tier3 import FaithfulnessResponseModel, SentenceResult

    # Build a response with only 1 result when 2 units are sent
    msg = MagicMock()
    msg.content = FaithfulnessResponseModel(
        sentence_results=[SentenceResult(sentence="x", verdict="Entailment", confidence=0.9)]
    ).model_dump_json()
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]

    mocker.patch(
        "groundguard.tiers.tier3_evaluation._completion_with_backoff",
        return_value=resp,
    )

    # claim has 2 sentences → 2 units; mock returns 1 sentence_result → ParseError
    src = Source(source_id="s1", content="Claim A. Claim B.")
    ctx = VerificationContext(claim="Claim A. Claim B.", sources=[src])
    chunk = Chunk(chunk_id="c1", source_id="s1", text_content="Claim A. Claim B.",
                  char_start=0, char_end=17, token_count=5)

    import pytest
    with pytest.raises(ParseError, match="faithfulness response has 1 results for 2 units"):
        evaluate_faithfulness(ctx, [chunk])
