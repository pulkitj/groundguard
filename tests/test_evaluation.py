"""Tier 3 evaluation tests — Part A: render_prompt and parse_response."""
import json
import re
import pytest
from unittest.mock import MagicMock
from agentic_verifier.models.internal import VerificationContext, ClassifiedAtom
from agentic_verifier.models.result import Source
from agentic_verifier.models.tier3 import (
    Tier3ResponseModel, TextualEntailment, ConceptualCoverage,
    AtomicVerification, SourceAttribution,
)
from agentic_verifier.tiers.tier3_evaluation import render_prompt, parse_response


def _make_ctx(claim="Revenue was $5M.") -> VerificationContext:
    return VerificationContext(
        claim=claim,
        original_sources=[Source(content="Revenue was $5M.", source_id="doc.pdf")],
        model="gpt-4o-mini",
    )


def _make_chunks():
    from agentic_verifier.loaders.chunker import Chunk
    return [Chunk(parent_source_id="doc.pdf", text_content="Revenue was $5M.", char_start=0, char_end=16)]


def _valid_t3_model():
    return Tier3ResponseModel(
        textual_entailment=TextualEntailment(label="Entailment", probability=0.95),
        conceptual_coverage=ConceptualCoverage(percentage=90.0, covered_concepts=["revenue"], missing_concepts=[]),
        factual_consistency_score=90.0,
        verifications=[AtomicVerification(claim_text="Revenue was $5M.", status="VERIFIED", source_id="doc.pdf", source_excerpt="Revenue was $5M.", reasoning_basis=None)],
        source_attributions=[SourceAttribution(source_id="doc.pdf", role="Supporting")],
        overall_verdict="The source fully supports the claim.",
    )


# TDD #8: boundary ID uniqueness
def test_render_prompt_uses_ctx_boundary_id():
    """render_prompt uses ctx._boundary_id — does NOT generate a new one."""
    ctx = _make_ctx()
    chunks = _make_chunks()
    prompt = render_prompt(ctx, chunks)
    assert ctx._boundary_id in prompt


def test_two_contexts_produce_different_boundary_ids_in_prompts():
    """TDD #8: Two VerificationContexts have unique _boundary_ids."""
    src = [Source(content="Revenue was $5M.", source_id="doc.pdf")]
    ctx1 = VerificationContext(claim="test", original_sources=src, model="gpt-4o-mini")
    ctx2 = VerificationContext(claim="test", original_sources=src, model="gpt-4o-mini")
    assert ctx1._boundary_id != ctx2._boundary_id
    p1 = render_prompt(ctx1, _make_chunks())
    p2 = render_prompt(ctx2, _make_chunks())
    m1 = re.findall(r'---CLAIM-([0-9a-f]{12})---', p1)
    m2 = re.findall(r'---CLAIM-([0-9a-f]{12})---', p2)
    assert m1 and m2
    assert m1[0] != m2[0]


def test_boundary_id_is_exactly_12_hex_chars():
    """_boundary_id must be exactly 12 lowercase hex characters."""
    ctx = _make_ctx()
    assert len(ctx._boundary_id) == 12
    assert all(c in "0123456789abcdef" for c in ctx._boundary_id)


# TDD #4: parse_response fence stripping
def test_parse_response_strips_json_fence():
    """TDD #4: parse_response handles ```json...``` fences."""
    model = _valid_t3_model()
    raw_json = model.model_dump_json()
    fenced_content = f"```json\n{raw_json}\n```"
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = None
    mock_response.choices[0].message.content = fenced_content
    result = parse_response(mock_response, "gpt-4o-mini")
    assert isinstance(result, Tier3ResponseModel)


def test_parse_response_strips_fence_without_language_tag():
    """parse_response handles ``` fences without 'json' tag."""
    model = _valid_t3_model()
    raw_json = model.model_dump_json()
    fenced = f"```\n{raw_json}\n```"
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = None
    mock_response.choices[0].message.content = fenced
    result = parse_response(mock_response, "gpt-4o-mini")
    assert isinstance(result, Tier3ResponseModel)


def test_parse_response_works_without_fence():
    """parse_response works on plain JSON without fences."""
    model = _valid_t3_model()
    raw_json = model.model_dump_json()
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = None
    mock_response.choices[0].message.content = raw_json
    result = parse_response(mock_response, "gpt-4o-mini")
    assert isinstance(result, Tier3ResponseModel)


def test_parse_response_uses_parsed_attribute_if_present():
    """Primary path: uses response.choices[0].message.parsed if it's a Tier3ResponseModel."""
    model = _valid_t3_model()
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = model
    result = parse_response(mock_response, "gpt-4o-mini")
    assert result is model


# TDD #6a and #6b — retry and ParseError (appended in T-22)
def test_evaluate_retries_once_on_validation_error(mocker):
    """TDD #6a: On first ValidationError, retries with temperature=0.1 and error appended."""
    import pydantic
    from agentic_verifier.tiers.tier3_evaluation import evaluate

    valid_model = _valid_t3_model()
    call_count = [0]

    def mock_completion(**kwargs):
        call_count[0] += 1
        mock_resp = MagicMock()
        mock_resp.choices[0].message.parsed = None
        if call_count[0] == 1:
            # First call: return invalid JSON so parse_response raises ValidationError
            mock_resp.choices[0].message.content = '{"invalid": "json_missing_fields"}'
        else:
            # Second call: return valid model
            mock_resp.choices[0].message.parsed = valid_model
        return mock_resp

    mocker.patch("litellm.completion", side_effect=mock_completion)
    mocker.patch("litellm.completion_cost", return_value=0.001)

    ctx = _make_ctx()
    result = evaluate(ctx, _make_chunks())
    assert call_count[0] == 2
    assert isinstance(result, Tier3ResponseModel)


def test_evaluate_raises_parse_error_after_two_failures(mocker):
    """TDD #6b: Two consecutive validation failures raise ParseError."""
    from agentic_verifier.tiers.tier3_evaluation import evaluate
    from agentic_verifier.exceptions import ParseError

    def mock_completion(**kwargs):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.parsed = None
        mock_resp.choices[0].message.content = '{"totally": "wrong"}'
        return mock_resp

    mocker.patch("litellm.completion", side_effect=mock_completion)
    mocker.patch("litellm.completion_cost", return_value=0.001)

    ctx = _make_ctx()
    with pytest.raises(ParseError):
        evaluate(ctx, _make_chunks())


# ---------------------------------------------------------------------------
# Open Issue #1 — evaluate_async() 0% test coverage
# ---------------------------------------------------------------------------

async def test_evaluate_async_returns_tier3_response_model(mocker):
    """Open Issue #1: evaluate_async() async path returns a valid Tier3ResponseModel."""
    from unittest.mock import AsyncMock, MagicMock
    from agentic_verifier.tiers.tier3_evaluation import evaluate_async

    valid_model = _valid_t3_model()

    mock_resp = MagicMock()
    mock_resp.choices[0].message.parsed = valid_model

    mocker.patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp)
    mocker.patch("litellm.completion_cost", return_value=0.001)

    ctx = _make_ctx()
    result = await evaluate_async(ctx, _make_chunks())

    assert isinstance(result, Tier3ResponseModel)
    assert result.textual_entailment.label == "Entailment"


async def test_evaluate_async_retries_once_on_validation_error(mocker):
    """Open Issue #1b: evaluate_async() retries once on parse failure, then returns valid model."""
    from unittest.mock import AsyncMock, MagicMock
    from agentic_verifier.tiers.tier3_evaluation import evaluate_async

    valid_model = _valid_t3_model()
    call_count = [0]

    async def mock_acompletion(**kwargs):
        call_count[0] += 1
        mock_resp = MagicMock()
        mock_resp.choices[0].message.parsed = None
        if call_count[0] == 1:
            mock_resp.choices[0].message.content = '{"invalid": "missing_fields"}'
        else:
            mock_resp.choices[0].message.parsed = valid_model
        return mock_resp

    mocker.patch("litellm.acompletion", side_effect=mock_acompletion)
    mocker.patch("litellm.completion_cost", return_value=0.001)

    ctx = _make_ctx()
    result = await evaluate_async(ctx, _make_chunks())

    assert call_count[0] == 2
    assert isinstance(result, Tier3ResponseModel)
