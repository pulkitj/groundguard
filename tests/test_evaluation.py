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


# ---------------------------------------------------------------------------
# T-63 — parse_response edge cases: think tags, None content, empty choices
# ---------------------------------------------------------------------------

def test_parse_response_strips_think_tags(mocker):
    """parse_response with <think> tags in content (Ollama thinking mode) — strips tags before JSON parse."""
    from unittest.mock import MagicMock
    from agentic_verifier.tiers.tier3_evaluation import parse_response

    model = _valid_t3_model()
    raw_json = model.model_dump_json()
    content_with_think = f"<think>some reasoning</think>\n{raw_json}"

    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = None
    mock_response.choices[0].message.content = content_with_think
    # Use ollama model so OLLAMA_ADAPTER is selected and think-tag stripping happens
    result = parse_response(mock_response, "ollama/qwen3:14b")
    assert isinstance(result, Tier3ResponseModel)


def test_parse_response_none_content_reasoning_content_fallback(mocker):
    """parse_response when message.content is None but reasoning_content has valid JSON."""
    from unittest.mock import MagicMock
    from agentic_verifier.tiers.tier3_evaluation import parse_response

    model = _valid_t3_model()
    raw_json = model.model_dump_json()

    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = None
    mock_response.choices[0].message.content = None
    mock_response.choices[0].message.reasoning_content = raw_json
    result = parse_response(mock_response, "ollama/qwen3:14b")
    assert isinstance(result, Tier3ResponseModel)


def test_parse_response_choices_empty_raises_descriptively(mocker):
    """litellm.completion returning choices=[] raises with a descriptive error, not IndexError."""
    from unittest.mock import MagicMock
    from agentic_verifier.tiers.tier3_evaluation import evaluate
    from agentic_verifier.exceptions import ParseError

    # Build a response with choices=[]
    mock_resp = MagicMock()
    mock_resp.choices = []

    mocker.patch("litellm.completion", return_value=mock_resp)
    mocker.patch("litellm.completion_cost", return_value=0.0)

    ctx = _make_ctx()
    # choices=[] causes IndexError in the adapter, caught by retry loop → ParseError after 2 attempts
    with pytest.raises(ParseError):
        evaluate(ctx, _make_chunks())


# ---------------------------------------------------------------------------
# T-66 — exponential backoff for transient LiteLLM errors
# ---------------------------------------------------------------------------

async def test_evaluate_async_exhausts_all_backoff_attempts(mocker):
    """BUG-03: evaluate_async retries _BACKOFF_MAX_ATTEMPTS times before re-raising."""
    from unittest.mock import AsyncMock
    import litellm
    from agentic_verifier.tiers.tier3_evaluation import evaluate_async, _BACKOFF_MAX_ATTEMPTS

    call_count = [0]

    async def always_fail(**kwargs):
        call_count[0] += 1
        raise litellm.ServiceUnavailableError(
            message="Service unavailable", llm_provider="test", model="test"
        )

    mocker.patch("litellm.acompletion", side_effect=always_fail)
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    with pytest.raises(litellm.ServiceUnavailableError):
        await evaluate_async(_make_ctx(), _make_chunks())

    assert call_count[0] == _BACKOFF_MAX_ATTEMPTS


async def test_evaluate_async_retries_on_transient_error(mocker):
    """T-66: evaluate_async retries transient errors with exponential backoff."""
    from unittest.mock import AsyncMock, MagicMock
    import litellm
    from agentic_verifier.tiers.tier3_evaluation import evaluate_async

    valid_model = _valid_t3_model()
    call_count = [0]

    async def mock_acompletion(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise litellm.ServiceUnavailableError(
                message="Service unavailable", llm_provider="test", model="test"
            )
        mock_resp = MagicMock()
        mock_resp.choices[0].message.parsed = valid_model
        return mock_resp

    mocker.patch("litellm.acompletion", side_effect=mock_acompletion)
    mocker.patch("litellm.completion_cost", return_value=0.001)
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)  # skip actual sleep

    ctx = _make_ctx()
    result = await evaluate_async(ctx, _make_chunks())

    assert call_count[0] == 2  # first call failed, second succeeded
    assert isinstance(result, Tier3ResponseModel)


# ---------------------------------------------------------------------------
# FIX-04 regression — reasoning_basis schema/prompt contract
# ---------------------------------------------------------------------------

def test_atomic_verification_accepts_reasoning_basis_as_list():
    """AtomicVerification.reasoning_basis accepts a list[str] value."""
    v = AtomicVerification(
        claim_text="Revenue grew faster than costs.",
        status="VERIFIED",
        reasoning_basis=["Source shows 20% revenue growth.", "Costs grew only 5%."],
    )
    assert v.reasoning_basis == ["Source shows 20% revenue growth.", "Costs grew only 5%."]


def test_atomic_verification_coerces_scalar_string_to_list():
    """A model returning reasoning_basis as a plain string is coerced to list[str], not rejected."""
    v = AtomicVerification(
        claim_text="Revenue grew faster than costs.",
        status="VERIFIED",
        reasoning_basis="Source shows revenue growth exceeded cost growth.",  # type: ignore[arg-type]
    )
    assert isinstance(v.reasoning_basis, list)
    assert v.reasoning_basis == ["Source shows revenue growth exceeded cost growth."]


def test_parse_response_accepts_reasoning_basis_list():
    """parse_response succeeds when reasoning_basis is a JSON array (inferential claim)."""
    from agentic_verifier.tiers.tier3_evaluation import parse_response

    model_with_reasoning = Tier3ResponseModel(
        textual_entailment=TextualEntailment(label="Entailment", probability=0.9),
        conceptual_coverage=ConceptualCoverage(percentage=85.0, covered_concepts=["growth"], missing_concepts=[]),
        factual_consistency_score=85.0,
        verifications=[AtomicVerification(
            claim_text="Revenue grew faster than costs.",
            status="VERIFIED",
            reasoning_basis=["Revenue up 20%.", "Costs up 5%."],
        )],
        source_attributions=[SourceAttribution(source_id="doc.pdf", role="Supporting")],
        overall_verdict="Inference is supported.",
    )
    raw_json = model_with_reasoning.model_dump_json()
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = None
    mock_response.choices[0].message.content = raw_json
    result = parse_response(mock_response, "gpt-4o-mini")
    assert isinstance(result, Tier3ResponseModel)
    assert result.verifications[0].reasoning_basis == ["Revenue up 20%.", "Costs up 5%."]


def test_parse_response_coerces_scalar_reasoning_basis():
    """parse_response succeeds when model returns reasoning_basis as a scalar string (regression guard)."""
    from agentic_verifier.tiers.tier3_evaluation import parse_response
    import json

    raw = {
        "textual_entailment": {"label": "Entailment", "probability": 0.9},
        "conceptual_coverage": {"percentage": 85.0, "covered_concepts": ["growth"], "missing_concepts": []},
        "factual_consistency_score": 85.0,
        "verifications": [{
            "claim_text": "Revenue grew faster than costs.",
            "status": "VERIFIED",
            "source_id": None,
            "source_excerpt": None,
            "reasoning_basis": "Revenue up 20%, costs up 5%.",  # scalar — old model output shape
        }],
        "source_attributions": [{"source_id": "doc.pdf", "role": "Supporting"}],
        "overall_verdict": "Inference is supported.",
    }
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = None
    mock_response.choices[0].message.content = json.dumps(raw)
    result = parse_response(mock_response, "gpt-4o-mini")
    assert isinstance(result, Tier3ResponseModel)
    assert result.verifications[0].reasoning_basis == ["Revenue up 20%, costs up 5%."]


def test_prompt_specifies_reasoning_basis_as_array():
    """The Tier 3 prompt explicitly instructs the model to emit reasoning_basis as a JSON array."""
    from agentic_verifier.tiers.tier3_evaluation import render_prompt
    prompt = render_prompt(_make_ctx(), _make_chunks())
    # Must mention reasoning_basis AND array/list in the same instruction, not just incidentally
    rb_idx = prompt.find("reasoning_basis")
    assert rb_idx != -1, "prompt must mention reasoning_basis"
    surrounding = prompt[rb_idx:rb_idx + 120].lower()
    assert "array" in surrounding or "list of" in surrounding, (
        "reasoning_basis instruction must explicitly say 'array' or 'list of'"
    )
