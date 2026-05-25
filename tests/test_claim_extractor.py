"""Tests for Phase 27: claim_extractor.py"""
import pytest
import json
from unittest.mock import MagicMock


def _mock_claims_response(claims: list):
    """Return a mock litellm response whose content is json: {"claims": [...]}"""
    msg = MagicMock()
    msg.content = json.dumps({"claims": claims})
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_extract_claims_returns_list(mocker):
    from groundguard.core.claim_extractor import extract_claims
    from groundguard.models.result import Source
    mocker.patch(
        "groundguard.core.claim_extractor._completion_with_backoff",
        return_value=_mock_claims_response(["Claim A.", "Claim B."]),
    )
    src = Source(source_id="s1", content="Claim A. Claim B.")
    claims = extract_claims("Claim A. Claim B.", [src], model="gpt-4o-mini")
    assert isinstance(claims, list)
    assert len(claims) == 2

def test_extract_claims_retry_on_parse_error(mocker):
    from groundguard.core.claim_extractor import extract_claims
    from groundguard.models.result import Source
    from groundguard.exceptions import ParseError
    import pydantic
    call_count = 0
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise pydantic.ValidationError.from_exception_data("_ClaimList", [])
        return _mock_claims_response(["Claim A."])
    mocker.patch("groundguard.core.claim_extractor._completion_with_backoff",
                 side_effect=side_effect)
    src = Source(source_id="s1", content="x")
    claims = extract_claims("x", [src], model="gpt-4o-mini")
    assert call_count == 2
    assert len(claims) == 1

def test_extract_claims_double_failure_raises_parse_error(mocker):
    from groundguard.core.claim_extractor import extract_claims
    from groundguard.models.result import Source
    from groundguard.exceptions import ParseError
    import pydantic
    mocker.patch(
        "groundguard.core.claim_extractor._completion_with_backoff",
        side_effect=pydantic.ValidationError.from_exception_data("_ClaimList", []),
    )
    src = Source(source_id="s1", content="x")
    with pytest.raises(ParseError):
        extract_claims("x", [src], model="gpt-4o-mini")

def test_extract_claims_boundary_id_in_prompt(mocker):
    from groundguard.core.claim_extractor import extract_claims
    from groundguard.models.result import Source
    calls = []
    def capture(*args, **kwargs):
        calls.append(kwargs)
        return _mock_claims_response(["x"])
    mocker.patch("groundguard.core.claim_extractor._completion_with_backoff",
                 side_effect=capture)
    src = Source(source_id="s1", content="x")
    extract_claims("x", [src], model="gpt-4o-mini")
    prompt_text = str(calls[0])
    import re
    assert re.search(r"[0-9a-f]{12}", prompt_text)

def test_extract_claims_async(mocker):
    import asyncio
    from unittest.mock import AsyncMock
    from groundguard.core.claim_extractor import extract_claims_async
    from groundguard.models.result import Source

    async_mock = mocker.patch(
        "groundguard.core.claim_extractor._acompletion_with_backoff",
        new_callable=AsyncMock,
        return_value=_mock_claims_response(["x"]),
    )
    src = Source(source_id="s1", content="x")
    result = asyncio.get_event_loop().run_until_complete(
        extract_claims_async("x", [src], model="gpt-4o-mini")
    )
    assert isinstance(result, list)
    assert async_mock.called, "_acompletion_with_backoff was not called — sync fallback may still be active"


# ---------------------------------------------------------------------------
# Audit, Cost Tracking, and Robustness Tests
# ---------------------------------------------------------------------------

def test_extract_claims_without_audit_omits_xml_fences(mocker):
    from groundguard.core.claim_extractor import extract_claims
    from groundguard.models.result import Source

    calls = []
    mocker.patch("groundguard.core.claim_extractor._completion_with_backoff",
                 side_effect=lambda *args, **kwargs: (calls.append(kwargs), _mock_claims_response(["x"]))[1])

    src = Source(source_id="s1", content="x")
    extract_claims("x", [src], model="gpt-4o-mini", audit=False)

    prompt_text = str(calls[0])
    assert "<audit_report>" not in prompt_text


def test_extract_claims_with_audit_includes_xml_fences(mocker):
    from groundguard.core.claim_extractor import extract_claims
    from groundguard.models.result import Source

    calls = []
    mocker.patch("groundguard.core.claim_extractor._completion_with_backoff",
                 side_effect=lambda *args, **kwargs: (calls.append(kwargs), _mock_claims_response(["x"]))[1])

    src = Source(source_id="s1", content="x")
    extract_claims("x", [src], model="gpt-4o-mini", audit=True)

    prompt_text = str(calls[0])
    assert "<audit_report>" in prompt_text


async def test_extract_claims_async_without_audit_omits_xml_fences(mocker):
    from groundguard.core.claim_extractor import extract_claims_async
    from groundguard.models.result import Source

    calls = []
    async def mock_acompleter(*args, **kwargs):
        calls.append(kwargs)
        return _mock_claims_response(["x"])

    mocker.patch("groundguard.core.claim_extractor._acompletion_with_backoff",
                 side_effect=mock_acompleter)

    src = Source(source_id="s1", content="x")
    await extract_claims_async("x", [src], model="gpt-4o-mini", audit=False)

    prompt_text = str(calls[0])
    assert "<audit_report>" not in prompt_text


async def test_extract_claims_async_with_audit_includes_xml_fences(mocker):
    from groundguard.core.claim_extractor import extract_claims_async
    from groundguard.models.result import Source

    calls = []
    async def mock_acompleter(*args, **kwargs):
        calls.append(kwargs)
        return _mock_claims_response(["x"])

    mocker.patch("groundguard.core.claim_extractor._acompletion_with_backoff",
                 side_effect=mock_acompleter)

    src = Source(source_id="s1", content="x")
    await extract_claims_async("x", [src], model="gpt-4o-mini", audit=True)

    prompt_text = str(calls[0])
    assert "<audit_report>" in prompt_text


def test_extract_claims_deducts_cost_from_tracker(mocker):
    """extract_claims deducts cost from the shared tracker."""
    from groundguard.core.claim_extractor import extract_claims
    from groundguard.models.internal import SharedCostTracker
    from groundguard.models.result import Source

    mocker.patch(
        "groundguard.core.claim_extractor._completion_with_backoff",
        return_value=_mock_claims_response(["Claim A."]),
    )
    mocker.patch("litellm.completion_cost", return_value=0.05)

    tracker = SharedCostTracker(max_spend=10.0)
    src = Source(source_id="s1", content="Claim A.")
    extract_claims("Claim A.", [src], model="gpt-4o-mini", cost_tracker=tracker)

    assert tracker.total_cost_usd == 0.05


def test_extract_claims_raises_on_budget_exceeded(mocker):
    """extract_claims raises VerificationCostExceededError if budget exceeded."""
    from groundguard.core.claim_extractor import extract_claims
    from groundguard.models.internal import SharedCostTracker
    from groundguard.models.result import Source
    from groundguard.exceptions import VerificationCostExceededError

    mocker.patch(
        "groundguard.core.claim_extractor._completion_with_backoff",
        return_value=_mock_claims_response(["Claim A."]),
    )
    mocker.patch("litellm.completion_cost", return_value=0.05)

    tracker = SharedCostTracker(max_spend=0.001)
    src = Source(source_id="s1", content="Claim A.")

    with pytest.raises(VerificationCostExceededError):
        extract_claims("Claim A.", [src], model="gpt-4o-mini", cost_tracker=tracker)


async def test_extract_claims_async_deducts_cost_from_tracker(mocker):
    from unittest.mock import AsyncMock
    from groundguard.core.claim_extractor import extract_claims_async
    from groundguard.models.internal import SharedCostTracker
    from groundguard.models.result import Source

    mocker.patch(
        "groundguard.core.claim_extractor._acompletion_with_backoff",
        new_callable=AsyncMock,
        return_value=_mock_claims_response(["Claim A."]),
    )
    mocker.patch("litellm.completion_cost", return_value=0.05)

    tracker = SharedCostTracker(max_spend=10.0)
    src = Source(source_id="s1", content="Claim A.")
    await extract_claims_async("Claim A.", [src], model="gpt-4o-mini", cost_tracker=tracker)

    assert tracker.total_cost_usd == 0.05


async def test_extract_claims_async_raises_on_budget_exceeded(mocker):
    from unittest.mock import AsyncMock
    from groundguard.core.claim_extractor import extract_claims_async
    from groundguard.models.internal import SharedCostTracker
    from groundguard.models.result import Source
    from groundguard.exceptions import VerificationCostExceededError

    mocker.patch(
        "groundguard.core.claim_extractor._acompletion_with_backoff",
        new_callable=AsyncMock,
        return_value=_mock_claims_response(["Claim A."]),
    )
    mocker.patch("litellm.completion_cost", return_value=0.05)

    tracker = SharedCostTracker(max_spend=0.001)
    src = Source(source_id="s1", content="Claim A.")

    with pytest.raises(VerificationCostExceededError):
        await extract_claims_async("Claim A.", [src], model="gpt-4o-mini", cost_tracker=tracker)


def test_extract_claims_hallucinated_valid_json_returns_list(mocker):
    """extract_claims accepts any claim content in the JSON list and returns it."""
    from groundguard.core.claim_extractor import extract_claims
    from groundguard.models.result import Source

    mocker.patch(
        "groundguard.core.claim_extractor._completion_with_backoff",
        return_value=_mock_claims_response(["hallucinated claim"]),
    )
    src = Source(source_id="s1", content="Claim A.")
    claims = extract_claims("Claim A.", [src], model="gpt-4o-mini")

    assert claims == ["hallucinated claim"]


def test_extract_claims_empty_list_in_response(mocker):
    """extract_claims handles an empty claims list response without error."""
    from groundguard.core.claim_extractor import extract_claims
    from groundguard.models.result import Source

    mocker.patch(
        "groundguard.core.claim_extractor._completion_with_backoff",
        return_value=_mock_claims_response([]),
    )
    src = Source(source_id="s1", content="Claim A.")
    claims = extract_claims("Claim A.", [src], model="gpt-4o-mini")

    assert claims == []


def test_extract_claims_extra_json_keys_ignored(mocker):
    """extract_claims ignores extra keys in the JSON response."""
    from groundguard.core.claim_extractor import extract_claims
    from groundguard.models.result import Source

    msg = MagicMock()
    msg.content = json.dumps({"claims": ["Claim A."], "unexpected_key": 42})
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]

    mocker.patch(
        "groundguard.core.claim_extractor._completion_with_backoff",
        return_value=resp,
    )
    src = Source(source_id="s1", content="Claim A.")
    claims = extract_claims("Claim A.", [src], model="gpt-4o-mini")

    assert claims == ["Claim A."]

