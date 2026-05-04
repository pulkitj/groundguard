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
    from groundguard.core.claim_extractor import extract_claims_async
    from groundguard.models.result import Source
    mocker.patch(
        "groundguard.core.claim_extractor._completion_with_backoff",
        return_value=_mock_claims_response(["x"]),
    )
    src = Source(source_id="s1", content="x")
    result = asyncio.get_event_loop().run_until_complete(
        extract_claims_async("x", [src], model="gpt-4o-mini")
    )
    assert isinstance(result, list)
