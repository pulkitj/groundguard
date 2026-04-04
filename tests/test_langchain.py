"""Tests for AgenticVerifierCallback LangChain integration — TDD T-32.

All tests are marked @pytest.mark.langchain and run zero real LLM calls.
langchain-core is NOT required to be installed; the tests mock sys.modules.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Inject fake langchain_core into sys.modules so the integration module can
# import it without langchain-core being installed.
# ---------------------------------------------------------------------------

_langchain_core_mock = MagicMock()
_langchain_core_documents_mock = MagicMock()
sys.modules.setdefault("langchain_core", _langchain_core_mock)
sys.modules.setdefault("langchain_core.documents", _langchain_core_documents_mock)

# Now safe to import the integration (it will find langchain_core in sys.modules)
from agentic_verifier.models.result import Source, VerificationResult  # noqa: E402
from agentic_verifier.exceptions import VerificationFailedError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_verification_result() -> VerificationResult:
    return VerificationResult(
        is_valid=True,
        overall_verdict="Verified",
        verification_method="tier2_lexical",
        atomic_claims=[],
        factual_consistency_score=1.0,
        sources_used=["report.pdf"],
        rationale="Match",
        offending_claim=None,
        status="VERIFIED",
        total_cost_usd=0.0,
    )


def _make_mock_document(page_content: str, metadata: dict) -> MagicMock:
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = metadata
    return doc


# ---------------------------------------------------------------------------
# Test 1 — RetrievalQA chain: source_documents extracted and verify() called
# ---------------------------------------------------------------------------

@pytest.mark.langchain
def test_on_chain_end_extracts_source_documents_as_source_objects(mocker):
    """T-32 #1a: Documents from outputs['source_documents'] are converted to Source objects."""
    mock_verify = mocker.patch(
        "agentic_verifier.integrations.langchain.verify",
        return_value=_make_verification_result(),
    )

    mock_doc = _make_mock_document(
        page_content="Revenue was $5M.",
        metadata={"source": "report.pdf"},
    )

    outputs = {
        "result": "Revenue was $5M.",
        "source_documents": [mock_doc],
    }

    from agentic_verifier.integrations.langchain import AgenticVerifierCallback

    callback = AgenticVerifierCallback(model="gpt-4o-mini")
    callback.on_chain_end(outputs)

    mock_verify.assert_called_once()
    _, call_kwargs = mock_verify.call_args
    sources_passed = call_kwargs.get("sources") or mock_verify.call_args[0][1]

    assert len(sources_passed) == 1
    assert isinstance(sources_passed[0], Source)
    assert sources_passed[0].content == "Revenue was $5M."


@pytest.mark.langchain
def test_on_chain_end_maps_document_metadata_source_to_source_id(mocker):
    """T-32 #1b: Document metadata['source'] becomes Source.source_id."""
    mock_verify = mocker.patch(
        "agentic_verifier.integrations.langchain.verify",
        return_value=_make_verification_result(),
    )

    mock_doc = _make_mock_document(
        page_content="Revenue was $5M.",
        metadata={"source": "report.pdf"},
    )

    outputs = {
        "result": "Revenue was $5M.",
        "source_documents": [mock_doc],
    }

    from agentic_verifier.integrations.langchain import AgenticVerifierCallback

    callback = AgenticVerifierCallback(model="gpt-4o-mini")
    callback.on_chain_end(outputs)

    _, call_kwargs = mock_verify.call_args
    sources_passed = call_kwargs.get("sources") or mock_verify.call_args[0][1]

    assert sources_passed[0].source_id == "report.pdf"


@pytest.mark.langchain
def test_on_chain_end_passes_result_string_as_claim_to_verify(mocker):
    """T-32 #1c: outputs['result'] is passed as the claim argument to verify()."""
    mock_verify = mocker.patch(
        "agentic_verifier.integrations.langchain.verify",
        return_value=_make_verification_result(),
    )

    mock_doc = _make_mock_document(
        page_content="Revenue was $5M.",
        metadata={"source": "report.pdf"},
    )

    outputs = {
        "result": "Revenue was $5M.",
        "source_documents": [mock_doc],
    }

    from agentic_verifier.integrations.langchain import AgenticVerifierCallback

    callback = AgenticVerifierCallback(model="gpt-4o-mini")
    callback.on_chain_end(outputs)

    mock_verify.assert_called_once()
    call_args, call_kwargs = mock_verify.call_args
    claim_passed = call_kwargs.get("claim") or call_args[0]

    assert claim_passed == "Revenue was $5M."


@pytest.mark.langchain
def test_on_chain_end_handles_multiple_source_documents(mocker):
    """T-32 #1d: All documents in source_documents are converted to Source objects."""
    mock_verify = mocker.patch(
        "agentic_verifier.integrations.langchain.verify",
        return_value=_make_verification_result(),
    )

    doc1 = _make_mock_document("Revenue was $5M.", {"source": "report.pdf"})
    doc2 = _make_mock_document("Net profit was $1M.", {"source": "financials.pdf"})

    outputs = {
        "result": "Revenue was $5M and net profit was $1M.",
        "source_documents": [doc1, doc2],
    }

    from agentic_verifier.integrations.langchain import AgenticVerifierCallback

    callback = AgenticVerifierCallback(model="gpt-4o-mini")
    callback.on_chain_end(outputs)

    _, call_kwargs = mock_verify.call_args
    sources_passed = call_kwargs.get("sources") or mock_verify.call_args[0][1]

    assert len(sources_passed) == 2
    assert all(isinstance(s, Source) for s in sources_passed)


@pytest.mark.langchain
def test_on_chain_end_calls_verify_exactly_once(mocker):
    """T-32 #1e: verify() is called exactly once per on_chain_end invocation."""
    mock_verify = mocker.patch(
        "agentic_verifier.integrations.langchain.verify",
        return_value=_make_verification_result(),
    )

    mock_doc = _make_mock_document("Revenue was $5M.", {"source": "report.pdf"})

    outputs = {
        "result": "Revenue was $5M.",
        "source_documents": [mock_doc],
    }

    from agentic_verifier.integrations.langchain import AgenticVerifierCallback

    callback = AgenticVerifierCallback(model="gpt-4o-mini")
    callback.on_chain_end(outputs)

    assert mock_verify.call_count == 1


# ---------------------------------------------------------------------------
# Test 2 — Unsupported chain: missing source_documents raises descriptive error
# ---------------------------------------------------------------------------

@pytest.mark.langchain
def test_on_chain_end_raises_verification_failed_error_when_no_source_documents(mocker):
    """T-32 #2a: Missing source_documents key raises VerificationFailedError, not AttributeError."""
    mocker.patch(
        "agentic_verifier.integrations.langchain.verify",
        return_value=_make_verification_result(),
    )

    outputs = {
        "result": "Revenue was $5M.",
        # No 'source_documents' key — simulates an unsupported chain type
    }

    from agentic_verifier.integrations.langchain import AgenticVerifierCallback

    callback = AgenticVerifierCallback(model="gpt-4o-mini")

    with pytest.raises(VerificationFailedError):
        callback.on_chain_end(outputs)


@pytest.mark.langchain
def test_on_chain_end_error_message_is_descriptive_for_missing_source_documents(mocker):
    """T-32 #2b: The VerificationFailedError message mentions source_documents or chain type."""
    mocker.patch(
        "agentic_verifier.integrations.langchain.verify",
        return_value=_make_verification_result(),
    )

    outputs = {"result": "Revenue was $5M."}

    from agentic_verifier.integrations.langchain import AgenticVerifierCallback

    callback = AgenticVerifierCallback(model="gpt-4o-mini")

    with pytest.raises(VerificationFailedError) as exc_info:
        callback.on_chain_end(outputs)

    error_message = str(exc_info.value).lower()
    # Message must mention source_documents, chain type, or a helpful hint — not a raw attr error
    assert any(
        keyword in error_message
        for keyword in ("source_documents", "chain", "unsupported", "retrieval")
    ), f"Error message was not descriptive enough: {exc_info.value}"


@pytest.mark.langchain
def test_on_chain_end_does_not_raise_attribute_error_for_missing_source_documents(mocker):
    """T-32 #2c: The exception type is NOT AttributeError — it must be VerificationFailedError."""
    mocker.patch(
        "agentic_verifier.integrations.langchain.verify",
        return_value=_make_verification_result(),
    )

    outputs = {"result": "Revenue was $5M."}

    from agentic_verifier.integrations.langchain import AgenticVerifierCallback

    callback = AgenticVerifierCallback(model="gpt-4o-mini")

    try:
        callback.on_chain_end(outputs)
    except AttributeError as exc:
        pytest.fail(
            f"Raised AttributeError instead of VerificationFailedError: {exc}"
        )
    except VerificationFailedError:
        pass  # expected
    except Exception as exc:
        pytest.fail(f"Raised unexpected exception type {type(exc).__name__}: {exc}")
