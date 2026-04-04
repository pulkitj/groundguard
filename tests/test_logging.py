"""Logging tests for agentic-verifier — TDD items #21, #22, #23 (T-31)."""
from __future__ import annotations

import logging
import re
from unittest.mock import MagicMock

import pytest

from agentic_verifier.core.verifier import verify
from agentic_verifier.exceptions import ParseError
from agentic_verifier.loaders.chunker import Chunk
from agentic_verifier.models.internal import RoutingDecision
from agentic_verifier.models.result import Source, VerificationResult
from agentic_verifier.models.tier3 import (
    AtomicVerification,
    ConceptualCoverage,
    SourceAttribution,
    TextualEntailment,
    Tier3ResponseModel,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

CLAIM = "Revenue was $5M."
SOURCES = [Source(content="Revenue was $5M.", source_id="doc.pdf")]

_CHUNK = Chunk(
    parent_source_id="doc.pdf",
    text_content="Revenue was $5M.",
    char_start=0,
    char_end=16,
)


def _valid_t3() -> Tier3ResponseModel:
    return Tier3ResponseModel(
        textual_entailment=TextualEntailment(label="Entailment", probability=0.95),
        conceptual_coverage=ConceptualCoverage(
            percentage=90.0, covered_concepts=["revenue"], missing_concepts=[]
        ),
        factual_consistency_score=90.0,
        verifications=[
            AtomicVerification(
                claim_text="Revenue was $5M.",
                status="VERIFIED",
                source_id="doc.pdf",
            )
        ],
        source_attributions=[SourceAttribution(source_id="doc.pdf", role="Supporting")],
        overall_verdict="Verified.",
    )


def _verified_result() -> VerificationResult:
    return VerificationResult(
        is_valid=True,
        overall_verdict="Verified",
        verification_method="tier2_lexical",
        atomic_claims=[],
        factual_consistency_score=1.0,
        sources_used=["doc.pdf"],
        rationale="Match",
        offending_claim=None,
        status="VERIFIED",
        total_cost_usd=0.0,
    )


def _patch_base(mocker):
    """Mock out classifier and chunker so pipeline focuses on tier2+."""
    mocker.patch(
        "agentic_verifier.core.verifier.classifier.parse_and_classify",
        return_value=[],
    )
    mocker.patch(
        "agentic_verifier.core.verifier.chunker.chunk_sources",
        return_value=[_CHUNK],
    )


def _skip_llm_tier2_result():
    from agentic_verifier.models.internal import Tier2Result as _T2R  # noqa: PLC0415
    return _T2R(
        decision=RoutingDecision.SKIP_LLM_HIGH_CONFIDENCE,
        top_k_chunks=[_CHUNK],
        highest_score=0.99,
    )


def _escalate_tier2_result():
    from agentic_verifier.models.internal import Tier2Result as _T2R  # noqa: PLC0415
    return _T2R(
        decision=RoutingDecision.ESCALATE_TO_LLM,
        top_k_chunks=[_CHUNK],
        highest_score=0.5,
    )


# ---------------------------------------------------------------------------
# #21 — Logger is silent by default (no handlers attached by library)
# ---------------------------------------------------------------------------

def test_logger_has_no_handlers_by_default():
    """#21: The 'agentic_verifier' logger must never attach its own handlers.

    The library is middleware — handler configuration is the host application's
    responsibility. We assert the logger has an empty handler list so that
    records propagate (or are silently discarded) based on the host's log config.
    """
    lib_logger = logging.getLogger("agentic_verifier")
    assert lib_logger.handlers == [], (
        "Library must not attach handlers to 'agentic_verifier' logger — "
        f"found: {lib_logger.handlers}"
    )


# ---------------------------------------------------------------------------
# #22 — _boundary_id appears in all DEBUG records; no sensitive content leaked
# ---------------------------------------------------------------------------

def test_boundary_id_in_all_records_and_no_sensitive_content(mocker, caplog):
    """#22: Every log record contains the call's boundary_id; claim/source text absent."""
    _patch_base(mocker)
    mocker.patch(
        "agentic_verifier.core.verifier.tier2_semantic.route_claim",
        return_value=_escalate_tier2_result(),
    )
    t3_model = _valid_t3()
    mocker.patch(
        "agentic_verifier.core.verifier.tier3_evaluation.evaluate",
        return_value=t3_model,
    )
    mocker.patch(
        "agentic_verifier.models.builder.ResultBuilder.build_llm_result",
        return_value=VerificationResult(
            is_valid=True,
            overall_verdict="Verified",
            verification_method="tier3_llm",
            atomic_claims=[],
            factual_consistency_score=0.9,
            sources_used=["doc.pdf"],
            rationale="LLM verified",
            offending_claim=None,
            status="VERIFIED",
            total_cost_usd=0.001,
        ),
    )

    with caplog.at_level(logging.DEBUG, logger="agentic_verifier"):
        verify(claim=CLAIM, sources=SOURCES)

    # Must have at least one log record (the INFO summary from verifier.py)
    assert len(caplog.records) >= 1, "Expected at least 1 log record from verify()"

    # Extract boundary_id from the first record that has a 12-hex-char token
    hex_pattern = re.compile(r"[0-9a-f]{12}")
    found_ids: set[str] = set()
    for record in caplog.records:
        matches = hex_pattern.findall(record.getMessage())
        found_ids.update(matches)

    # There must be exactly one unique boundary_id across all records
    assert len(found_ids) == 1, (
        f"Expected exactly 1 unique boundary_id across all log records, found: {found_ids}"
    )

    boundary_id = next(iter(found_ids))
    assert len(boundary_id) == 12
    assert all(c in "0123456789abcdef" for c in boundary_id)

    # Assert every record's message contains that boundary_id
    for record in caplog.records:
        assert boundary_id in record.getMessage(), (
            f"Log record missing boundary_id '{boundary_id}': {record.getMessage()!r}"
        )

    # Assert no record leaks raw claim text or source document text
    for record in caplog.records:
        msg = record.getMessage()
        assert CLAIM not in msg, (
            f"Log record contains sensitive claim text: {msg!r}"
        )
        assert SOURCES[0].content not in msg, (
            f"Log record contains sensitive source content: {msg!r}"
        )


# ---------------------------------------------------------------------------
# #23 — Log levels fire at correct tiers (parameterized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ["lexical_pass", "retry_then_succeed", "double_parse_fail"])
def test_log_levels_by_scenario(scenario, mocker, caplog):
    """#23: Correct log levels emitted per routing scenario."""
    _patch_base(mocker)

    if scenario == "lexical_pass":
        # (a) Tier 2 SKIP_LLM_HIGH_CONFIDENCE — should emit exactly 1 INFO, 0 WARNING, 0 ERROR
        mocker.patch(
            "agentic_verifier.core.verifier.tier2_semantic.route_claim",
            return_value=_skip_llm_tier2_result(),
        )
        mocker.patch(
            "agentic_verifier.models.builder.ResultBuilder.build_lexical_pass",
            return_value=_verified_result(),
        )

        with caplog.at_level(logging.DEBUG, logger="agentic_verifier"):
            verify(claim=CLAIM, sources=SOURCES)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]

        assert len(info_records) == 1, (
            f"Lexical pass: expected 1 INFO record, got {len(info_records)}: "
            f"{[r.getMessage() for r in info_records]}"
        )
        assert len(warning_records) == 0, (
            f"Lexical pass: expected 0 WARNING records, got {len(warning_records)}"
        )
        assert len(error_records) == 0, (
            f"Lexical pass: expected 0 ERROR records, got {len(error_records)}"
        )

    elif scenario == "retry_then_succeed":
        # (b) Tier 3 retries on first invalid JSON — WARNING "attempt 1/2" from tier3_evaluation
        mocker.patch(
            "agentic_verifier.core.verifier.tier2_semantic.route_claim",
            return_value=_escalate_tier2_result(),
        )

        valid_model = _valid_t3()
        call_count = [0]

        def mock_completion(**kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.choices[0].message.parsed = None
            if call_count[0] == 1:
                # First call: invalid JSON triggers ValidationError in parse_response
                mock_resp.choices[0].message.content = '{"invalid": "missing_required_fields"}'
            else:
                # Second call: return valid model via .parsed attribute
                mock_resp.choices[0].message.parsed = valid_model
            return mock_resp

        mocker.patch("litellm.completion", side_effect=mock_completion)
        mocker.patch("litellm.completion_cost", return_value=0.001)

        with caplog.at_level(logging.DEBUG, logger="agentic_verifier"):
            verify(claim=CLAIM, sources=SOURCES)

        assert call_count[0] == 2, "Expected exactly 2 litellm.completion calls (retry)"

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1, (
            f"Expected at least 1 WARNING record for retry, got: {[r.getMessage() for r in caplog.records]}"
        )
        # The retry warning from tier3_evaluation must mention "attempt 1/2"
        retry_warnings = [r for r in warning_records if "attempt 1/2" in r.getMessage()]
        assert len(retry_warnings) == 1, (
            f"Expected 1 WARNING containing 'attempt 1/2', found: "
            f"{[r.getMessage() for r in warning_records]}"
        )

    elif scenario == "double_parse_fail":
        # (c) evaluate() raises ParseError → verifier catches it → ERROR + INFO summary
        mocker.patch(
            "agentic_verifier.core.verifier.tier2_semantic.route_claim",
            return_value=_escalate_tier2_result(),
        )
        mocker.patch(
            "agentic_verifier.core.verifier.tier3_evaluation.evaluate",
            side_effect=ParseError("Mocked double parse failure"),
        )

        with caplog.at_level(logging.DEBUG, logger="agentic_verifier"):
            result = verify(claim=CLAIM, sources=SOURCES)

        assert result.status == "PARSE_ERROR", (
            f"Expected PARSE_ERROR result, got: {result.status}"
        )

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1, (
            f"Expected at least 1 ERROR record for ParseError scenario, "
            f"got: {[r.getMessage() for r in caplog.records]}"
        )

        # The ERROR summary emitted by verifier.py must mention PARSE_ERROR
        error_msgs = [r.getMessage() for r in error_records]
        assert any("PARSE_ERROR" in msg for msg in error_msgs), (
            f"Expected an ERROR record containing 'PARSE_ERROR', got: {error_msgs}"
        )
