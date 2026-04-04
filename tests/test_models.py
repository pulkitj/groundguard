"""Test data models — TDD item #15."""
import math
from agentic_verifier.models.internal import VerificationContext
from agentic_verifier.models.result import Source


def test_verification_context_default_cost_tracker_is_inf():
    """TDD #15: VerificationContext without explicit cost_tracker uses max_spend=inf."""
    ctx = VerificationContext(
        claim="Revenue was $5M.",
        original_sources=[Source(content="Revenue was $5M.", source_id="doc1.pdf")],
        model="gpt-4o-mini",
    )
    assert ctx.cost_tracker.max_spend == float('inf')


def test_verification_context_boundary_id_is_12_hex_chars():
    """Boundary ID must be 48-bit (12 hex chars from secrets.token_hex(6))."""
    ctx = VerificationContext(
        claim="test",
        original_sources=[Source(content="test", source_id="s1")],
        model="gpt-4o-mini",
    )
    assert len(ctx._boundary_id) == 12
    assert all(c in "0123456789abcdef" for c in ctx._boundary_id)


def test_two_contexts_have_unique_boundary_ids():
    """Each VerificationContext generates a unique _boundary_id."""
    src = [Source(content="test", source_id="s1")]
    ctx1 = VerificationContext(claim="test", original_sources=src, model="gpt-4o-mini")
    ctx2 = VerificationContext(claim="test", original_sources=src, model="gpt-4o-mini")
    assert ctx1._boundary_id != ctx2._boundary_id
