"""Full-output verification example.

Prerequisites:
    pip install groundguard

Runs small, hardcoded examples with no external dependencies.
Configure provider credentials through environment variables if your selected
groundguard model requires them.
"""
from __future__ import annotations

import os

import groundguard
from groundguard.models.result import GroundingResult, Source


def run_full_output_example() -> GroundingResult:
    """Verify a multi-sentence answer with task context."""
    analysis = (
        "GroundGuard verifies generated answers against developer-provided "
        "sources. It is designed as a deterministic assertion layer rather "
        "than a retrieval pipeline."
    )
    sources = [
        Source(
            content=(
                "GroundGuard is a Python middleware library that verifies "
                "AI-generated text is factually grounded in developer-provided "
                "source documents. It is not a RAG pipeline."
            ),
            source_id="project_summary",
        )
    ]
    model = os.getenv("GROUNDGUARD_MODEL", "gpt-4o-mini")
    result = groundguard.verify_analysis(
        analysis,
        sources,
        model=model,
        context="Technical documentation verification for the groundguard library.",
    )
    print(
        f"Grounded: {result.is_grounded} | "
        f"Score: {result.score:.2f} | "
        f"Status: {result.status}"
    )
    return result


def run_large_context_example() -> GroundingResult:
    """Verify against a long document without BM25 chunking (auto_chunk=False).

    Recommended for large-context models (Gemini 1.5 Pro, Claude 3.5+) when the
    source is long and negating clauses in low-scoring chunks must not be dropped
    by BM25 ranking.
    """
    analysis = (
        "The contract requires payment within 30 days of invoice receipt. "
        "Late payments incur a 1.5% monthly penalty."
    )
    long_source = Source(
        content=(
            "PAYMENT TERMS: All invoices are due and payable within thirty (30) "
            "days of the invoice date. Notwithstanding the foregoing, any payment "
            "received after the due date shall accrue interest at a rate of one and "
            "one-half percent (1.5%) per month on the outstanding balance. "
            "This provision was amended by Addendum C and supersedes all prior "
            "payment schedules. The parties agree that electronic fund transfer "
            "is the preferred payment method."
        ),
        source_id="contract_payment_terms",
    )
    model = os.getenv("GROUNDGUARD_MODEL", "gpt-4o-mini")
    result = groundguard.verify_analysis(
        analysis,
        sources=[long_source],
        model=model,
        auto_chunk=False,  # pass the full document — no BM25 chunking
        context="Contract clause verification for payment terms.",
    )
    print(
        f"Grounded: {result.is_grounded} | "
        f"Score: {result.score:.2f} | "
        f"Status: {result.status}"
    )
    return result


if __name__ == "__main__":
    print("=== Standard example (with context) ===")
    run_full_output_example()
    print()
    print("=== Large-context example (auto_chunk=False) ===")
    run_large_context_example()
