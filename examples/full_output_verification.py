"""Full-output verification example.

Prerequisites:
    pip install groundguard

Runs a small, hardcoded multi-sentence example with no external dependencies.
Configure provider credentials through environment variables if your selected
groundguard model requires them.
"""
from __future__ import annotations

import os

import groundguard
from groundguard.models.result import GroundingResult, Source


def run_full_output_example() -> GroundingResult:
    """Verify a multi-sentence answer and print a short result summary."""
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
    result = groundguard.verify_analysis(analysis, sources, model=model)
    print(
        f"Grounded: {result.is_grounded} | "
        f"Score: {result.score:.2f} | "
        f"Status: {result.status}"
    )
    return result


if __name__ == "__main__":
    run_full_output_example()
