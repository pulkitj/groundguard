"""Result builder."""
from __future__ import annotations
from typing import TYPE_CHECKING

from agentic_verifier.models.result import VerificationResult

if TYPE_CHECKING:
    from agentic_verifier.models.internal import VerificationContext
    from agentic_verifier.models.tier3 import Tier3ResponseModel
    from agentic_verifier.loaders.chunker import Chunk


class ResultBuilder:
    """Builds VerificationResult from pipeline outputs."""

    @staticmethod
    def build_lexical_pass(ctx: "VerificationContext", matched_chunks: "list[Chunk]") -> "VerificationResult":
        """Build result for Tier 2 high-confidence lexical pass (no LLM call)."""
        raise NotImplementedError("Phase 8: T-25")

    @staticmethod
    def build_llm_result(ctx: "VerificationContext", t3_model: "Tier3ResponseModel", method: str) -> "VerificationResult":
        """Build result from Tier 3 LLM evaluation output."""
        raise NotImplementedError("Phase 8: T-25")
