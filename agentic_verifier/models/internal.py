"""Internal pipeline models."""
from __future__ import annotations
import secrets
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agentic_verifier.loaders.chunker import Chunk
    from agentic_verifier.models.result import Source

from agentic_verifier.exceptions import VerificationCostExceededError


class SharedCostTracker:
    """
    Thread-safe accumulator for LLM API costs across batch verification calls.

    The spend cap is a SOFT cap — it is enforced AFTER an LLM call completes.
    The triggering call has already been billed. All subsequent calls are blocked.
    """
    def __init__(self, max_spend: float = 0.50):
        self.max_spend = max_spend
        self.total_cost_usd = 0.0
        self._lock = threading.Lock()

    def add_cost(self, cost: float | None) -> float:
        """
        Thread-safe cost accumulation. Cap check runs INSIDE the lock to
        prevent race conditions between lock release and comparison.

        Args:
            cost: The cost to add. None is treated as 0.0 (local models).

        Returns:
            The new total cost.

        Raises:
            VerificationCostExceededError: If total exceeds max_spend.
        """
        with self._lock:
            self.total_cost_usd += (cost or 0.0)
            new_total = self.total_cost_usd
            if new_total > self.max_spend:
                raise VerificationCostExceededError(
                    f"Spend cap of ${self.max_spend:.2f} exceeded "
                    f"(current total: ${new_total:.4f}). "
                    "Note: this is a soft cap — the triggering call has already completed."
                )
            return new_total


@dataclass
class ClassifiedAtom:
    """A single atomic claim sentence with its Extractive/Inferential classification."""
    claim_text: str
    claim_type: Literal["Extractive", "Inferential"]


@dataclass
class VerificationContext:
    """
    Immutable-ish per-call context object passed through the verification pipeline.
    All configuration parameters are set at construction time.
    """
    claim: str
    original_sources: list[Source]
    model: str

    auto_chunk: bool = True
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    max_source_tokens: int = 8000
    tier1_min_similarity: float = 0.90

    tier2_lexical_threshold: float = 0.85
    tier2_low_score_floor: float = 0.01
    top_k_chunks: int = 5

    agent_provided_evidence: str | None = None

    cost_tracker: SharedCostTracker = field(
        default_factory=lambda: SharedCostTracker(max_spend=float('inf'))
    )

    _boundary_id: str = field(default_factory=lambda: secrets.token_hex(6))

    tier0_atoms: list[ClassifiedAtom] = field(default_factory=list)


class RoutingDecision:
    """
    Tier 2 routing decision constants.

    SKIP_LLM_HIGH_CONFIDENCE: BM25 score above lexical threshold — no LLM needed.
    ESCALATE_TO_LLM: BM25 score in mid range — send top-k chunks to LLM.
    ESCALATE_ALL_LOW_SCORE: All BM25 scores at/below floor — send all chunks.
    """
    SKIP_LLM_HIGH_CONFIDENCE = "lexical_pass"
    ESCALATE_TO_LLM = "semantic_review"
    ESCALATE_ALL_LOW_SCORE = "all_low_escalate"


@dataclass
class Tier2Result:
    """Output of the Tier 2 routing step."""
    decision: str  # One of RoutingDecision constants
    top_k_chunks: list[Chunk] = field(default_factory=list)
    highest_score: float = 0.0


@dataclass
class ClaimInput:
    """Input for a single item in verify_batch()."""
    claim: str
    sources: list[Source]
    agent_provided_evidence: str | None = None
    model: str | None = None
