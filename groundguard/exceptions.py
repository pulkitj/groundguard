"""Custom exceptions for the groundguard library."""


class HallucinatedEvidenceError(Exception):
    """Raised when agent-provided evidence cannot be found in source documents.

    This occurs during the verification pipeline when a claim or citation
    supplied by the agent does not correspond to any content present in the
    retrieved or provided source material.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class VerificationCostExceededError(Exception):
    """Raised when the verification spend cap is exceeded.

    This is raised when the cumulative token or monetary cost of running the
    verification pipeline surpasses the configured budget limit, preventing
    further LLM calls from being made.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class VerificationFailedError(Exception):
    """Raised when the verification pipeline encounters an unrecoverable error.

    Examples include LiteLLM connection failures, upstream API errors, or any
    other condition that prevents the pipeline from completing normally.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ParseError(Exception):
    """Raised when a Tier 3 LLM response cannot be parsed as valid JSON after retry.

    This exception is thrown when the structured JSON output expected from the
    Tier 3 verification LLM is malformed or missing, even after the configured
    number of retry attempts have been exhausted.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvariantError(Exception):
    """Raised when a citation invariant is violated.

    Attributes:
        cost_usd: LLM spend incurred before the invariant was detected.
                  Populated by verify()/averify() before re-raising.
    """

    cost_usd: float

    def __init__(self, message: str, cost_usd: float = 0.0) -> None:
        super().__init__(message)
        self.cost_usd = cost_usd
