"""Shared constants for the groundguard library."""
from __future__ import annotations

import litellm

# FIX-02: Unified transient error tuple used by both tier3_evaluation.py and verifier.py.
# Previously tier3_evaluation.py was missing litellm.exceptions.Timeout, meaning timeouts
# were not retried by the backoff loop — only caught and re-raised by the orchestrator.
TRANSIENT_LITELLM_ERRORS = (
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
)
