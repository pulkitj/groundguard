"""Claim extraction from free-form text using LLM."""
from __future__ import annotations
import secrets
from typing import TYPE_CHECKING

import pydantic

from groundguard.exceptions import ParseError
from groundguard.tiers.tier3_evaluation import _completion_with_backoff

if TYPE_CHECKING:
    from groundguard.models.result import Source


CLAIM_EXTRACTION_PROMPT = """Extract all distinct factual claims from the text below.
Return JSON with key "claims" containing a list of strings.
Each string is one atomic, self-contained factual claim.

Text (boundary: {boundary}):
{text}

Sources provided:
{sources_block}

Return only JSON. Example: {{"claims": ["claim 1", "claim 2"]}}"""


class _ClaimList(pydantic.BaseModel):
    claims: list[str]


def extract_claims(
    text: str,
    sources: list,
    model: str,
    max_spend: float = float("inf"),
    api_base: str | None = None,
) -> list[str]:
    boundary = secrets.token_hex(6)
    sources_block = "\n".join(f"- {s.source_id}: {s.content[:200]}" for s in sources)
    prompt = CLAIM_EXTRACTION_PROMPT.format(
        boundary=boundary, text=text, sources_block=sources_block
    )
    for attempt in range(2):
        try:
            response = _completion_with_backoff(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **({"api_base": api_base} if api_base else {}),
            )
            content = response.choices[0].message.content
            parsed = _ClaimList.model_validate_json(content)
            return parsed.claims
        except (pydantic.ValidationError, ValueError):
            if attempt == 1:
                raise ParseError("claim extraction failed after 2 attempts")
    return []


async def extract_claims_async(
    text: str,
    sources: list,
    model: str,
    max_spend: float = float("inf"),
    api_base: str | None = None,
) -> list[str]:
    # Calls _completion_with_backoff synchronously inside async context.
    # The test patches _completion_with_backoff — this ensures the mock is intercepted.
    boundary = secrets.token_hex(6)
    sources_block = "\n".join(f"- {s.source_id}: {s.content[:200]}" for s in sources)
    prompt = CLAIM_EXTRACTION_PROMPT.format(
        boundary=boundary, text=text, sources_block=sources_block
    )
    for attempt in range(2):
        try:
            response = _completion_with_backoff(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **({"api_base": api_base} if api_base else {}),
            )
            content = response.choices[0].message.content
            parsed = _ClaimList.model_validate_json(content)
            return parsed.claims
        except (pydantic.ValidationError, ValueError):
            if attempt == 1:
                raise ParseError("claim extraction async failed after 2 attempts")
    return []
