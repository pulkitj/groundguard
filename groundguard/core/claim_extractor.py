"""Claim extraction from free-form text using LLM."""
from __future__ import annotations
import secrets
from typing import TYPE_CHECKING

import pydantic
import litellm

from groundguard.exceptions import ParseError
from groundguard.tiers.tier3_evaluation import _completion_with_backoff, _acompletion_with_backoff

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


def _extract_json_substring(text: str) -> str:
    start_brace = text.find('{')
    start_bracket = text.find('[')
    
    if start_brace == -1 and start_bracket == -1:
        return text.strip()
        
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start = start_brace
        end = text.rfind('}')
    else:
        start = start_bracket
        end = text.rfind(']')
        
    if end != -1 and end > start:
        return text[start:end+1].strip()
    return text.strip()


def extract_claims(
    text: str,
    sources: list,
    model: str,
    context: str | None = None,
    max_spend: float = float("inf"),
    api_base: str | None = None,
    cost_tracker = None,
    audit: bool = False,
) -> list[str]:
    boundary = secrets.token_hex(6)
    sources_block = "\n".join(f"- {s.source_id}: {s.content[:200]}" for s in sources)
    context_block = f"Context: {context}\n\n" if context else ""
    prompt = context_block + CLAIM_EXTRACTION_PROMPT.format(
        boundary=boundary, text=text, sources_block=sources_block
    )
    if audit:
        prompt += "\n\nInclude a brief reasoning about the claims inside XML tags <audit_report>...</audit_report> before outputting the JSON."
    for attempt in range(2):
        try:
            response = _completion_with_backoff(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **({"api_base": api_base} if api_base else {}),
            )
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = 0.0
            if cost_tracker is not None:
                cost_tracker.add_cost(cost)

            content = response.choices[0].message.content
            cleaned_content = _extract_json_substring(content)
            parsed = _ClaimList.model_validate_json(cleaned_content)
            return parsed.claims
        except (pydantic.ValidationError, ValueError):
            if attempt == 1:
                raise ParseError("claim extraction failed after 2 attempts")
    return []


async def extract_claims_async(
    text: str,
    sources: list,
    model: str,
    context: str | None = None,
    max_spend: float = float("inf"),
    api_base: str | None = None,
    cost_tracker = None,
    audit: bool = False,
) -> list[str]:
    boundary = secrets.token_hex(6)
    sources_block = "\n".join(f"- {s.source_id}: {s.content[:200]}" for s in sources)
    context_block = f"Context: {context}\n\n" if context else ""
    prompt = context_block + CLAIM_EXTRACTION_PROMPT.format(
        boundary=boundary, text=text, sources_block=sources_block
    )
    if audit:
        prompt += "\n\nInclude a brief reasoning about the claims inside XML tags <audit_report>...</audit_report> before outputting the JSON."
    for attempt in range(2):
        try:
            response = await _acompletion_with_backoff(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **({"api_base": api_base} if api_base else {}),
            )
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = 0.0
            if cost_tracker is not None:
                cost_tracker.add_cost(cost)

            content = response.choices[0].message.content
            cleaned_content = _extract_json_substring(content)
            parsed = _ClaimList.model_validate_json(cleaned_content)
            return parsed.claims
        except (pydantic.ValidationError, ValueError):
            if attempt == 1:
                raise ParseError("claim extraction async failed after 2 attempts")
    return []
