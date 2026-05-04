"""Legal document loaders and preprocessing utilities."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from groundguard.models.result import Source

_LEGAL_STOPWORDS = frozenset({
    "This", "That", "These", "Those", "The", "Each", "Any", "All",
    "Such", "Said", "When", "Where", "Which", "With", "Upon", "After",
    "Before", "During", "Within", "Without", "Unless", "Until",
})


class PassiveVoiceNormalizer:
    """Converts passive legal voice to active voice."""

    _PATTERNS = [
        # "X shall not be exceeded by Y" -> "Y shall not exceed X"
        (r'^(.+?)\s+shall not be exceeded by\s+(.+?)\.?\s*$',
         lambda m: f"{m.group(2)} shall not exceed {m.group(1)}"),
        # "X shall be determined by Y" -> "Y shall determine X"
        (r'^(.+?)\s+shall be determined by\s+(.+?)\.?\s*$',
         lambda m: f"{m.group(2)} shall determine {m.group(1)}"),
        # "Term is defined as D in Schedule N" -> "Schedule N defines Term as D"
        (r'^(.+?)\s+is defined as\s+(.+?)\s+in\s+(.+?)\.?\s*$',
         lambda m: f"{m.group(3)} defines {m.group(1)} as {m.group(2)}"),
    ]

    def normalize(self, text: str) -> str:
        try:
            for pattern, rewriter in self._PATTERNS:
                m = re.match(pattern, text.strip(), re.IGNORECASE)
                if m:
                    return rewriter(m)
            return text
        except Exception:
            return text  # fail-safe


class TermRegistry:
    def __init__(self):
        self._terms: dict[str, Source] = {}

    @classmethod
    def from_sources(cls, sources: list[Source]) -> "TermRegistry":
        registry = cls()
        for src in sources:
            if src.source_type == "legal_definition":
                m = re.search(r'"([^"]+)"\s+means\s+', src.content)
                if m:
                    registry.register(m.group(1), src)
        return registry

    def resolve(self, term: str) -> Source | None:
        return self._terms.get(term.lower())

    def register(self, term: str, source: Source) -> None:
        self._terms[term.lower()] = source

    def known_terms(self) -> list[str]:
        return list(self._terms.keys())


def load_legal_docx(path: str) -> list[Source]:
    from groundguard.loaders.structured import load_docx
    sources = load_docx(path, split_by="heading", source_type="legal_clause")
    for src in sources:
        if '"' in src.content and " means " in src.content:
            src.source_type = "legal_definition"
    return sources


def load_financial_table(path: str) -> list[Source]:
    from groundguard.loaders.structured import load_xlsx
    return load_xlsx(path, source_type="financial_table_row")


@dataclass
class StructuredClaimUnit:
    main_proposition: str
    original_text: str
    modal_operator: str | None = None
    subordinate_modifiers: list[str] = field(default_factory=list)
    defined_terms_referenced: list[str] = field(default_factory=list)


def decompose_clause(text: str) -> StructuredClaimUnit:
    normalizer = PassiveVoiceNormalizer()
    active_text = normalizer.normalize(text)

    modal_patterns = [
        (r'\bshall not\b', "shall not"),
        (r'\bshall\b', "shall"),
        (r'\bmay not\b', "may not"),
        (r'\bmay\b', "may"),
        (r'\bmust\b', "must"),
    ]
    modal_operator = None
    for pattern, modal in modal_patterns:
        if re.search(pattern, active_text, re.IGNORECASE):
            modal_operator = modal
            break

    subordinate_modifiers = []
    for m in re.finditer(
        r'(?:as defined in|pursuant to|subject to|in accordance with|as set forth in)\s+[^,\.]+',
        text, re.IGNORECASE
    ):
        subordinate_modifiers.append(m.group(0).strip())

    defined_terms_referenced = []
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
        term = m.group(1)
        if term not in _LEGAL_STOPWORDS and len(term) >= 4:
            if term not in defined_terms_referenced:
                defined_terms_referenced.append(term)

    return StructuredClaimUnit(
        main_proposition=active_text,
        original_text=text,
        modal_operator=modal_operator,
        subordinate_modifiers=subordinate_modifiers,
        defined_terms_referenced=defined_terms_referenced,
    )
