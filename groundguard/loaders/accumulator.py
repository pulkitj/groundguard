"""Accumulators for multi-turn grounding and multi-step source collection."""
from __future__ import annotations

from groundguard.models.result import GroundingResult, Source


class GroundingAccumulator:
    """Accumulates verify_answer/verify_analysis results over a multi-turn conversation."""

    def __init__(self) -> None:
        self._results: list[GroundingResult] = []

    def add(self, result: GroundingResult) -> None:
        self._results.append(result)

    @property
    def overall_score(self) -> float:
        if not self._results:
            return 0.0
        return sum(r.score for r in self._results) / len(self._results)

    @property
    def is_grounded(self) -> bool:
        return all(r.is_grounded for r in self._results)

    def reset(self) -> None:
        self._results.clear()


class SourceAccumulator:
    """
    Accumulates source documents across multi-step agent pipelines.

    Deduplicates by source_id. Supports marking sources as LLM-derived
    (sets derived_from_llm=True). When populate_boundary_context=True and
    consecutive sources share the same source_id prefix (split on `::`),
    auto-populates prev_context/next_context between adjacent chunks.
    """

    def __init__(self) -> None:
        self._sources: list[Source] = []
        self._seen_ids: set[str] = set()

    def add(
        self,
        sources: list[Source],
        step: str | None = None,
        mark_llm_derived: bool = False,
        populate_boundary_context: bool = False,
    ) -> "SourceAccumulator":
        new_sources: list[Source] = []
        for src in sources:
            if src.source_id in self._seen_ids:
                continue
            self._seen_ids.add(src.source_id)
            if mark_llm_derived:
                src = Source(
                    content=src.content,
                    source_id=src.source_id,
                    source_type=src.source_type,
                    page_hint=src.page_hint,
                    section_id=src.section_id,
                    derived_from_llm=True,
                    original_document_id=src.original_document_id,
                    as_of_date=src.as_of_date,
                    prev_context=src.prev_context,
                    next_context=src.next_context,
                )
            new_sources.append(src)

        if populate_boundary_context and new_sources:
            all_after = self._sources + new_sources
            for idx_in_all, s in enumerate(all_after):
                if idx_in_all == 0:
                    continue
                prev = all_after[idx_in_all - 1]
                if prev.source_id.split("::")[0] != s.source_id.split("::")[0]:
                    continue
                # last sentence of prev -> prev_context of current
                parts = [p.strip() for p in prev.content.split(".") if p.strip()]
                last_sent = (parts[-1] + ".") if parts else prev.content
                s.prev_context = last_sent
                # first sentence of current -> next_context of prev
                first_sent = (s.content.split(".")[0] + ".") if "." in s.content else s.content
                prev.next_context = first_sent

        self._sources.extend(new_sources)
        return self

    def all_sources(self) -> list[Source]:
        return list(self._sources)

    def has_llm_derived(self) -> bool:
        return any(s.derived_from_llm for s in self._sources)

    def clear(self) -> "SourceAccumulator":
        self._sources.clear()
        self._seen_ids.clear()
        return self

    def __len__(self) -> int:
        return len(self._sources)
