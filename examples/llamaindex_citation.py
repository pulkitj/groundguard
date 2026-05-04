"""LlamaIndex CitationQueryEngine verification example.

Prerequisites:
    pip install groundguard llama-index

Configure any LlamaIndex LLM or embedding provider credentials through that
provider's environment variables.
"""
from __future__ import annotations

from collections.abc import Callable

from groundguard import verify_analysis
from groundguard.models.result import Source


def _require_llama_index():
    try:
        from llama_index.core.query_engine import CitationQueryEngine
    except ImportError as exc:
        raise ImportError("This example requires LlamaIndex: pip install llama-index") from exc
    return CitationQueryEngine


def _sources_from_nodes(response) -> list[Source]:
    sources = []
    for index, node_with_score in enumerate(getattr(response, "source_nodes", []) or [], start=1):
        node = getattr(node_with_score, "node", node_with_score)
        get_text = getattr(node, "get_text", None)
        content = get_text() if callable(get_text) else str(node)
        node_id = getattr(node, "node_id", f"citation_node_{index}")
        sources.append(Source(content=content, source_id=str(node_id)))
    return sources


def build_verified_citation_engine(
    index, groundguard_model: str = "gpt-4o-mini"
) -> Callable:
    """Build a callable that verifies CitationQueryEngine responses."""
    CitationQueryEngine = _require_llama_index()
    query_engine = CitationQueryEngine.from_args(index)

    def ask(question: str):
        response = query_engine.query(question)
        answer = getattr(response, "response", str(response))
        sources = _sources_from_nodes(response)
        verification = verify_analysis(answer, sources, model=groundguard_model)
        return {"answer": answer, "verification": verification, "sources": sources}

    return ask


if __name__ == "__main__":
    print("Build a LlamaIndex index, then pass it to build_verified_citation_engine(index).")
