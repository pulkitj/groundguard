"""Cohere chat RAG verification example.

Prerequisites:
    pip install groundguard cohere

Set COHERE_API_KEY in your environment when constructing a real Cohere client.
"""
from __future__ import annotations

import os
from collections.abc import Callable

from groundguard import verify
from groundguard.models.result import Source


def _require_cohere() -> None:
    try:
        import cohere  # noqa: F401
    except ImportError as exc:
        raise ImportError("This example requires Cohere: pip install cohere") from exc


def _document_sources(documents: list[str]) -> list[Source]:
    return [
        Source(content=document, source_id=f"cohere_doc_{index}")
        for index, document in enumerate(documents, start=1)
    ]


def build_verified_cohere_rag(
    co_client, documents: list[str], groundguard_model: str = "gpt-4o-mini"
) -> Callable:
    """Build a callable that verifies Cohere chat responses against documents."""
    _require_cohere()
    sources = _document_sources(documents)
    cohere_documents = [{"text": document} for document in documents]

    def ask(question: str):
        response = co_client.chat(message=question, documents=cohere_documents)
        answer = getattr(response, "text", "")
        verification = verify(answer, sources, model=groundguard_model)
        return {"answer": answer, "verification": verification, "sources": sources}

    return ask


if __name__ == "__main__":
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("Set COHERE_API_KEY and pass a Cohere client to build_verified_cohere_rag().")
    else:
        print("Create cohere.Client(os.getenv('COHERE_API_KEY')) and call the builder.")
