"""LangChain RetrievalQA verification example.

Prerequisites:
    pip install groundguard langchain langchain-community

Provide your own LangChain LLM and retriever. Any provider credentials should be
configured with environment variables used by that provider.
"""
from __future__ import annotations

from collections.abc import Callable

from groundguard import verify
from groundguard.models.result import Source


def _require_langchain():
    try:
        from langchain.chains import RetrievalQA
    except ImportError as exc:
        raise ImportError(
            "This example requires optional dependencies: "
            "pip install langchain langchain-community"
        ) from exc
    return RetrievalQA


def _sources_from_documents(documents) -> list[Source]:
    sources = []
    for index, document in enumerate(documents or [], start=1):
        content = getattr(document, "page_content", str(document))
        metadata = getattr(document, "metadata", {}) or {}
        source_id = metadata.get("source") or metadata.get("id") or f"retrieved_doc_{index}"
        sources.append(Source(content=content, source_id=str(source_id)))
    return sources


def build_verified_retrieval_chain(
    llm, retriever, groundguard_model: str = "gpt-4o-mini"
) -> Callable:
    """Build a callable that answers with RetrievalQA and verifies the answer."""
    RetrievalQA = _require_langchain()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    def ask(question: str):
        response = chain.invoke({"query": question})
        answer = response.get("result", "")
        sources = _sources_from_documents(response.get("source_documents", []))
        verification = verify(answer, sources, model=groundguard_model)
        return {"answer": answer, "verification": verification, "sources": sources}

    return ask


if __name__ == "__main__":
    print(
        "Create your LangChain LLM and retriever, then pass them to "
        "build_verified_retrieval_chain(llm, retriever)."
    )
