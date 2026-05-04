"""LangChain integration — AgenticVerifierCallback."""
from __future__ import annotations

try:
    from langchain_core.documents import Document  # noqa: F401
except ImportError as e:
    raise ImportError(
        "langchain-core is required for the LangChain integration. "
        "Install it with: pip install agentic-verifier[langchain]"
    ) from e

from groundguard.core.verifier import verify
from groundguard.exceptions import VerificationFailedError
from groundguard.models.result import Source


class AgenticVerifierCallback:
    """
    LangChain callback that intercepts chain output and runs agentic verification.
    Supports RetrievalQA and LCEL RAG chains only (chains that produce source_documents).

    Usage:
        callback = AgenticVerifierCallback(model="gpt-4o-mini")
        chain.invoke({"query": "..."}, callbacks=[callback])
    """

    def __init__(self, model: str = "gpt-4o-mini", **verify_kwargs):
        self.model = model
        self.verify_kwargs = verify_kwargs

    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        """
        Called after the chain completes. Extracts source_documents, maps them
        to Source objects, and runs verify() with the chain's answer as the claim.

        Raises:
            VerificationFailedError: If the chain output does not contain
                'source_documents' (unsupported chain type).
        """
        if "source_documents" not in outputs:
            raise VerificationFailedError(
                "AgenticVerifierCallback requires 'source_documents' in chain outputs. "
                "Only RetrievalQA and LCEL RAG chains are supported. "
                "Unsupported chain type: 'source_documents' key is missing."
            )

        if "result" not in outputs:
            raise VerificationFailedError(
                "AgenticVerifierCallback requires 'result' in chain outputs. "
                "Only RetrievalQA and LCEL RAG chains (that emit 'result') are supported. "
                "Unsupported chain type: 'result' key is missing."
            )

        sources = [
            Source(
                content=doc.page_content,
                source_id=doc.metadata.get("source", f"source_{i}"),
            )
            for i, doc in enumerate(outputs["source_documents"])
        ]

        claim = outputs["result"]

        verify(claim=claim, sources=sources, model=self.model, **self.verify_kwargs)
