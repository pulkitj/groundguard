"""AWS Bedrock RetrieveAndGenerate verification example.

Prerequisites:
    pip install groundguard boto3

Configure AWS credentials with the standard AWS environment variables or
credential files before creating a Bedrock Agent Runtime client.
"""
from __future__ import annotations

import os
from collections.abc import Callable

from groundguard import verify
from groundguard.models.result import Source


def _require_boto3() -> None:
    try:
        import boto3  # noqa: F401
    except ImportError as exc:
        raise ImportError("This example requires boto3: pip install boto3") from exc


def _sources_from_citations(response) -> list[Source]:
    sources = []
    citations = response.get("citations", []) if isinstance(response, dict) else []
    for index, citation in enumerate(citations, start=1):
        references = citation.get("retrievedReferences", [])
        for ref_index, reference in enumerate(references, start=1):
            content = reference.get("content", {}).get("text", "")
            location = reference.get("location", {})
            source_id = (
                location.get("s3Location", {}).get("uri")
                or location.get("webLocation", {}).get("url")
                or f"bedrock_reference_{index}_{ref_index}"
            )
            if content:
                sources.append(Source(content=content, source_id=str(source_id)))
    return sources


def build_verified_bedrock_rag(
    bedrock_client, knowledge_base_id: str, groundguard_model: str = "gpt-4o-mini"
) -> Callable:
    """Build a callable that verifies Bedrock RetrieveAndGenerate responses."""
    _require_boto3()

    def ask(question: str):
        response = bedrock_client.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": knowledge_base_id,
                    "modelArn": os.getenv("BEDROCK_RAG_MODEL_ARN", ""),
                },
            },
        )
        answer = response.get("output", {}).get("text", "")
        sources = _sources_from_citations(response)
        verification = verify(answer, sources, model=groundguard_model)
        return {"answer": answer, "verification": verification, "sources": sources}

    return ask


if __name__ == "__main__":
    if not os.getenv("AWS_REGION"):
        print("Set AWS_REGION and AWS credentials before creating the Bedrock client.")
    print("Pass your bedrock-agent-runtime client to build_verified_bedrock_rag().")
