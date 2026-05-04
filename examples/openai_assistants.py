"""OpenAI Assistants thread verification example.

Prerequisites:
    pip install groundguard openai

Set OPENAI_API_KEY in your environment before creating an OpenAI client.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable

from groundguard import verify
from groundguard.models.result import Source


def _require_openai() -> None:
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise ImportError("This example requires OpenAI: pip install openai") from exc


def _normalise_sources(sources: list) -> list[Source]:
    normalised = []
    for index, source in enumerate(sources, start=1):
        if isinstance(source, Source):
            normalised.append(source)
        else:
            normalised.append(Source(content=str(source), source_id=f"assistant_source_{index}"))
    return normalised


def _message_text(message) -> str:
    parts = []
    for content in getattr(message, "content", []) or []:
        text = getattr(content, "text", None)
        value = getattr(text, "value", None)
        if value:
            parts.append(value)
    return "\n".join(parts)


def build_verified_assistant_thread(
    client, assistant_id: str, sources: list, groundguard_model: str = "gpt-4o-mini"
) -> Callable:
    """Build a callable that verifies the final assistant message in a thread."""
    _require_openai()
    ground_sources = _normalise_sources(sources)

    def ask(user_message: str):
        thread = client.beta.threads.create(
            messages=[{"role": "user", "content": user_message}]
        )
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )
        while run.status in {"queued", "in_progress", "requires_action"}:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status != "completed":
            raise RuntimeError(f"Assistant run ended with status: {run.status}")

        messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")
        final_message = next(
            message for message in messages.data if getattr(message, "role", "") == "assistant"
        )
        answer = _message_text(final_message)
        verification = verify(answer, ground_sources, model=groundguard_model)
        return {"answer": answer, "verification": verification, "sources": ground_sources}

    return ask


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY before creating an OpenAI client.")
    print("Pass your OpenAI client and assistant ID to build_verified_assistant_thread().")
