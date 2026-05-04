"""Auto-chunker and Source wrapper."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groundguard.models.internal import VerificationContext
    from groundguard.models.result import Source


@dataclass
class Chunk:
    """A chunk of text from a source document, with character offset tracking."""
    parent_source_id: str
    text_content: str
    char_start: int
    char_end: int


def wrap_as_chunks(sources: list[Source]) -> list[Chunk]:
    """Wraps each Source as a single Chunk (used when auto_chunk=False)."""
    return [
        Chunk(
            parent_source_id=s.source_id,
            text_content=s.content,
            char_start=0,
            char_end=len(s.content),
        )
        for s in sources
    ]


def chunk_sources(ctx: VerificationContext) -> list[Chunk]:
    """
    Splits sources into chunks for downstream tier processing.

    - auto_chunk=False: wraps each source as one Chunk
    - auto_chunk=True + small source (< max_source_tokens): one Chunk per source
    - auto_chunk=True + large source (>= max_source_tokens): sliding window

    Token estimate: 1 token ≈ 4 characters.
    """
    if not ctx.auto_chunk:
        return wrap_as_chunks(ctx.original_sources)

    if ctx.chunk_overlap_tokens >= ctx.chunk_size_tokens:
        raise ValueError(
            f"chunk_overlap_tokens ({ctx.chunk_overlap_tokens}) must be strictly less "
            f"than chunk_size_tokens ({ctx.chunk_size_tokens})."
        )

    chunks: list[Chunk] = []
    for source in ctx.original_sources:
        approx_tokens = len(source.content) // 4
        if approx_tokens <= ctx.max_source_tokens:
            chunks.append(Chunk(
                parent_source_id=source.source_id,
                text_content=source.content,
                char_start=0,
                char_end=len(source.content),
            ))
        else:
            chunks.extend(_sliding_window_chunks(source, ctx))

    return chunks


def _sliding_window_chunks(source: Source, ctx: VerificationContext) -> list[Chunk]:
    """
    Splits a large source into overlapping token windows, tracking char offsets.

    Uses whitespace tokenization for simplicity (compatible with BM25Okapi).
    """
    content = source.content
    # Simple whitespace tokenization — matches BM25Okapi's expected input
    words = content.split()
    if not words:
        return [Chunk(
            parent_source_id=source.source_id,
            text_content=content,
            char_start=0,
            char_end=len(content),
        )]

    size = ctx.chunk_size_tokens
    overlap = ctx.chunk_overlap_tokens
    stride = size - overlap

    # Build a list of (word, char_start_in_content) pairs
    word_positions: list[tuple[str, int]] = []
    pos = 0
    for word in words:
        idx = content.find(word, pos)
        word_positions.append((word, idx))
        pos = idx + len(word)

    result: list[Chunk] = []
    start_idx = 0
    while start_idx < len(word_positions):
        end_idx = min(start_idx + size, len(word_positions))

        char_start = word_positions[start_idx][1]
        last_word, last_word_pos = word_positions[end_idx - 1]
        char_end = last_word_pos + len(last_word)

        text_content = content[char_start:char_end]

        result.append(Chunk(
            parent_source_id=source.source_id,
            text_content=text_content,
            char_start=char_start,
            char_end=char_end,
        ))

        if end_idx >= len(word_positions):
            break
        start_idx += stride

    return result
