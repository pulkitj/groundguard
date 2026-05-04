"""Tests for auto-chunker — TDD items #5, #14, #17."""
import pytest
from groundguard.loaders.chunker import Chunk, chunk_sources, wrap_as_chunks
from groundguard.models.result import Source
from groundguard.models.internal import VerificationContext


def _make_ctx(content: str, auto_chunk: bool = True,
              chunk_size: int = 500, chunk_overlap: int = 50,
              max_source_tokens: int = 8000) -> VerificationContext:
    return VerificationContext(
        claim="test claim",
        original_sources=[Source(content=content, source_id="test.pdf")],
        model="gpt-4o-mini",
        auto_chunk=auto_chunk,
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=chunk_overlap,
        max_source_tokens=max_source_tokens,
    )


def test_small_source_produces_single_chunk():
    """TDD #5: Source below max_source_tokens produces exactly one chunk."""
    short_text = "This is a short source document."
    ctx = _make_ctx(short_text)
    chunks = chunk_sources(ctx)
    assert len(chunks) == 1
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(short_text)
    assert chunks[0].source_id == "test.pdf"


def test_overlap_equal_to_size_raises_value_error():
    """TDD #14: chunk_overlap_tokens == chunk_size_tokens must raise ValueError."""
    ctx = _make_ctx("Some content", chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError, match="chunk_overlap_tokens"):
        chunk_sources(ctx)


def test_overlap_greater_than_size_raises_value_error():
    """TDD #14: chunk_overlap_tokens > chunk_size_tokens must raise ValueError."""
    ctx = _make_ctx("Some content", chunk_size=100, chunk_overlap=101)
    with pytest.raises(ValueError, match="chunk_overlap_tokens"):
        chunk_sources(ctx)


def test_sliding_window_char_offsets_correct():
    """TDD #17: For every returned Chunk, source.content[char_start:char_end] == chunk.text_content."""
    # Create a long source that exceeds max_source_tokens (set low for test)
    # 1 token ≈ 4 chars, so max_source_tokens=10 means threshold at 40 chars
    long_text = "word " * 50  # ~250 chars, will exceed threshold of 40 chars
    ctx = VerificationContext(
        claim="test",
        original_sources=[Source(content=long_text, source_id="long.pdf")],
        model="gpt-4o-mini",
        auto_chunk=True,
        chunk_size_tokens=20,
        chunk_overlap_tokens=5,
        max_source_tokens=10,  # Forces sliding window: len(text)//4 = ~62 > 10
    )
    chunks = chunk_sources(ctx)
    assert len(chunks) > 1, "Long source should produce multiple chunks"
    for chunk in chunks:
        extracted = long_text[chunk.char_start:chunk.char_end]
        assert extracted == chunk.text_content, (
            f"Chunk text mismatch: content[{chunk.char_start}:{chunk.char_end}] = "
            f"{repr(extracted)!r} != chunk.text_content = {repr(chunk.text_content)!r}"
        )


def test_auto_chunk_false_wraps_as_single_chunks():
    """auto_chunk=False wraps each source as one chunk regardless of size."""
    content = "word " * 100
    ctx = _make_ctx(content, auto_chunk=False)
    chunks = chunk_sources(ctx)
    assert len(chunks) == 1
    assert chunks[0].text_content == content


def test_wrap_as_chunks_preserves_lineage():
    """wrap_as_chunks assigns correct source_id."""
    sources = [
        Source(content="Doc A content", source_id="a.pdf"),
        Source(content="Doc B content", source_id="b.pdf"),
    ]
    chunks = wrap_as_chunks(sources)
    assert len(chunks) == 2
    assert chunks[0].source_id == "a.pdf"
    assert chunks[1].source_id == "b.pdf"
