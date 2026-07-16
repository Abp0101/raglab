from datetime import UTC, datetime
from uuid import uuid4

import pytest

from raglab.chunking import (
    FixedTokenChunker,
    ParentChildChunker,
    SectionAwareChunker,
    create_chunker,
)
from raglab.core.schemas import (
    ChunkingConfig,
    ChunkingStrategy,
    Document,
    DocumentPage,
    DocumentStatus,
    ParsedDocument,
    SectionHeading,
)


def make_document(text: str, headings: tuple[SectionHeading, ...] = ()) -> ParsedDocument:
    return ParsedDocument(
        document=Document(
            document_id=uuid4(),
            collection_id=uuid4(),
            file_name="benchmark.txt",
            display_title="Benchmark",
            uploaded_at=datetime.now(UTC),
            file_type="text/plain",
            content_hash="1" * 64,
            page_count=1,
            status=DocumentStatus.PROCESSING,
        ),
        pages=(DocumentPage(page_number=1, text=text, section_headings=headings),),
        parser_name="test",
    )


def test_fixed_token_chunker_uses_token_windows_and_exact_offsets() -> None:
    text = " ".join(f"token{index}" for index in range(75))
    document = make_document(text)
    config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED_TOKEN,
        chunk_size=32,
        chunk_overlap=8,
    )

    chunks = FixedTokenChunker().chunk(document, config)

    assert [chunk.token_count for chunk in chunks] == [32, 32, 27]
    assert chunks[0].text.split()[24] == chunks[1].text.split()[0]
    for chunk in chunks:
        assert chunk.text_span is not None
        assert text[chunk.text_span.start : chunk.text_span.end] == chunk.text


def test_section_aware_chunker_never_crosses_detected_heading() -> None:
    methods = "METHODS\n" + "Calibration evidence. " * 8
    results = "RESULTS\n" + "Validation evidence. " * 8
    text = methods + "\n\n" + results
    results_start = text.index("RESULTS")
    document = make_document(
        text,
        (
            SectionHeading(text="METHODS", start=0),
            SectionHeading(text="RESULTS", start=results_start),
        ),
    )
    config = ChunkingConfig(
        strategy=ChunkingStrategy.SECTION_AWARE,
        chunk_size=80,
        chunk_overlap=10,
    )

    chunks = SectionAwareChunker().chunk(document, config)

    assert {chunk.metadata.section_heading for chunk in chunks} == {"METHODS", "RESULTS"}
    assert all(
        chunk.text_span is not None
        and not (chunk.text_span.start < results_start < chunk.text_span.end)
        for chunk in chunks
    )


def test_parent_child_chunker_links_children_inside_parent_spans() -> None:
    text = "METHODS\n\n" + "The sensor captured a calibrated measurement. " * 15
    document = make_document(text, (SectionHeading(text="METHODS", start=0),))
    config = ChunkingConfig(
        strategy=ChunkingStrategy.PARENT_CHILD,
        chunk_size=64,
        chunk_overlap=8,
        parent_chunk_size=180,
        parent_chunk_overlap=20,
    )

    chunks = ParentChildChunker().chunk(document, config)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    children = [chunk for chunk in chunks if chunk.metadata.parent_chunk_id is not None]
    parent_ids = {child.metadata.parent_chunk_id for child in children}

    assert children
    assert parent_ids <= chunks_by_id.keys()
    assert [chunk.metadata.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    for child in children:
        parent = chunks_by_id[child.metadata.parent_chunk_id]
        assert child.text_span is not None
        assert parent.text_span is not None
        assert parent.text_span.start <= child.text_span.start
        assert parent.text_span.end >= child.text_span.end


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_registry_creates_the_requested_strategy(strategy: ChunkingStrategy) -> None:
    assert create_chunker(strategy).name == strategy.value


def test_parent_child_config_requires_larger_parent() -> None:
    with pytest.raises(ValueError, match="parent chunk size"):
        ChunkingConfig(
            strategy=ChunkingStrategy.PARENT_CHILD,
            chunk_size=128,
            parent_chunk_size=128,
        )
