from datetime import UTC, datetime
from uuid import uuid4

from raglab.chunking import RecursiveCharacterChunker
from raglab.core.schemas import (
    ChunkingConfig,
    Document,
    DocumentPage,
    DocumentStatus,
    ParsedDocument,
    SectionHeading,
)


def make_parsed_document(text: str) -> ParsedDocument:
    document = Document(
        document_id=uuid4(),
        collection_id=uuid4(),
        file_name="sensors.pdf",
        display_title="Sensors",
        uploaded_at=datetime.now(UTC),
        file_type="application/pdf",
        content_hash="b" * 64,
        page_count=1,
        status=DocumentStatus.PROCESSING,
    )
    return ParsedDocument(
        document=document,
        pages=(
            DocumentPage(
                page_number=1,
                text=text,
                section_headings=(SectionHeading(text="METHODS", start=0),),
            ),
        ),
        parser_name="test-parser",
    )


def test_chunker_preserves_offsets_overlap_and_provenance() -> None:
    text = "METHODS\n\n" + "The sensor measured knee motion repeatedly. " * 8
    parsed = make_parsed_document(text)
    config = ChunkingConfig(chunk_size=100, chunk_overlap=20)

    chunks = RecursiveCharacterChunker().chunk(parsed, config)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= config.chunk_size for chunk in chunks)
    assert all(chunk.metadata.page_number == 1 for chunk in chunks)
    assert all(chunk.metadata.section_heading == "METHODS" for chunk in chunks)
    assert [chunk.metadata.chunk_index for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.text_span is not None
        assert text[chunk.text_span.start : chunk.text_span.end] == chunk.text


def test_chunk_ids_are_deterministic() -> None:
    parsed = make_parsed_document("METHODS\n\n" + "Evidence sentence. " * 10)
    chunker = RecursiveCharacterChunker()
    config = ChunkingConfig(chunk_size=80, chunk_overlap=10)

    first = chunker.chunk(parsed, config)
    second = chunker.chunk(parsed, config)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
