from uuid import uuid4

import pymupdf
import pytest

from raglab.core.exceptions import DocumentParsingError, DocumentValidationError
from raglab.core.schemas import DocumentInput, DocumentStatus
from raglab.ingestion.parsers import PyMuPDFParser
from raglab.ingestion.validation import PdfUploadValidator


def make_pdf(*, blank_second_page: bool = False) -> bytes:
    pdf = pymupdf.open()
    first_page = pdf.new_page()
    first_page.insert_text((72, 72), "METHODS", fontsize=18)
    first_page.insert_text((72, 110), "The wearable IMU sampled motion at 100 Hz.", fontsize=11)
    if blank_second_page:
        pdf.new_page()
    pdf.set_metadata(
        {
            "title": "Wearable Sensor Study",
            "author": "Ada Engineer; Ben Researcher",
            "creationDate": "D:20250102000000Z",
        }
    )
    content = pdf.tobytes()
    pdf.close()
    return content


@pytest.mark.asyncio
async def test_parser_extracts_pages_metadata_and_headings() -> None:
    parser = PyMuPDFParser(PdfUploadValidator(max_size_bytes=1_000_000), max_pages=10)
    document_input = DocumentInput(
        file_name="wearable-study.pdf",
        content=make_pdf(blank_second_page=True),
        collection_id=uuid4(),
    )

    parsed = await parser.parse(document_input)

    assert parsed.document.display_title == "Wearable Sensor Study"
    assert parsed.document.authors == ("Ada Engineer", "Ben Researcher")
    assert parsed.document.publication_date is not None
    assert parsed.document.publication_date.isoformat() == "2025-01-02"
    assert parsed.document.page_count == 2
    assert parsed.document.status is DocumentStatus.PROCESSING
    assert len(parsed.pages) == 1
    assert parsed.pages[0].page_number == 1
    assert parsed.pages[0].section_headings[0].text == "METHODS"
    assert parsed.warnings == ("page 2 contained no extractable text",)


@pytest.mark.asyncio
async def test_parser_document_identity_is_deterministic_within_collection() -> None:
    parser = PyMuPDFParser(PdfUploadValidator(max_size_bytes=1_000_000), max_pages=10)
    collection_id = uuid4()
    document = DocumentInput(
        file_name="wearable-study.pdf",
        content=make_pdf(),
        collection_id=collection_id,
    )

    first = await parser.parse(document)
    second = await parser.parse(document)

    assert first.document.document_id == second.document.document_id


@pytest.mark.asyncio
async def test_parser_rejects_structurally_invalid_pdf() -> None:
    parser = PyMuPDFParser(PdfUploadValidator(max_size_bytes=1_000), max_pages=10)

    with pytest.raises(DocumentParsingError, match="invalid or unsupported"):
        await parser.parse(
            DocumentInput(
                file_name="broken.pdf",
                content=b"%PDF-this-is-not-valid",
                collection_id=uuid4(),
            )
        )


@pytest.mark.asyncio
async def test_parser_enforces_page_limit() -> None:
    parser = PyMuPDFParser(PdfUploadValidator(max_size_bytes=1_000_000), max_pages=1)

    with pytest.raises(DocumentValidationError, match="page processing limit"):
        await parser.parse(
            DocumentInput(
                file_name="long.pdf",
                content=make_pdf(blank_second_page=True),
                collection_id=uuid4(),
            )
        )


@pytest.mark.asyncio
async def test_parser_rejects_pdf_without_extractable_text() -> None:
    pdf = pymupdf.open()
    pdf.new_page()
    content = pdf.tobytes()
    pdf.close()
    parser = PyMuPDFParser(PdfUploadValidator(max_size_bytes=1_000_000), max_pages=10)

    with pytest.raises(DocumentParsingError, match="no extractable text"):
        await parser.parse(
            DocumentInput(file_name="scan.pdf", content=content, collection_id=uuid4())
        )
