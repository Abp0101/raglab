"""PDF parsing with page provenance and conservative metadata extraction."""

import asyncio
import hashlib
import statistics
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

import pymupdf

from raglab.core.exceptions import DocumentParsingError, DocumentValidationError
from raglab.core.schemas import (
    Document,
    DocumentInput,
    DocumentPage,
    DocumentStatus,
    ParsedDocument,
)
from raglab.ingestion.metadata import detect_section_headings, normalize_page_text
from raglab.ingestion.validation import PdfUploadValidator


class _PdfPage(Protocol):
    def get_text(self, option: str, *, sort: bool) -> Any: ...


class _PdfDocument(Protocol):
    needs_pass: bool
    page_count: int
    metadata: Mapping[str, Any] | None

    def load_page(self, page_id: int) -> _PdfPage: ...

    def close(self) -> None: ...


class PyMuPDFParser:
    """Parse text PDFs from memory without creating user-controlled paths."""

    name = "pymupdf"

    def __init__(self, validator: PdfUploadValidator, *, max_pages: int) -> None:
        self._validator = validator
        self._max_pages = max_pages

    async def parse(self, document: DocumentInput) -> ParsedDocument:
        """Parse in a worker thread because PyMuPDF extraction is synchronous."""
        self._validator.validate(document)
        return await asyncio.to_thread(self._parse_sync, document)

    def _parse_sync(self, document_input: DocumentInput) -> ParsedDocument:
        try:
            pdf = cast(
                _PdfDocument,
                pymupdf.open(  # type: ignore[no-untyped-call]
                    stream=document_input.content, filetype="pdf"
                ),
            )
        except (pymupdf.FileDataError, RuntimeError) as error:
            raise DocumentParsingError("PDF structure is invalid or unsupported") from error

        try:
            if pdf.needs_pass:
                raise DocumentValidationError("encrypted PDFs are not supported")
            if pdf.page_count == 0:
                raise DocumentParsingError("PDF contains no pages")
            if pdf.page_count > self._max_pages:
                raise DocumentValidationError(
                    f"PDF exceeds the {self._max_pages}-page processing limit"
                )

            metadata = pdf.metadata or {}
            pages, warnings = self._extract_pages(pdf)
            if not pages:
                raise DocumentParsingError(
                    "PDF contains no extractable text; scanned-document OCR is not yet supported"
                )
            parsed_document = self._build_document(document_input, metadata, pdf.page_count)
            return ParsedDocument(
                document=parsed_document,
                pages=tuple(pages),
                parser_name=self.name,
                warnings=tuple(warnings),
            )
        finally:
            pdf.close()

    def _extract_pages(self, pdf: _PdfDocument) -> tuple[list[DocumentPage], list[str]]:
        pages: list[DocumentPage] = []
        warnings: list[str] = []
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            raw_text = page.get_text("text", sort=True)
            text = normalize_page_text(raw_text)
            if not text:
                warnings.append(f"page {page_index + 1} contained no extractable text")
                continue
            font_candidates = self._font_heading_candidates(page)
            pages.append(
                DocumentPage(
                    page_number=page_index + 1,
                    text=text,
                    section_headings=detect_section_headings(text, font_candidates),
                )
            )
        return pages, warnings

    @staticmethod
    def _font_heading_candidates(page: _PdfPage) -> tuple[str, ...]:
        page_dict: Mapping[str, Any] = page.get_text("dict", sort=True)
        spans = [
            span
            for block in page_dict.get("blocks", [])
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]
        if not spans:
            return ()
        body_size = statistics.median(float(span["size"]) for span in spans)
        return tuple(
            str(span["text"]).strip()
            for span in spans
            if float(span["size"]) >= body_size * 1.2 and len(str(span["text"]).split()) <= 14
        )

    @staticmethod
    def _build_document(
        document_input: DocumentInput,
        metadata: Mapping[str, Any],
        page_count: int,
    ) -> Document:
        title = document_input.display_title or str(metadata.get("title") or "").strip()
        if not title:
            title = Path(document_input.file_name).stem.replace("_", " ").replace("-", " ")
        author = str(metadata.get("author") or "").strip()
        authors = tuple(
            part.strip() for part in author.replace(";", ",").split(",") if part.strip()
        )
        return Document(
            document_id=uuid4(),
            collection_id=document_input.collection_id,
            file_name=document_input.file_name,
            display_title=title,
            authors=authors,
            source_url=document_input.source_url,
            uploaded_at=datetime.now(UTC),
            publication_date=_parse_pdf_date(str(metadata.get("creationDate") or "")),
            file_type="application/pdf",
            content_hash=hashlib.sha256(document_input.content).hexdigest(),
            page_count=page_count,
            status=DocumentStatus.PROCESSING,
        )


def _parse_pdf_date(value: str) -> date | None:
    """Parse the stable date portion of a PDF `D:YYYYMMDD...` timestamp."""
    normalized = value.removeprefix("D:")
    if len(normalized) < 4 or not normalized[:4].isdigit():
        return None
    month = int(normalized[4:6]) if len(normalized) >= 6 and normalized[4:6].isdigit() else 1
    day = int(normalized[6:8]) if len(normalized) >= 8 and normalized[6:8].isdigit() else 1
    try:
        return date(int(normalized[:4]), month, day)
    except ValueError:
        return None
