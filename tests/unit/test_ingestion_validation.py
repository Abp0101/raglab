from uuid import uuid4

import pytest

from raglab.core.exceptions import DocumentValidationError
from raglab.core.schemas import DocumentInput
from raglab.ingestion.validation import PdfUploadValidator


def make_input(file_name: str = "study.pdf", content: bytes = b"%PDF-valid") -> DocumentInput:
    return DocumentInput(file_name=file_name, content=content, collection_id=uuid4())


@pytest.mark.parametrize("file_name", ["../study.pdf", "folder/study.pdf", "folder\\study.pdf"])
def test_validator_rejects_path_components(file_name: str) -> None:
    with pytest.raises(DocumentValidationError, match="path components"):
        PdfUploadValidator(max_size_bytes=100).validate(make_input(file_name=file_name))


def test_validator_rejects_unsupported_extension() -> None:
    with pytest.raises(DocumentValidationError, match=r"only \.pdf"):
        PdfUploadValidator(max_size_bytes=100).validate(make_input(file_name="study.txt"))


def test_validator_rejects_oversized_file() -> None:
    with pytest.raises(DocumentValidationError, match="upload limit"):
        PdfUploadValidator(max_size_bytes=5).validate(make_input())


def test_validator_rejects_spoofed_pdf_extension() -> None:
    with pytest.raises(DocumentValidationError, match="PDF signature"):
        PdfUploadValidator(max_size_bytes=100).validate(make_input(content=b"not a pdf"))


def test_validator_accepts_case_insensitive_pdf_extension() -> None:
    PdfUploadValidator(max_size_bytes=100).validate(make_input(file_name="STUDY.PDF"))
