"""Validation for untrusted PDF uploads before parser invocation."""

from dataclasses import dataclass

from raglab.core.exceptions import DocumentValidationError
from raglab.core.schemas import DocumentInput

PDF_SIGNATURE = b"%PDF-"


@dataclass(frozen=True, slots=True)
class PdfUploadValidator:
    """Enforce bounded, path-safe, signature-checked PDF input."""

    max_size_bytes: int

    def validate(self, document: DocumentInput) -> None:
        """Raise a safe validation error when an upload is not an allowed PDF."""
        file_name = document.file_name
        if "/" in file_name or "\\" in file_name or "\x00" in file_name:
            raise DocumentValidationError("file name must not contain path components")
        if not file_name.casefold().endswith(".pdf"):
            raise DocumentValidationError("only .pdf files are currently supported")
        if len(document.content) > self.max_size_bytes:
            raise DocumentValidationError(
                f"file exceeds the {self.max_size_bytes}-byte upload limit"
            )
        if not document.content.startswith(PDF_SIGNATURE):
            raise DocumentValidationError("file content does not have a PDF signature")
