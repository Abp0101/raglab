"""Secure, framework-independent document ingestion."""

from raglab.ingestion.deletion import CoordinatedDocumentDeletionService
from raglab.ingestion.jobs import BackgroundIngestionManager
from raglab.ingestion.langchain_pipeline import LangChainIngestionPipeline
from raglab.ingestion.parsers import PyMuPDFParser
from raglab.ingestion.pipeline import DocumentIngestionPipeline
from raglab.ingestion.validation import PdfUploadValidator

__all__ = [
    "BackgroundIngestionManager",
    "CoordinatedDocumentDeletionService",
    "DocumentIngestionPipeline",
    "LangChainIngestionPipeline",
    "PdfUploadValidator",
    "PyMuPDFParser",
]
