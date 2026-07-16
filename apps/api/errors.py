"""Safe HTTP translations for expected domain and provider failures."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from raglab.core.exceptions import (
    CollectionNotFoundError,
    DocumentNotFoundError,
    DocumentParsingError,
    DocumentValidationError,
    DuplicateDocumentError,
    MalformedProviderResponseError,
    PaidProviderDisabledError,
    ProviderUnavailableError,
    RAGLabError,
    UnsupportedFrameworkError,
)


def register_exception_handlers(application: FastAPI) -> None:
    """Register stable public errors without leaking internal provider responses."""

    async def handle_expected(_: Request, error: Exception) -> JSONResponse:
        code = _status_code(error)
        return JSONResponse(
            status_code=code,
            content={"error": {"type": _error_type(error), "message": str(error)}},
        )

    application.add_exception_handler(RAGLabError, handle_expected)


def _status_code(error: Exception) -> int:
    if isinstance(error, (CollectionNotFoundError, DocumentNotFoundError)):
        return status.HTTP_404_NOT_FOUND
    if isinstance(error, DuplicateDocumentError):
        return status.HTTP_409_CONFLICT
    if isinstance(error, (DocumentValidationError, DocumentParsingError)):
        return status.HTTP_422_UNPROCESSABLE_CONTENT
    if isinstance(error, PaidProviderDisabledError):
        return status.HTTP_403_FORBIDDEN
    if isinstance(error, UnsupportedFrameworkError):
        return status.HTTP_501_NOT_IMPLEMENTED
    if isinstance(error, ProviderUnavailableError):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    if isinstance(error, MalformedProviderResponseError):
        return status.HTTP_502_BAD_GATEWAY
    return status.HTTP_400_BAD_REQUEST


def _error_type(error: Exception) -> str:
    name = type(error).__name__
    return name.removesuffix("Error")
