"""Safe HTTP translations for expected domain and provider failures."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from raglab.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    CollectionNotFoundError,
    DocumentDeletionConflictError,
    DocumentNotFoundError,
    DocumentParsingError,
    DocumentValidationError,
    DuplicateDocumentError,
    IngestionJobNotFoundError,
    InvalidCursorError,
    MalformedProviderResponseError,
    PaidProviderDisabledError,
    ProviderUnavailableError,
    RAGLabError,
    UnsupportedFrameworkError,
)
from raglab.core.metrics import LocalMetrics

logger = logging.getLogger(__name__)


def register_exception_handlers(application: FastAPI) -> None:
    """Register stable public errors without leaking internal provider responses."""

    async def handle_expected(request: Request, error: Exception) -> JSONResponse:
        code = _status_code(error)
        error_type = _error_type(error)
        _observe_error(request, error_type)
        logger.log(
            logging.WARNING if code >= 500 else logging.INFO,
            "request_failed",
            extra={
                "operation": _operation(request),
                "error_type": error_type,
                "status_code": code,
            },
        )
        headers: dict[str, str] = {}
        if isinstance(error, AuthenticationError):
            headers["WWW-Authenticate"] = "Bearer"
        if isinstance(error, ProviderUnavailableError):
            headers["Retry-After"] = "5"
        return JSONResponse(
            status_code=code,
            content=public_error_payload(error),
            headers=headers or None,
        )

    async def handle_unexpected(request: Request, error: Exception) -> JSONResponse:
        error_type = type(error).__name__
        _observe_error(request, "Internal")
        logger.error(
            "unexpected_request_failure",
            extra={
                "operation": _operation(request),
                "error_type": error_type,
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=public_error_payload(error),
        )

    application.add_exception_handler(RAGLabError, handle_expected)
    application.add_exception_handler(Exception, handle_unexpected)


def _status_code(error: Exception) -> int:
    if isinstance(error, AuthenticationError):
        return status.HTTP_401_UNAUTHORIZED
    if isinstance(error, AuthorizationError):
        return status.HTTP_403_FORBIDDEN
    if isinstance(
        error,
        (CollectionNotFoundError, DocumentNotFoundError, IngestionJobNotFoundError),
    ):
        return status.HTTP_404_NOT_FOUND
    if isinstance(error, (DocumentDeletionConflictError, DuplicateDocumentError)):
        return status.HTTP_409_CONFLICT
    if isinstance(error, (DocumentValidationError, DocumentParsingError, InvalidCursorError)):
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


def public_error_payload(error: Exception) -> dict[str, dict[str, str]]:
    """Build the same safe error envelope for HTTP and already-open SSE responses."""
    if isinstance(error, RAGLabError):
        error_type = _error_type(error)
        message = str(error)
    else:
        error_type = "Internal"
        message = "request processing failed"
    return {"error": {"type": error_type, "message": message}}


def _operation(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "unmatched"


def _observe_error(request: Request, error_type: str) -> None:
    metrics = getattr(request.app.state, "metrics", None)
    if isinstance(metrics, LocalMetrics):
        metrics.observe_error(_operation(request), error_type)
