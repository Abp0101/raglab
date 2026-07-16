"""Safe, typed failures shared across framework implementations."""


class RAGLabError(Exception):
    """Base class for expected application failures."""


class DocumentValidationError(RAGLabError):
    """An uploaded document failed validation."""


class DocumentParsingError(RAGLabError):
    """A validated document could not be parsed safely."""


class DuplicateDocumentError(RAGLabError):
    """A document with the same content hash already exists."""


class CollectionNotFoundError(RAGLabError):
    """The requested collection does not exist."""


class DocumentNotFoundError(RAGLabError):
    """The requested document does not exist."""


class PaidProviderDisabledError(RAGLabError):
    """A metered remote provider was selected without an explicit safety opt-in."""


class UnsupportedFrameworkError(RAGLabError):
    """The requested framework adapter has not been registered."""


class ProviderUnavailableError(RAGLabError):
    """An embedding, model, or storage provider could not respond."""


class MalformedProviderResponseError(RAGLabError):
    """A provider response could not be parsed into the required contract."""


class InsufficientEvidenceError(RAGLabError):
    """Retrieved evidence does not support a grounded answer."""
