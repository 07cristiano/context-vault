"""Stable application errors that can be translated at the API boundary."""


class ContextVaultError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(ContextVaultError):
    """The local application configuration is unsafe or invalid."""


class DatabaseError(ContextVaultError):
    """The local database could not complete an operation."""


class ModelUnavailableError(ContextVaultError):
    """Ollama or a required local model is unavailable."""


class ModelResponseError(ContextVaultError):
    """A local model returned malformed or unsafe output."""


class UploadValidationError(ContextVaultError):
    """An uploaded file is unsupported, unsafe, or malformed."""


class DuplicateDocumentError(ContextVaultError):
    """The same file content is already indexed."""


class CapacityError(ContextVaultError):
    """A documented small-corpus capacity limit would be exceeded."""


class DocumentNotFoundError(ContextVaultError):
    """The requested document does not exist."""


class ExtractionError(ContextVaultError):
    """A supported file did not contain usable content."""
