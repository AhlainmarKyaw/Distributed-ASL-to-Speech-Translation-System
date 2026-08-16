"""Explicit worker failure categories used by Celery retry policy."""


class TransientWorkerError(RuntimeError):
    """A temporary worker-side failure that is safe to retry."""


class PermanentWorkerError(RuntimeError):
    """A non-retryable model, input, or inference failure."""


class ModelUnavailableError(PermanentWorkerError):
    """The configured model cannot currently serve predictions."""
