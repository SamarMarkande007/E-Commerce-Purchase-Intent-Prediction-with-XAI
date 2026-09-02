"""Project-specific exceptions.

Using named exceptions instead of bare ``except:`` or generic
``Exception`` lets callers (the API, the dashboard, tests) catch exactly
the failure mode they care about and respond appropriately — e.g. the
API can turn a ``DataValidationError`` into a 422 instead of a 500.
"""

from __future__ import annotations


class PurchaseIntentError(Exception):
    """Base class for all project-specific exceptions."""


class DataValidationError(PurchaseIntentError):
    """Raised when input data fails schema/dtype/content validation.

    Examples: the CSV is missing, a required column is absent, a column
    has an unexpected dtype, or the target column contains values other
    than 0/1.
    """


class ConfigError(PurchaseIntentError):
    """Raised when configuration is missing, malformed, or inconsistent."""


class ModelNotFoundError(PurchaseIntentError):
    """Raised when a saved model or preprocessing artifact cannot be loaded."""


class PredictionError(PurchaseIntentError):
    """Raised when scoring a session fails after the model has loaded."""
