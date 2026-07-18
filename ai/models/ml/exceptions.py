"""
Exception hierarchy for `ai/models/ml/`.

Mirrors the pattern established in `ai/data_collection/exceptions.py` and
`ai/feature_engineering/exceptions.py`: every failure mode gets its own
typed exception carrying structured context, instead of a bare
`Exception`/`ValueError`, so callers (Decision Support Engine in Sprint 7,
the backend API in Sprint 8) can catch precisely what they need to handle
and errors are never swallowed silently (SRS NFR-7).
"""

from typing import Any, Optional


class MLPipelineError(Exception):
    """Base class for all errors raised by `ai/models/ml/`."""

    def __init__(self, message: str, *, context: Optional[dict] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self._format())

    def _format(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({ctx_str})"
        return self.message


class ProcessedDatasetNotFoundError(MLPipelineError):
    """No Sprint 2 processed dataset could be located under `datasets/processed/`."""

    def __init__(self, ticker: str, search_dir: Any):
        super().__init__(
            f"no processed dataset found for ticker '{ticker}'",
            context={"ticker": ticker, "search_dir": str(search_dir)},
        )
        self.ticker = ticker


class SchemaValidationError(MLPipelineError):
    """The processed DataFrame is missing required columns (targets or raw features)."""

    def __init__(self, reason: str, *, missing_columns: Optional[list] = None):
        super().__init__(reason, context={"missing_columns": missing_columns or []})
        self.missing_columns = missing_columns or []


class InvalidTargetColumnError(MLPipelineError):
    """The requested target column doesn't exist, or doesn't match the requested task type."""

    def __init__(self, target_column: str, task: str, *, available: Optional[list] = None):
        super().__init__(
            f"'{target_column}' is not a valid {task} target",
            context={"target_column": target_column, "task": task, "available": available or []},
        )
        self.target_column = target_column
        self.task = task


class InsufficientTrainingDataError(MLPipelineError):
    """Not enough clean rows remain after NaN-handling to form a meaningful
    train/validation/test split.
    """

    def __init__(self, required_rows: int, available_rows: int, *, context: Optional[dict] = None):
        super().__init__(
            f"insufficient data: need at least {required_rows} rows, got {available_rows}",
            context={"required_rows": required_rows, "available_rows": available_rows, **(context or {})},
        )
        self.required_rows = required_rows
        self.available_rows = available_rows


class UnknownModelError(MLPipelineError):
    """A model name was requested that isn't registered in the model factory."""

    def __init__(self, model_name: str, *, available: Optional[list] = None):
        super().__init__(
            f"unknown model '{model_name}'",
            context={"model_name": model_name, "available": available or []},
        )
        self.model_name = model_name


class ModelNotFittedError(MLPipelineError):
    """`.predict()` / `.get_feature_importance()` called before `.fit()`."""

    def __init__(self, model_name: str):
        super().__init__(f"model '{model_name}' has not been fitted yet", context={"model_name": model_name})
        self.model_name = model_name


class ModelTrainingError(MLPipelineError):
    """The underlying estimator raised while fitting."""

    def __init__(self, model_name: str, reason: str, *, context: Optional[dict] = None):
        super().__init__(
            f"training failed for model '{model_name}': {reason}",
            context={"model_name": model_name, **(context or {})},
        )
        self.model_name = model_name


class PredictionError(MLPipelineError):
    """The underlying estimator raised while predicting, or a feature mismatch was detected."""

    def __init__(self, model_name: str, reason: str, *, context: Optional[dict] = None):
        super().__init__(
            f"prediction failed for model '{model_name}': {reason}",
            context={"model_name": model_name, **(context or {})},
        )
        self.model_name = model_name


class ModelPersistenceError(MLPipelineError):
    """Saving/loading a model artifact or its metadata failed."""


class FeatureSelectionError(MLPipelineError):
    """A feature selection method failed (e.g. RFE on zero-variance input)."""

    def __init__(self, method: str, reason: str, *, context: Optional[dict] = None):
        super().__init__(
            f"feature selection method '{method}' failed: {reason}",
            context={"method": method, **(context or {})},
        )
        self.method = method


class HyperparameterTuningError(MLPipelineError):
    """GridSearchCV / RandomizedSearchCV raised during tuning."""

    def __init__(self, model_name: str, reason: str, *, context: Optional[dict] = None):
        super().__init__(
            f"hyperparameter tuning failed for model '{model_name}': {reason}",
            context={"model_name": model_name, **(context or {})},
        )
        self.model_name = model_name