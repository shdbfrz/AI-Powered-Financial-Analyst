"""
Exception hierarchy for `ai/models/time_series/`.

Mirrors `ai/models/ml/exceptions.py`: every failure mode gets its own typed
exception carrying structured context, instead of a bare `Exception` /
`ValueError`, so callers (ForecastService, Decision Support Engine in
Sprint 7, backend API in Sprint 8) can catch precisely what they need to
handle and errors are never swallowed silently (SRS NFR-7).
"""

from typing import Any, Optional


class TimeSeriesError(Exception):
    """Base class for all errors raised by `ai/models/time_series/`."""

    def __init__(self, message: str, *, context: Optional[dict] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self._format())

    def _format(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({ctx_str})"
        return self.message


class ProcessedDatasetNotFoundError(TimeSeriesError):
    """No Sprint 2 processed dataset could be located under `datasets/processed/`."""

    def __init__(self, ticker: str, search_dir: Any):
        super().__init__(
            f"no processed dataset found for ticker '{ticker}'",
            context={"ticker": ticker, "search_dir": str(search_dir)},
        )
        self.ticker = ticker


class SchemaValidationError(TimeSeriesError):
    """The processed DataFrame is missing the date/price columns required for forecasting."""

    def __init__(self, reason: str, *, missing_columns: Optional[list] = None):
        super().__init__(reason, context={"missing_columns": missing_columns or []})
        self.missing_columns = missing_columns or []


class InvalidDateSeriesError(TimeSeriesError):
    """Dates are missing, duplicated, unsorted, or not chronologically contiguous enough to model."""


class InsufficientTrainingDataError(TimeSeriesError):
    """Not enough clean rows remain to form a meaningful train/validation/test split."""

    def __init__(self, required_rows: int, available_rows: int, *, context: Optional[dict] = None):
        super().__init__(
            f"insufficient data: need at least {required_rows} rows, got {available_rows}",
            context={"required_rows": required_rows, "available_rows": available_rows, **(context or {})},
        )
        self.required_rows = required_rows
        self.available_rows = available_rows


class NonStationarySeriesError(TimeSeriesError):
    """The series remains non-stationary even after the configured maximum differencing order."""

    def __init__(self, max_order: int, *, context: Optional[dict] = None):
        super().__init__(
            f"series still non-stationary after {max_order} rounds of differencing",
            context=context,
        )
        self.max_order = max_order


class UnknownModelError(TimeSeriesError):
    """A model name was requested that isn't registered in the model factory."""

    def __init__(self, model_name: str, *, available: Optional[list] = None):
        super().__init__(
            f"unknown time series model '{model_name}'",
            context={"model_name": model_name, "available": available or []},
        )
        self.model_name = model_name


class ModelUnavailableError(TimeSeriesError):
    """The requested model's optional dependency (pmdarima / prophet) is not installed.

    Mirrors Sprint 3's graceful-skip handling of LightGBM/CatBoost: an
    optional model being unavailable in a given environment must never crash
    the pipeline, only skip that one model with a clear log message.
    """

    def __init__(self, model_name: str, package: str):
        super().__init__(
            f"model '{model_name}' requires optional package '{package}', which is not installed",
            context={"model_name": model_name, "package": package},
        )
        self.model_name = model_name
        self.package = package


class ModelNotFittedError(TimeSeriesError):
    """`.forecast()` / `.get_confidence_intervals()` called before `.fit()`."""

    def __init__(self, model_name: str):
        super().__init__(f"model '{model_name}' has not been fitted yet", context={"model_name": model_name})
        self.model_name = model_name


class ModelTrainingError(TimeSeriesError):
    """The underlying statistical/forecasting library raised while fitting."""

    def __init__(self, model_name: str, reason: str, *, context: Optional[dict] = None):
        super().__init__(
            f"training failed for model '{model_name}': {reason}",
            context={"model_name": model_name, **(context or {})},
        )
        self.model_name = model_name


class ForecastError(TimeSeriesError):
    """The underlying library raised while forecasting, or an invalid horizon was requested."""

    def __init__(self, model_name: str, reason: str, *, context: Optional[dict] = None):
        super().__init__(
            f"forecasting failed for model '{model_name}': {reason}",
            context={"model_name": model_name, **(context or {})},
        )
        self.model_name = model_name


class ModelPersistenceError(TimeSeriesError):
    """Saving/loading a model artifact or its metadata failed."""


class HyperparameterSearchError(TimeSeriesError):
    """(S)ARIMA order search / auto_arima / Prophet tuning failed for every candidate."""

    def __init__(self, model_name: str, reason: str, *, context: Optional[dict] = None):
        super().__init__(
            f"hyperparameter search failed for model '{model_name}': {reason}",
            context={"model_name": model_name, **(context or {})},
        )
        self.model_name = model_name