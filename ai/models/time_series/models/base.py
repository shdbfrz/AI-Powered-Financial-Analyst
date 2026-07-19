"""
Base class for every Sprint 4 forecasting model.

Template Method pattern, same shape as Sprint 3's `ai.models.ml.models.base
.BaseMLModel`: the public `fit()` / `forecast()` methods own validation,
timing, logging, and fitted-state bookkeeping, while each concrete model
only implements the two library-specific hooks (`_fit`, `_forecast`). This
keeps ARIMA/SARIMA/Auto-ARIMA/Prophet/Exponential-Smoothing interchangeable
behind one interface — which is exactly what `ForecastService` and
`ModelComparator` depend on.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import pandas as pd

from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.models.time_series.exceptions import ForecastError, ModelNotFittedError, ModelTrainingError
from ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastResult:
    """Standardized output of every model's `.forecast()` call."""

    model_name: str
    forecast: pd.Series  # indexed by future business-day dates
    lower_bound: Optional[pd.Series] = None
    upper_bound: Optional[pd.Series] = None
    confidence_level: float = 0.95
    params: dict = field(default_factory=dict)
    fit_time_seconds: float = 0.0
    forecast_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "confidence_level": self.confidence_level,
            "params": self.params,
            "fit_time_seconds": round(self.fit_time_seconds, 4),
            "forecast_time_seconds": round(self.forecast_time_seconds, 4),
            "forecast": {str(k): float(v) for k, v in self.forecast.items()},
            "lower_bound": (
                {str(k): float(v) for k, v in self.lower_bound.items()}
                if self.lower_bound is not None
                else None
            ),
            "upper_bound": (
                {str(k): float(v) for k, v in self.upper_bound.items()}
                if self.upper_bound is not None
                else None
            ),
        }


class BaseTimeSeriesModel(ABC):
    """Template Method base for all forecasting models.

    Subclasses implement `_fit(series)` and `_forecast(steps) ->
    ForecastResult`; everything else (fitted-state guard, timing, logging,
    typed error translation) is handled once here.
    """

    model_name: str = "base"

    def __init__(self, config: TimeSeriesConfig = DEFAULT_TS_CONFIG, **hyperparams):
        self.config = config
        self.hyperparams = hyperparams
        self._is_fitted = False
        self._fit_time_seconds = 0.0
        self._train_series: Optional[pd.Series] = None
        self.logger = logger

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, series: pd.Series) -> "BaseTimeSeriesModel":
        if series.dropna().empty:
            raise ModelTrainingError(self.model_name, "training series is empty after dropping NaNs")

        self.logger.info("fitting model", extra={"model": self.model_name, "rows": len(series)})
        start = perf_counter()
        try:
            self._fit(series)
        except ModelTrainingError:
            raise
        except Exception as exc:  # noqa: BLE001 — translate any library error into our typed hierarchy
            raise ModelTrainingError(self.model_name, str(exc)) from exc

        self._fit_time_seconds = perf_counter() - start
        self._is_fitted = True
        self._train_series = series
        self.logger.info(
            "model fit complete",
            extra={"model": self.model_name, "seconds": round(self._fit_time_seconds, 4)},
        )
        return self

    def forecast(self, steps: int) -> ForecastResult:
        if not self._is_fitted:
            raise ModelNotFittedError(self.model_name)
        if steps <= 0:
            raise ForecastError(self.model_name, f"steps must be a positive integer, got {steps}")

        start = perf_counter()
        try:
            result = self._forecast(steps)
        except ForecastError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ForecastError(self.model_name, str(exc)) from exc

        result.forecast_time_seconds = perf_counter() - start
        result.fit_time_seconds = self._fit_time_seconds
        self.logger.info(
            "forecast generated",
            extra={"model": self.model_name, "steps": steps, "seconds": round(result.forecast_time_seconds, 4)},
        )
        return result

    def get_params(self) -> dict:
        return dict(self.hyperparams)

    @staticmethod
    def _future_business_index(last_date: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
        """Next `steps` business days after `last_date` — matches the
        trading-day cadence of the input OHLCV data (no weekends)."""
        return pd.bdate_range(start=last_date, periods=steps + 1, freq="B")[1:]

    @abstractmethod
    def _fit(self, series: pd.Series) -> None:
        ...

    @abstractmethod
    def _forecast(self, steps: int) -> ForecastResult:
        ...