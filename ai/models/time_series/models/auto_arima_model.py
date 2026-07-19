"""
Auto ARIMA model, backed by `pmdarima.auto_arima`.

`pmdarima` is an OPTIONAL dependency (see `ai/requirements-core.txt`'s
comment) for the same reason CatBoost is optional in Sprint 3: it's a
Cython extension with a build toolchain that doesn't always cooperate with
newer Python/numpy combinations (Python 3.13 in particular). Its absence
must never crash the pipeline — `TimeSeriesModelFactory` catches
`ModelUnavailableError` and skips this model with a log message, exactly
like Sprint 3's LightGBM/CatBoost handling.
"""

import warnings
from typing import Optional

import pandas as pd

from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.models.time_series.exceptions import ModelUnavailableError
from ai.models.time_series.models.base import BaseTimeSeriesModel, ForecastResult

try:
    import pmdarima as pm

    PMDARIMA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when pmdarima truly isn't installed
    PMDARIMA_AVAILABLE = False


class AutoArimaModel(BaseTimeSeriesModel):
    model_name = "auto_arima"

    def __init__(self, config: TimeSeriesConfig = DEFAULT_TS_CONFIG, seasonal_period: Optional[int] = None):
        super().__init__(config, seasonal_period=seasonal_period)
        if not PMDARIMA_AVAILABLE:
            raise ModelUnavailableError(self.model_name, "pmdarima")
        self._seasonal_period = seasonal_period or config.default_seasonal_period
        self._fitted_model = None

    def _fit(self, series: pd.Series) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted_model = pm.auto_arima(
                series,
                start_p=0,
                start_q=0,
                max_p=self.config.arima_max_p,
                max_q=self.config.arima_max_q,
                max_d=self.config.max_differencing_order,
                seasonal=True,
                m=self._seasonal_period,
                start_P=0,
                start_Q=0,
                max_P=self.config.sarima_max_P,
                max_Q=self.config.sarima_max_Q,
                max_D=self.config.sarima_max_D,
                information_criterion="aic",
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                random_state=self.config.random_state,
            )
        self.hyperparams.update(
            order=self._fitted_model.order,
            seasonal_order=self._fitted_model.seasonal_order,
            aic=float(self._fitted_model.aic()),
        )

    def _forecast(self, steps: int) -> ForecastResult:
        mean, conf_int = self._fitted_model.predict(
            n_periods=steps, return_conf_int=True, alpha=1 - self.config.confidence_level
        )
        future_index = self._future_business_index(self._train_series.index[-1], steps)
        mean_series = pd.Series(mean, index=future_index)
        lower = pd.Series(conf_int[:, 0], index=future_index)
        upper = pd.Series(conf_int[:, 1], index=future_index)

        return ForecastResult(
            model_name=self.model_name,
            forecast=mean_series,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=self.config.confidence_level,
            params={
                "order": self._fitted_model.order,
                "seasonal_order": self._fitted_model.seasonal_order,
                "aic": float(self._fitted_model.aic()),
            },
        )