"""
Facebook Prophet model.

`prophet` (and its `cmdstanpy` backend, which compiles a Stan binary on
first use) is an OPTIONAL dependency for the same reason `pmdarima` is —
the compiled-backend install step can fail in constrained/offline Windows
environments. Its absence must never crash the pipeline; see
`auto_arima_model.py`'s docstring for the shared rationale.
"""

import logging
from typing import Optional

import pandas as pd

from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.models.time_series.exceptions import ModelUnavailableError
from ai.models.time_series.models.base import BaseTimeSeriesModel, ForecastResult

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when prophet truly isn't installed
    PROPHET_AVAILABLE = False


class ProphetModel(BaseTimeSeriesModel):
    model_name = "prophet"

    def __init__(self, config: TimeSeriesConfig = DEFAULT_TS_CONFIG):
        super().__init__(config)
        if not PROPHET_AVAILABLE:
            raise ModelUnavailableError(self.model_name, "prophet")
        self._fitted_model: Optional[Prophet] = None

    def _fit(self, series: pd.Series) -> None:
        # Prophet's internal loggers (cmdstanpy) are noisy at INFO level by
        # default; keep pipeline logs readable, our own get_logger() output
        # is still emitted around this call.
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.WARNING)

        train_df = pd.DataFrame({"ds": series.index, "y": series.values})
        self._fitted_model = Prophet(
            seasonality_mode=self.config.prophet_seasonality_mode,
            weekly_seasonality=self.config.prophet_weekly_seasonality,
            yearly_seasonality=self.config.prophet_yearly_seasonality,
            daily_seasonality=self.config.prophet_daily_seasonality,
            changepoint_prior_scale=self.config.prophet_changepoint_prior_scale,
            interval_width=self.config.confidence_level,
        )
        self._fitted_model.fit(train_df)
        self.hyperparams.update(
            seasonality_mode=self.config.prophet_seasonality_mode,
            changepoint_prior_scale=self.config.prophet_changepoint_prior_scale,
        )

    def _forecast(self, steps: int) -> ForecastResult:
        future_index = self._future_business_index(self._train_series.index[-1], steps)
        future_df = pd.DataFrame({"ds": future_index})
        prediction = self._fitted_model.predict(future_df)

        mean = pd.Series(prediction["yhat"].values, index=future_index)
        lower = pd.Series(prediction["yhat_lower"].values, index=future_index)
        upper = pd.Series(prediction["yhat_upper"].values, index=future_index)

        return ForecastResult(
            model_name=self.model_name,
            forecast=mean,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=self.config.confidence_level,
            params=self.hyperparams,
        )