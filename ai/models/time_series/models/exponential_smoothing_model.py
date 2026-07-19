"""
Holt-Winters Exponential Smoothing model (Sprint 4 spec: optional/bonus).

Included as a fast, dependency-light baseline that complements the
statistical (S)ARIMA family and Prophet — useful in `ModelComparator` as a
sanity floor ("did SARIMA actually beat a much simpler smoother?").
"""

import warnings

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.models.time_series.models.base import BaseTimeSeriesModel, ForecastResult


class ExponentialSmoothingModel(BaseTimeSeriesModel):
    model_name = "exponential_smoothing"

    def __init__(
        self,
        config: TimeSeriesConfig = DEFAULT_TS_CONFIG,
        seasonal_period: int | None = None,
        trend: str = "add",
        seasonal: str = "add",
    ):
        super().__init__(config, trend=trend, seasonal=seasonal, seasonal_period=seasonal_period)
        self._seasonal_period = seasonal_period or config.default_seasonal_period
        self._trend = trend
        self._seasonal = seasonal
        self._fitted_model = None

    def _fit(self, series: pd.Series) -> None:
        # Holt-Winters needs at least 2 full seasonal cycles to fit a
        # seasonal component; fall back to trend-only smoothing on short series.
        use_seasonal = len(series) >= 2 * self._seasonal_period
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted_model = ExponentialSmoothing(
                series,
                trend=self._trend,
                seasonal=self._seasonal if use_seasonal else None,
                seasonal_periods=self._seasonal_period if use_seasonal else None,
                initialization_method="estimated",
            ).fit()
        self.hyperparams["seasonal_applied"] = use_seasonal

    def _forecast(self, steps: int) -> ForecastResult:
        mean = self._fitted_model.forecast(steps)
        future_index = self._future_business_index(self._train_series.index[-1], steps)
        mean.index = future_index

        # Holt-Winters has no native confidence interval; approximate with
        # the in-sample residual std, widening with the forecast horizon —
        # a standard, clearly-labeled approximation rather than a fabricated
        # exact interval.
        resid_std = float(self._fitted_model.resid.std())
        z = 1.959963985 if self.config.confidence_level >= 0.95 else 1.6448536
        margin = pd.Series(
            [z * resid_std * ((i + 1) ** 0.5) for i in range(steps)], index=future_index
        )

        return ForecastResult(
            model_name=self.model_name,
            forecast=mean,
            lower_bound=mean - margin,
            upper_bound=mean + margin,
            confidence_level=self.config.confidence_level,
            params={**self.hyperparams, "confidence_interval_method": "residual_std_approximation"},
        )