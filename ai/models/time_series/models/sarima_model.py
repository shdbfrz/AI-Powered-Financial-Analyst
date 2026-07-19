"""
SARIMA(p, d, q)(P, D, Q, m) model.

Extends ARIMA's AIC-driven `(p, d, q)` selection with a seasonal
`(P, D, Q, m)` component. `m` (seasonal period) defaults to
`analysis.detect_seasonal_period` if not supplied. Grid search is kept
small (`P, Q` in `{0, 1}` by default) because SARIMAX fits scale poorly with
combinatorial seasonal orders — this mirrors the Sprint 4 spec's own
guidance to optimize P/D/Q/seasonal-period rather than exhaustively search
every combination.
"""

import warnings
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ai.models.time_series.analysis import detect_seasonal_period
from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.models.time_series.exceptions import HyperparameterSearchError
from ai.models.time_series.models.base import BaseTimeSeriesModel, ForecastResult
from ai.models.time_series.stationarity import analyze_stationarity


class SarimaModel(BaseTimeSeriesModel):
    model_name = "sarima"

    def __init__(
        self,
        config: TimeSeriesConfig = DEFAULT_TS_CONFIG,
        order: Optional[tuple[int, int, int]] = None,
        seasonal_order: Optional[tuple[int, int, int, int]] = None,
        max_p: Optional[int] = None,
        max_q: Optional[int] = None,
    ):
        super().__init__(config, order=order, seasonal_order=seasonal_order)
        self._order = order
        self._seasonal_order = seasonal_order
        self._max_p = max_p if max_p is not None else min(config.arima_max_p, 2)
        self._max_q = max_q if max_q is not None else min(config.arima_max_q, 2)
        self._fitted_model = None
        self._best_aic: Optional[float] = None

    def _select_orders(self, series: pd.Series) -> tuple[tuple, tuple]:
        stationarity_report = analyze_stationarity(series, config=self.config)
        d = stationarity_report.recommended_differencing_order

        seasonal_period, _ = detect_seasonal_period(series, config=self.config)
        max_P = min(self.config.sarima_max_P, 1)
        max_Q = min(self.config.sarima_max_Q, 1)
        max_D = min(self.config.sarima_max_D, 1)

        best_order, best_seasonal, best_aic, best_result = None, None, np.inf, None
        candidates = list(product(range(self._max_p + 1), range(self._max_q + 1), range(max_P + 1), range(max_D + 1), range(max_Q + 1)))
        for p, q, P, D, Q in candidates:
            if p == 0 and q == 0 and P == 0 and Q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = SARIMAX(
                        series,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, seasonal_period),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)
                if fitted.aic < best_aic:
                    best_order, best_seasonal, best_aic, best_result = (p, d, q), (P, D, Q, seasonal_period), fitted.aic, fitted
            except Exception:  # noqa: BLE001 — skip non-converging combos
                continue

        if best_order is None:
            raise HyperparameterSearchError(
                self.model_name, "no (p,d,q)(P,D,Q,m) combination converged", context={"d": d, "m": seasonal_period}
            )

        self.logger.info(
            "SARIMA order selected by AIC grid search",
            extra={"order": best_order, "seasonal_order": best_seasonal, "aic": round(best_aic, 4)},
        )
        return best_order, best_seasonal

    def _fit(self, series: pd.Series) -> None:
        if self._order is None or self._seasonal_order is None:
            self._order, self._seasonal_order = self._select_orders(series)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted_model = SARIMAX(
                series,
                order=self._order,
                seasonal_order=self._seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        self._best_aic = float(self._fitted_model.aic)
        self.hyperparams.update(order=self._order, seasonal_order=self._seasonal_order, aic=self._best_aic)

    def _forecast(self, steps: int) -> ForecastResult:
        forecast_res = self._fitted_model.get_forecast(steps=steps)
        mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=1 - self.config.confidence_level)

        future_index = self._future_business_index(self._train_series.index[-1], steps)
        mean.index = future_index
        conf_int.index = future_index

        return ForecastResult(
            model_name=self.model_name,
            forecast=mean,
            lower_bound=conf_int.iloc[:, 0],
            upper_bound=conf_int.iloc[:, 1],
            confidence_level=self.config.confidence_level,
            params={"order": self._order, "seasonal_order": self._seasonal_order, "aic": self._best_aic},
        )