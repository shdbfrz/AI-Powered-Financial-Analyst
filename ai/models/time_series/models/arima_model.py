"""
ARIMA(p, d, q) model.

If `order` isn't supplied explicitly, `_fit` determines `d` from
`ai.models.time_series.stationarity.analyze_stationarity` and then grid
searches `(p, q)` by AIC — the standard Box-Jenkins order-selection
procedure, automated instead of eyeballed off ACF/PACF plots (those plots
are still produced by `analysis.py` for the required visualizations /
documentation, but order selection itself is AIC-driven for
reproducibility).
"""

import warnings
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.models.time_series.exceptions import HyperparameterSearchError, ModelTrainingError
from ai.models.time_series.models.base import BaseTimeSeriesModel, ForecastResult
from ai.models.time_series.stationarity import analyze_stationarity


class ArimaModel(BaseTimeSeriesModel):
    model_name = "arima"

    def __init__(
        self,
        config: TimeSeriesConfig = DEFAULT_TS_CONFIG,
        order: Optional[tuple[int, int, int]] = None,
        max_p: Optional[int] = None,
        max_q: Optional[int] = None,
    ):
        super().__init__(config, order=order, max_p=max_p, max_q=max_q)
        self._order = order
        self._max_p = max_p if max_p is not None else min(config.arima_max_p, 3)
        self._max_q = max_q if max_q is not None else min(config.arima_max_q, 3)
        self._fitted_model = None
        self._best_aic: Optional[float] = None

    def _select_order(self, series: pd.Series) -> tuple[int, int, int]:
        stationarity_report = analyze_stationarity(series, config=self.config)
        d = stationarity_report.recommended_differencing_order

        best_order, best_aic, best_result = None, np.inf, None
        for p, q in product(range(self._max_p + 1), range(self._max_q + 1)):
            if p == 0 and q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = ARIMA(series, order=(p, d, q)).fit()
                if fitted.aic < best_aic:
                    best_order, best_aic, best_result = (p, d, q), fitted.aic, fitted
            except Exception:  # noqa: BLE001 — some (p,q) combos fail to converge; skip and continue search
                continue

        if best_order is None:
            raise HyperparameterSearchError(
                self.model_name, "no (p, d, q) combination converged", context={"d": d}
            )

        self.logger.info(
            "ARIMA order selected by AIC grid search",
            extra={"order": best_order, "aic": round(best_aic, 4)},
        )
        return best_order

    def _fit(self, series: pd.Series) -> None:
        self._order = self._order or self._select_order(series)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted_model = ARIMA(series, order=self._order).fit()
        self._best_aic = float(self._fitted_model.aic)
        self.hyperparams["order"] = self._order
        self.hyperparams["aic"] = self._best_aic

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
            params={"order": self._order, "aic": self._best_aic},
        )