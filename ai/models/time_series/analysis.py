"""
Descriptive time series analysis for `ai/models/time_series/`.

Produces the exploratory artifacts Sprint 4 requires before any model is
fit: classical trend/seasonality/residual decomposition, autocorrelation
(ACF) and partial autocorrelation (PACF) — used to eyeball candidate ARIMA
(p, q) orders — and rolling mean/std, which double as the Bollinger-style
volatility view already familiar from Sprint 2's feature set.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DecompositionResult:
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    observed: pd.Series
    seasonal_period: int
    model: str  # "additive" | "multiplicative"

    def seasonality_strength(self) -> float:
        """Var(seasonal) / (Var(seasonal) + Var(residual)) — 0 (no seasonal
        pattern) to ~1 (dominated by seasonality). Standard decomposition
        strength metric (Hyndman & Athanasopoulos)."""
        seasonal_var = float(np.nanvar(self.seasonal))
        residual_var = float(np.nanvar(self.residual))
        denom = seasonal_var + residual_var
        return 0.0 if denom == 0 else seasonal_var / denom

    def trend_strength(self) -> float:
        detrended_var = float(np.nanvar(self.residual + self.seasonal - np.nanmean(self.seasonal)))
        trend_plus_residual_var = float(np.nanvar(self.trend + self.residual))
        denom = trend_plus_residual_var
        return 0.0 if denom == 0 else max(0.0, 1 - float(np.nanvar(self.residual)) / denom)


def decompose_series(
    series: pd.Series,
    *,
    seasonal_period: int,
    model: str = "additive",
) -> DecompositionResult:
    """Classical seasonal decomposition (trend + seasonal + residual)."""
    result: DecomposeResult = seasonal_decompose(
        series, model=model, period=seasonal_period, extrapolate_trend="freq"
    )
    return DecompositionResult(
        trend=result.trend,
        seasonal=result.seasonal,
        residual=result.resid,
        observed=result.observed,
        seasonal_period=seasonal_period,
        model=model,
    )


def detect_seasonal_period(
    series: pd.Series, *, config: TimeSeriesConfig = DEFAULT_TS_CONFIG
) -> tuple[int, float]:
    """Estimate the dominant seasonal period from the ACF: the lag (beyond
    lag 1) with the highest autocorrelation, within
    `config.seasonality_detection_max_lag`. Returns
    `(period, acf_value_at_that_lag)`. Falls back to
    `config.default_seasonal_period` if no lag clears a weak-correlation
    floor of 0.1, since daily equity prices frequently show no strong
    seasonal cycle at all.
    """
    clean = series.dropna()
    max_lag = min(config.seasonality_detection_max_lag, len(clean) // 2 - 1)
    if max_lag < 2:
        return config.default_seasonal_period, 0.0

    acf_values = acf(clean, nlags=max_lag, fft=True)
    # Skip lag 0 (always 1.0) and lag 1 (trivial short-range correlation).
    candidate_lags = np.arange(2, max_lag + 1)
    candidate_values = acf_values[2 : max_lag + 1]
    if len(candidate_values) == 0:
        return config.default_seasonal_period, 0.0

    best_idx = int(np.argmax(np.abs(candidate_values)))
    best_lag = int(candidate_lags[best_idx])
    best_value = float(candidate_values[best_idx])

    if abs(best_value) < 0.1:
        logger.info(
            "no strong seasonal lag detected; falling back to default_seasonal_period",
            extra={"best_lag": best_lag, "best_acf": round(best_value, 4)},
        )
        return config.default_seasonal_period, best_value

    return best_lag, best_value


@dataclass
class AutocorrelationResult:
    acf_values: np.ndarray
    acf_confint: np.ndarray
    pacf_values: np.ndarray
    pacf_confint: np.ndarray
    n_lags: int


def compute_acf_pacf(
    series: pd.Series, *, config: TimeSeriesConfig = DEFAULT_TS_CONFIG
) -> AutocorrelationResult:
    """ACF and PACF with 95% confidence intervals, used both for the
    required ACF/PACF plots and to inform manual (p, q) order selection
    (Box-Jenkins methodology: PACF cutoff -> AR order p, ACF cutoff -> MA
    order q)."""
    clean = series.dropna()
    max_lags = min(config.acf_pacf_max_lags, len(clean) // 2 - 1)
    acf_values, acf_confint = acf(clean, nlags=max_lags, alpha=0.05, fft=True)
    pacf_values, pacf_confint = pacf(clean, nlags=max_lags, alpha=0.05)
    return AutocorrelationResult(
        acf_values=acf_values,
        acf_confint=acf_confint,
        pacf_values=pacf_values,
        pacf_confint=pacf_confint,
        n_lags=max_lags,
    )


def rolling_statistics(
    series: pd.Series, *, window: int = 20
) -> pd.DataFrame:
    """Rolling mean and rolling standard deviation — the same visual check
    used to eyeball non-constant variance before running ADF/KPSS formally."""
    return pd.DataFrame(
        {
            "observed": series,
            "rolling_mean": series.rolling(window=window).mean(),
            "rolling_std": series.rolling(window=window).std(),
        }
    )


def detect_trend_direction(series: pd.Series) -> dict:
    """Simple linear-regression-slope trend read: fits `close ~ time_index`
    and reports the sign/magnitude, used as a plain-English trend summary
    alongside the formal decomposition."""
    clean = series.dropna()
    x = np.arange(len(clean))
    slope, intercept = np.polyfit(x, clean.values, 1)
    direction = "upward" if slope > 0 else ("downward" if slope < 0 else "flat")
    return {
        "slope_per_day": float(slope),
        "intercept": float(intercept),
        "direction": direction,
        "total_change_over_period": float(slope * len(clean)),
    }