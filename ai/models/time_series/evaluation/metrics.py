"""
Forecast evaluation metrics for `ai/models/time_series/`.

All metrics required by the Sprint 4 spec, plus the same near-zero-safe
MAPE guard Sprint 3's `ai.models.ml.evaluation.metrics` uses (a price series
is always positive so this mostly matters for `future_return`-style deltas
elsewhere, but the guard is kept for defensive consistency).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

_MAPE_EPSILON = 1e-8


@dataclass
class ForecastMetrics:
    mae: float
    mse: float
    rmse: float
    mape: float
    smape: float
    r2: float
    directional_accuracy: float
    forecast_bias: float
    n_observations: int

    def to_dict(self) -> dict:
        return {
            "mae": round(self.mae, 6),
            "mse": round(self.mse, 6),
            "rmse": round(self.rmse, 6),
            "mape": round(self.mape, 6),
            "smape": round(self.smape, 6),
            "r2": round(self.r2, 6),
            "directional_accuracy": round(self.directional_accuracy, 6),
            "forecast_bias": round(self.forecast_bias, 6),
            "n_observations": self.n_observations,
        }


def _align(actual: pd.Series, predicted: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    joined = pd.concat({"actual": actual, "predicted": predicted}, axis=1).dropna()
    if joined.empty:
        raise ValueError("actual and predicted series have no overlapping, non-NaN index")
    return joined["actual"].to_numpy(dtype=float), joined["predicted"].to_numpy(dtype=float)


def mean_absolute_error(actual: pd.Series, predicted: pd.Series) -> float:
    a, p = _align(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def mean_squared_error(actual: pd.Series, predicted: pd.Series) -> float:
    a, p = _align(actual, predicted)
    return float(np.mean((a - p) ** 2))


def root_mean_squared_error(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def mean_absolute_percentage_error(actual: pd.Series, predicted: pd.Series) -> float:
    """MAPE, guarded against division-by-near-zero (SRS FR-3.2 precedent
    from Sprint 3): denominators below `_MAPE_EPSILON` are excluded rather
    than producing an exploding/undefined percentage."""
    a, p = _align(actual, predicted)
    mask = np.abs(a) > _MAPE_EPSILON
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def symmetric_mean_absolute_percentage_error(actual: pd.Series, predicted: pd.Series) -> float:
    """SMAPE — bounded in [0, 200], symmetric in over/under-forecasting,
    more robust than MAPE when `actual` is near zero."""
    a, p = _align(actual, predicted)
    denom = np.abs(a) + np.abs(p)
    mask = denom > _MAPE_EPSILON
    if not mask.any():
        return float("nan")
    return float(np.mean(2 * np.abs(p[mask] - a[mask]) / denom[mask]) * 100)


def r_squared(actual: pd.Series, predicted: pd.Series) -> float:
    a, p = _align(actual, predicted)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def directional_accuracy(actual: pd.Series, predicted: pd.Series, *, previous_value: float) -> float:
    """Fraction of forecast steps where the predicted direction of change
    (up/down relative to the last known actual value, then relative to the
    previous forecasted step) matches the actual direction of change.
    Financially, getting the *direction* right often matters more than the
    exact price level (SRS §3.4/3.5 precedent: directional metrics are
    tracked alongside error metrics throughout the project)."""
    a, p = _align(actual, predicted)
    if len(a) == 0:
        return float("nan")
    actual_prev = np.concatenate(([previous_value], a[:-1]))
    predicted_prev = np.concatenate(([previous_value], p[:-1]))
    actual_direction = np.sign(a - actual_prev)
    predicted_direction = np.sign(p - predicted_prev)
    return float(np.mean(actual_direction == predicted_direction))


def forecast_bias(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean signed error: positive = model systematically over-forecasts,
    negative = systematically under-forecasts. Unlike MAE, sign matters."""
    a, p = _align(actual, predicted)
    return float(np.mean(p - a))


def evaluate_forecast(
    actual: pd.Series, predicted: pd.Series, *, previous_value: float
) -> ForecastMetrics:
    """Compute every Sprint 4-required metric in one call."""
    a, p = _align(actual, predicted)
    return ForecastMetrics(
        mae=mean_absolute_error(actual, predicted),
        mse=mean_squared_error(actual, predicted),
        rmse=root_mean_squared_error(actual, predicted),
        mape=mean_absolute_percentage_error(actual, predicted),
        smape=symmetric_mean_absolute_percentage_error(actual, predicted),
        r2=r_squared(actual, predicted),
        directional_accuracy=directional_accuracy(actual, predicted, previous_value=previous_value),
        forecast_bias=forecast_bias(actual, predicted),
        n_observations=len(a),
    )