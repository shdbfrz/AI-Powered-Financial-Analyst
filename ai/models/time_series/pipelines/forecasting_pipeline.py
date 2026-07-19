"""
Forecast generation utilities for `ai/models/time_series/`: rolling forecasts
and walk-forward validation.

Walk-forward validation (a.k.a. rolling-origin evaluation) re-fits the model
at each step on an expanding training window and forecasts one step ahead,
which is a far more realistic estimate of live performance than a single
static multi-step forecast — but it is `O(n / step)` model fits, so
`step` exists specifically to keep it tractable on classical models that
each take real wall-clock time to fit (SARIMA in particular).
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.models.time_series.evaluation.metrics import ForecastMetrics, evaluate_forecast
from ai.models.time_series.exceptions import ForecastError
from ai.models.time_series.models.base import BaseTimeSeriesModel
from ai.utils.logger import get_logger

logger = get_logger(__name__)


def generate_rolling_forecast(model: BaseTimeSeriesModel, steps: int):
    """Thin, explicitly-named wrapper around `model.forecast(steps)` — a
    "rolling forecast" here means the single multi-step-ahead forecast
    produced from the model's last fitted state, as distinct from
    `walk_forward_validate`'s repeated re-fitting. Kept as its own function
    so pipeline code reads as "which forecasting strategy" rather than a
    bare method call.
    """
    return model.forecast(steps)


@dataclass
class WalkForwardResult:
    model_name: str
    step: int
    n_folds: int
    metrics: ForecastMetrics
    fold_errors: list[float]


def walk_forward_validate(
    model_factory: Callable[[], BaseTimeSeriesModel],
    series: pd.Series,
    *,
    config: TimeSeriesConfig = DEFAULT_TS_CONFIG,
    min_train_size: int | None = None,
    step: int | None = None,
) -> WalkForwardResult:
    """Expanding-window walk-forward validation: starting from
    `min_train_size` observations, repeatedly fit-forecast `step` steps
    ahead, append those `step` actuals to the training window, and repeat
    until the series is exhausted. Returns pooled metrics across every fold.

    `model_factory` must return a *fresh, unfitted* model instance each call
    (e.g. `lambda: ArimaModel(config=config)`) since each fold fits from
    scratch on a larger window.
    """
    min_train_size = min_train_size or config.walk_forward_min_train_size
    step = step or config.walk_forward_step

    if min_train_size >= len(series):
        raise ForecastError(
            "walk_forward_validate", "min_train_size must be smaller than the series length"
        )

    all_actual, all_predicted, fold_errors = [], [], []
    n_folds = 0
    origin = min_train_size

    while origin + step <= len(series):
        train_window = series.iloc[:origin]
        actual_window = series.iloc[origin : origin + step]

        model = model_factory()
        try:
            model.fit(train_window)
            result = model.forecast(step)
        except Exception as exc:  # noqa: BLE001 — one bad fold shouldn't kill the whole validation run
            logger.info("walk-forward fold failed, skipping", extra={"origin": origin, "reason": str(exc)})
            origin += step
            continue

        predicted_aligned = pd.Series(result.forecast.values, index=actual_window.index)
        fold_mae = float((actual_window - predicted_aligned).abs().mean())

        all_actual.append(actual_window)
        all_predicted.append(predicted_aligned)
        fold_errors.append(fold_mae)
        n_folds += 1
        origin += step

    if n_folds == 0:
        raise ForecastError(
            "walk_forward_validate", "every walk-forward fold failed to fit/forecast"
        )

    pooled_actual = pd.concat(all_actual)
    pooled_predicted = pd.concat(all_predicted)
    metrics = evaluate_forecast(
        pooled_actual, pooled_predicted, previous_value=float(series.iloc[min_train_size - 1])
    )

    model_name = model_factory().model_name
    logger.info(
        "walk-forward validation complete",
        extra={"model": model_name, "n_folds": n_folds, "step": step, "rmse": round(metrics.rmse, 4)},
    )
    return WalkForwardResult(model_name=model_name, step=step, n_folds=n_folds, metrics=metrics, fold_errors=fold_errors)