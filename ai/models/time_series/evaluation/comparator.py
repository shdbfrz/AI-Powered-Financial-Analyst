"""
Model comparison and reporting for `ai/models/time_series/`.

`TimeSeriesModelComparator` turns a list of `TimeSeriesModelResult` into a
ranked `Model_Comparison.csv`-ready DataFrame and a short narrative, mirroring
`ai.models.ml.evaluation.comparator.ModelComparator`'s shape. It additionally
accepts Sprint 3's ML `ModelResult` rows (already-computed, passed in — this
module does not re-run Sprint 3) so ARIMA/SARIMA/Prophet can be ranked
side-by-side with Sprint 3's tree/linear models on a shared metric, per the
Sprint 4 spec's "Compare ML (Sprint 3) vs ARIMA/SARIMA/Prophet" requirement.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ai.models.time_series.evaluation.metrics import ForecastMetrics
from ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TimeSeriesModelResult:
    """Everything collected about one trained/evaluated forecasting model."""

    model_name: str
    horizon_days: int
    hyperparameters: dict
    validation_metrics: ForecastMetrics
    test_metrics: ForecastMetrics
    training_time_seconds: float
    prediction_time_seconds: float
    family: str = "time_series"  # "time_series" | "machine_learning" — lets ML rows join the same table
    error: Optional[str] = None

    def flat_row(self, primary_metric: str) -> dict:
        row = {
            "model_name": self.model_name,
            "family": self.family,
            "horizon_days": self.horizon_days,
            "training_time_seconds": round(self.training_time_seconds, 4),
            "prediction_time_seconds": round(self.prediction_time_seconds, 6),
        }
        for split_name, metrics in (("validation", self.validation_metrics), ("test", self.test_metrics)):
            if metrics is None:
                continue
            for metric_name, value in metrics.to_dict().items():
                row[f"{split_name}_{metric_name}"] = value
        row["primary_metric"] = primary_metric
        row["primary_metric_value"] = row.get(f"test_{primary_metric}")
        return row


@dataclass
class TimeSeriesComparisonTable:
    horizon_days: int
    primary_metric: str
    primary_metric_direction: str
    dataframe: pd.DataFrame
    narrative: str


class TimeSeriesModelComparator:
    """Ranks time series (and, optionally, injected ML) results and renders
    the sprint's `Model_Comparison.csv` + narrative."""

    def __init__(self, primary_metric: str = "rmse", primary_metric_direction: str = "minimize"):
        self.primary_metric = primary_metric
        self.primary_metric_direction = primary_metric_direction
        self.logger = logger

    def compare(
        self, results: list[TimeSeriesModelResult], *, horizon_days: int
    ) -> TimeSeriesComparisonTable:
        usable = [r for r in results if r.error is None and r.test_metrics is not None]
        if not usable:
            raise ValueError("no successfully evaluated models to compare")

        rows = [r.flat_row(self.primary_metric) for r in usable]
        df = pd.DataFrame(rows)
        ascending = self.primary_metric_direction == "minimize"
        df = df.sort_values("primary_metric_value", ascending=ascending).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))

        best = df.iloc[0]
        narrative = self._narrative(df, best, horizon_days)

        self.logger.info(
            "model comparison complete",
            extra={"horizon_days": horizon_days, "best_model": best["model_name"], "primary_metric": self.primary_metric},
        )
        return TimeSeriesComparisonTable(
            horizon_days=horizon_days,
            primary_metric=self.primary_metric,
            primary_metric_direction=self.primary_metric_direction,
            dataframe=df,
            narrative=narrative,
        )

    def _narrative(self, df: pd.DataFrame, best: pd.Series, horizon_days: int) -> str:
        lines = [
            f"For the {horizon_days}-day horizon, **{best['model_name']}** ({best['family']}) ranks best on "
            f"{self.primary_metric} ({best['primary_metric_value']:.4f}), with a directional accuracy of "
            f"{best.get('test_directional_accuracy', float('nan')):.2%}." if "test_directional_accuracy" in best else
            f"For the {horizon_days}-day horizon, **{best['model_name']}** ({best['family']}) ranks best on "
            f"{self.primary_metric} ({best['primary_metric_value']:.4f}).",
        ]
        ts_rows = df[df["family"] == "time_series"]
        ml_rows = df[df["family"] == "machine_learning"]
        if len(ts_rows) and len(ml_rows):
            ts_best_rank = int(ts_rows["rank"].min())
            ml_best_rank = int(ml_rows["rank"].min())
            if ts_best_rank < ml_best_rank:
                lines.append(
                    "Classical time series models outperformed Sprint 3's ML models on this split — "
                    "unsurprising when the series is close to a random walk with a stable seasonal/trend "
                    "component, which ARIMA-family models are structurally built to capture directly, "
                    "whereas the ML models only see it indirectly through engineered lag/rolling features."
                )
            else:
                lines.append(
                    "Sprint 3's ML models outperformed the classical time series models on this split — "
                    "plausible when the 205-feature engineered set captures cross-sectional structure "
                    "(momentum, volatility regime, candlestick/zone signals) that a univariate close-price "
                    "series alone cannot represent."
                )
        return " ".join(lines)

    @staticmethod
    def merge_ml_results(
        ts_results: list[TimeSeriesModelResult], ml_rows: list[dict], *, horizon_days: int
    ) -> list[TimeSeriesModelResult]:
        """Wrap Sprint 3 `ModelResult.flat_row(...)`-style dicts (already
        computed by `ai.models.ml.evaluation.comparator`) as
        `TimeSeriesModelResult` stand-ins so they can be ranked in the same
        table. Only the fields needed for `flat_row`/ranking are populated;
        `validation_metrics`/`test_metrics` stay as raw dict-like proxies
        rather than being coerced into `ForecastMetrics` (which requires the
        forecast-specific directional/bias fields Sprint 3 doesn't compute).
        """
        merged = list(ts_results)
        for row in ml_rows:
            proxy = _MLRowMetricsProxy(row)
            merged.append(
                TimeSeriesModelResult(
                    model_name=row.get("model_name", "unknown_ml_model"),
                    horizon_days=horizon_days,
                    hyperparameters={},
                    validation_metrics=proxy,
                    test_metrics=proxy,
                    training_time_seconds=row.get("training_time_seconds", 0.0),
                    prediction_time_seconds=row.get("prediction_time_seconds", 0.0),
                    family="machine_learning",
                )
            )
        return merged


@dataclass
class _MLRowMetricsProxy:
    """Adapts a flattened Sprint 3 comparison row into something with a
    `.to_dict()` matching enough of `ForecastMetrics`'s shape for
    `TimeSeriesModelResult.flat_row` to consume without special-casing."""

    row: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        keys = ("mae", "mse", "rmse", "mape", "r2")
        return {k: self.row.get(f"test_{k}", self.row.get(k)) for k in keys if f"test_{k}" in self.row or k in self.row}