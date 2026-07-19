"""
Plot generation for `ai/models/time_series/`.

Mirrors `ai.models.ml.visualization.plots.PlotGenerator`'s shape: one class,
one method per required chart, each returning the saved `Path` so the
pipeline/report code can reference it without re-deriving filenames. Saved
under `storage/reports/time_series/` (`ARCHITECTURE.md` §4's per-module
reports convention — `ml_reports_dir`'s sibling, not a new top-level dir).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — this module never opens an interactive window
import matplotlib.pyplot as plt
import pandas as pd

from ai.models.time_series.analysis import AutocorrelationResult, DecompositionResult
from ai.models.time_series.models.base import ForecastResult
from ai.utils.config import settings
from ai.utils.logger import get_logger

logger = get_logger(__name__)

_TS_REPORTS_SUBDIR = "time_series"


def _reports_dir() -> Path:
    path = settings.resolve(settings.ml_reports_dir).parent / _TS_REPORTS_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


class TimeSeriesPlotGenerator:
    def __init__(self):
        self.logger = logger

    def _save(self, fig, filename: str) -> Path:
        path = _reports_dir() / filename
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("plot saved", extra={"path": str(path)})
        return path

    def actual_vs_forecast(
        self, actual: pd.Series, result: ForecastResult, *, ticker: str
    ) -> Path:
        fig, ax = plt.subplots(figsize=(10, 5))
        actual.plot(ax=ax, label="Actual", color="#1f77b4")
        result.forecast.plot(ax=ax, label=f"{result.model_name} forecast", color="#d62728")
        if result.lower_bound is not None and result.upper_bound is not None:
            ax.fill_between(
                result.forecast.index, result.lower_bound, result.upper_bound,
                color="#d62728", alpha=0.2, label=f"{int(result.confidence_level * 100)}% CI",
            )
        ax.set_title(f"{ticker} — Actual vs {result.model_name} Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        return self._save(fig, f"{ticker}_{result.model_name}_actual_vs_forecast.png")

    def residuals(self, actual: pd.Series, predicted: pd.Series, *, ticker: str, model_name: str) -> Path:
        joined = pd.concat({"actual": actual, "predicted": predicted}, axis=1).dropna()
        residual = joined["actual"] - joined["predicted"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(residual.index, residual.values, color="#2ca02c")
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_title("Residuals over time")

        axes[1].hist(residual.values, bins=30, color="#2ca02c", alpha=0.8)
        axes[1].set_title("Residual distribution")
        fig.suptitle(f"{ticker} — {model_name} residual analysis")
        return self._save(fig, f"{ticker}_{model_name}_residuals.png")

    def trend_and_seasonality(self, decomposition: DecompositionResult, *, ticker: str) -> Path:
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        decomposition.observed.plot(ax=axes[0], title="Observed")
        decomposition.trend.plot(ax=axes[1], title="Trend", color="#ff7f0e")
        decomposition.seasonal.plot(ax=axes[2], title=f"Seasonal (period={decomposition.seasonal_period})", color="#2ca02c")
        decomposition.residual.plot(ax=axes[3], title="Residual", color="#7f7f7f")
        fig.suptitle(f"{ticker} — Seasonal Decomposition ({decomposition.model})")
        fig.tight_layout()
        return self._save(fig, f"{ticker}_decomposition.png")

    def acf_pacf(self, result: AutocorrelationResult, *, ticker: str) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        lags = range(len(result.acf_values))
        axes[0].stem(lags, result.acf_values)
        axes[0].fill_between(lags, result.acf_confint[:, 0] - result.acf_values, result.acf_confint[:, 1] - result.acf_values, alpha=0.15)
        axes[0].set_title("ACF")

        pacf_lags = range(len(result.pacf_values))
        axes[1].stem(pacf_lags, result.pacf_values)
        axes[1].fill_between(pacf_lags, result.pacf_confint[:, 0] - result.pacf_values, result.pacf_confint[:, 1] - result.pacf_values, alpha=0.15)
        axes[1].set_title("PACF")

        fig.suptitle(f"{ticker} — ACF / PACF")
        return self._save(fig, f"{ticker}_acf_pacf.png")

    def rolling_mean_std(self, rolling_df: pd.DataFrame, *, ticker: str, window: int) -> Path:
        fig, ax = plt.subplots(figsize=(10, 5))
        rolling_df["observed"].plot(ax=ax, label="Close", alpha=0.5)
        rolling_df["rolling_mean"].plot(ax=ax, label=f"Rolling Mean ({window}d)", color="#ff7f0e")
        rolling_df["rolling_std"].plot(ax=ax, label=f"Rolling Std ({window}d)", color="#d62728", secondary_y=True)
        ax.set_title(f"{ticker} — Rolling Mean & Std")
        ax.legend(loc="upper left")
        return self._save(fig, f"{ticker}_rolling_stats.png")

    def forecast_comparison(self, results: list[ForecastResult], *, ticker: str) -> Path:
        fig, ax = plt.subplots(figsize=(10, 5))
        for result in results:
            result.forecast.plot(ax=ax, label=result.model_name, marker="o")
        ax.set_title(f"{ticker} — Forecast Comparison Across Models")
        ax.set_xlabel("Date")
        ax.set_ylabel("Forecasted Close Price")
        ax.legend()
        return self._save(fig, f"{ticker}_forecast_comparison.png")