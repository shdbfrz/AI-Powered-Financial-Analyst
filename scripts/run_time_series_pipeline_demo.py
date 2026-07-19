"""
Sprint 4 demo script.

Usage (PowerShell, from project root, venv activated):

    python scripts/run_time_series_pipeline_demo.py AAPL

Requires a processed dataset already produced by Sprint 2 for the given
ticker under `datasets/processed/`. Trains every available model (ARIMA,
SARIMA, Auto ARIMA if installed, Prophet if installed, Exponential
Smoothing) across the configured forecast horizons, prints the ranked
comparison table for each horizon, and saves model artifacts + plots to
`storage/models/time_series/` and `storage/reports/time_series/`.
"""

import sys
from pathlib import Path

# Allow running this script directly (python scripts/run_time_series_pipeline_demo.py)
# — same fix as Sprint 3's run_ml_pipeline_demo.py, needed because plain
# `python scripts/foo.py` does NOT read pytest.ini's `pythonpath = .` (that
# only applies to pytest itself), so the project root must be added to
# sys.path manually before importing anything under `ai.`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.models.time_series.analysis import compute_acf_pacf, rolling_statistics  # noqa: E402
from ai.models.time_series.models.registry import TimeSeriesModelFactory  # noqa: E402
from ai.models.time_series.pipelines.training_pipeline import TimeSeriesTrainingPipeline  # noqa: E402
from ai.models.time_series.visualization.plots import TimeSeriesPlotGenerator  # noqa: E402
from ai.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main(ticker: str) -> None:
    print(f"\n=== Sprint 4 Time Series Pipeline Demo — {ticker} ===\n")

    print("Available models in this environment:", TimeSeriesModelFactory.available_models())

    pipeline = TimeSeriesTrainingPipeline(persist=True)
    result = pipeline.run(ticker)

    print(f"\nStationarity verdict (train split): {result.stationarity_report.verdict}")
    print(f"Recommended differencing order (d): {result.stationarity_report.recommended_differencing_order}")
    print(f"Detected seasonal period (m): {result.seasonal_period}")

    for horizon, table in result.comparison_tables.items():
        print(f"\n--- {horizon}-Day Horizon: Model Comparison ---")
        cols = ["rank", "model_name", "family", "test_rmse", "test_mae", "test_mape", "test_directional_accuracy"]
        available_cols = [c for c in cols if c in table.dataframe.columns]
        print(table.dataframe[available_cols].to_string(index=False))
        print(f"\n{table.narrative}")

    # Plots for the best model on the shortest configured horizon.
    shortest_horizon = min(result.comparison_tables)
    best_table = result.comparison_tables[shortest_horizon]
    best_model_name = best_table.dataframe.iloc[0]["model_name"]
    if best_table.dataframe.iloc[0]["family"] == "time_series":
        print(f"\nGenerating plots for best model: {best_model_name} (horizon={shortest_horizon})")
        model = TimeSeriesModelFactory.create(best_model_name)
        model.fit(result.split.train)
        forecast = model.forecast(len(result.split.validation))

        plots = TimeSeriesPlotGenerator()
        plots.actual_vs_forecast(result.split.validation, forecast, ticker=ticker)
        plots.residuals(result.split.validation, forecast.forecast, ticker=ticker, model_name=best_model_name)
        plots.trend_and_seasonality(result.decomposition, ticker=ticker)
        plots.acf_pacf(compute_acf_pacf(result.split.train), ticker=ticker)
        plots.rolling_mean_std(rolling_statistics(result.split.train, window=20), ticker=ticker, window=20)
        print("Plots saved under storage/reports/time_series/")

    print("\nModel artifacts saved under storage/models/time_series/")
    print("\n=== Demo complete ===\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_time_series_pipeline_demo.py <TICKER>")
        sys.exit(1)
    main(sys.argv[1].upper())