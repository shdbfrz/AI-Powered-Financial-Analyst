"""
scripts/run_ml_pipeline_demo.py
================================
One-off script to run the full Sprint 3 ML pipeline end-to-end (regression
+ classification) and print a summary, for manual verification (not part of
CI).

Usage:
    # from the project root, with a venv that has ai/requirements-core.txt installed
    python scripts/run_ml_pipeline_demo.py AAPL

If no processed dataset exists yet for the given ticker under
`datasets/processed/` (i.e. Sprint 2's feature engineering pipeline hasn't
been run for it), this script generates a small synthetic OHLCV series and
runs the real Sprint 2 pipeline on it first, purely so Sprint 3 can be
exercised end-to-end without depending on live data or a prior manual step.

What it checks, in order:
    1. Load (or generate + build) a Sprint 2 processed dataset
    2. Run MLTrainingPipeline for the default regression target (future_return_5_day)
    3. Run MLTrainingPipeline for the default classification target (target_direction_5_day)
    4. Print the winning model, comparison table, and every artifact path written
"""

import sys
import warnings
from pathlib import Path

# Allow running this script directly (python scripts/run_ml_pipeline_demo.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai.feature_engineering import FeatureEngineeringPipeline  # noqa: E402
from ai.models.ml import MLTrainingPipeline  # noqa: E402
from ai.models.ml.data_loader import find_processed_dataset_files  # noqa: E402

PASS = "\033[92mPASS\033[0m"
INFO = "\033[94mINFO\033[0m"


def _synthetic_ohlcv(ticker: str, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range("2016-01-04", periods=n)
    regime = np.sin(np.linspace(0, 6 * np.pi, n)) * 0.05
    close = 100 + np.cumsum(rng.normal(0.03, 1.4, n) + regime)
    close = np.maximum(close, 5)
    open_ = close + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.3, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.3, 0.3, n))
    volume = rng.integers(1_000_000, 8_000_000, n)
    return pd.DataFrame({
        "date": dates, "ticker": ticker, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    })


def _print_result(result, label: str) -> None:
    print(f"\n--- {label} ---")
    print(f"{PASS} best model: {result.best_model_name}")
    cols = ["performance_rank", "model_name", "primary_metric_value", "training_time_seconds", "efficiency_rank"]
    print(result.comparison_table.dataframe[cols].to_string(index=False))
    print(f"{INFO} Model_Comparison.csv -> {result.comparison_csv_path}")
    print(f"{INFO} Evaluation_Report.md -> {result.evaluation_report_path}")
    print(f"{INFO} Model_Documentation.md -> {result.model_documentation_path}")
    print(f"{INFO} Model_Metadata.json -> {result.metadata_json_path}")
    print(f"{INFO} {len(result.saved_models)} model artifact(s) saved, {len(result.plot_paths)} plot(s) generated")


def main() -> int:
    warnings.filterwarnings("ignore")
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "DEMO"

    print("=" * 70)
    print(f"Sprint 3 Machine Learning Pipeline Demo — ticker={ticker}")
    print("=" * 70)

    if find_processed_dataset_files(ticker):
        print(f"{INFO} Found an existing processed dataset for {ticker}; using the most recent one.")
        df = None  # MLTrainingPipeline will locate it via ticker
    else:
        print(f"{INFO} No processed dataset found for {ticker}; generating synthetic OHLCV "
              f"and running the Sprint 2 pipeline first.")
        raw_df = _synthetic_ohlcv(ticker)
        fe_result = FeatureEngineeringPipeline().run(raw_df=raw_df, version="demo")
        print(f"{PASS} Sprint 2 pipeline produced {fe_result.rows_out} rows x {fe_result.columns_out} columns "
              f"-> {fe_result.processed_csv_path}")
        df = None  # re-load from disk (proves the on-disk contract, not just the in-memory object)

    pipeline = MLTrainingPipeline()

    regression_result = pipeline.run(ticker=ticker, df=df, task="regression", version="demo")
    _print_result(regression_result, "Regression — future_return_5_day")

    classification_result = pipeline.run(ticker=ticker, df=df, task="classification", version="demo")
    _print_result(classification_result, "Classification — target_direction_5_day")

    print("\n" + "=" * 70)
    print(f"{PASS} Sprint 3 demo complete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())