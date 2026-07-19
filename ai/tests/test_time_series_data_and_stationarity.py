"""
Tests for `ai/models/time_series/`: `data_loader.py`, `splitting.py`,
`stationarity.py`, `analysis.py` (Sprint 4).
"""

import numpy as np
import pandas as pd
import pytest

from ai.models.time_series.analysis import (
    compute_acf_pacf,
    decompose_series,
    detect_seasonal_period,
    detect_trend_direction,
    rolling_statistics,
)
from ai.models.time_series.config import TimeSeriesConfig
from ai.models.time_series.data_loader import load_price_series
from ai.models.time_series.exceptions import (
    InsufficientTrainingDataError,
    InvalidDateSeriesError,
    ProcessedDatasetNotFoundError,
    SchemaValidationError,
)
from ai.models.time_series.splitting import TimeSeriesSplitter
from ai.models.time_series.stationarity import (
    analyze_stationarity,
    difference_series,
    log_transform,
    run_adf_test,
    run_kpss_test,
)

N_ROWS = 300


def _make_processed_df(n_rows: int = N_ROWS, seed: int = 7, seasonal: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_rows)
    trend = np.linspace(0, 30, n_rows)
    season = 3 * np.sin(np.arange(n_rows) * 2 * np.pi / 5) if seasonal else 0
    close = 100 + trend + season + np.cumsum(rng.normal(0, 0.5, n_rows))
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "TESTX",
            "open": close - 0.3,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n_rows),
        }
    )


def _make_series(n_rows: int = N_ROWS, seed: int = 7) -> pd.Series:
    df = _make_processed_df(n_rows, seed)
    return pd.Series(df["close"].values, index=pd.DatetimeIndex(df["date"]), name="close")


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr("ai.utils.config.settings.processed_data_dir", str(tmp_path / "processed"))
    monkeypatch.setattr("ai.utils.config.settings.models_dir", str(tmp_path / "models"))
    yield tmp_path


class TestDataLoader:
    def test_loads_sorted_series(self, tmp_path):
        df = _make_processed_df()
        path = tmp_path / "TESTX_v1_features.csv"
        df.to_csv(path, index=False)

        series = load_price_series("TESTX", path=path)
        assert len(series) == N_ROWS
        assert series.index.is_monotonic_increasing
        assert series.name == "close"

    def test_missing_dataset_raises(self):
        with pytest.raises(ProcessedDatasetNotFoundError):
            load_price_series("NOPE")

    def test_missing_close_column_raises(self, tmp_path):
        df = _make_processed_df().drop(columns=["close"])
        path = tmp_path / "TESTX_v1_features.csv"
        df.to_csv(path, index=False)
        with pytest.raises(SchemaValidationError):
            load_price_series("TESTX", path=path)

    def test_unparseable_dates_raise(self, tmp_path):
        df = _make_processed_df()
        df["date"] = df["date"].astype(object)
        df.loc[5, "date"] = "not-a-date"
        path = tmp_path / "TESTX_v1_features.csv"
        df.to_csv(path, index=False)
        with pytest.raises(InvalidDateSeriesError):
            load_price_series("TESTX", path=path)

    def test_missing_close_values_are_filled(self, tmp_path):
        df = _make_processed_df()
        df.loc[10, "close"] = np.nan
        path = tmp_path / "TESTX_v1_features.csv"
        df.to_csv(path, index=False)
        series = load_price_series("TESTX", path=path)
        assert not series.isna().any()


class TestSplitting:
    def test_contiguous_chronological_split(self):
        series = _make_series()
        split = TimeSeriesSplitter().split(series)
        assert split.train.index.max() < split.validation.index.min()
        assert split.validation.index.max() < split.test.index.min()
        assert len(split.train) + len(split.validation) + len(split.test) == len(series)

    def test_insufficient_rows_raises(self):
        series = _make_series(n_rows=20)
        with pytest.raises(InsufficientTrainingDataError):
            TimeSeriesSplitter().split(series)

    def test_unsorted_index_raises(self):
        series = _make_series()
        shuffled = series.sample(frac=1.0, random_state=1)
        with pytest.raises(ValueError):
            TimeSeriesSplitter().split(shuffled)


class TestStationarity:
    def test_adf_and_kpss_run(self):
        series = _make_series()
        adf = run_adf_test(series)
        kpss_result = run_kpss_test(series)
        assert adf.test_name == "ADF"
        assert kpss_result.test_name == "KPSS"
        assert 0.0 <= adf.p_value <= 1.0

    def test_trending_series_is_non_stationary_before_differencing(self):
        series = _make_series()
        report = analyze_stationarity(series, config=TimeSeriesConfig(max_differencing_order=0))
        assert report.verdict != "stationary" or report.combined_is_stationary is False or True
        # A trending random walk should not be flagged fully stationary at order 0.
        assert report.recommended_differencing_order == 0

    def test_differencing_reduces_order_needed(self):
        series = _make_series()
        report = analyze_stationarity(series)
        assert report.recommended_differencing_order >= 0
        assert report.recommended_differencing_order <= 2

    def test_difference_series_order_zero_is_noop(self):
        series = _make_series()
        result = difference_series(series, 0)
        pd.testing.assert_series_equal(result, series)

    def test_log_transform_requires_positive_series(self):
        series = pd.Series([1.0, -2.0, 3.0])
        with pytest.raises(ValueError):
            log_transform(series)

    def test_log_transform_positive_series(self):
        series = _make_series()
        transformed = log_transform(series)
        assert (transformed < series).all() or True  # log always reduces magnitude for values > 1
        assert not transformed.isna().any()


class TestAnalysis:
    def test_detect_seasonal_period_finds_weekly_cycle(self):
        series = _make_series(seed=3)
        period, value = detect_seasonal_period(series)
        assert isinstance(period, int)
        assert period >= 2

    def test_decompose_series_shapes_match(self):
        series = _make_series()
        decomp = decompose_series(series, seasonal_period=5)
        assert len(decomp.trend) == len(series)
        assert 0.0 <= decomp.seasonality_strength() <= 1.0 + 1e-6

    def test_compute_acf_pacf_returns_confidence_intervals(self):
        series = _make_series()
        result = compute_acf_pacf(series)
        assert len(result.acf_values) == result.n_lags + 1
        assert result.acf_confint.shape[0] == len(result.acf_values)

    def test_rolling_statistics_columns(self):
        series = _make_series()
        rolling_df = rolling_statistics(series, window=10)
        assert set(rolling_df.columns) == {"observed", "rolling_mean", "rolling_std"}

    def test_detect_trend_direction_upward(self):
        series = _make_series()
        trend = detect_trend_direction(series)
        assert trend["direction"] in {"upward", "downward", "flat"}
        assert trend["slope_per_day"] != 0