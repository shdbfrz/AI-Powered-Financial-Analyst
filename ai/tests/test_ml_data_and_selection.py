"""
Tests for `ai/models/ml/data_loader.py`, `splitting.py`, `preprocessing.py`,
and `selection/feature_selector.py` (Sprint 3).

Fully offline — a small synthetic *processed* DataFrame (mirroring Sprint
2's real output schema: raw OHLCV columns, a handful of numeric features, a
categorical feature, and target columns for horizons 1/3/5) is generated
in-memory rather than depending on `ai/feature_engineering/` having already
run, so these tests stay fast and independent.
"""

import numpy as np
import pandas as pd
import pytest

from ai.models.ml.config import MLConfig
from ai.models.ml.data_loader import (
    build_feature_matrix,
    build_inference_matrix,
    find_processed_dataset_files,
    load_processed_dataset,
    resolve_task_and_validate_target,
)
from ai.models.ml.exceptions import (
    InsufficientTrainingDataError,
    InvalidTargetColumnError,
    ProcessedDatasetNotFoundError,
    SchemaValidationError,
)
from ai.models.ml.preprocessing import FeaturePreprocessor, PreprocessorNotFittedError
from ai.models.ml.selection import FeatureSelector
from ai.models.ml.splitting import TimeSeriesSplitter

N_ROWS = 400


def _make_processed_df(n_rows: int = N_ROWS, seed: int = 7) -> pd.DataFrame:
    """A small synthetic frame matching Sprint 2's real schema closely enough
    to exercise Sprint 3 against: raw OHLCV + identifiers, a few numeric
    engineered features (one of them mostly-NaN, mirroring
    `demand_zone_strength`-style "sparse event" columns), one categorical
    feature, and target columns for horizons 1/3/5.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_rows)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n_rows))

    df = pd.DataFrame({
        "date": dates, "ticker": "TESTX",
        "open": close + rng.normal(0, 0.2, n_rows), "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": rng.integers(1_000_000, 5_000_000, n_rows),
        "rsi": rng.uniform(20, 80, n_rows),
        "sma_10": close + rng.normal(0, 0.3, n_rows),
        "sma_10_duplicate": close + rng.normal(0, 0.3, n_rows),  # near-duplicate -> correlation_analysis should catch this
        "volume_zscore": rng.normal(0, 1, n_rows),
        "constant_feature": 1.0,  # zero variance -> variance_threshold should catch this
        "trend_label": rng.choice(["Uptrend", "Downtrend", "Sideways"], n_rows),
    })
    # Mostly-NaN "sparse event" style column, like Sprint 2's demand_zone_strength.
    zone_strength = np.full(n_rows, np.nan)
    zone_strength[::11] = rng.uniform(0, 1, len(zone_strength[::11]))
    df["zone_strength"] = zone_strength

    for horizon in (1, 3, 5):
        future_close = df["close"].shift(-horizon)
        df[f"target_{horizon}_day"] = future_close
        df[f"future_return_{horizon}_day"] = (future_close - df["close"]) / df["close"]
        df[f"target_direction_{horizon}_day"] = (future_close > df["close"])
        df[f"target_regression_{horizon}_day"] = df[f"future_return_{horizon}_day"]

    return df


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr("ai.utils.config.settings.processed_data_dir", str(tmp_path / "processed"))
    monkeypatch.setattr("ai.utils.config.settings.models_dir", str(tmp_path / "models"))
    monkeypatch.setattr("ai.utils.config.settings.ml_reports_dir", str(tmp_path / "reports" / "ml"))
    yield tmp_path


@pytest.fixture
def processed_df() -> pd.DataFrame:
    return _make_processed_df()


class TestLoadProcessedDataset:
    def test_load_from_in_memory_df(self, processed_df):
        out = load_processed_dataset(df=processed_df)
        assert len(out) == len(processed_df)
        assert out["date"].is_monotonic_increasing

    def test_missing_ticker_raises(self):
        with pytest.raises(ProcessedDatasetNotFoundError):
            load_processed_dataset(ticker="ZZZZ")

    def test_no_args_raises_value_error(self):
        with pytest.raises(ValueError):
            load_processed_dataset()

    def test_missing_target_columns_raises_schema_error(self):
        bad_df = pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=10), "close": range(10)})
        with pytest.raises(SchemaValidationError):
            load_processed_dataset(df=bad_df)

    def test_find_and_load_by_ticker(self, isolated_storage, processed_df):
        processed_dir = isolated_storage / "processed"
        processed_dir.mkdir(parents=True)
        processed_df.to_csv(processed_dir / "TESTX_v1_features.csv", index=False)

        files = find_processed_dataset_files("TESTX")
        assert len(files) == 1

        out = load_processed_dataset(ticker="TESTX")
        assert len(out) == len(processed_df)


class TestResolveTaskAndValidateTarget:
    def test_regression_target(self):
        assert resolve_task_and_validate_target("future_return_5_day") == "regression"

    def test_classification_target(self):
        assert resolve_task_and_validate_target("target_direction_5_day") == "classification"

    def test_unknown_target_raises(self):
        with pytest.raises(InvalidTargetColumnError):
            resolve_task_and_validate_target("not_a_target")


class TestBuildFeatureMatrix:
    def test_excludes_all_target_family_columns(self, processed_df):
        X, y, dates = build_feature_matrix(processed_df, "future_return_5_day")
        assert not any(c.startswith(("target_", "future_return_")) for c in X.columns)
        assert "date" not in X.columns and "ticker" not in X.columns

    def test_drops_unlabeled_trailing_rows(self, processed_df):
        X, y, dates = build_feature_matrix(processed_df, "target_5_day")
        assert len(X) == N_ROWS - 5  # last 5 rows have no known 5-day-future close
        assert y.isna().sum() == 0

    def test_classification_target_cast_to_int(self, processed_df):
        X, y, dates = build_feature_matrix(processed_df, "target_direction_5_day")
        assert set(y.unique()).issubset({0, 1})

    def test_invalid_target_raises(self, processed_df):
        with pytest.raises(InvalidTargetColumnError):
            build_feature_matrix(processed_df, "not_a_real_target")

    def test_raw_matrix_is_unencoded(self, processed_df):
        """build_feature_matrix must NOT one-hot encode or impute — that's
        preprocessing.FeaturePreprocessor's job, and only after splitting.
        """
        X, y, dates = build_feature_matrix(processed_df, "future_return_5_day")
        assert "trend_label" in X.columns  # still categorical, not dummy-encoded
        assert X["zone_strength"].isna().any()  # still has NaNs


class TestBuildInferenceMatrix:
    def test_keeps_every_row_even_with_nan_targets(self, processed_df):
        X, dates = build_inference_matrix(processed_df)
        assert len(X) == len(processed_df)  # unlike build_feature_matrix, nothing is dropped


class TestTimeSeriesSplitter:
    def test_split_is_contiguous_and_chronological(self, processed_df):
        X, y, dates = build_feature_matrix(processed_df, "future_return_5_day")
        split = TimeSeriesSplitter().split(X, y, dates)

        assert len(split.X_train) + len(split.X_val) + len(split.X_test) == len(X)
        assert split.dates_train.max() <= split.dates_val.min()
        assert split.dates_val.max() <= split.dates_test.min()

    def test_ratios_approximately_respected(self, processed_df):
        X, y, dates = build_feature_matrix(processed_df, "future_return_5_day")
        config = MLConfig(train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2)
        split = TimeSeriesSplitter(config).split(X, y, dates)
        n = len(X)
        assert abs(len(split.X_train) / n - 0.6) < 0.02

    def test_insufficient_data_raises(self, processed_df):
        tiny = processed_df.iloc[:5]
        X, y, dates = build_feature_matrix(tiny, "future_return_5_day")
        with pytest.raises(InsufficientTrainingDataError):
            TimeSeriesSplitter().split(X, y, dates)


class TestFeaturePreprocessorNoLeakage:
    def test_transform_before_fit_raises(self, processed_df):
        X, y, dates = build_feature_matrix(processed_df, "future_return_5_day")
        with pytest.raises(PreprocessorNotFittedError):
            FeaturePreprocessor().transform(X)

    def test_split_and_preprocess_leaves_no_nans(self, processed_df):
        X, y, dates = build_feature_matrix(processed_df, "future_return_5_day")
        split, preprocessor = TimeSeriesSplitter().split_and_preprocess(X, y, dates)
        assert split.X_train.isna().sum().sum() == 0
        assert split.X_val.isna().sum().sum() == 0
        assert split.X_test.isna().sum().sum() == 0

    def test_categorical_column_is_one_hot_encoded(self, processed_df):
        X, y, dates = build_feature_matrix(processed_df, "future_return_5_day")
        split, preprocessor = TimeSeriesSplitter().split_and_preprocess(X, y, dates)
        assert "trend_label" not in split.X_train.columns
        assert any(c.startswith("trend_label_") for c in split.X_train.columns)

    def test_imputation_statistics_come_from_train_only(self):
        """Regression test for the leakage bug this sprint deliberately
        designed around: a column that's NaN in train but has an extreme
        value in the validation/test period must be imputed using ONLY the
        training-period statistic, never a statistic that "knows about" the
        extreme future value.
        """
        n = N_ROWS
        df = _make_processed_df(n_rows=n)
        # Force a column to be NaN everywhere in the training period (first
        # 70%) and a large constant everywhere after — if the preprocessor's
        # median leaked from val/test, the train-period fill value would be
        # pulled toward that large constant.
        train_cutoff = int(n * 0.70)
        df["leak_probe"] = np.nan
        df.loc[df.index[train_cutoff:], "leak_probe"] = 999.0

        X, y, dates = build_feature_matrix(df, "future_return_5_day")
        split, preprocessor = TimeSeriesSplitter().split_and_preprocess(X, y, dates)

        assert preprocessor.fill_values_["leak_probe"] == 0.0  # median of an all-NaN train column, defensively filled to 0.0
        assert (split.X_train["leak_probe"] == 0.0).all()

    def test_drop_rows_strategy_never_leaves_nans(self, processed_df):
        config = MLConfig(nan_strategy="drop_rows", min_rows_required=5)
        X, y, dates = build_feature_matrix(processed_df, "future_return_5_day", config)
        split, preprocessor = TimeSeriesSplitter(config).split_and_preprocess(X, y, dates)
        assert split.X_train.isna().sum().sum() == 0
        assert len(split.X_train) < len(X)  # rows were genuinely dropped, not imputed


class TestFeatureSelector:
    def test_correlation_analysis_drops_near_duplicate(self, processed_df):
        X, y, dates = build_feature_matrix(processed_df, "future_return_5_day")
        split, _ = TimeSeriesSplitter().split_and_preprocess(X, y, dates)
        dropped = FeatureSelector().correlation_analysis(split.X_train)
        assert "sma_10_duplicate" in dropped

    def test_variance_threshold_drops_constant_column(self, processed_df):
        X, y, dates = build_feature_matrix(processed_df, "future_return_5_day")
        split, _ = TimeSeriesSplitter().split_and_preprocess(X, y, dates)
        dropped = FeatureSelector().variance_threshold(split.X_train)
        assert "constant_feature" in dropped

    def test_recommend_feature_subset_respects_top_k(self, processed_df):
        X, y, dates = build_feature_matrix(processed_df, "future_return_5_day")
        split, _ = TimeSeriesSplitter().split_and_preprocess(X, y, dates)
        config = MLConfig(top_k_recommended_features=5)
        result = FeatureSelector(config).recommend_feature_subset(split.X_train, split.y_train, "regression")
        assert len(result.recommended_features) == 5
        assert "constant_feature" not in result.recommended_features
        assert "sma_10_duplicate" not in result.recommended_features