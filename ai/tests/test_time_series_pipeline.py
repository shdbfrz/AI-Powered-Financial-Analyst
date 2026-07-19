"""
Tests for `ai/models/time_series/pipelines/`, `persistence/`, and
`prediction/forecast_service.py` (Sprint 4).
"""

import numpy as np
import pandas as pd
import pytest

from ai.models.time_series.exceptions import ModelPersistenceError
from ai.models.time_series.models.arima_model import ArimaModel
from ai.models.time_series.persistence.model_storage import TimeSeriesModelStorage
from ai.models.time_series.pipelines.forecasting_pipeline import walk_forward_validate
from ai.models.time_series.pipelines.training_pipeline import TimeSeriesTrainingPipeline
from ai.models.time_series.prediction.forecast_service import ForecastService

N_ROWS = 250


def _make_processed_df(n_rows: int = N_ROWS, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_rows)
    trend = np.linspace(0, 25, n_rows)
    season = 2.5 * np.sin(np.arange(n_rows) * 2 * np.pi / 5)
    close = 100 + trend + season + np.cumsum(rng.normal(0, 0.4, n_rows))
    return pd.DataFrame(
        {
            "date": dates, "ticker": "TESTX",
            "open": close - 0.3, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": rng.integers(1_000_000, 5_000_000, n_rows),
        }
    )


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr("ai.utils.config.settings.processed_data_dir", str(tmp_path / "processed"))
    monkeypatch.setattr("ai.utils.config.settings.models_dir", str(tmp_path / "models"))
    yield tmp_path


@pytest.fixture
def processed_dataset(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / "TESTX_20250101T000000Z_features.csv"
    _make_processed_df().to_csv(path, index=False)
    return path


class TestTrainingPipeline:
    def test_run_produces_results_for_every_horizon(self, processed_dataset):
        pipeline = TimeSeriesTrainingPipeline(persist=False)
        result = pipeline.run("TESTX", model_names=["arima", "exponential_smoothing"], horizons=(1, 3))

        assert set(result.model_results.keys()) == {1, 3}
        for horizon, models in result.model_results.items():
            assert len(models) == 2
            assert any(r.error is None for r in models)

        for horizon, table in result.comparison_tables.items():
            assert len(table.dataframe) >= 1
            assert "rank" in table.dataframe.columns

    def test_one_bad_model_does_not_abort_the_run(self, processed_dataset):
        pipeline = TimeSeriesTrainingPipeline(persist=False)
        result = pipeline.run("TESTX", model_names=["arima", "not_a_real_model"], horizons=(1,))
        results = result.model_results[1]
        names_with_error = {r.model_name for r in results if r.error is not None}
        assert "not_a_real_model" in names_with_error
        assert any(r.error is None for r in results)

    def test_persists_artifacts_when_enabled(self, processed_dataset, tmp_path):
        pipeline = TimeSeriesTrainingPipeline(persist=True)
        pipeline.run("TESTX", model_names=["arima"], horizons=(1,))
        metadata_path = tmp_path / "models" / "time_series" / "Model_Metadata.json"
        assert metadata_path.exists()


class TestPersistence:
    def test_save_before_fit_raises(self, tmp_path):
        storage = TimeSeriesModelStorage()
        model = ArimaModel(order=(1, 1, 1))
        with pytest.raises(ModelPersistenceError):
            storage.save(model, ticker="TESTX", horizon_days=5, test_metrics={})

    def test_save_and_load_roundtrip(self, processed_dataset, tmp_path):
        from ai.models.time_series.data_loader import load_price_series

        series = load_price_series("TESTX", path=processed_dataset)
        model = ArimaModel(order=(1, 1, 1)).fit(series)

        storage = TimeSeriesModelStorage()
        info = storage.save(model, ticker="TESTX", horizon_days=5, test_metrics={"rmse": 1.0})
        assert info.path.exists()

        loaded = storage.load(info.path)
        result = loaded.forecast(3)
        assert len(result.forecast) == 3


class TestWalkForwardValidation:
    def test_produces_pooled_metrics(self, processed_dataset):
        from ai.models.time_series.data_loader import load_price_series

        series = load_price_series("TESTX", path=processed_dataset)
        result = walk_forward_validate(
            lambda: ArimaModel(order=(1, 1, 1)), series, min_train_size=150, step=25
        )
        assert result.n_folds >= 1
        assert result.metrics.rmse >= 0


class TestForecastService:
    def test_fresh_fit_forecast(self, processed_dataset):
        service = ForecastService()
        response = service.forecast(model="arima", ticker="TESTX", horizon=5, model_kwargs={"order": (1, 1, 1)})
        assert len(response.points) == 5
        assert response.educational_disclaimer
        assert response.model_name == "arima"

    def test_cached_artifact_requires_prior_save(self, processed_dataset):
        service = ForecastService()
        with pytest.raises(ModelPersistenceError):
            service.forecast(model="arima", ticker="TESTX", horizon=5, use_cached_artifact=True)

    def test_persist_then_use_cached(self, processed_dataset):
        service = ForecastService()
        service.forecast(
            model="arima", ticker="TESTX", horizon=5, persist=True, model_kwargs={"order": (1, 1, 1)}
        )
        response = service.forecast(model="arima", ticker="TESTX", horizon=5, use_cached_artifact=True)
        assert len(response.points) == 5

    def test_forecast_all_models_skips_failures(self, processed_dataset):
        service = ForecastService()
        responses = service.forecast_all_models("TESTX", horizon=3, models=["arima", "exponential_smoothing"])
        assert "arima" in responses
        assert "exponential_smoothing" in responses

    def test_invalid_horizon_raises(self, processed_dataset):
        service = ForecastService()
        with pytest.raises(ValueError):
            service.forecast(model="arima", ticker="TESTX", horizon=0)