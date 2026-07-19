"""
Tests for `ai/models/time_series/models/` and `evaluation/` (Sprint 4).
"""

import numpy as np
import pandas as pd
import pytest

from ai.models.time_series.evaluation.comparator import TimeSeriesModelComparator, TimeSeriesModelResult
from ai.models.time_series.evaluation.metrics import (
    directional_accuracy,
    evaluate_forecast,
    forecast_bias,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
)
from ai.models.time_series.exceptions import ForecastError, ModelNotFittedError, ModelTrainingError, UnknownModelError
from ai.models.time_series.models.arima_model import ArimaModel
from ai.models.time_series.models.exponential_smoothing_model import ExponentialSmoothingModel
from ai.models.time_series.models.registry import TimeSeriesModelFactory
from ai.models.time_series.models.sarima_model import SarimaModel

N_ROWS = 200


def _make_series(n_rows: int = N_ROWS, seed: int = 5) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_rows)
    trend = np.linspace(0, 20, n_rows)
    season = 2 * np.sin(np.arange(n_rows) * 2 * np.pi / 5)
    close = 100 + trend + season + np.cumsum(rng.normal(0, 0.4, n_rows))
    return pd.Series(close, index=pd.DatetimeIndex(dates), name="close")


class TestRegressionMetrics:
    def test_known_values(self):
        actual = pd.Series([1.0, 2.0, 3.0, 4.0])
        predicted = pd.Series([1.0, 2.0, 3.0, 5.0])
        assert mean_absolute_error(actual, predicted) == pytest.approx(0.25)
        assert root_mean_squared_error(actual, predicted) == pytest.approx(0.5)

    def test_mape_excludes_near_zero_actuals(self):
        actual = pd.Series([0.0, 0.0000001, 10.0, 10.0])
        predicted = pd.Series([5.0, 5.0, 11.0, 9.0])
        mape = mean_absolute_percentage_error(actual, predicted)
        assert np.isfinite(mape)

    def test_smape_is_bounded(self):
        actual = pd.Series([10.0, 20.0, 30.0])
        predicted = pd.Series([12.0, 18.0, 33.0])
        smape = symmetric_mean_absolute_percentage_error(actual, predicted)
        assert 0.0 <= smape <= 200.0

    def test_directional_accuracy_perfect(self):
        previous = 100.0
        actual = pd.Series([101.0, 103.0, 102.0])
        predicted = pd.Series([101.5, 103.5, 102.5])  # same up/down/down pattern
        acc = directional_accuracy(actual, predicted, previous_value=previous)
        assert acc == pytest.approx(1.0)

    def test_forecast_bias_sign(self):
        actual = pd.Series([10.0, 10.0, 10.0])
        over_forecast = pd.Series([11.0, 11.0, 11.0])
        assert forecast_bias(actual, over_forecast) > 0

    def test_evaluate_forecast_bundles_all_metrics(self):
        actual = pd.Series([100.0, 101.0, 102.0])
        predicted = pd.Series([100.5, 101.5, 101.5])
        metrics = evaluate_forecast(actual, predicted, previous_value=99.0)
        d = metrics.to_dict()
        for key in ("mae", "mse", "rmse", "mape", "smape", "r2", "directional_accuracy", "forecast_bias"):
            assert key in d


class TestBaseModelTemplateMethod:
    def test_forecast_before_fit_raises(self):
        model = ArimaModel(order=(1, 1, 1))
        with pytest.raises(ModelNotFittedError):
            model.forecast(5)

    def test_forecast_non_positive_steps_raises(self):
        model = ArimaModel(order=(1, 1, 1))
        model.fit(_make_series())
        with pytest.raises(ForecastError):
            model.forecast(0)

    def test_empty_series_raises_training_error(self):
        model = ArimaModel(order=(1, 1, 1))
        empty = pd.Series([np.nan, np.nan], index=pd.bdate_range("2023-01-02", periods=2))
        with pytest.raises(ModelTrainingError):
            model.fit(empty)

    def test_is_fitted_flag(self):
        model = ArimaModel(order=(1, 1, 1))
        assert model.is_fitted is False
        model.fit(_make_series())
        assert model.is_fitted is True


class TestArimaModel:
    def test_fit_forecast_explicit_order(self):
        model = ArimaModel(order=(1, 1, 1))
        model.fit(_make_series())
        result = model.forecast(5)
        assert len(result.forecast) == 5
        assert result.lower_bound is not None
        assert (result.lower_bound <= result.upper_bound).all()

    def test_auto_order_selection(self):
        model = ArimaModel(max_p=2, max_q=2)
        model.fit(_make_series())
        result = model.forecast(3)
        assert model.get_params()["order"] is not None
        assert len(result.forecast) == 3

    def test_forecast_index_is_future_business_days(self):
        series = _make_series()
        model = ArimaModel(order=(1, 1, 1))
        model.fit(series)
        result = model.forecast(5)
        assert result.forecast.index.min() > series.index.max()


class TestSarimaModel:
    def test_fit_forecast_with_explicit_orders(self):
        model = SarimaModel(order=(1, 1, 0), seasonal_order=(0, 1, 0, 5))
        model.fit(_make_series())
        result = model.forecast(5)
        assert len(result.forecast) == 5
        assert result.lower_bound is not None


class TestExponentialSmoothingModel:
    def test_fit_forecast(self):
        model = ExponentialSmoothingModel(seasonal_period=5)
        model.fit(_make_series())
        result = model.forecast(5)
        assert len(result.forecast) == 5
        assert (result.lower_bound <= result.forecast).all()
        assert (result.forecast <= result.upper_bound).all()

    def test_short_series_skips_seasonal_component(self):
        model = ExponentialSmoothingModel(seasonal_period=5)
        model.fit(_make_series(n_rows=8))
        assert model.get_params()["seasonal_applied"] is False


class TestModelRegistry:
    def test_unknown_model_raises(self):
        with pytest.raises(UnknownModelError):
            TimeSeriesModelFactory.create("not_a_real_model")

    def test_required_models_always_available(self):
        available = TimeSeriesModelFactory.available_models()
        for required in ("arima", "sarima", "exponential_smoothing"):
            assert required in available

    def test_registered_model_names_includes_optional(self):
        names = TimeSeriesModelFactory.registered_model_names()
        assert "prophet" in names
        assert "auto_arima" in names


class TestModelComparator:
    def test_ranks_by_primary_metric(self):
        good_metrics = evaluate_forecast(pd.Series([1, 2, 3]), pd.Series([1.0, 2.0, 3.0]), previous_value=0.5)
        bad_metrics = evaluate_forecast(pd.Series([1, 2, 3]), pd.Series([5.0, 6.0, 7.0]), previous_value=0.5)

        results = [
            TimeSeriesModelResult(
                model_name="good_model", horizon_days=5, hyperparameters={},
                validation_metrics=good_metrics, test_metrics=good_metrics,
                training_time_seconds=1.0, prediction_time_seconds=0.1,
            ),
            TimeSeriesModelResult(
                model_name="bad_model", horizon_days=5, hyperparameters={},
                validation_metrics=bad_metrics, test_metrics=bad_metrics,
                training_time_seconds=1.0, prediction_time_seconds=0.1,
            ),
        ]
        table = TimeSeriesModelComparator(primary_metric="rmse", primary_metric_direction="minimize").compare(
            results, horizon_days=5
        )
        assert table.dataframe.iloc[0]["model_name"] == "good_model"

    def test_raises_when_all_models_failed(self):
        results = [
            TimeSeriesModelResult(
                model_name="failed", horizon_days=5, hyperparameters={},
                validation_metrics=None, test_metrics=None,
                training_time_seconds=0.0, prediction_time_seconds=0.0, error="boom",
            )
        ]
        with pytest.raises(ValueError):
            TimeSeriesModelComparator().compare(results, horizon_days=5)