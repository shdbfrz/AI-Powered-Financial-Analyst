"""
Tests for `ai/models/ml/evaluation/`, `persistence/`, `tuning/`,
`pipelines/training_pipeline.py`, and `prediction/inference.py` (Sprint 3).
"""

import numpy as np
import pandas as pd
import pytest

from ai.models.ml.evaluation.comparator import ModelComparator, ModelResult
from ai.models.ml.evaluation.metrics import compute_classification_metrics, compute_regression_metrics
from ai.models.ml.exceptions import ModelPersistenceError
from ai.models.ml.models import ModelFactory
from ai.models.ml.persistence import ModelStorage, estimate_memory_bytes
from ai.models.ml.pipelines import MLTrainingPipeline
from ai.models.ml.prediction import InferenceService
from ai.models.ml.tuning import HyperparameterTuner

N_ROWS = 400


def _make_processed_df(n_rows: int = N_ROWS, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_rows)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n_rows))
    df = pd.DataFrame({
        "date": dates, "ticker": "TESTX",
        "open": close + rng.normal(0, 0.2, n_rows), "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": rng.integers(1_000_000, 5_000_000, n_rows),
        "rsi": rng.uniform(20, 80, n_rows),
        "momentum": rng.normal(0, 1, n_rows),
        "volatility": rng.uniform(0.1, 2.0, n_rows),
        "trend_label": rng.choice(["Uptrend", "Downtrend", "Sideways"], n_rows),
    })
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


class TestRegressionMetrics:
    def test_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 5.0])  # one point off by 1
        metrics = compute_regression_metrics(y_true, y_pred, n_features=1)
        assert metrics.mae == pytest.approx(0.25)
        assert metrics.mse == pytest.approx(0.25)
        assert metrics.rmse == pytest.approx(0.5)
        assert metrics.r2 < 1.0

    def test_mape_excludes_near_zero_true_values(self):
        y_true = np.array([0.0, 0.0000001, 10.0, 10.0])
        y_pred = np.array([5.0, 5.0, 11.0, 9.0])
        metrics = compute_regression_metrics(y_true, y_pred, n_features=1)
        assert metrics.mape is not None
        assert metrics.mape < 1000  # would be astronomically large if the near-zero rows weren't excluded

    def test_adjusted_r2_none_when_underdetermined(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 2.1])
        metrics = compute_regression_metrics(y_true, y_pred, n_features=5)  # n - n_features - 1 <= 0
        assert metrics.adjusted_r2 is None


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 1, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics.accuracy == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.confusion_matrix == [[2, 0], [0, 3]]

    def test_roc_auc_none_with_single_class(self):
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 0, 1]
        proba = [0.9, 0.8, 0.3, 0.7]
        metrics = compute_classification_metrics(y_true, y_pred, proba)
        assert metrics.roc_auc is None  # undefined with only one class present

    def test_roc_auc_computed_with_both_classes(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        proba = [0.2, 0.9, 0.6, 0.8]
        metrics = compute_classification_metrics(y_true, y_pred, proba)
        assert metrics.roc_auc is not None


class TestModelComparator:
    def _fake_result(self, name, rmse, training_time=1.0, complexity=100, memory=1000):
        return ModelResult(
            model_name=name, display_name=name.title(), task_type="regression", target_column="future_return_5_day",
            hyperparameters={}, train_metrics={"rmse": rmse}, validation_metrics={"rmse": rmse},
            test_metrics={"rmse": rmse}, training_time_seconds=training_time, prediction_time_seconds=0.001,
            complexity_score=complexity, memory_bytes=memory,
        )

    def test_lower_rmse_ranks_first(self):
        results = [self._fake_result("bad", 0.5), self._fake_result("good", 0.1), self._fake_result("medium", 0.3)]
        table = ModelComparator().build_comparison_table(results, "rmse", "minimize")
        assert table.dataframe.iloc[0]["model_name"] == "good"
        assert table.dataframe.iloc[0]["performance_rank"] == 1
        assert table.narrative  # non-empty narrative generated

    def test_failed_models_excluded_from_ranking(self):
        ok = self._fake_result("ok", 0.2)
        failed = ModelResult(
            model_name="broken", display_name="Broken", task_type="regression", target_column="x",
            hyperparameters={}, train_metrics={}, validation_metrics={}, test_metrics={},
            training_time_seconds=0, prediction_time_seconds=0, complexity_score=0, memory_bytes=0,
            error="did not converge",
        )
        table = ModelComparator().build_comparison_table([ok, failed], "rmse", "minimize")
        assert len(table.dataframe) == 1
        assert table.dataframe.iloc[0]["model_name"] == "ok"

    def test_all_models_failed_returns_empty_table(self):
        failed = ModelResult(
            model_name="broken", display_name="Broken", task_type="regression", target_column="x",
            hyperparameters={}, train_metrics={}, validation_metrics={}, test_metrics={},
            training_time_seconds=0, prediction_time_seconds=0, complexity_score=0, memory_bytes=0,
            error="boom",
        )
        table = ModelComparator().build_comparison_table([failed], "rmse", "minimize")
        assert table.dataframe.empty
        assert "No model trained successfully" in table.narrative


class TestModelStorage:
    def test_save_load_roundtrip(self):
        X = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 3)), columns=["a", "b", "c"])
        y = X["a"] * 2 + 1
        model = ModelFactory.create("ridge_regression")
        model.fit(X, y)
        preds_before = model.predict(X)

        storage = ModelStorage()
        saved = storage.save(model, ticker="TESTX", target_column="future_return_5_day", version="v1", test_metrics={"rmse": 0.1})
        assert saved.path.exists()

        loaded = storage.load(saved.path)
        assert loaded.is_fitted
        assert np.allclose(loaded.predict(X), preds_before)

    def test_save_unfitted_model_raises(self):
        model = ModelFactory.create("ridge_regression")
        with pytest.raises(ModelPersistenceError):
            ModelStorage().save(model, ticker="TESTX", target_column="x", version="v1", test_metrics={})

    def test_load_missing_path_raises(self, tmp_path):
        with pytest.raises(ModelPersistenceError):
            ModelStorage().load(tmp_path / "does_not_exist.joblib")

    def test_metadata_json_written(self):
        X = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 2)), columns=["a", "b"])
        y = X["a"]
        model = ModelFactory.create("linear_regression")
        model.fit(X, y)
        storage = ModelStorage()
        saved = storage.save(model, ticker="TESTX", target_column="future_return_5_day", version="v1", test_metrics={"rmse": 0.2})
        path = storage.save_metadata_json([saved], "TESTX", "v1")
        assert path.exists()
        import json
        payload = json.loads(path.read_text())
        assert payload["models"][0]["model_name"] == "linear_regression"

    def test_estimate_memory_bytes_positive(self):
        X = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 2)), columns=["a", "b"])
        y = X["a"]
        model = ModelFactory.create("random_forest")
        model.fit(X, y)
        assert estimate_memory_bytes(model.estimator) > 0


class TestHyperparameterTuner:
    def test_tune_ridge_returns_best_params(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
        y = X["a"] * 3 + rng.normal(scale=0.1, size=200)
        model = ModelFactory.create("ridge_regression")
        tuner = HyperparameterTuner()
        tuned_model, result = tuner.tune(model, X, y, method="random", n_iter=4, cv_splits=3)
        assert result.method == "random"
        assert "alpha" in result.best_params
        assert tuned_model.is_fitted is False  # tune() returns an unfitted model; caller fits the final split

    def test_model_with_no_grid_skips_tuning(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc"))
        y = X["a"]
        model = ModelFactory.create("linear_regression")  # no registered hyperparameter grid
        tuner = HyperparameterTuner()
        _, result = tuner.tune(model, X, y)
        assert result.method == "none"


class TestMLTrainingPipelineIntegration:
    def test_full_regression_run_produces_expected_artifacts(self, processed_df):
        pipeline = MLTrainingPipeline()
        result = pipeline.run(
            df=processed_df, task="regression", target_column="future_return_5_day",
            model_names=["linear_regression", "ridge_regression"], version="test-v1",
        )
        assert result.best_model_name in {"linear_regression", "ridge_regression"}
        assert len(result.model_results) == 2
        assert result.comparison_csv_path.exists()
        assert result.evaluation_report_path.exists()
        assert result.model_documentation_path.exists()
        assert result.metadata_json_path.exists()
        assert len(result.saved_models) == 2
        assert len(result.plot_paths) > 0
        assert all(p.exists() for p in result.plot_paths)

    def test_full_classification_run(self, processed_df):
        pipeline = MLTrainingPipeline()
        result = pipeline.run(
            df=processed_df, task="classification", target_column="target_direction_5_day",
            model_names=["logistic_regression"], generate_plots=False, version="test-v1",
        )
        assert result.best_model_name == "logistic_regression"
        assert result.model_results[0].test_metrics["accuracy"] is not None

    def test_partial_model_failure_does_not_abort_run(self, processed_df, monkeypatch):
        """One model raising during fit must not prevent the others from
        completing — this is exactly the resilience the Sprint 3 spec's
        "Model Training Failure" error-handling requirement is about.
        """
        pipeline = MLTrainingPipeline()
        original_create = ModelFactory.create

        def _flaky_create(name, **kwargs):
            model = original_create(name, **kwargs)
            if name == "ridge_regression":
                def _broken_fit(*args, **kwargs):
                    raise RuntimeError("simulated training failure")
                model.estimator.fit = _broken_fit
            return model

        monkeypatch.setattr("ai.models.ml.pipelines.training_pipeline.ModelFactory.create", _flaky_create)

        result = pipeline.run(
            df=processed_df, task="regression", target_column="future_return_5_day",
            model_names=["linear_regression", "ridge_regression"], generate_plots=False, version="test-v1",
        )
        by_name = {r.model_name: r for r in result.model_results}
        assert by_name["linear_regression"].error is None
        assert by_name["ridge_regression"].error is not None

    def test_use_feature_selection_reduces_feature_count(self, processed_df):
        from ai.models.ml.config import MLConfig
        pipeline = MLTrainingPipeline(MLConfig(top_k_recommended_features=3))
        result = pipeline.run(
            df=processed_df, task="regression", target_column="future_return_5_day",
            model_names=["linear_regression"], use_feature_selection=True, generate_plots=False, version="test-v1",
        )
        assert result.feature_selection_result is not None
        assert result.dataset_info["n_features"] == 3


class TestInferenceServiceIntegration:
    def test_predict_latest_after_training(self, processed_df):
        pipeline = MLTrainingPipeline()
        pipeline.run(
            df=processed_df, task="regression", target_column="future_return_5_day",
            model_names=["ridge_regression"], generate_plots=False, version="test-v1",
        )

        service = InferenceService()
        result = service.predict_latest(
            ticker="TESTX", target_column="future_return_5_day", model_name="ridge_regression",
            processed_df=processed_df, n_rows=3,
        )
        assert len(result.predictions) == 3
        assert result.task_type == "regression"

    def test_unknown_model_name_raises(self):
        with pytest.raises(Exception):
            InferenceService().predict_latest(
                ticker="TESTX", target_column="future_return_5_day", model_name="not_a_model",
            )