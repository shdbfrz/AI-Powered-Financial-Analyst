"""
Tests for `ai/models/ml/models/` (Sprint 3): the `ModelFactory` registry and
every registered `BaseMLModel` implementation.

Fully offline — a small synthetic feature matrix is generated in-memory,
mirroring the pattern in `ai/tests/test_feature_engineering.py`.
"""

import numpy as np
import pandas as pd
import pytest

from ai.models.ml.exceptions import ModelNotFittedError, PredictionError, UnknownModelError
from ai.models.ml.models import ModelFactory
from ai.models.ml.models.base import ModelInfo


def _make_xy(n_rows: int = 300, n_features: int = 6, seed: int = 42, classification: bool = False):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n_rows, n_features)), columns=[f"feature_{i}" for i in range(n_features)])
    signal = 2.0 * X["feature_0"] - 1.0 * X["feature_1"] + 0.5 * X["feature_2"]
    y = signal + rng.normal(scale=0.2, size=n_rows)
    if classification:
        y = (y > y.median()).astype(int)
    return X, y


class TestModelFactory:
    def test_available_models_counts(self):
        regression = ModelFactory.available_models("regression")
        classification = ModelFactory.available_models("classification")
        # 10 required + up to 2 optional (lightgbm/catboost) regression models
        assert 10 <= len(regression) <= 12
        assert len(classification) == 3
        assert set(classification) == {"logistic_regression", "random_forest_classifier", "xgboost_classifier"}

    def test_create_unknown_model_raises(self):
        with pytest.raises(UnknownModelError):
            ModelFactory.create("not_a_real_model")

    def test_create_all_unknown_name_raises(self):
        with pytest.raises(UnknownModelError):
            ModelFactory.create_all("regression", names=["random_forest", "not_a_real_model"])

    def test_get_model_info_returns_model_info(self):
        info = ModelFactory.get_model_info("xgboost")
        assert isinstance(info, ModelInfo)
        assert info.name == "xgboost"
        assert info.task_type == "regression"

    def test_all_model_info_covers_every_registered_model(self):
        infos = ModelFactory.all_model_info()
        assert {i.name for i in infos} == set(ModelFactory.available_models())
        # Every model must document all four fields the Sprint 3 spec asks for.
        for info in infos:
            assert info.purpose and info.advantages and info.limitations and info.best_use_cases

    def test_create_all_regression(self):
        models = ModelFactory.create_all("regression")
        assert set(models) == set(ModelFactory.available_models("regression"))


class TestEveryRegisteredModel:
    """Every model registered for either task must satisfy the same contract."""

    @pytest.mark.parametrize("model_name", ModelFactory.available_models("regression"))
    def test_regression_model_fit_predict_roundtrip(self, model_name):
        X, y = _make_xy(classification=False)
        X_train, X_test = X.iloc[:240], X.iloc[240:]
        y_train, y_test = y.iloc[:240], y.iloc[240:]

        model = ModelFactory.create(model_name)
        assert model.is_fitted is False

        model.fit(X_train, y_train)
        assert model.is_fitted is True
        assert model.last_fit_seconds >= 0.0

        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test),) == y_test.shape
        assert np.isfinite(predictions).all()

        assert model.estimate_complexity() > 0

        importance = model.get_feature_importance()
        assert importance is not None
        assert set(importance.index) == set(X_train.columns)
        assert (importance >= 0).all()

    @pytest.mark.parametrize("model_name", ModelFactory.available_models("classification"))
    def test_classification_model_fit_predict_proba_roundtrip(self, model_name):
        X, y = _make_xy(classification=True)
        X_train, X_test = X.iloc[:240], X.iloc[240:]
        y_train, y_test = y.iloc[:240], y.iloc[240:]

        model = ModelFactory.create(model_name)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
        assert set(np.unique(predictions)).issubset({0, 1})

        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestBaseMLModelGuards:
    def test_predict_before_fit_raises(self):
        model = ModelFactory.create("random_forest")
        X, _ = _make_xy()
        with pytest.raises(ModelNotFittedError):
            model.predict(X)

    def test_feature_importance_before_fit_raises(self):
        model = ModelFactory.create("ridge_regression")
        with pytest.raises(ModelNotFittedError):
            model.get_feature_importance()

    def test_predict_with_missing_feature_raises(self):
        X, y = _make_xy()
        model = ModelFactory.create("random_forest")
        model.fit(X, y)
        with pytest.raises(PredictionError):
            model.predict(X.drop(columns=["feature_0"]))

    def test_fit_on_empty_data_raises(self):
        model = ModelFactory.create("linear_regression")
        empty_X = pd.DataFrame(columns=["a", "b"])
        empty_y = pd.Series(dtype=float)
        with pytest.raises(Exception):  # ModelTrainingError
            model.fit(empty_X, empty_y)


class TestLinearModelsAreScaled:
    """Ridge/Lasso/ElasticNet/Logistic must be wrapped in a StandardScaler
    Pipeline (scale-sensitive regularization); tree/ensemble/boosting models
    must not be (scaling a tree split is a no-op, so wrapping would just add
    overhead).
    """

    @pytest.mark.parametrize("model_name", ["ridge_regression", "lasso_regression", "elastic_net", "logistic_regression"])
    def test_linear_family_is_pipeline_wrapped(self, model_name):
        from sklearn.pipeline import Pipeline
        model = ModelFactory.create(model_name)
        assert isinstance(model.estimator, Pipeline)
        assert "scaler" in model.estimator.named_steps

    @pytest.mark.parametrize("model_name", ["random_forest", "xgboost", "decision_tree"])
    def test_tree_family_is_not_pipeline_wrapped(self, model_name):
        from sklearn.pipeline import Pipeline
        model = ModelFactory.create(model_name)
        assert not isinstance(model.estimator, Pipeline)


class TestOptionalDependencies:
    def test_lightgbm_catboost_flagged_optional_when_present(self):
        for name in ("lightgbm", "catboost"):
            if name in ModelFactory.available_models("regression"):
                info = ModelFactory.get_model_info(name)
                assert info.is_optional_dependency is True

    def test_required_models_not_flagged_optional(self):
        for name in ("linear_regression", "random_forest", "xgboost"):
            assert ModelFactory.get_model_info(name).is_optional_dependency is False