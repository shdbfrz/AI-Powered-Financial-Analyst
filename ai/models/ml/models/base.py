"""
Base abstractions for `ai/models/ml/models/`.

Every concrete model (Linear Regression, Random Forest, XGBoost, ...)
implements `BaseMLModel` so the rest of the pipeline (training, tuning,
evaluation, persistence) can treat all fifteen models uniformly (Strategy
pattern, mirroring `ai.data_collection.base_provider.BaseDataProvider`):
fit/predict/feature-importance/complexity all go through one contract, so
adding model #16 later never requires touching the pipeline.

Each concrete class also declares a class-level `info: ModelInfo` — its
purpose, advantages, limitations, and best use cases — which feeds
`Model_Documentation.md` (Sprint 3 output requirement) the same way
`FeatureDefinition` feeds `Feature_Report.md` in Sprint 2, so the
documentation is generated from a single source of truth instead of being
maintained by hand in two places.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from ai.models.ml.exceptions import ModelNotFittedError, ModelTrainingError, PredictionError
from ai.utils.logger import get_logger

TaskType = Literal["regression", "classification"]
ModelFamily = Literal["linear", "tree", "ensemble", "boosting"]


@dataclass(frozen=True)
class ModelInfo:
    """Documents one model for `Model_Documentation.md` / `Model_Metadata.json`."""

    name: str  # short, stable identifier used in the registry and file names, e.g. "random_forest"
    display_name: str  # human-readable, e.g. "Random Forest"
    family: ModelFamily
    task_type: TaskType
    purpose: str
    advantages: str
    limitations: str
    best_use_cases: str
    is_optional_dependency: bool = False  # True for LightGBM/CatBoost: pipeline degrades gracefully if not installed
    recommended_for: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "family": self.family,
            "task_type": self.task_type,
            "purpose": self.purpose,
            "advantages": self.advantages,
            "limitations": self.limitations,
            "best_use_cases": self.best_use_cases,
            "is_optional_dependency": self.is_optional_dependency,
            "recommended_for": list(self.recommended_for),
        }


class BaseMLModel(ABC):
    """Template-method base class for a single ML model.

    Subclasses implement `_build_estimator` (return a scikit-learn-compatible
    estimator) and declare `default_hyperparameters()` + a class-level
    `info`. `fit`/`predict` wrap the estimator with validation, timing, and
    consistent error handling so subclasses stay focused on which estimator
    to construct.
    """

    info: ModelInfo  # set by every concrete subclass

    def __init__(self, **hyperparameter_overrides: Any):
        self.hyperparameters: dict = {**self.default_hyperparameters(), **hyperparameter_overrides}
        self.estimator = self._build_estimator(self.hyperparameters)
        self.is_fitted: bool = False
        self.feature_names_: list[str] = []
        self.last_fit_seconds: float = 0.0
        self.last_predict_seconds: float = 0.0
        self.logger = get_logger(f"models.ml.{self.info.name}")

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def default_hyperparameters(cls) -> dict:
        """Sane, documented defaults — used whenever the caller doesn't
        override a hyperparameter and as the tuner's search-space center.
        """

    @abstractmethod
    def _build_estimator(self, params: dict):
        """Construct and return the underlying scikit-learn-compatible estimator."""

    # ------------------------------------------------------------------
    # Template methods (shared by every model)
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseMLModel":
        """Fit the estimator, recording feature names and wall-clock time.
        Never mutates the caller's `X`/`y`.
        """
        if len(X) == 0 or len(y) == 0:
            raise ModelTrainingError(self.info.name, "cannot fit on empty X/y")
        if len(X) != len(y):
            raise ModelTrainingError(self.info.name, f"X has {len(X)} rows but y has {len(y)}")

        start = time.perf_counter()
        try:
            self.estimator.fit(X, y)
        except Exception as e:  # noqa: BLE001 - convert to typed error, never fail silently
            raise ModelTrainingError(self.info.name, str(e)) from e
        self.last_fit_seconds = time.perf_counter() - start

        self.is_fitted = True
        self.feature_names_ = list(X.columns)
        self.logger.info(
            "Fitted %s on %d rows x %d features in %.3fs",
            self.info.display_name, len(X), X.shape[1], self.last_fit_seconds,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict, recording wall-clock prediction time."""
        self._check_fitted()
        self._check_feature_alignment(X)
        start = time.perf_counter()
        try:
            predictions = self.estimator.predict(X)
        except Exception as e:  # noqa: BLE001
            raise PredictionError(self.info.name, str(e)) from e
        self.last_predict_seconds = time.perf_counter() - start
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class probabilities. Only meaningful for classifiers whose
        estimator implements `predict_proba` (Logistic Regression, Random
        Forest Classifier, XGBoost Classifier all do).
        """
        self._check_fitted()
        if not hasattr(self.estimator, "predict_proba"):
            raise PredictionError(self.info.name, "underlying estimator does not support predict_proba")
        self._check_feature_alignment(X)
        try:
            return self.estimator.predict_proba(X)
        except Exception as e:  # noqa: BLE001
            raise PredictionError(self.info.name, str(e)) from e

    def _final_estimator(self):
        """Unwrap a scikit-learn `Pipeline` (used by the linear model family
        to bundle `StandardScaler` + the estimator) to its last step, so
        `coef_` / `feature_importances_` / `tree_` introspection works
        regardless of whether `self.estimator` is a raw estimator or a
        Pipeline — `Pipeline` proxies `fit`/`predict`/`predict_proba`
        automatically but not these fitted attributes.
        """
        from sklearn.pipeline import Pipeline
        return self.estimator[-1] if isinstance(self.estimator, Pipeline) else self.estimator

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Feature importance/coefficient magnitude, descending. `None` if
        the underlying estimator exposes neither (none of the fifteen models
        registered in Sprint 3 fall into this bucket, but the contract stays
        defensive for future additions).
        """
        self._check_fitted()
        est = self._final_estimator()
        if hasattr(est, "feature_importances_"):
            values = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            coef = np.asarray(est.coef_, dtype=float)
            values = np.abs(coef.ravel()) if coef.ndim > 1 else np.abs(coef)
        else:
            return None
        return pd.Series(values, index=self.feature_names_, name="importance").sort_values(ascending=False)

    def estimate_complexity(self) -> int:
        """A relative complexity score for the Model Comparison table — NOT a
        precise parameter/FLOP count, just a consistent, comparable proxy:
        (a) tree ensembles: n_estimators x 2^depth (bounded)
        (b) single trees: actual node count from the fitted tree
        (c) linear models: number of coefficients
        (d) fallback: number of input features
        """
        self._check_fitted()
        est = self._final_estimator()
        if hasattr(est, "tree_") and hasattr(est.tree_, "node_count"):
            return int(est.tree_.node_count)
        if hasattr(est, "tree_count_"):
            # CatBoost: exposes the actual number of trees built (may differ
            # from the `iterations` hyperparameter under early stopping) but
            # not a sklearn-style `max_depth` attribute; approximate using
            # CatBoost's own default depth (6) since this is a relative
            # ranking signal, not an exact count.
            return int(est.tree_count_) * int(2 ** 6)
        if hasattr(est, "n_estimators"):
            depth = getattr(est, "max_depth", None)
            if depth is None or depth <= 0:  # -1/None conventionally mean "unbounded" (e.g. LightGBM)
                depth = 8
            return int(est.n_estimators) * int(2 ** min(int(depth), 16))
        if hasattr(est, "coef_"):
            return int(np.asarray(est.coef_).size) + 1
        return len(self.feature_names_) or 1

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise ModelNotFittedError(self.info.name)

    def _check_feature_alignment(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise PredictionError(
                self.info.name,
                f"input is missing {len(missing)} feature(s) the model was trained on",
                context={"missing_features": missing[:10]},
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"<{self.__class__.__name__} name='{self.info.name}' status={status}>"