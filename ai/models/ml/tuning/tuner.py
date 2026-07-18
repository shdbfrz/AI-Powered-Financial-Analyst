"""
Hyperparameter tuning orchestration for `ai/models/ml/`.

Wraps `GridSearchCV` / `RandomizedSearchCV`, always cross-validated with
`TimeSeriesSplit` (never `KFold`'s random folds — a validation fold must
come chronologically after its training fold, or tuning would pick
hyperparameters that look good only because of leaked future information).
"""

import time
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from ai.models.ml.config import DEFAULT_ML_CONFIG, MLConfig
from ai.models.ml.exceptions import HyperparameterTuningError
from ai.models.ml.models.base import BaseMLModel
from ai.models.ml.models.registry import ModelFactory
from ai.models.ml.tuning.param_grids import get_param_grid
from ai.utils.logger import get_logger

logger = get_logger(__name__)

TuningMethod = Literal["grid", "random"]

_DEFAULT_SCORING = {"regression": "neg_root_mean_squared_error", "classification": "f1"}


@dataclass
class TuningResult:
    model_name: str
    method: str  # "grid" | "random" | "none" (nothing tunable was registered)
    best_params: dict
    best_score: float
    scoring: str
    cv_splits: int
    n_candidates_evaluated: int
    duration_seconds: float


def _grid_size(grid: dict[str, list]) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


class HyperparameterTuner:
    """`GridSearchCV`/`RandomizedSearchCV` over a `BaseMLModel`'s registered
    search space (`ai.models.ml.tuning.param_grids`), returning a freshly
    constructed, untrained `BaseMLModel` with the best-found hyperparameters
    — the caller still calls `.fit()` on the *final* train split themselves
    (tuning only ever fits on CV folds internally), keeping "select
    hyperparameters" and "fit the model we'll actually evaluate/save" as two
    explicit, auditable steps.
    """

    def __init__(self, config: MLConfig = DEFAULT_ML_CONFIG):
        self.config = config
        self.logger = logger

    def tune(
        self,
        model: BaseMLModel,
        X: pd.DataFrame,
        y: pd.Series,
        method: TuningMethod = "random",
        cv_splits: Optional[int] = None,
        n_iter: Optional[int] = None,
        scoring: Optional[str] = None,
    ) -> tuple[BaseMLModel, TuningResult]:
        """Search `model`'s registered hyperparameter grid and return a new,
        unfitted `BaseMLModel` instance built with the best-found params.

        Raises:
            HyperparameterTuningError: the underlying search failed (e.g. an
                estimator raised on every candidate).
        """
        param_grid = get_param_grid(model.info.name)
        if not param_grid:
            self.logger.info("No tunable hyperparameters registered for '%s'; skipping tuning.", model.info.name)
            return model, TuningResult(
                model_name=model.info.name, method="none", best_params=model.hyperparameters,
                best_score=float("nan"), scoring="n/a", cv_splits=0, n_candidates_evaluated=0, duration_seconds=0.0,
            )

        is_pipeline = isinstance(model.estimator, Pipeline)
        prefixed_grid = {f"model__{k}": v for k, v in param_grid.items()} if is_pipeline else param_grid

        cv = TimeSeriesSplit(n_splits=cv_splits or self.config.default_cv_splits)
        resolved_scoring = scoring or _DEFAULT_SCORING[model.info.task_type]

        if method == "grid":
            search = GridSearchCV(
                estimator=model.estimator, param_grid=prefixed_grid, cv=cv,
                scoring=resolved_scoring, n_jobs=-1,
            )
        elif method == "random":
            requested_iter = n_iter or self.config.default_random_search_iterations
            search = RandomizedSearchCV(
                estimator=model.estimator, param_distributions=prefixed_grid,
                n_iter=min(requested_iter, _grid_size(prefixed_grid)), cv=cv,
                scoring=resolved_scoring, n_jobs=-1, random_state=self.config.random_state,
            )
        else:
            raise ValueError(f"Unknown tuning method: {method!r}. Expected 'grid' or 'random'.")

        self.logger.info(
            "Tuning '%s' via %s search (%d candidate combination(s) in grid, cv_splits=%d, scoring=%s)",
            model.info.name, method, _grid_size(prefixed_grid), cv.n_splits, resolved_scoring,
        )

        start = time.perf_counter()
        try:
            search.fit(X, y)
        except Exception as e:  # noqa: BLE001
            raise HyperparameterTuningError(model.info.name, str(e)) from e
        duration = time.perf_counter() - start

        best_params = {
            (k.removeprefix("model__") if is_pipeline else k): v
            for k, v in search.best_params_.items()
        }
        tuned_model = ModelFactory.create(model.info.name, **best_params)

        result = TuningResult(
            model_name=model.info.name, method=method, best_params=best_params,
            best_score=float(search.best_score_), scoring=resolved_scoring,
            cv_splits=cv.n_splits, n_candidates_evaluated=len(search.cv_results_["params"]),
            duration_seconds=duration,
        )
        self.logger.info(
            "Tuning finished for '%s' in %.2fs: best_score=%.6f best_params=%s",
            model.info.name, duration, result.best_score, best_params,
        )
        return tuned_model, result