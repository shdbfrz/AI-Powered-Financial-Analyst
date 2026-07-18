"""
Model registry / factory for `ai/models/ml/models/`.

A Factory (mirrors `ai.data_collection.manager._PROVIDER_REGISTRY` and
`ai.feature_engineering.features.build_generators`) mapping a short, stable
model name to its `BaseMLModel` subclass. The training pipeline asks this
factory for models by name instead of importing every model class itself —
adding model #16 means registering it here, nowhere else needs to change.

LightGBM/CatBoost are only present in the registry if their package is
installed (`ai.models.ml.models.boosting_models.LIGHTGBM_AVAILABLE` /
`CATBOOST_AVAILABLE`) — the Sprint 3 spec marks both "(Optional)".
"""

from typing import Optional

from ai.models.ml.exceptions import UnknownModelError
from ai.models.ml.models.base import BaseMLModel, ModelInfo, TaskType
from ai.models.ml.models.boosting_models import (
    CATBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    CatBoostRegressorModel,
    LightGBMRegressorModel,
    XGBoostClassifierModel,
    XGBoostRegressorModel,
)
from ai.models.ml.models.linear_models import (
    ElasticNetModel,
    LassoRegressionModel,
    LinearRegressionModel,
    LogisticRegressionModel,
    RidgeRegressionModel,
)
from ai.models.ml.models.tree_ensemble_models import (
    AdaBoostRegressorModel,
    DecisionTreeRegressorModel,
    ExtraTreesRegressorModel,
    GradientBoostingRegressorModel,
    RandomForestClassifierModel,
    RandomForestRegressorModel,
)
from ai.utils.logger import get_logger

logger = get_logger(__name__)

# Order mirrors the Sprint 3 spec's model list, grouped by task.
_REGRESSION_REGISTRY: dict[str, type[BaseMLModel]] = {
    "linear_regression": LinearRegressionModel,
    "ridge_regression": RidgeRegressionModel,
    "lasso_regression": LassoRegressionModel,
    "elastic_net": ElasticNetModel,
    "decision_tree": DecisionTreeRegressorModel,
    "random_forest": RandomForestRegressorModel,
    "extra_trees": ExtraTreesRegressorModel,
    "gradient_boosting": GradientBoostingRegressorModel,
    "adaboost": AdaBoostRegressorModel,
    "xgboost": XGBoostRegressorModel,
}
if LIGHTGBM_AVAILABLE:
    _REGRESSION_REGISTRY["lightgbm"] = LightGBMRegressorModel
if CATBOOST_AVAILABLE:
    _REGRESSION_REGISTRY["catboost"] = CatBoostRegressorModel

_CLASSIFICATION_REGISTRY: dict[str, type[BaseMLModel]] = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest_classifier": RandomForestClassifierModel,
    "xgboost_classifier": XGBoostClassifierModel,
}

_ALL_REGISTRIES: dict[TaskType, dict[str, type[BaseMLModel]]] = {
    "regression": _REGRESSION_REGISTRY,
    "classification": _CLASSIFICATION_REGISTRY,
}

logger.info(
    "Model registry initialized: %d regression model(s), %d classification model(s) "
    "(lightgbm_available=%s, catboost_available=%s)",
    len(_REGRESSION_REGISTRY), len(_CLASSIFICATION_REGISTRY), LIGHTGBM_AVAILABLE, CATBOOST_AVAILABLE,
)


class ModelFactory:
    """Single entry point for instantiating any registered model by name."""

    @staticmethod
    def available_models(task_type: Optional[TaskType] = None) -> list[str]:
        """List registered model names, optionally filtered to one task type."""
        if task_type is not None:
            return list(_ALL_REGISTRIES[task_type])
        return list(_REGRESSION_REGISTRY) + list(_CLASSIFICATION_REGISTRY)

    @staticmethod
    def _resolve(name: str) -> type[BaseMLModel]:
        for registry in _ALL_REGISTRIES.values():
            if name in registry:
                return registry[name]
        raise UnknownModelError(name, available=ModelFactory.available_models())

    @staticmethod
    def create(name: str, **hyperparameter_overrides) -> BaseMLModel:
        """Instantiate a registered model by name.

        Raises:
            UnknownModelError: `name` isn't registered for either task.
        """
        model_cls = ModelFactory._resolve(name)
        return model_cls(**hyperparameter_overrides)

    @staticmethod
    def create_all(task_type: TaskType, names: Optional[list[str]] = None) -> dict[str, BaseMLModel]:
        """Instantiate every registered model for `task_type` (or just `names`,
        if given — still validated against the registry).
        """
        registry = _ALL_REGISTRIES[task_type]
        selected = names if names is not None else list(registry)
        unknown = set(selected) - set(registry)
        if unknown:
            raise UnknownModelError(
                next(iter(unknown)),
                available=list(registry),
            )
        return {name: registry[name]() for name in selected}

    @staticmethod
    def get_model_info(name: str) -> ModelInfo:
        return ModelFactory._resolve(name).info

    @staticmethod
    def all_model_info() -> list[ModelInfo]:
        """Every registered model's metadata, in registry order — feeds
        `Model_Documentation.md`.
        """
        return [cls.info for registry in _ALL_REGISTRIES.values() for cls in registry.values()]