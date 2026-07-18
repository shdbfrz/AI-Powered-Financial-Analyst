"""
`ai/models/ml/models/` — the fifteen baseline model implementations
(ten required regression models, two optional regression models, three
classification models) plus the `BaseMLModel` contract and `ModelFactory`
registry they're all instantiated through.

    from ai.models.ml.models import ModelFactory

    model = ModelFactory.create("random_forest")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
"""

from ai.models.ml.models.base import BaseMLModel, ModelInfo
from ai.models.ml.models.boosting_models import CATBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
from ai.models.ml.models.registry import ModelFactory

__all__ = [
    "BaseMLModel",
    "ModelInfo",
    "ModelFactory",
    "LIGHTGBM_AVAILABLE",
    "CATBOOST_AVAILABLE",
]