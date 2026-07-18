"""
`ai/models/ml/tuning/` — GridSearchCV / RandomizedSearchCV hyperparameter
tuning with `TimeSeriesSplit` cross-validation.

    from ai.models.ml.tuning import HyperparameterTuner

    tuner = HyperparameterTuner()
    tuned_model, result = tuner.tune(model, X_train, y_train, method="random")
"""

from ai.models.ml.tuning.tuner import HyperparameterTuner, TuningResult

__all__ = ["HyperparameterTuner", "TuningResult"]