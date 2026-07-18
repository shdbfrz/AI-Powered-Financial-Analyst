"""
Hyperparameter search spaces for `ai/models/ml/tuning/`.

One dict per tunable model, keyed by the *unprefixed* hyperparameter name
(`ai.models.ml.tuning.tuner.HyperparameterTuner` adds the `model__` prefix
automatically for models wrapped in a `[Scaler, Model]` Pipeline — see
`ai.models.ml.models.linear_models`). Deliberately modest (2-4 params x
2-4 values each): this is a portfolio/coursework-scale pipeline meant to
run on a laptop, not a hyperparameter sweep on a cluster — grids are sized
so `GridSearchCV` finishes in seconds-to-low-minutes per model on a few
thousand rows, not hours.

`LinearRegressionModel` has no entry: Ordinary Least Squares has no
regularization/complexity hyperparameter worth searching over.
"""

PARAM_GRIDS: dict[str, dict[str, list]] = {
    # --- Linear family (regression) ---
    "ridge_regression": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "lasso_regression": {"alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01]},
    "elastic_net": {"alpha": [0.0001, 0.001, 0.01], "l1_ratio": [0.2, 0.5, 0.8]},

    # --- Linear family (classification) ---
    "logistic_regression": {"C": [0.01, 0.1, 1.0, 10.0]},

    # --- Tree / ensemble family (regression) ---
    "decision_tree": {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [5, 10, 20, 40]},
    "random_forest": {"n_estimators": [150, 300, 500], "max_depth": [6, 10, 14], "min_samples_leaf": [5, 10, 20]},
    "extra_trees": {"n_estimators": [150, 300, 500], "max_depth": [6, 10, 14], "min_samples_leaf": [5, 10, 20]},
    "gradient_boosting": {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [2, 3, 4]},
    "adaboost": {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0]},

    # --- Tree / ensemble family (classification) ---
    "random_forest_classifier": {"n_estimators": [150, 300, 500], "max_depth": [4, 8, 12], "min_samples_leaf": [5, 10, 20]},

    # --- Boosting family (regression) ---
    "xgboost": {"n_estimators": [200, 400, 600], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.03, 0.1]},
    "lightgbm": {"n_estimators": [200, 400, 600], "num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.03, 0.1]},
    "catboost": {"iterations": [200, 400, 600], "depth": [4, 6, 8], "learning_rate": [0.01, 0.03, 0.1]},

    # --- Boosting family (classification) ---
    "xgboost_classifier": {"n_estimators": [200, 400, 600], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.03, 0.1]},
}


def get_param_grid(model_name: str) -> dict[str, list]:
    """Return the search space for `model_name`, or `{}` if it has none
    registered (the tuner treats an empty grid as "nothing to tune").
    """
    return PARAM_GRIDS.get(model_name, {})