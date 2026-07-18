"""
Tree and ensemble model family for `ai/models/ml/models/`.

Unlike the linear family, none of these need `StandardScaler` — tree-based
splits are invariant to monotonic feature scaling, so wrapping them in a
Pipeline would only add overhead without changing results.
"""

from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor

from ai.models.ml.models.base import BaseMLModel, ModelInfo


class DecisionTreeRegressorModel(BaseMLModel):
    """A single regression tree."""

    info = ModelInfo(
        name="decision_tree",
        display_name="Decision Tree Regressor",
        family="tree",
        task_type="regression",
        purpose="Learns a piecewise-constant prediction by recursively splitting the "
                "feature space on the single most informative feature/threshold at each "
                "node — the building block every ensemble in this sprint (Random Forest, "
                "Extra Trees, Gradient Boosting, AdaBoost) is made of.",
        advantages="Captures non-linear relationships and feature interactions with zero "
                    "preprocessing; trivially interpretable for shallow depths (can be "
                    "printed/plotted as an actual flowchart); fast to train.",
        limitations="A single tree overfits easily — deep enough to fit training noise, it "
                     "memorizes rather than generalizes, which is precisely why it's used "
                     "as a component of ensembles rather than deployed alone.",
        best_use_cases="Diagnostic tool (visualize *why* a specific split matters) and as "
                        "the weak learner inside AdaBoost; rarely the final production model.",
        recommended_for=("interpretability", "ensemble building block"),
    )

    @classmethod
    def default_hyperparameters(cls) -> dict:
        return {"max_depth": 8, "min_samples_leaf": 20, "random_state": 42}

    def _build_estimator(self, params: dict) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(**params)


class RandomForestRegressorModel(BaseMLModel):
    """Bagged ensemble of decorrelated regression trees."""

    info = ModelInfo(
        name="random_forest",
        display_name="Random Forest",
        family="ensemble",
        task_type="regression",
        purpose="Averages many regression trees, each trained on a bootstrap sample with a "
                "random feature subset per split, to reduce the variance/overfitting of any "
                "single tree while keeping its ability to model non-linear interactions.",
        advantages="Strong out-of-the-box accuracy with minimal tuning, robust to outliers "
                    "and irrelevant features (of which 205 candidates surely include some), "
                    "provides feature importances for free, trains in parallel (`n_jobs=-1`).",
        limitations="Less interpretable than a single tree or a linear model; can still "
                     "overfit with very deep trees on a small dataset; prediction requires "
                     "walking every tree, so it's slower at inference than a linear model.",
        best_use_cases="Strong default choice for tabular financial features — this "
                        "project's ARCHITECTURE.md names it explicitly as one of the two "
                        "required core models alongside XGBoost.",
        recommended_for=("strong default", "feature importance"),
    )

    @classmethod
    def default_hyperparameters(cls) -> dict:
        return {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 10, "n_jobs": -1, "random_state": 42}

    def _build_estimator(self, params: dict) -> RandomForestRegressor:
        return RandomForestRegressor(**params)


class ExtraTreesRegressorModel(BaseMLModel):
    """Extremely Randomized Trees — Random Forest with randomized split thresholds too."""

    info = ModelInfo(
        name="extra_trees",
        display_name="Extra Trees",
        family="ensemble",
        task_type="regression",
        purpose="Like Random Forest, but each tree also picks its split *thresholds* "
                "randomly (rather than searching for the optimal one), then averages "
                "across many such trees — trading a bit of per-tree accuracy for a "
                "further reduction in variance and faster training.",
        advantages="Typically trains faster than Random Forest (no per-split threshold "
                    "search) and can generalize slightly better on noisy tabular data such "
                    "as engineered technical indicators, which are inherently noisy.",
        limitations="The extra randomization can underfit if the true signal is subtle and "
                     "the dataset is small; like Random Forest, still an ensemble of black "
                     "boxes rather than a single interpretable model.",
        best_use_cases="Direct A/B comparison against Random Forest on the same features — "
                        "this sprint's Model Comparison table is exactly the mechanism for "
                        "deciding which one wins on a given ticker.",
        recommended_for=("variance reduction", "fast training"),
    )

    @classmethod
    def default_hyperparameters(cls) -> dict:
        return {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 10, "n_jobs": -1, "random_state": 42}

    def _build_estimator(self, params: dict) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(**params)


class GradientBoostingRegressorModel(BaseMLModel):
    """Sequential (boosted) ensemble of shallow regression trees."""

    info = ModelInfo(
        name="gradient_boosting",
        display_name="Gradient Boosting",
        family="boosting",
        task_type="regression",
        purpose="Builds trees sequentially, each one fit to the *residual errors* of the "
                "ensemble so far, gradually reducing bias — scikit-learn's native gradient "
                "boosting implementation (as distinct from the external XGBoost library).",
        advantages="Often reaches lower bias than bagged ensembles (Random Forest/Extra "
                    "Trees) on structured tabular data; `learning_rate` gives fine control "
                    "over the bias/variance trade-off.",
        limitations="Trains sequentially (trees depend on prior trees), so it cannot "
                     "parallelize across estimators the way Random Forest can, making it "
                     "noticeably slower to train; more sensitive to hyperparameters "
                     "(`learning_rate` x `n_estimators` x `max_depth` all interact) and to "
                     "overfitting if boosted too long.",
        best_use_cases="When XGBoost isn't available/desired and a boosted (bias-reducing) "
                        "rather than bagged (variance-reducing) tree ensemble is wanted, "
                        "or as a second, dependency-light boosting reference point next to "
                        "XGBoost in the comparison table.",
        recommended_for=("bias reduction", "boosting baseline"),
    )

    @classmethod
    def default_hyperparameters(cls) -> dict:
        return {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "random_state": 42}

    def _build_estimator(self, params: dict) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(**params)


class AdaBoostRegressorModel(BaseMLModel):
    """Adaptive Boosting ensemble."""

    info = ModelInfo(
        name="adaboost",
        display_name="AdaBoost",
        family="boosting",
        task_type="regression",
        purpose="Fits a sequence of weak learners (shallow decision trees), each one "
                "reweighting the training samples the previous learner predicted worst, "
                "then combines all of them via a weighted vote/average.",
        advantages="Simple, few hyperparameters to tune, tends to be robust when the base "
                    "learner is kept intentionally weak (shallow trees), fast to train "
                    "relative to Gradient Boosting.",
        limitations="Sensitive to noisy targets/outliers — a mislabeled or extreme-outlier "
                     "sample gets increasingly upweighted across rounds, which is a real "
                     "risk on raw financial return data (single-day price shocks). Generally "
                     "underperforms Gradient Boosting/XGBoost on complex tabular data.",
        best_use_cases="A lightweight boosting reference point in the comparison table, "
                        "particularly useful for illustrating the bias/variance trade-off "
                        "against the bagged ensembles (Random Forest/Extra Trees) in the "
                        "sprint's model-comparison writeup.",
        recommended_for=("boosting baseline", "low-noise targets"),
    )

    @classmethod
    def default_hyperparameters(cls) -> dict:
        return {"n_estimators": 150, "learning_rate": 0.5, "random_state": 42}

    def _build_estimator(self, params: dict) -> AdaBoostRegressor:
        return AdaBoostRegressor(**params)


class RandomForestClassifierModel(BaseMLModel):
    """Bagged ensemble of decorrelated classification trees."""

    info = ModelInfo(
        name="random_forest_classifier",
        display_name="Random Forest Classifier",
        family="ensemble",
        task_type="classification",
        purpose="The classification counterpart of Random Forest Regressor: averages many "
                "classification trees' votes to predict `target_direction_{h}_day` "
                "(up/down) with a well-behaved probability estimate from `predict_proba`.",
        advantages="Handles non-linear feature interactions naturally (e.g. RSI is only "
                    "predictive conditional on trend regime — exactly the kind of "
                    "interaction a linear classifier misses); robust to the class-imbalance "
                    "and outliers common in direction-prediction tasks; feature importances "
                    "for free.",
        limitations="Predicted probabilities from tree ensembles are less well-calibrated "
                     "out of the box than Logistic Regression's (worth checking before "
                     "feeding `confidence_score` in Sprint 7's Decision Support Engine); "
                     "less interpretable than the linear baseline.",
        best_use_cases="Primary classification model for the Buy/Sell direction signal — "
                        "named explicitly alongside XGBoost Classifier as a required model "
                        "in this sprint's specification.",
        recommended_for=("strong default", "direction classification"),
    )

    @classmethod
    def default_hyperparameters(cls) -> dict:
        return {
            "n_estimators": 300, "max_depth": 8, "min_samples_leaf": 10,
            "n_jobs": -1, "random_state": 42, "class_weight": "balanced",
        }

    def _build_estimator(self, params: dict) -> RandomForestClassifier:
        return RandomForestClassifier(**params)