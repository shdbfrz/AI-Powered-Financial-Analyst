"""
External gradient-boosting library model family for `ai/models/ml/models/`.

XGBoost is a required dependency (already pinned in `ai/requirements.txt`).
LightGBM and CatBoost are explicitly marked "(Optional)" in the Sprint 3
spec, so — mirroring how Sprint 1's `AlphaVantageProvider` is registered but
degrades gracefully when its API key isn't configured — these two are
import-guarded: if the package isn't installed, `*_AVAILABLE` is `False` and
`ai.models.ml.models.registry` simply omits them from the registry (with a
logged warning) instead of the whole module failing to import.
"""

from xgboost import XGBClassifier, XGBRegressor

from ai.models.ml.models.base import BaseMLModel, ModelInfo
from ai.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in environments without lightgbm
    LIGHTGBM_AVAILABLE = False
    logger.warning("lightgbm is not installed; LightGBM Regressor will be omitted from the model registry.")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in environments without catboost
    CATBOOST_AVAILABLE = False
    logger.warning("catboost is not installed; CatBoost Regressor will be omitted from the model registry.")


class XGBoostRegressorModel(BaseMLModel):
    """Extreme Gradient Boosting regressor."""

    info = ModelInfo(
        name="xgboost",
        display_name="XGBoost",
        family="boosting",
        task_type="regression",
        purpose="A highly optimized, regularized gradient-boosted tree ensemble — adds "
                "L1/L2 regularization on leaf weights and second-order gradient "
                "information on top of scikit-learn's Gradient Boosting, generally "
                "reaching lower error for the same training budget.",
        advantages="State-of-the-art accuracy on structured/tabular data in most published "
                    "benchmarks; built-in handling of missing values; trains fast via "
                    "histogram-based split finding; explicit regularization "
                    "(`reg_alpha`/`reg_lambda`) reduces overfitting risk versus plain "
                    "Gradient Boosting.",
        limitations="More hyperparameters to tune than Random Forest for full performance; "
                     "an external dependency (not part of scikit-learn) that must be "
                     "version-pinned (`ai/requirements.txt` pins `xgboost==2.1.1`); like "
                     "every tree ensemble here, a black box relative to the linear family.",
        best_use_cases="Named explicitly in this project's ARCHITECTURE.md and PDR as one "
                        "of the two required core ML models (alongside Random Forest) — the "
                        "expected top performer in the regression comparison table for most "
                        "tickers.",
        recommended_for=("strong default", "production candidate"),
    )

    @classmethod
    def default_hyperparameters(cls) -> dict:
        return {
            "n_estimators": 400, "max_depth": 5, "learning_rate": 0.03,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "n_jobs": -1, "random_state": 42,
        }

    def _build_estimator(self, params: dict) -> XGBRegressor:
        return XGBRegressor(**params)


class XGBoostClassifierModel(BaseMLModel):
    """Extreme Gradient Boosting classifier."""

    info = ModelInfo(
        name="xgboost_classifier",
        display_name="XGBoost Classifier",
        family="boosting",
        task_type="classification",
        purpose="The classification counterpart of XGBoost Regressor: a regularized, "
                "boosted tree ensemble predicting `target_direction_{h}_day`, typically the "
                "strongest of the three classification models on structured features.",
        advantages="Same regularization/accuracy advantages as the regressor variant, plus "
                    "`scale_pos_weight` for handling any class imbalance between up/down "
                    "days; well-calibrated `predict_proba` relative to bagged forests.",
        limitations="Same as the regressor: more hyperparameters, external dependency, "
                     "black-box relative to Logistic Regression.",
        best_use_cases="Expected top performer for the Buy/Sell direction signal that "
                        "feeds the Decision Support Engine's `confidence_score` (Sprint 7) "
                        "— named explicitly as a required classification model in this "
                        "sprint's specification.",
        recommended_for=("strong default", "direction classification", "production candidate"),
    )

    @classmethod
    def default_hyperparameters(cls) -> dict:
        return {
            "n_estimators": 400, "max_depth": 5, "learning_rate": 0.03,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "n_jobs": -1, "random_state": 42, "eval_metric": "logloss",
        }

    def _build_estimator(self, params: dict) -> XGBClassifier:
        return XGBClassifier(**params)


if LIGHTGBM_AVAILABLE:

    class LightGBMRegressorModel(BaseMLModel):
        """Light Gradient Boosting Machine regressor (optional dependency)."""

        info = ModelInfo(
            name="lightgbm",
            display_name="LightGBM",
            family="boosting",
            task_type="regression",
            is_optional_dependency=True,
            purpose="A histogram-based gradient boosting framework using leaf-wise "
                    "(best-first) tree growth instead of XGBoost's level-wise growth, "
                    "generally trading a bit of overfitting risk for faster training on "
                    "larger datasets.",
            advantages="Typically the fastest-training boosted ensemble of the three "
                        "(native XGBoost, LightGBM, CatBoost) here, with competitive "
                        "accuracy; native categorical-feature support (relevant for "
                        "Sprint 2's `price_action_label`/`trend_label`/`market_bias` "
                        "columns without one-hot expansion, though this pipeline currently "
                        "one-hot-encodes them upstream for consistency across all models).",
            limitations="Leaf-wise growth can overfit small datasets more readily than "
                         "level-wise growth (XGBoost's default) unless `num_leaves`/"
                         "`min_child_samples` are constrained; optional dependency — the "
                         "pipeline must run correctly without it installed.",
            best_use_cases="Useful third boosting data point in the comparison table, "
                            "particularly valuable once the dataset grows large enough "
                            "(multi-ticker, multi-year) that XGBoost's training time becomes "
                            "a bottleneck.",
            recommended_for=("fast training", "large datasets"),
        )

        @classmethod
        def default_hyperparameters(cls) -> dict:
            return {
                "n_estimators": 400, "max_depth": -1, "num_leaves": 31, "learning_rate": 0.03,
                "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1, "verbose": -1,
            }

        def _build_estimator(self, params: dict) -> "LGBMRegressor":
            return LGBMRegressor(**params)

else:
    LightGBMRegressorModel = None  # registry.py checks LIGHTGBM_AVAILABLE before referencing this


if CATBOOST_AVAILABLE:

    class CatBoostRegressorModel(BaseMLModel):
        """Categorical Boosting regressor (optional dependency)."""

        info = ModelInfo(
            name="catboost",
            display_name="CatBoost",
            family="boosting",
            task_type="regression",
            is_optional_dependency=True,
            purpose="A gradient boosting framework built around ordered boosting and native "
                    "categorical-feature handling, designed to reduce the prediction shift "
                    "(target leakage) that naive gradient boosting can suffer on categorical "
                    "columns.",
            advantages="Strong out-of-the-box defaults (often competitive without any "
                        "tuning), native handling of Sprint 2's categorical Phase-4 columns "
                        "(`price_action_label`, `trend_label`, `breakout_label`, "
                        "`market_bias`, `market_structure`), generally robust to overfitting "
                        "via ordered boosting.",
            limitations="Slower to train than LightGBM in most benchmarks; larger install "
                         "footprint; optional dependency — the pipeline must run correctly "
                         "without it installed; verbose output must be explicitly silenced.",
            best_use_cases="Fourth boosting reference point, particularly worth including "
                            "once Sprint 2's categorical columns are passed in natively "
                            "(rather than one-hot encoded) to see whether native categorical "
                            "handling measurably helps on this feature set.",
            recommended_for=("categorical features", "robust defaults"),
        )

        @classmethod
        def default_hyperparameters(cls) -> dict:
            return {
                "iterations": 400, "depth": 6, "learning_rate": 0.03,
                "l2_leaf_reg": 3.0, "random_seed": 42, "verbose": False, "allow_writing_files": False,
            }

        def _build_estimator(self, params: dict) -> "CatBoostRegressor":
            return CatBoostRegressor(**params)

else:
    CatBoostRegressorModel = None  # registry.py checks CATBOOST_AVAILABLE before referencing this