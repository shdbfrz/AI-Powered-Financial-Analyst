"""
Linear model family for `ai/models/ml/models/`.

Every model here is wrapped as `Pipeline([StandardScaler, estimator])`
rather than the bare scikit-learn estimator. This is a deliberate,
documented choice: Ridge/Lasso/ElasticNet/Logistic Regression are all
regularized, and the penalty term is scale-dependent — an unscaled `volume`
column (millions) would dominate an unscaled `rsi` column (0-100) or a
`future_return_5_day`-style feature (~0.01) purely because of its magnitude,
not its predictive value. `BaseMLModel._final_estimator()` unwraps the
Pipeline for coefficient introspection, so this is transparent to callers.
"""

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai.models.ml.models.base import BaseMLModel, ModelInfo


def _scaled(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


class LinearRegressionModel(BaseMLModel):
    """Ordinary Least Squares regression — the pipeline's baseline-of-baselines."""

    info = ModelInfo(
        name="linear_regression",
        display_name="Linear Regression",
        family="linear",
        task_type="regression",
        purpose="Fits a straight-line (hyperplane) relationship between the engineered "
                "features and the target with no regularization — the simplest possible "
                "baseline every other model must beat to justify its extra complexity.",
        advantages="Fast to train and predict, fully interpretable coefficients, no "
                    "hyperparameters to tune, deterministic.",
        limitations="Assumes a linear relationship; cannot capture the non-linear "
                     "interactions (e.g. RSI x volatility regime) that drive real price "
                     "moves. Sensitive to multicollinearity among the 205 engineered "
                     "features (many indicators are derived from the same close series).",
        best_use_cases="Sanity-check baseline and as an interpretable reference point when "
                        "explaining *why* a more complex model's extra accuracy is worth its "
                        "extra opacity (relevant for the LLM Explanation Layer in Sprint 8).",
        recommended_for=("baseline", "interpretability"),
    )

    @classmethod
    def default_hyperparameters(cls) -> dict:
        return {"fit_intercept": True, "n_jobs": -1}

    def _build_estimator(self, params: dict) -> Pipeline:
        return _scaled(LinearRegression(**params))


class RidgeRegressionModel(BaseMLModel):
    """L2-regularized linear regression."""

    info = ModelInfo(
        name="ridge_regression",
        display_name="Ridge Regression",
        family="linear",
        task_type="regression",
        purpose="Linear regression with an L2 penalty that shrinks all coefficients "
                "toward zero, trading a small amount of bias for a large reduction in "
                "variance when features are correlated.",
        advantages="Handles the multicollinearity that plain Linear Regression struggles "
                    "with (e.g. sma_5/sma_10/sma_20 are highly correlated by construction); "
                    "still closed-form and fast; every coefficient stays in the model "
                    "(no hard feature selection), so nothing is discarded silently.",
        limitations="Does not perform feature selection — with 205 candidate features, a "
                     "Ridge model keeps (and must be interpreted alongside) all of them. "
                     "The regularization strength `alpha` needs tuning per dataset.",
        best_use_cases="Preferred over plain Linear Regression whenever the feature set has "
                        "known collinear groups, which is the norm here (Bollinger/SMA/EMA "
                        "families all move together).",
        recommended_for=("regularization", "collinear features"),
    )

    @classmethod
    def default_hyperparameters(cls) -> dict:
        return {"alpha": 1.0, "fit_intercept": True, "random_state": 42}

    def _build_estimator(self, params: dict) -> Pipeline:
        return _scaled(Ridge(**params))


class LassoRegressionModel(BaseMLModel):
    """L1-regularized linear regression (sparse coefficients)."""

    info = ModelInfo(
        name="lasso_regression",
        display_name="Lasso Regression",
        family="linear",
        task_type="regression",
        purpose="Linear regression with an L1 penalty that can shrink coefficients "
                "exactly to zero, performing embedded feature selection while fitting.",
        advantages="Produces a sparse, more interpretable model by automatically dropping "
                    "features it finds uninformative — useful as a cheap complement to the "
                    "explicit Recursive Feature Elimination step in this sprint's feature "
                    "selection module.",
        limitations="Arbitrarily picks one feature from a correlated group and zeroes out "
                     "the rest, which can be unstable (a slightly different data window "
                     "may keep a different member of the same indicator family). Requires "
                     "tuning `alpha`; too large a value can zero out every coefficient.",
        best_use_cases="Exploratory feature-importance signal ('which of these 205 "
                        "indicators does a sparse linear model keep?') more than a final "
                        "production model on its own.",
        recommended_for=("feature selection", "sparse models"),
    )

    @classmethod
    def default_hyperparameters(cls) -> dict:
        return {"alpha": 0.001, "fit_intercept": True, "random_state": 42, "max_iter": 10_000}

    def _build_estimator(self, params: dict) -> Pipeline:
        return _scaled(Lasso(**params))


class ElasticNetModel(BaseMLModel):
    """Combined L1 + L2 regularized linear regression."""

    info = ModelInfo(
        name="elastic_net",
        display_name="ElasticNet",
        family="linear",
        task_type="regression",
        purpose="Linear regression with a weighted mix of L1 and L2 penalties "
                "(`l1_ratio`), combining Lasso's sparsity with Ridge's stability on "
                "correlated feature groups.",
        advantages="More stable than pure Lasso when features are highly correlated (the "
                    "common case here), while still capable of zeroing out irrelevant "
                    "features; two knobs (`alpha`, `l1_ratio`) give finer control than "
                    "either Ridge or Lasso alone.",
        limitations="Two hyperparameters instead of one means a larger tuning search space; "
                     "still fundamentally a linear model and cannot capture non-linear "
                     "interactions.",
        best_use_cases="A middle-ground default when it's unclear upfront whether Ridge- or "
                        "Lasso-style regularization suits the feature set better.",
        recommended_for=("regularization", "collinear features", "feature selection"),
    )

    @classmethod
    def default_hyperparameters(cls) -> dict:
        return {"alpha": 0.001, "l1_ratio": 0.5, "fit_intercept": True, "random_state": 42, "max_iter": 10_000}

    def _build_estimator(self, params: dict) -> Pipeline:
        return _scaled(ElasticNet(**params))


class LogisticRegressionModel(BaseMLModel):
    """Regularized linear classifier for the Buy/Sell direction target."""

    info = ModelInfo(
        name="logistic_regression",
        display_name="Logistic Regression",
        family="linear",
        task_type="classification",
        purpose="Linear decision boundary over the engineered features, predicting the "
                "probability that `target_direction_{h}_day` is True (price rises over "
                "the next h day(s)).",
        advantages="Fast, well-calibrated probabilities (useful downstream for the "
                    "Decision Support Engine's confidence_score in Sprint 7), interpretable "
                    "coefficients, a strong and cheap baseline for a binary signal.",
        limitations="Linear decision boundary — cannot represent the non-linear regime "
                     "changes (e.g. a bullish RSI crossover only mattering when volatility "
                     "is also low) that tree-based classifiers pick up naturally.",
        best_use_cases="Baseline for the Buy/Sell direction classification task and as the "
                        "reference point for whether Random Forest/XGBoost's extra "
                        "complexity is actually earning better accuracy on this ticker.",
        recommended_for=("baseline", "probability calibration"),
    )

    @classmethod
    def default_hyperparameters(cls) -> dict:
        return {"C": 1.0, "max_iter": 2000, "random_state": 42, "n_jobs": -1}

    def _build_estimator(self, params: dict) -> Pipeline:
        return _scaled(LogisticRegression(**params))