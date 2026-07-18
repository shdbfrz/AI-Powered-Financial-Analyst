"""
Configuration for `ai/models/ml/` (Sprint 3 — Baseline Machine Learning Pipeline).

Same two-tier split established in `ai/feature_engineering/config.py`:

- `ai/utils/config.py::settings` — environment-variable-backed settings
  (storage paths, log level). Every module reads from there.
- `MLConfig` (this file) — non-secret, purely computational tunables (split
  ratios, target column names, feature-selection thresholds, tuning
  defaults). Not secrets, change per-experiment, passed as a plain frozen
  dataclass through the pipeline instead of living in `.env`.
"""

from dataclasses import dataclass, field
from typing import Literal

NanStrategy = Literal["drop_rows", "impute_median", "impute_mean"]
RankingDirection = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class MLConfig:
    """All tunable parameters for the Sprint 3 ML pipeline.

    Frozen (immutable) so a config object can be safely reused/shared across
    models and pipeline runs and logged verbatim as part of run metadata for
    reproducibility (ARCHITECTURE.md §2.3).
    """

    # --- Target columns (produced by Sprint 2's TargetFeatureGenerator for
    # target_horizons=(1, 3, 5); see ai/feature_engineering/features/target.py) ---
    regression_targets: tuple[str, ...] = (
        "target_1_day", "target_3_day", "target_5_day",
        "future_return_1_day", "future_return_3_day", "future_return_5_day",
    )
    classification_targets: tuple[str, ...] = (
        "target_direction_1_day", "target_direction_3_day", "target_direction_5_day",
    )
    # future_return_* is scale-invariant across tickers (Sprint 2's own
    # FeatureDefinition recommends it over the raw price target_*_day for
    # exactly this reason) — default regression target unless overridden.
    default_regression_target: str = "future_return_5_day"
    default_classification_target: str = "target_direction_5_day"

    # --- Columns that are never candidate features -----------------------------
    non_feature_columns: tuple[str, ...] = ("date", "ticker")
    # Every target/label column (any horizon) shares one of these prefixes;
    # matches the exclusion rule already established in
    # ai/feature_engineering/selection.py so Sprint 2 and Sprint 3 agree on
    # what counts as a "feature" vs. a "label".
    target_column_prefixes: tuple[str, ...] = ("target_", "future_return_")

    # --- Data splitting (time-based; SRS/Sprint 3 spec: never shuffle) ---------
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    # --- Missing-value handling for the feature matrix --------------------------
    # Sprint 2 deliberately does NOT drop warm-up NaN rows (e.g. the first 199
    # rows before a 200-day SMA is defined) so shorter-window consumers keep
    # those rows. It also leaves several indicators (demand_zone_strength,
    # supply_zone_strength, zone_width, and similar "is an event currently
    # active" columns) NaN on the majority of rows by design — a zone/pattern
    # simply isn't present most days. Empirically, requiring *every* feature
    # to be simultaneously non-NaN ("drop_rows") drops effectively all rows
    # on the real 205-feature schema, so the default here is per-column
    # median imputation, fit on the training split only (see
    # `ai.models.ml.preprocessing.FeaturePreprocessor`) and applied to
    # validation/test — never the reverse, which would leak future
    # information backward into training features. "drop_rows" remains
    # available for feature subsets dense enough for it to be viable (e.g.
    # after feature selection narrows to indicators that are rarely NaN).
    nan_strategy: NanStrategy = "impute_median"

    # --- Minimum data volume guard ------------------------------------------------
    min_rows_required: int = 50  # per split (train/validation/test) — below this a split is meaningless

    # --- Feature selection --------------------------------------------------------
    correlation_threshold: float = 0.95
    variance_threshold: float = 1e-6
    rfe_n_features_to_select: int = 40
    permutation_importance_repeats: int = 10
    permutation_importance_max_samples: int = 1000  # subsample for speed on large feature sets
    reference_estimator_n_estimators: int = 200  # used by RFE / tree-importance's internal reference model
    top_k_recommended_features: int = 40

    # --- Hyperparameter tuning ------------------------------------------------------
    default_cv_splits: int = 5
    default_random_search_iterations: int = 20

    # --- Reproducibility -------------------------------------------------------------
    random_state: int = 42

    # --- Model ranking (Model Comparison table) ---------------------------------------
    regression_primary_metric: str = "rmse"
    regression_primary_metric_direction: RankingDirection = "minimize"
    classification_primary_metric: str = "f1_score"
    classification_primary_metric_direction: RankingDirection = "maximize"

    # --- Visualization ------------------------------------------------------------------
    top_n_models_to_plot: int = 3  # prediction-vs-actual / residual / feature-importance plots
    learning_curve_train_sizes: tuple[float, ...] = field(
        default_factory=lambda: (0.2, 0.4, 0.6, 0.8, 1.0)
    )

    def __post_init__(self) -> None:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + validation_ratio + test_ratio must sum to 1.0, got {total:.4f}"
            )
        if not (0 < self.train_ratio < 1) or not (0 < self.test_ratio < 1):
            raise ValueError("train_ratio and test_ratio must both be in (0, 1)")

    def all_target_columns(self) -> tuple[str, ...]:
        """Every valid target column across both tasks (used for input validation)."""
        return self.regression_targets + self.classification_targets

    def target_prefix_columns(self, df_columns) -> list[str]:
        """Every column in `df_columns` that is a target/label column for *any*
        horizon — used to exclude other horizons' targets from the feature
        matrix when training on one specific target.
        """
        return [c for c in df_columns if c.startswith(self.target_column_prefixes)]


DEFAULT_ML_CONFIG = MLConfig()