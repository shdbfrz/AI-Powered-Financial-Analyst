"""
Feature preprocessing for `ai/models/ml/` — categorical encoding and NaN
imputation, fit on the training split only.

This is a deliberate, separate step from `ai.models.ml.data_loader` (which
only excludes non-feature/target columns) and from
`ai.models.ml.splitting` (which only partitions rows by time). Encoding and
imputation both derive statistics from data (which categories exist, what
the "typical" value of a column is) — computing those statistics from the
full dataset and then applying them everywhere, including back onto
training rows, would leak validation/test-period information into the
training set. The standard, leakage-free discipline — identical in spirit
to fitting a `StandardScaler` on `X_train` only — is: `fit()` on the
training split, `transform()` everything (including the training split
itself) with what was learned from training data alone.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ai.models.ml.config import DEFAULT_ML_CONFIG, MLConfig
from ai.models.ml.exceptions import MLPipelineError
from ai.utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessorNotFittedError(MLPipelineError):
    """`.transform()` called before `.fit()`."""

    def __init__(self):
        super().__init__("FeaturePreprocessor.transform() called before .fit()")


@dataclass
class PreprocessingReport:
    categorical_columns: list = field(default_factory=list)
    dummy_columns_created: list = field(default_factory=list)
    numeric_columns: list = field(default_factory=list)
    imputation_strategy: str = ""
    columns_imputed: dict = field(default_factory=dict)  # column -> fraction of training rows that were NaN
    rows_dropped_train: int = 0

    def as_dict(self) -> dict:
        return {
            "n_categorical_columns": len(self.categorical_columns),
            "n_dummy_columns_created": len(self.dummy_columns_created),
            "n_numeric_columns": len(self.numeric_columns),
            "imputation_strategy": self.imputation_strategy,
            "top_10_most_imputed_columns": dict(
                sorted(self.columns_imputed.items(), key=lambda kv: -kv[1])[:10]
            ),
            "rows_dropped_train": self.rows_dropped_train,
        }


class FeaturePreprocessor:
    """One-hot encodes categorical columns and imputes numeric NaNs, with all
    learned statistics coming exclusively from `fit()`'s input (the training
    split).

    Usage:
        preprocessor = FeaturePreprocessor(config).fit(X_train_raw)
        X_train = preprocessor.transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)
        X_test = preprocessor.transform(X_test_raw)
    """

    def __init__(self, config: MLConfig = DEFAULT_ML_CONFIG):
        self.config = config
        self.categorical_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.dummy_columns_: list[str] = []
        self.fill_values_: dict[str, float] = {}
        self.report: Optional[PreprocessingReport] = None
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame) -> "FeaturePreprocessor":
        self.categorical_columns_ = X_train.select_dtypes(exclude=[np.number, "bool", "boolean"]).columns.tolist()
        self.numeric_columns_ = [c for c in X_train.columns if c not in self.categorical_columns_]

        dummied = self._dummy_encode(X_train[self.categorical_columns_]) if self.categorical_columns_ else pd.DataFrame(index=X_train.index)
        self.dummy_columns_ = dummied.columns.tolist()

        numeric_train = X_train[self.numeric_columns_].astype(float)
        nan_fraction = numeric_train.isna().mean()

        if self.config.nan_strategy == "impute_mean":
            self.fill_values_ = numeric_train.mean(numeric_only=True).fillna(0.0).to_dict()
        else:  # "impute_median" (default) and "drop_rows" (which still needs values for any leftover partial NaNs on val/test rows it won't drop)
            self.fill_values_ = numeric_train.median(numeric_only=True).fillna(0.0).to_dict()

        self.report = PreprocessingReport(
            categorical_columns=self.categorical_columns_,
            dummy_columns_created=self.dummy_columns_,
            numeric_columns=self.numeric_columns_,
            imputation_strategy=self.config.nan_strategy,
            columns_imputed={c: round(float(nan_fraction[c]), 4) for c in self.numeric_columns_ if nan_fraction[c] > 0},
        )
        self.is_fitted = True
        logger.info(
            "FeaturePreprocessor fitted: %d categorical -> %d dummy column(s), %d numeric column(s), "
            "%d of them had train-set NaNs (strategy=%s)",
            len(self.categorical_columns_), len(self.dummy_columns_), len(self.numeric_columns_),
            len(self.report.columns_imputed), self.config.nan_strategy,
        )
        return self

    def _dummy_encode(self, categorical: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(categorical, columns=list(categorical.columns), dummy_na=False)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted encoding/imputation to `X` (any split — train,
        validation, test, or new inference-time data).
        """
        if not self.is_fitted:
            raise PreprocessorNotFittedError()

        numeric_part = X[self.numeric_columns_].astype(float)
        if self.config.nan_strategy != "drop_rows":
            numeric_part = numeric_part.fillna(self.fill_values_)

        if self.categorical_columns_:
            dummied = self._dummy_encode(X[self.categorical_columns_])
            # Align to the columns seen during fit(): a category present only
            # in val/test becomes an all-zero row (never seen at train time,
            # so the model has nothing learned about it); a category from
            # training absent in this split is added back as all-zero.
            dummied = dummied.reindex(columns=self.dummy_columns_, fill_value=False).astype(float)
        else:
            dummied = pd.DataFrame(index=X.index)

        return pd.concat([numeric_part, dummied], axis=1)

    def transform_and_align(
        self, X: pd.DataFrame, y: pd.Series, dates: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """`transform(X)`, then — only under `nan_strategy="drop_rows"` —
        drop any row still containing a NaN and apply the same row mask to
        `y`/`dates` so the three stay aligned. A no-op under the imputation
        strategies, since `transform()` already leaves no NaNs behind.
        """
        X_out = self.transform(X)
        if self.config.nan_strategy == "drop_rows":
            mask = X_out.notna().all(axis=1)
            return (
                X_out.loc[mask].reset_index(drop=True),
                y.loc[mask].reset_index(drop=True),
                dates.loc[mask].reset_index(drop=True),
            )
        return X_out.reset_index(drop=True), y.reset_index(drop=True), dates.reset_index(drop=True)