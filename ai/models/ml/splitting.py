"""
Data splitting for `ai/models/ml/`.

Financial time series must never be randomly shuffled before splitting —
doing so lets a model train on rows that come *after* its validation/test
rows chronologically, which leaks future information and produces
optimistic, non-reproducible-in-production metrics. Every split in this
module is a contiguous, chronologically-ordered block (train = earliest,
validation = middle, test = most recent).
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ai.models.ml.config import DEFAULT_ML_CONFIG, MLConfig
from ai.models.ml.exceptions import InsufficientTrainingDataError
from ai.models.ml.preprocessing import FeaturePreprocessor
from ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataSplit:
    """Contiguous, time-ordered train/validation/test partitions."""

    X_train: pd.DataFrame
    y_train: pd.Series
    dates_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    dates_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    dates_test: pd.Series

    def summary(self) -> dict:
        return {
            "train": {"rows": len(self.X_train), "from": str(self.dates_train.min()), "to": str(self.dates_train.max())},
            "validation": {"rows": len(self.X_val), "from": str(self.dates_val.min()), "to": str(self.dates_val.max())},
            "test": {"rows": len(self.X_test), "from": str(self.dates_test.min()), "to": str(self.dates_test.max())},
        }


class TimeSeriesSplitter:
    """Splits a chronologically-sorted `(X, y, dates)` triple into contiguous
    train/validation/test blocks by ratio (Sprint 3 spec: "Time-based Split.
    Do NOT randomly shuffle financial data.").
    """

    def __init__(self, config: MLConfig = DEFAULT_ML_CONFIG):
        self.config = config
        self.logger = logger

    def split(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> DataSplit:
        """Split assuming `X`/`y`/`dates` are already sorted ascending by date
        (guaranteed by `data_loader.load_processed_dataset`).

        Raises:
            InsufficientTrainingDataError: any resulting partition would be empty.
        """
        n = len(X)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.validation_ratio)

        if train_end == 0 or val_end == train_end or val_end >= n:
            raise InsufficientTrainingDataError(
                required_rows=3, available_rows=n,
                context={"train_ratio": self.config.train_ratio, "validation_ratio": self.config.validation_ratio,
                         "test_ratio": self.config.test_ratio, "reason": "one or more split partitions would be empty"},
            )

        split = DataSplit(
            X_train=X.iloc[:train_end].reset_index(drop=True),
            y_train=y.iloc[:train_end].reset_index(drop=True),
            dates_train=dates.iloc[:train_end].reset_index(drop=True),
            X_val=X.iloc[train_end:val_end].reset_index(drop=True),
            y_val=y.iloc[train_end:val_end].reset_index(drop=True),
            dates_val=dates.iloc[train_end:val_end].reset_index(drop=True),
            X_test=X.iloc[val_end:].reset_index(drop=True),
            y_test=y.iloc[val_end:].reset_index(drop=True),
            dates_test=dates.iloc[val_end:].reset_index(drop=True),
        )
        self.logger.info("Time-based split: %s", split.summary())
        return split

    def split_and_preprocess(
        self, X_raw: pd.DataFrame, y: pd.Series, dates: pd.Series,
    ) -> tuple[DataSplit, FeaturePreprocessor]:
        """The method pipeline code actually calls: `split()` the raw
        (unencoded, un-imputed) matrix `ai.models.ml.data_loader.build_feature_matrix`
        returns, then fit a `FeaturePreprocessor` on the training partition
        only and apply it to all three partitions — never the other way
        around, which would leak validation/test-period statistics into the
        training features.

        Returns:
            A fully preprocessed `DataSplit` ready for `BaseMLModel.fit()`,
            plus the fitted `FeaturePreprocessor` (the training pipeline
            persists this alongside the model so inference-time data can be
            transformed identically — see `ai.models.ml.prediction`).

        Raises:
            InsufficientTrainingDataError: any resulting partition has fewer
                than `config.min_rows_required` rows, either before or after
                preprocessing (relevant under `nan_strategy="drop_rows"`,
                where preprocessing itself can shrink a partition further).
        """
        raw_split = self.split(X_raw, y, dates)
        self._check_min_rows("train (pre-preprocessing)", len(raw_split.X_train))

        preprocessor = FeaturePreprocessor(self.config).fit(raw_split.X_train)

        X_train, y_train, dates_train = preprocessor.transform_and_align(raw_split.X_train, raw_split.y_train, raw_split.dates_train)
        X_val, y_val, dates_val = preprocessor.transform_and_align(raw_split.X_val, raw_split.y_val, raw_split.dates_val)
        X_test, y_test, dates_test = preprocessor.transform_and_align(raw_split.X_test, raw_split.y_test, raw_split.dates_test)

        for split_name, frame in (("train", X_train), ("validation", X_val), ("test", X_test)):
            self._check_min_rows(f"{split_name} (post-preprocessing)", len(frame))

        split = DataSplit(
            X_train=X_train, y_train=y_train, dates_train=dates_train,
            X_val=X_val, y_val=y_val, dates_val=dates_val,
            X_test=X_test, y_test=y_test, dates_test=dates_test,
        )
        self.logger.info("Split + preprocessed: %s", split.summary())
        return split, preprocessor

    def _check_min_rows(self, label: str, n_rows: int) -> None:
        if n_rows < self.config.min_rows_required:
            raise InsufficientTrainingDataError(
                self.config.min_rows_required, n_rows, context={"partition": label},
            )

    def time_series_cv(self, n_splits: int = None) -> TimeSeriesSplit:
        """A scikit-learn-compatible `cv` object for `GridSearchCV` /
        `RandomizedSearchCV` that respects chronological order — each fold's
        validation block comes strictly after its training block, unlike
        `KFold`'s random partitioning (Sprint 3 spec: "TimeSeries Cross
        Validation").
        """
        return TimeSeriesSplit(n_splits=n_splits or self.config.default_cv_splits)