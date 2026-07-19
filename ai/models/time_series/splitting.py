"""
Data splitting for `ai/models/time_series/`.

Mirrors `ai.models.ml.splitting.TimeSeriesSplitter`'s core rule: financial
time series must never be randomly shuffled. Every split here is a
contiguous, chronologically-ordered block (train = earliest, validation =
middle, test = most recent) so no model ever trains on rows that come after
its evaluation rows.
"""

from dataclasses import dataclass

import pandas as pd

from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.models.time_series.exceptions import InsufficientTrainingDataError
from ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SeriesSplit:
    """Contiguous, time-ordered train/validation/test partitions of a single
    univariate price series."""

    train: pd.Series
    validation: pd.Series
    test: pd.Series

    def summary(self) -> dict:
        return {
            name: {
                "rows": len(part),
                "from": str(part.index.min()) if len(part) else None,
                "to": str(part.index.max()) if len(part) else None,
            }
            for name, part in (("train", self.train), ("validation", self.validation), ("test", self.test))
        }


class TimeSeriesSplitter:
    """Splits a chronologically-sorted price `Series` into contiguous
    train/validation/test blocks by ratio."""

    def __init__(self, config: TimeSeriesConfig = DEFAULT_TS_CONFIG):
        self.config = config
        self.logger = logger

    def split(self, series: pd.Series) -> SeriesSplit:
        if not series.index.is_monotonic_increasing:
            raise ValueError("series index must be sorted ascending before splitting")

        n = len(series)
        if n < self.config.min_rows_required:
            raise InsufficientTrainingDataError(self.config.min_rows_required, n)

        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.validation_ratio)

        result = SeriesSplit(
            train=series.iloc[:train_end],
            validation=series.iloc[train_end:val_end],
            test=series.iloc[val_end:],
        )

        for name, part in (("train", result.train), ("validation", result.validation), ("test", result.test)):
            if len(part) == 0:
                raise InsufficientTrainingDataError(
                    1, 0, context={"split": name, "total_rows": n}
                )

        self.logger.info("time series split complete", extra=result.summary())
        return result