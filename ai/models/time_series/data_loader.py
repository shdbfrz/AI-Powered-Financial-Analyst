"""
Processed-dataset loading for `ai/models/time_series/`.

Reuses the exact same `datasets/processed/{ticker}_{version}_features.csv`
files Sprint 2 writes and Sprint 3's `ai.models.ml.data_loader` already
reads — no new storage location is invented (ARCHITECTURE.md §4). Unlike
Sprint 3, this module only needs two columns: `date` and `close`. Everything
else in the 213-column schema (engineered indicators) is Sprint 3's concern,
not this one's — classical time series models forecast the raw price series
itself.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.models.time_series.exceptions import (
    InvalidDateSeriesError,
    ProcessedDatasetNotFoundError,
    SchemaValidationError,
)
from ai.utils.config import settings
from ai.utils.logger import get_logger
from ai.utils.validators import validate_ticker

logger = get_logger(__name__)

_PROCESSED_SUFFIX = "_features.csv"
_NON_DATASET_SUFFIXES = ("_Feature_Metadata.json", "_Feature_Summary.csv", "_Feature_Report.md")


def _processed_data_dir() -> Path:
    return settings.resolve(settings.processed_data_dir)


def find_processed_dataset_files(ticker: str) -> list[Path]:
    """Every processed dataset file for `ticker`, across any run version,
    sorted chronologically (version is a UTC timestamp string, so
    lexicographic order == chronological order). Mirrors
    `ai.models.ml.data_loader.find_processed_dataset_files` exactly.
    """
    ticker = validate_ticker(ticker)
    processed_dir = _processed_data_dir()
    if not processed_dir.exists():
        return []

    matches = []
    for path in processed_dir.glob(f"*{_PROCESSED_SUFFIX}"):
        stem = path.name[: -len(_PROCESSED_SUFFIX)]
        if any(path.name.endswith(suffix) for suffix in _NON_DATASET_SUFFIXES):
            continue
        if "_" not in stem:
            continue
        file_ticker = stem.split("_", 1)[0]
        if file_ticker.upper() == ticker.upper():
            matches.append(path)

    return sorted(matches)


def _latest_processed_dataset_path(ticker: str) -> Path:
    matches = find_processed_dataset_files(ticker)
    if not matches:
        raise ProcessedDatasetNotFoundError(ticker, _processed_data_dir())
    return matches[-1]


def load_price_series(
    ticker: str,
    *,
    path: Optional[Path] = None,
    config: TimeSeriesConfig = DEFAULT_TS_CONFIG,
) -> pd.Series:
    """Load the chronologically-sorted `close` price series for `ticker`,
    indexed by `date`, from the latest Sprint 2 processed dataset (or an
    explicit `path`, mainly for tests).

    Raises:
        ProcessedDatasetNotFoundError: no matching file under datasets/processed/.
        SchemaValidationError: the file is missing `date` or `close`.
        InvalidDateSeriesError: dates are unparseable, duplicated, or unsorted
            after coercion.
    """
    source = path or _latest_processed_dataset_path(ticker)
    logger.info("loading processed dataset", extra={"ticker": ticker, "path": str(source)})

    df = pd.read_csv(source)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in (config.date_column, config.price_column) if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"processed dataset for '{ticker}' is missing required column(s)",
            missing_columns=missing,
        )

    df[config.date_column] = pd.to_datetime(df[config.date_column], errors="coerce")
    if df[config.date_column].isna().any():
        raise InvalidDateSeriesError(f"'{ticker}' dataset has unparseable date values")

    df = df.sort_values(config.date_column).reset_index(drop=True)
    if df[config.date_column].duplicated().any():
        dupes = int(df[config.date_column].duplicated().sum())
        raise InvalidDateSeriesError(
            f"'{ticker}' dataset has {dupes} duplicate date(s) after sorting"
        )

    series = pd.Series(
        df[config.price_column].astype(float).values,
        index=pd.DatetimeIndex(df[config.date_column], name="date"),
        name=config.price_column,
    )

    if series.isna().any():
        n_missing = int(series.isna().sum())
        logger.info(
            "forward/back-filling missing close prices",
            extra={"ticker": ticker, "missing_count": n_missing},
        )
        series = series.ffill().bfill()

    logger.info(
        "price series loaded",
        extra={"ticker": ticker, "rows": len(series), "from": str(series.index.min()), "to": str(series.index.max())},
    )
    return series