"""
Processed-dataset loading for `ai/models/ml/`.

Reads the CSVs Sprint 2's `ai.feature_engineering.storage.save_processed_dataset`
writes to `datasets/processed/{ticker}_{version}_features.csv` (213 columns:
7 raw OHLCV/identifier columns + 205 generated features + 12 target/label
columns for target_horizons=(1, 3, 5) — see
`ai/feature_engineering/features/target.py`).

This module owns two responsibilities: locating/loading that CSV, and
building a *raw* `(X, y, dates)` triple for one chosen target column, with
non-feature and other-horizon target columns removed. Categorical encoding
and NaN imputation are deliberately NOT done here — see the module
docstring on `build_feature_matrix` for why, and
`ai.models.ml.preprocessing.FeaturePreprocessor` for where that happens.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from ai.models.ml.config import DEFAULT_ML_CONFIG, MLConfig
from ai.models.ml.exceptions import (
    InsufficientTrainingDataError,
    InvalidTargetColumnError,
    ProcessedDatasetNotFoundError,
    SchemaValidationError,
)
from ai.utils.config import settings
from ai.utils.logger import get_logger
from ai.utils.validators import validate_ticker

logger = get_logger(__name__)

_PROCESSED_SUFFIX = "_features.csv"
# Sibling files written by Sprint 2 alongside the processed dataset that must
# never be mistaken for it (same ticker/version prefix, different suffix).
_NON_DATASET_SUFFIXES = ("_Feature_Metadata.json", "_Feature_Summary.csv", "_Feature_Report.md")


def _processed_data_dir() -> Path:
    return settings.resolve(settings.processed_data_dir)


def find_processed_dataset_files(ticker: str) -> list[Path]:
    """Return every processed dataset file under `datasets/processed/` for
    `ticker`, across any run version, sorted by filename (version is a UTC
    timestamp string, so lexicographic order == chronological order).
    """
    ticker = validate_ticker(ticker)
    processed_dir = _processed_data_dir()
    if not processed_dir.exists():
        return []

    matches = []
    for path in processed_dir.glob(f"*{_PROCESSED_SUFFIX}"):
        stem = path.name[: -len(_PROCESSED_SUFFIX)]
        if "_" not in stem:
            continue
        file_ticker, _, _version = stem.partition("_")
        if file_ticker.upper() == ticker:
            matches.append(path)

    matches.sort(key=lambda p: p.name)
    return matches


def latest_processed_dataset_path(ticker: str) -> Path:
    """Return the most recently generated processed dataset for `ticker`.

    Raises:
        ProcessedDatasetNotFoundError: no matching file exists under `datasets/processed/`.
    """
    files = find_processed_dataset_files(ticker)
    if not files:
        raise ProcessedDatasetNotFoundError(ticker, _processed_data_dir())
    return files[-1]


def _validate_schema(df: pd.DataFrame, source, config: MLConfig) -> None:
    missing = [c for c in ("date",) if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"processed dataset '{source}' is missing required column(s)", missing_columns=missing
        )
    present_targets = [c for c in config.all_target_columns() if c in df.columns]
    if not present_targets:
        raise SchemaValidationError(
            f"processed dataset '{source}' contains none of the expected target columns "
            f"— was it produced by the Sprint 2 pipeline with target_horizons enabled?",
            missing_columns=list(config.all_target_columns()),
        )


def load_processed_dataset(
    ticker: Optional[str] = None,
    path: Optional[Path] = None,
    df: Optional[pd.DataFrame] = None,
    config: MLConfig = DEFAULT_ML_CONFIG,
) -> pd.DataFrame:
    """Load a Sprint 2 processed dataset into a DataFrame, sorted by date ascending.

    Args:
        ticker: looked up via `find_processed_dataset_files` if `path`/`df` not given
            (uses the most recent run for that ticker).
        path: an explicit processed CSV path, bypassing ticker lookup.
        df: an already-loaded processed DataFrame, bypasses disk entirely
            (used by tests and by callers chaining Sprint 2 -> Sprint 3 in-memory).
        config: MLConfig, for target-column schema validation.

    Raises:
        ValueError: none of `ticker`, `path`, `df` was provided.
        ProcessedDatasetNotFoundError: no processed file found for `ticker`.
        SchemaValidationError: the dataset is missing `date` or every target column.
    """
    if df is not None:
        out = df.copy()
        source = "<in-memory DataFrame>"
    else:
        if path is None and ticker is None:
            raise ValueError("load_processed_dataset requires one of `ticker`, `path`, or `df`")
        source = Path(path) if path is not None else latest_processed_dataset_path(validate_ticker(ticker))
        logger.info("Loading processed dataset from %s", source)
        out = pd.read_csv(source)

    _validate_schema(out, source, config)

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    logger.info("Loaded %d processed rows x %d columns from %s", len(out), out.shape[1], source)
    return out


def build_inference_matrix(
    df: pd.DataFrame, config: MLConfig = DEFAULT_ML_CONFIG,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a raw (unencoded) feature matrix for *inference* rather than
    training: every row is kept, with no target-based row dropping —
    unlike `build_feature_matrix`, this never requires a target column to be
    present or non-NaN, since the rows we most want to predict for (the
    latest ones) by definition don't have a known future target yet.
    """
    exclude = set(config.non_feature_columns) | set(config.target_prefix_columns(df.columns))
    feature_columns = [c for c in df.columns if c not in exclude]
    X = df[feature_columns].copy()
    dates = df["date"].copy()
    return X, dates


def resolve_task_and_validate_target(target_column: str, config: MLConfig = DEFAULT_ML_CONFIG) -> str:
    """Return "regression" or "classification" for `target_column`.

    Raises:
        InvalidTargetColumnError: `target_column` isn't one of the columns
            Sprint 2's TargetFeatureGenerator produces.
    """
    if target_column in config.regression_targets:
        return "regression"
    if target_column in config.classification_targets:
        return "classification"
    raise InvalidTargetColumnError(
        target_column, "regression or classification", available=list(config.all_target_columns())
    )


def build_feature_matrix(
    df: pd.DataFrame,
    target_column: str,
    config: MLConfig = DEFAULT_ML_CONFIG,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the *raw* `(X, y, dates)` triple for one target — deliberately
    unencoded and un-imputed. Categorical encoding and NaN imputation are
    fit-on-train/applied-to-val-test concerns (see
    `ai.models.ml.preprocessing.FeaturePreprocessor`), so they must happen
    *after* `ai.models.ml.splitting.TimeSeriesSplitter` partitions the data,
    never before — imputing with a statistic (e.g. median) computed across
    the whole dataset would let validation/test-period values influence what
    a training-period row looks like, a subtle but real form of leakage.

    - Drops `date`/`ticker` and *every* target-family column (all horizons,
      not just the chosen one) from the feature matrix — training on
      `future_return_5_day` must never see `target_5_day` or
      `target_direction_5_day`, which are trivially derived from it.
    - Drops rows where `y` is NaN (Sprint 2's forward-looking targets leave
      the last `max(target_horizons)` rows unlabeled by design). This alone
      never leaks: it only removes rows with no label at all.

    Raises:
        InvalidTargetColumnError: `target_column` not present in `df` or not
            a recognized target column.
    """
    task = resolve_task_and_validate_target(target_column, config)
    if target_column not in df.columns:
        raise InvalidTargetColumnError(target_column, task, available=list(df.columns))

    labeled = df.dropna(subset=[target_column]).reset_index(drop=True)
    if labeled.empty:
        raise InsufficientTrainingDataError(
            required_rows=1, available_rows=0,
            context={"target_column": target_column, "reason": "every row has a NaN target"},
        )

    exclude = set(config.non_feature_columns) | set(config.target_prefix_columns(labeled.columns))
    feature_columns = [c for c in labeled.columns if c not in exclude]

    X = labeled[feature_columns].copy()
    y = labeled[target_column].copy()
    dates = labeled["date"].copy()

    # Boolean/"boolean" (nullable) target columns -> plain 0/1 ints for classifiers.
    if task == "classification":
        y = y.astype("boolean").astype(int)

    logger.info(
        "build_feature_matrix: target=%s task=%s -> X=%s (raw, unencoded) y=%s (dropped %d unlabeled row(s))",
        target_column, task, X.shape, y.shape, len(df) - len(labeled),
    )
    return X, y, dates