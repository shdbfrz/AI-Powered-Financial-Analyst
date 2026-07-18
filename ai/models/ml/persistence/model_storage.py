"""
Model artifact persistence for `ai/models/ml/`.

Trained models are saved to `storage/models/ml/` (the canonical location
`ARCHITECTURE.md` §4 and `storage/README.md` already define for trained
model artifacts across ML/Time Series/Deep Learning — not a new
`ai/models/ml/saved_models/` directory, to avoid two competing locations for
the same artifact type). Like Sprint 2's `storage.py`, writes are atomic
(temp file + rename) so a crash mid-write never leaves a corrupt `.joblib`
file where the Decision Support Engine (Sprint 7) or a backend request
handler (Sprint 8) expects a valid one.
"""

import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import joblib

from ai.models.ml.exceptions import ModelPersistenceError
from ai.models.ml.models.base import BaseMLModel
from ai.utils.config import settings
from ai.utils.logger import get_logger

logger = get_logger(__name__)


def estimate_memory_bytes(estimator) -> int:
    """Serialize `estimator` with joblib into an in-memory buffer and measure
    its byte size — reuses the exact serialization path used for saving, so
    the "Memory Usage" figure in the Model Comparison table matches the
    actual `.joblib` file size on disk, not an approximation.
    """
    buffer = io.BytesIO()
    joblib.dump(estimator, buffer)
    return buffer.tell()


@dataclass
class SavedModelInfo:
    """One entry in `Model_Metadata.json` — everything needed to locate,
    reload, and audit a saved model artifact.
    """

    path: Path
    model_name: str
    display_name: str
    ticker: str
    target_column: str
    task_type: str
    version: str
    training_date: str
    features_used: list
    hyperparameters: dict
    test_metrics: dict
    file_size_bytes: int

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "model_name": self.model_name,
            "display_name": self.display_name,
            "ticker": self.ticker,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "version": self.version,
            "training_date": self.training_date,
            "n_features_used": len(self.features_used),
            "features_used": self.features_used,
            "hyperparameters": self.hyperparameters,
            "test_metrics": self.test_metrics,
            "file_size_bytes": self.file_size_bytes,
        }


def _atomic_dump(obj, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        joblib.dump(obj, tmp_path)
        os.replace(tmp_path, path)
    except OSError as e:
        raise ModelPersistenceError(f"failed writing model artifact '{path}': {e}") from e
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _atomic_write_json(path: Path, data: Union[dict, list]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except OSError as e:
        raise ModelPersistenceError(f"failed writing '{path}': {e}") from e
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


class ModelStorage:
    """Saves/loads whole `BaseMLModel` wrapper objects (not just the raw
    scikit-learn estimator) via joblib, so `feature_names_`, `hyperparameters`,
    and `info` all travel with the artifact — the Inference Service
    (`ai.models.ml.prediction.inference`) can load one file and immediately
    call `.predict()` without reconstructing any of that state.
    """

    def __init__(self):
        self.logger = logger

    def _models_dir(self) -> Path:
        path = settings.resolve(settings.models_dir, "ml")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def build_filename(self, ticker: str, target_column: str, model_name: str, version: str) -> str:
        return f"{ticker}_{target_column}_{model_name}_{version}.joblib"

    def save(
        self, model: BaseMLModel, *, ticker: str, target_column: str, version: str, test_metrics: dict,
    ) -> SavedModelInfo:
        """Persist a fitted model to `storage/models/ml/` and return its metadata record.

        Raises:
            ModelPersistenceError: the model isn't fitted, or the write failed.
        """
        if not model.is_fitted:
            raise ModelPersistenceError(f"refusing to save unfitted model '{model.info.name}'")

        filename = self.build_filename(ticker.upper(), target_column, model.info.name, version)
        path = self._models_dir() / filename
        _atomic_dump(model, path)
        file_size = path.stat().st_size

        info = SavedModelInfo(
            path=path, model_name=model.info.name, display_name=model.info.display_name,
            ticker=ticker.upper(), target_column=target_column, task_type=model.info.task_type,
            version=version, training_date=datetime.now(timezone.utc).isoformat(),
            features_used=model.feature_names_, hyperparameters=model.hyperparameters,
            test_metrics=test_metrics, file_size_bytes=file_size,
        )
        self.logger.info("Saved model '%s' (%d bytes) to %s", model.info.name, file_size, path)
        return info

    def load(self, path: Path) -> BaseMLModel:
        """Load a previously-saved `BaseMLModel` wrapper.

        Raises:
            ModelPersistenceError: the file doesn't exist or fails to deserialize.
        """
        path = Path(path)
        if not path.exists():
            raise ModelPersistenceError(f"model artifact not found: {path}")
        try:
            model = joblib.load(path)
        except Exception as e:  # noqa: BLE001
            raise ModelPersistenceError(f"failed to load model artifact '{path}': {e}") from e
        self.logger.info("Loaded model '%s' from %s", getattr(model, "info", None) and model.info.name, path)
        return model

    def save_preprocessor(self, preprocessor, *, ticker: str, target_column: str, version: str) -> Path:
        """Persist a fitted `FeaturePreprocessor` as a sibling artifact to the
        model(s) trained on its output — inference-time data must go through
        the exact same fit-on-train encoding/imputation, never a freshly
        re-fitted one (that would silently reintroduce the leakage this
        module's docstring warns about).
        """
        filename = f"{ticker.upper()}_{target_column}_preprocessor_{version}.joblib"
        path = self._models_dir() / filename
        _atomic_dump(preprocessor, path)
        self.logger.info("Saved feature preprocessor to %s", path)
        return path

    def load_preprocessor(self, path: Path):
        path = Path(path)
        if not path.exists():
            raise ModelPersistenceError(f"preprocessor artifact not found: {path}")
        try:
            return joblib.load(path)
        except Exception as e:  # noqa: BLE001
            raise ModelPersistenceError(f"failed to load preprocessor artifact '{path}': {e}") from e

    def find_preprocessor(self, ticker: str, target_column: str) -> Path:
        """Most recent preprocessor artifact for `(ticker, target_column)`."""
        matches = sorted(self._models_dir().glob(f"{ticker.upper()}_{target_column}_preprocessor_*.joblib"))
        if not matches:
            raise ModelPersistenceError(
                f"no saved preprocessor found for ticker='{ticker}' target_column='{target_column}'"
            )
        return matches[-1]

    def find_models(self, ticker: str, target_column: str = None, model_name: str = None) -> list[Path]:
        """Every saved artifact matching the given filters, most recent version last."""
        pattern_ticker = ticker.upper()
        matches = []
        for path in self._models_dir().glob(f"{pattern_ticker}_*.joblib"):
            stem = path.stem
            parts = stem.split("_")
            if len(parts) < 4:
                continue
            if target_column and target_column not in stem:
                continue
            if model_name and model_name not in stem:
                continue
            matches.append(path)
        matches.sort(key=lambda p: p.name)
        return matches

    def save_metadata_json(self, saved_infos: list[SavedModelInfo], ticker: str, version: str, target_column: str = "") -> Path:
        """Save `Model_Metadata.json`: one entry per saved artifact (Sprint 3
        output requirement: "Model Name, Training Date, Features Used,
        Metrics, Version").

        `target_column` is included in the filename (when provided) so that
        running regression and classification for the same `(ticker,
        version)` — a common pattern, since `MLTrainingPipeline.run()`
        defaults `version` to the same timestamp for both calls in one
        session — never causes one task's metadata file to silently
        overwrite the other's; the underlying `.joblib` model files were
        never at risk of this (their filenames already include
        `target_column`), only this aggregate index file was.
        """
        suffix = f"_{target_column}" if target_column else ""
        path = self._models_dir() / f"{ticker.upper()}_{version}{suffix}_Model_Metadata.json"
        payload = {
            "ticker": ticker.upper(), "version": version, "target_column": target_column,
            "models": [info.as_dict() for info in saved_infos],
        }
        _atomic_write_json(path, payload)
        self.logger.info("Saved metadata for %d model(s) to %s", len(saved_infos), path)
        return path