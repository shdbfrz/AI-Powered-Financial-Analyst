"""
Model artifact persistence for `ai/models/time_series/`.

Trained models are saved to `storage/models/time_series/` — a sibling of
Sprint 3's `storage/models/ml/`, both under the single `models_dir` location
`ARCHITECTURE.md` §4 already defines for trained artifacts across
ML/Time-Series/Deep-Learning, not a newly invented directory. Writes are
atomic (temp file + `os.replace`), same as Sprint 3's `ModelStorage`, so a
crash mid-save never leaves a corrupt artifact behind.

Prophet models are NOT joblib-safe in all versions (the underlying Stan fit
object can contain unpicklable state) — this module uses Prophet's own
`model_to_json`/`model_from_json` for that one model family and joblib for
every statsmodels-backed model, transparently, behind one interface.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib

from ai.models.time_series.exceptions import ModelPersistenceError
from ai.models.time_series.models.base import BaseTimeSeriesModel
from ai.utils.config import settings
from ai.utils.logger import get_logger

logger = get_logger(__name__)

_TS_MODELS_SUBDIR = "time_series"


@dataclass
class SavedTimeSeriesModelInfo:
    path: Path
    model_name: str
    ticker: str
    horizon_days: int
    version: str
    training_date: str
    hyperparameters: dict
    test_metrics: dict
    file_size_bytes: int

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "model_name": self.model_name,
            "ticker": self.ticker,
            "horizon_days": self.horizon_days,
            "version": self.version,
            "training_date": self.training_date,
            "hyperparameters": self.hyperparameters,
            "test_metrics": self.test_metrics,
            "file_size_bytes": self.file_size_bytes,
        }


def _models_dir() -> Path:
    return settings.resolve(settings.models_dir, _TS_MODELS_SUBDIR)


def _artifact_path(ticker: str, model_name: str, horizon_days: int, version: str) -> Path:
    return _models_dir() / f"{ticker}_{model_name}_h{horizon_days}_{version}.{'json' if model_name == 'prophet' else 'joblib'}"


class TimeSeriesModelStorage:
    """Atomic save/load of trained time series model artifacts + metadata."""

    def __init__(self):
        self.logger = logger

    def save(
        self,
        model: BaseTimeSeriesModel,
        *,
        ticker: str,
        horizon_days: int,
        test_metrics: dict,
        version: str | None = None,
    ) -> SavedTimeSeriesModelInfo:
        if not model.is_fitted:
            raise ModelPersistenceError("cannot save a model that has not been fitted")

        version = version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        models_dir = _models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        path = _artifact_path(ticker, model.model_name, horizon_days, version)
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        try:
            if model.model_name == "prophet":
                from prophet.serialize import model_to_json

                tmp_path.write_text(model_to_json(model._fitted_model))
            else:
                joblib.dump(model, tmp_path)
            os.replace(tmp_path, path)
        except OSError as e:
            raise ModelPersistenceError(f"failed writing model artifact '{path}': {e}") from e
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        info = SavedTimeSeriesModelInfo(
            path=path,
            model_name=model.model_name,
            ticker=ticker,
            horizon_days=horizon_days,
            version=version,
            training_date=datetime.now(timezone.utc).isoformat(),
            hyperparameters=model.get_params(),
            test_metrics=test_metrics,
            file_size_bytes=path.stat().st_size,
        )
        self._write_metadata_entry(info)
        self.logger.info(
            "model artifact saved",
            extra={"ticker": ticker, "model": model.model_name, "horizon": horizon_days, "path": str(path)},
        )
        return info

    def load(self, path: Path):
        if not path.exists():
            raise ModelPersistenceError(f"model artifact not found: '{path}'")
        try:
            if path.suffix == ".json":
                from prophet.serialize import model_from_json

                return model_from_json(path.read_text())
            return joblib.load(path)
        except Exception as e:  # noqa: BLE001
            raise ModelPersistenceError(f"failed loading model artifact '{path}': {e}") from e

    def _write_metadata_entry(self, info: SavedTimeSeriesModelInfo) -> None:
        metadata_path = _models_dir() / "Model_Metadata.json"
        entries = []
        if metadata_path.exists():
            try:
                entries = json.loads(metadata_path.read_text())
            except (json.JSONDecodeError, OSError):
                entries = []
        entries.append(info.as_dict())

        tmp_path = metadata_path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(entries, indent=2, default=str))
            os.replace(tmp_path, metadata_path)
        except OSError as e:
            raise ModelPersistenceError(f"failed updating '{metadata_path}': {e}") from e
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass