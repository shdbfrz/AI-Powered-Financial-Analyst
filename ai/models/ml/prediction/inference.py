"""
Prediction for `ai/models/ml/`.

Two distinct, deliberately separate responsibilities (Sprint 3 spec asks for
both a "reusable prediction pipeline" and a separate "inference service"):

- `Predictor`: a thin, stateless-ish wrapper around one already-in-memory
  `BaseMLModel` (+ its `FeaturePreprocessor`) — used right after training,
  before anything is persisted, and by `InferenceService` once it has loaded
  both from disk.
- `InferenceService`: the production-facing facade (mirrors
  `ai.data_collection.manager.DataCollectionManager`) that *loads* a saved
  model + its matching preprocessor from `storage/models/ml/` and predicts
  for the latest available data for a ticker — this is the entry point the
  Decision Support Engine (Sprint 7) and backend API (Sprint 8) will call.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ai.models.ml.config import DEFAULT_ML_CONFIG, MLConfig
from ai.models.ml.data_loader import build_inference_matrix, load_processed_dataset
from ai.models.ml.exceptions import ModelPersistenceError, PredictionError, UnknownModelError
from ai.models.ml.models.base import BaseMLModel
from ai.models.ml.persistence.model_storage import ModelStorage
from ai.models.ml.preprocessing import FeaturePreprocessor
from ai.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """Reusable prediction pipeline for one already-in-memory model.

    Handles the two things every caller would otherwise duplicate: routing
    raw features through the model's preprocessor (if any), and defensively
    re-aligning columns to exactly what the model was trained on (extra
    columns dropped, missing columns filled with 0) so a slightly different
    upstream schema degrades gracefully instead of raising.
    """

    def __init__(self, model: BaseMLModel, preprocessor: Optional[FeaturePreprocessor] = None):
        self.model = model
        self.preprocessor = preprocessor
        self.logger = logger

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X_ready = self.preprocessor.transform(X) if self.preprocessor is not None else X
        missing = [c for c in self.model.feature_names_ if c not in X_ready.columns]
        if missing:
            self.logger.warning(
                "Predictor: %d feature(s) the model expects are absent from the input; filling with 0.0: %s",
                len(missing), missing[:10],
            )
        return X_ready.reindex(columns=self.model.feature_names_, fill_value=0.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare(X))

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self.model.info.task_type != "classification":
            return None
        return self.model.predict_proba(self._prepare(X))


@dataclass
class InferenceResult:
    ticker: str
    target_column: str
    model_name: str
    task_type: str
    dates: list
    predictions: list
    probabilities: Optional[list]
    model_artifact_path: str

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "target_column": self.target_column,
            "model_name": self.model_name,
            "task_type": self.task_type,
            "predictions": [
                {
                    "date": str(d),
                    "prediction": float(p),
                    **({"probability_positive": float(prob)} if prob is not None else {}),
                }
                for d, p, prob in zip(
                    self.dates, self.predictions,
                    self.probabilities if self.probabilities is not None else [None] * len(self.predictions),
                )
            ],
            "model_artifact_path": self.model_artifact_path,
        }


class InferenceService:
    """Facade over `ModelStorage` + `Predictor`: load the latest saved model
    (and its matching preprocessor) for `(ticker, target_column, model_name)`
    and predict for the most recent processed rows.

    Note: per `ARCHITECTURE.md` §2.2, this service *only* produces a raw
    model prediction — it is not the Decision Support Engine (Sprint 7,
    which combines this with sentiment/risk into a signal) and it is not
    the LLM Explanation Layer (Sprint 8, which only narrates an already
    -computed decision and never predicts).
    """

    def __init__(self, storage: Optional[ModelStorage] = None, config: MLConfig = DEFAULT_ML_CONFIG):
        self.storage = storage or ModelStorage()
        self.config = config
        self.logger = logger

    def predict_latest(
        self,
        ticker: str,
        target_column: str,
        model_name: str,
        processed_df: Optional[pd.DataFrame] = None,
        n_rows: int = 1,
    ) -> InferenceResult:
        """Predict for the most recent `n_rows` processed rows of `ticker`.

        Raises:
            ModelPersistenceError: no saved model/preprocessor matches.
            UnknownModelError: `model_name` isn't a recognized registry name
                (surfaced early, before touching disk, for a clearer error).
        """
        from ai.models.ml.models.registry import ModelFactory
        if model_name not in ModelFactory.available_models():
            raise UnknownModelError(model_name, available=ModelFactory.available_models())

        df = processed_df if processed_df is not None else load_processed_dataset(ticker=ticker, config=self.config)
        X_raw, dates = build_inference_matrix(df, self.config)

        model_matches = self.storage.find_models(ticker, target_column=target_column, model_name=model_name)
        if not model_matches:
            raise ModelPersistenceError(
                f"no saved model found for ticker='{ticker}' target_column='{target_column}' model_name='{model_name}'"
            )
        model_path = model_matches[-1]
        model = self.storage.load(model_path)

        preprocessor = None
        try:
            preprocessor_path = self.storage.find_preprocessor(ticker, target_column)
            preprocessor = self.storage.load_preprocessor(preprocessor_path)
        except ModelPersistenceError:
            self.logger.warning(
                "No saved preprocessor found for ticker='%s' target_column='%s'; predicting on raw features "
                "as-is (only safe if every feature is already numeric).", ticker, target_column,
            )

        predictor = Predictor(model, preprocessor)
        latest_X = X_raw.tail(n_rows)
        latest_dates = dates.tail(n_rows)

        try:
            predictions = predictor.predict(latest_X)
        except Exception as e:  # noqa: BLE001
            raise PredictionError(model_name, str(e)) from e
        probabilities = None
        if model.info.task_type == "classification":
            proba = predictor.predict_proba(latest_X)
            probabilities = proba[:, 1].tolist() if proba is not None else None

        self.logger.info(
            "InferenceService.predict_latest: ticker=%s target=%s model=%s -> %d prediction(s)",
            ticker, target_column, model_name, len(predictions),
        )
        return InferenceResult(
            ticker=ticker.upper(), target_column=target_column, model_name=model_name,
            task_type=model.info.task_type, dates=latest_dates.tolist(), predictions=predictions.tolist(),
            probabilities=probabilities, model_artifact_path=str(model_path),
        )