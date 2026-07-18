"""
`ai/models/ml/prediction/` — reusable prediction pipeline (`Predictor`) and
the production-facing facade (`InferenceService`) that loads a saved model +
its matching preprocessor and predicts for new data.

    from ai.models.ml.prediction import InferenceService

    service = InferenceService()
    result = service.predict_latest(ticker="AAPL", target_column="future_return_5_day", model_name="xgboost")
"""

from ai.models.ml.prediction.inference import InferenceResult, InferenceService, Predictor

__all__ = ["Predictor", "InferenceService", "InferenceResult"]