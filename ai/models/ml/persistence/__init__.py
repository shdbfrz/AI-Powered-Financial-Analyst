"""
`ai/models/ml/persistence/` — joblib model artifact persistence + metadata.

    from ai.models.ml.persistence import ModelStorage

    storage = ModelStorage()
    saved = storage.save(model, ticker="AAPL", target_column="future_return_5_day", version="v1")
    loaded_model = storage.load(saved.path)
"""

from ai.models.ml.persistence.model_storage import ModelStorage, SavedModelInfo, estimate_memory_bytes

__all__ = ["ModelStorage", "SavedModelInfo", "estimate_memory_bytes"]