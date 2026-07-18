"""
`ai/models/ml/` — Sprint 3: Baseline Machine Learning Pipeline.

Trains, tunes, evaluates, compares, persists, and visualizes fifteen
baseline models (ten regression + two optional regression + three
classification) on Sprint 2's engineered features, for the regression
targets `target_{1,3,5}_day` / `future_return_{1,3,5}_day` and the
classification targets `target_direction_{1,3,5}_day`.

Quick start:

    from ai.models.ml import MLTrainingPipeline

    pipeline = MLTrainingPipeline()
    result = pipeline.run(ticker="AAPL", task="regression")
    print(result.best_model_name, result.comparison_table.dataframe)

See `ai/models/ml/README.md` for the full architecture, and
`docs/architecture/ML_PIPELINE_ARCHITECTURE.md` /
`docs/architecture/ML_PIPELINE_WORKFLOW.md` for diagrams.
"""

from ai.models.ml.config import DEFAULT_ML_CONFIG, MLConfig
from ai.models.ml.data_loader import load_processed_dataset
from ai.models.ml.models import ModelFactory
from ai.models.ml.persistence import ModelStorage
from ai.models.ml.pipelines import MLTrainingPipeline, MLTrainingResult
from ai.models.ml.prediction import InferenceService

__all__ = [
    "MLConfig",
    "DEFAULT_ML_CONFIG",
    "MLTrainingPipeline",
    "MLTrainingResult",
    "ModelFactory",
    "ModelStorage",
    "InferenceService",
    "load_processed_dataset",
]