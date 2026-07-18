"""
`ai/models/ml/pipelines/` — the top-level orchestrator that ties data
loading, splitting, preprocessing, feature selection, tuning, training,
evaluation, comparison, persistence, and visualization into one call.

    from ai.models.ml.pipelines import MLTrainingPipeline

    pipeline = MLTrainingPipeline()
    result = pipeline.run(ticker="AAPL", task="regression")
"""

from ai.models.ml.pipelines.training_pipeline import MLTrainingPipeline, MLTrainingResult

__all__ = ["MLTrainingPipeline", "MLTrainingResult"]