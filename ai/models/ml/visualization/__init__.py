"""
`ai/models/ml/visualization/` — matplotlib plot generation for
`storage/reports/ml/`: prediction-vs-actual, residuals, feature importance,
learning curves, validation curves, and the model comparison chart.

    from ai.models.ml.visualization import PlotGenerator

    plots = PlotGenerator()
    plots.prediction_vs_actual(y_test, predictions, title="XGBoost", path=out_path)
"""

from ai.models.ml.visualization.plots import PlotGenerator

__all__ = ["PlotGenerator"]