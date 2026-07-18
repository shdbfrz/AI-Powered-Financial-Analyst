"""
`ai/models/ml/selection/` — supervised feature selection for the ML
training pipeline (correlation pruning, variance thresholding, RFE,
permutation importance, tree-based importance), combined into one
recommended feature subset.

    from ai.models.ml.selection import FeatureSelector

    selector = FeatureSelector()
    result = selector.recommend_feature_subset(X_train, y_train, task_type="regression")
    X_train_selected = X_train[result.recommended_features]
"""

from ai.models.ml.selection.feature_selector import FeatureSelectionResult, FeatureSelector

__all__ = ["FeatureSelector", "FeatureSelectionResult"]