"""
Supervised feature selection for `ai/models/ml/`.

Distinct from (and complementary to) `ai.feature_engineering.selection`,
which reports pairwise correlation/low-variance/duplicate-column findings
*without* looking at any target (Sprint 2 doesn't have one yet at that
stage). This module is target-aware: RFE and permutation importance need
`y` to rank features by how much they actually help predict it, which is
exactly what Sprint 3 needs and Sprint 2 architecturally cannot do.
"""

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression

from ai.models.ml.config import DEFAULT_ML_CONFIG, MLConfig
from ai.models.ml.exceptions import FeatureSelectionError
from ai.utils.logger import get_logger

logger = get_logger(__name__)

TaskType = Literal["regression", "classification"]


@dataclass
class FeatureSelectionResult:
    """Every selection method's findings, plus one combined recommendation."""

    correlation_dropped: list[str] = field(default_factory=list)
    variance_dropped: list[str] = field(default_factory=list)
    rfe_selected: list[str] = field(default_factory=list)
    permutation_importance_ranking: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    tree_importance_ranking: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    recommended_features: list[str] = field(default_factory=list)
    candidate_features_after_pruning: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "correlation_dropped": self.correlation_dropped,
            "variance_dropped": self.variance_dropped,
            "rfe_selected": self.rfe_selected,
            "permutation_importance_top20": self.permutation_importance_ranking.head(20).round(6).to_dict(),
            "tree_importance_top20": self.tree_importance_ranking.head(20).round(6).to_dict(),
            "recommended_features": self.recommended_features,
            "candidate_features_after_pruning": len(self.candidate_features_after_pruning),
        }


class FeatureSelector:
    """Runs correlation/variance/RFE/permutation/tree-importance analysis
    over a training split and recommends a final feature subset by
    rank-voting across the target-aware methods (RFE, permutation
    importance, tree importance) after excluding features flagged by the
    target-agnostic filters (correlation, variance).
    """

    def __init__(self, config: MLConfig = DEFAULT_ML_CONFIG):
        self.config = config
        self.logger = logger

    # ------------------------------------------------------------------
    # Individual methods (each usable standalone)
    # ------------------------------------------------------------------

    def correlation_analysis(self, X: pd.DataFrame) -> list[str]:
        """Return features to drop: for every pair with |correlation| >=
        threshold, drop the second-listed one (keep the first), so only one
        representative of each highly-correlated group survives.
        """
        if X.shape[1] < 2:
            return []
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = {
            column for column in upper.columns
            if (upper[column] >= self.config.correlation_threshold).any()
        }
        dropped = sorted(to_drop)
        self.logger.info("correlation_analysis: dropping %d of %d feature(s) (threshold=%.2f)",
                          len(dropped), X.shape[1], self.config.correlation_threshold)
        return dropped

    def variance_threshold(self, X: pd.DataFrame) -> list[str]:
        """Return near-constant features (variance below `config.variance_threshold`)."""
        try:
            selector = VarianceThreshold(threshold=self.config.variance_threshold)
            selector.fit(X)
        except ValueError as e:
            raise FeatureSelectionError("variance_threshold", str(e)) from e
        dropped = sorted(X.columns[~selector.get_support()].tolist())
        self.logger.info("variance_threshold: dropping %d of %d feature(s) (threshold=%.2e)",
                          len(dropped), X.shape[1], self.config.variance_threshold)
        return dropped

    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, task_type: TaskType) -> list[str]:
        """RFE with a fast linear reference estimator (LinearRegression /
        LogisticRegression) — deliberately not a tree ensemble, since RFE
        refits the estimator `n_features / step` times and a 200-tree forest
        at that cadence would make this step the pipeline's bottleneck.
        `step` removes multiple features per iteration for the same reason.
        """
        n_to_select = min(self.config.rfe_n_features_to_select, X.shape[1])
        estimator = LinearRegression() if task_type == "regression" else LogisticRegression(max_iter=1000)
        step = max(1, X.shape[1] // 20)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rfe = RFE(estimator=estimator, n_features_to_select=n_to_select, step=step)
                rfe.fit(X, y)
        except Exception as e:  # noqa: BLE001
            raise FeatureSelectionError("rfe", str(e), context={"n_features": X.shape[1]}) from e
        selected = sorted(X.columns[rfe.support_].tolist())
        self.logger.info("recursive_feature_elimination: selected %d of %d feature(s)", len(selected), X.shape[1])
        return selected

    def _reference_estimator(self, task_type: TaskType):
        """A modest tree ensemble used purely as a feature-ranking signal —
        not a candidate for the final Model Comparison table.
        """
        common = dict(n_estimators=self.config.reference_estimator_n_estimators, max_depth=8,
                      n_jobs=-1, random_state=self.config.random_state)
        return ExtraTreesRegressor(**common) if task_type == "regression" else ExtraTreesClassifier(**common)

    def permutation_importance_ranking(self, X: pd.DataFrame, y: pd.Series, task_type: TaskType) -> pd.Series:
        """Fit a reference estimator, then measure how much each feature's
        permutation degrades held-out performance — more reliable than
        tree-based importance for correlated features, at the cost of being
        slower (re-scores the model once per feature per repeat).
        """
        estimator = self._reference_estimator(task_type)
        try:
            estimator.fit(X, y)
            n_samples = min(len(X), self.config.permutation_importance_max_samples)
            sample_idx = X.sample(n=n_samples, random_state=self.config.random_state).index
            result = permutation_importance(
                estimator, X.loc[sample_idx], y.loc[sample_idx],
                n_repeats=self.config.permutation_importance_repeats,
                random_state=self.config.random_state, n_jobs=-1,
            )
        except Exception as e:  # noqa: BLE001
            raise FeatureSelectionError("permutation_importance", str(e)) from e
        ranking = pd.Series(result.importances_mean, index=X.columns, name="permutation_importance")
        return ranking.sort_values(ascending=False)

    def tree_based_importance(self, X: pd.DataFrame, y: pd.Series, task_type: TaskType) -> pd.Series:
        """Fit a reference tree ensemble and return its native `feature_importances_`
        (mean decrease in impurity) — cheap since it's a byproduct of a fit
        we'd do anyway, but biased toward high-cardinality/continuous features.
        """
        estimator = self._reference_estimator(task_type)
        try:
            estimator.fit(X, y)
        except Exception as e:  # noqa: BLE001
            raise FeatureSelectionError("tree_based_importance", str(e)) from e
        ranking = pd.Series(estimator.feature_importances_, index=X.columns, name="tree_importance")
        return ranking.sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Combined recommendation
    # ------------------------------------------------------------------

    def recommend_feature_subset(self, X: pd.DataFrame, y: pd.Series, task_type: TaskType) -> FeatureSelectionResult:
        """Run every method above and combine them into one recommendation.

        Strategy:
            1. Drop features flagged by the target-agnostic filters
               (correlation, variance) — these are always safe to remove.
            2. Rank the survivors by RFE membership + permutation importance
               rank + tree importance rank (simple rank-voting: a feature
               that's strong by all three target-aware signals ranks above
               one that's only strong by one).
            3. Recommend the top `config.top_k_recommended_features`.
        """
        correlation_dropped = self.correlation_analysis(X)
        variance_dropped = self.variance_threshold(X)
        candidates = [c for c in X.columns if c not in set(correlation_dropped) | set(variance_dropped)]

        if len(candidates) < 2:
            raise FeatureSelectionError(
                "recommend_feature_subset", "fewer than 2 candidate features survived correlation/variance pruning",
                context={"candidates": candidates},
            )

        X_candidates = X[candidates]
        rfe_selected = self.recursive_feature_elimination(X_candidates, y, task_type)
        permutation_ranking = self.permutation_importance_ranking(X_candidates, y, task_type)
        tree_ranking = self.tree_based_importance(X_candidates, y, task_type)

        votes = pd.Series(0.0, index=candidates)
        votes.loc[rfe_selected] += 1.0
        # Rank-based (not raw-magnitude) scoring: robust to permutation
        # importance / impurity importance living on completely different
        # numeric scales.
        votes += (1.0 - (permutation_ranking.rank(ascending=False) / len(permutation_ranking)))
        votes += (1.0 - (tree_ranking.rank(ascending=False) / len(tree_ranking)))

        top_k = min(self.config.top_k_recommended_features, len(candidates))
        recommended = votes.sort_values(ascending=False).head(top_k).index.tolist()

        self.logger.info(
            "recommend_feature_subset: %d -> %d candidates (after pruning) -> %d recommended",
            X.shape[1], len(candidates), len(recommended),
        )

        return FeatureSelectionResult(
            correlation_dropped=correlation_dropped,
            variance_dropped=variance_dropped,
            rfe_selected=rfe_selected,
            permutation_importance_ranking=permutation_ranking,
            tree_importance_ranking=tree_ranking,
            recommended_features=recommended,
            candidate_features_after_pruning=candidates,
        )