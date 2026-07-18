"""
Plot generation for `ai/models/ml/`.

Headless (`Agg` backend) since this runs inside a training pipeline / CI
job / backend server, never an interactive session — every method here
saves directly to a PNG under `storage/reports/ml/` and returns the path,
rather than calling `plt.show()`.
"""

from pathlib import Path


import matplotlib
matplotlib.use("Agg")  # noqa: E402 - must precede pyplot import; this module only ever writes files
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import learning_curve, validation_curve  # noqa: E402

from ai.utils.logger import get_logger

logger = get_logger(__name__)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


class PlotGenerator:
    """Every method saves one PNG and returns its path. Callers (mainly
    `ai.models.ml.pipelines.training_pipeline`) decide *which* models/plots
    to generate and where — this class only knows how to draw them.
    """

    def __init__(self):
        self.logger = logger

    def _save(self, fig, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("Saved plot to %s", path)
        return path

    def prediction_vs_actual(self, y_true, y_pred, title: str, path: Path) -> Path:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.4, s=14, edgecolors="none")
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect prediction (y = x)")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Prediction vs Actual — {title}")
        ax.legend()
        return self._save(fig, path)

    def residual_plot(self, y_true, y_pred, title: str, path: Path) -> Path:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter(y_pred, residuals, alpha=0.4, s=14, edgecolors="none")
        axes[0].axhline(0, color="r", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residual (actual - predicted)")
        axes[0].set_title("Residuals vs Predicted")
        axes[1].hist(residuals, bins=30, color="steelblue")
        axes[1].set_xlabel("Residual")
        axes[1].set_title("Residual Distribution")
        fig.suptitle(f"Residual Analysis — {title}")
        return self._save(fig, path)

    def feature_importance_plot(self, importance: pd.Series, title: str, path: Path, top_n: int = 20) -> Path:
        top = importance.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.32 * len(top))))
        ax.barh(top.index.astype(str), top.values, color="teal")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {len(top)} Feature Importances — {title}")
        return self._save(fig, path)

    def learning_curve_plot(
        self, estimator, X: pd.DataFrame, y: pd.Series, title: str, path: Path,
        cv, scoring: str, train_sizes,
    ) -> Path:
        """Training-set-size vs score, for both train and held-out CV folds
        — reveals whether a model would benefit from more data (still
        improving) or has plateaued (more data won't help; the bottleneck is
        the features/model, not the data volume).
        """
        sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, scoring=scoring, train_sizes=list(train_sizes), n_jobs=-1,
        )
        fig, ax = plt.subplots(figsize=(7.5, 5))
        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)
        ax.plot(sizes, train_mean, "o-", label="Training score", color="darkorange")
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="darkorange")
        ax.plot(sizes, val_mean, "o-", label="Cross-validation score", color="steelblue")
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="steelblue")
        ax.set_xlabel("Training examples")
        ax.set_ylabel(scoring)
        ax.set_title(f"Learning Curve — {title}")
        ax.legend(loc="best")
        return self._save(fig, path)

    def validation_curve_plot(
        self, estimator, X: pd.DataFrame, y: pd.Series, param_name: str, param_range: list,
        title: str, path: Path, cv, scoring: str,
    ) -> Path:
        """Score vs one hyperparameter's value — reveals under/overfitting
        as that hyperparameter varies (distinct from the learning curve,
        which varies *data volume* instead).
        """
        train_scores, val_scores = validation_curve(
            estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring, n_jobs=-1,
        )
        fig, ax = plt.subplots(figsize=(7.5, 5))
        x_positions = range(len(param_range))
        ax.plot(x_positions, train_scores.mean(axis=1), "o-", label="Training score", color="darkorange")
        ax.plot(x_positions, val_scores.mean(axis=1), "o-", label="Cross-validation score", color="steelblue")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels([str(p) for p in param_range])
        ax.set_xlabel(param_name)
        ax.set_ylabel(scoring)
        ax.set_title(f"Validation Curve ({param_name}) — {title}")
        ax.legend(loc="best")
        return self._save(fig, path)

    def model_comparison_chart(
        self, comparison_df: pd.DataFrame, primary_metric: str, title: str, path: Path,
    ) -> Path:
        ordered = comparison_df.sort_values("primary_metric_value")
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(ordered))))
        colors = ["seagreen" if i == 0 else "steelblue" for i in range(len(ordered))][::-1]
        ax.barh(ordered["display_name"], ordered["primary_metric_value"], color=colors)
        ax.set_xlabel(primary_metric)
        ax.set_title(title)
        return self._save(fig, path)