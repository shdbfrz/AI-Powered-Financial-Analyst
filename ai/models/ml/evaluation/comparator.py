"""
Model comparison and reporting for `ai/models/ml/`.

`ModelComparator` turns a list of `ModelResult` (one per trained model) into:
  - a ranked `Model_Comparison.csv`-ready DataFrame
  - a short, templated "why one model performs better" narrative
  - a full `Evaluation_Report.md`
  - `Model_Documentation.md`, rendered from each model's `ModelInfo` (the
    same "generate documentation from metadata instead of maintaining it by
    hand twice" approach `ai.feature_engineering.storage.save_feature_report`
    uses for `Feature_Report.md`).
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ai.models.ml.models.base import ModelInfo
from ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """Everything collected about one trained model — the unit `ModelComparator` ranks."""

    model_name: str
    display_name: str
    task_type: str
    target_column: str
    hyperparameters: dict
    train_metrics: dict
    validation_metrics: dict
    test_metrics: dict
    training_time_seconds: float
    prediction_time_seconds: float
    complexity_score: int
    memory_bytes: int
    was_tuned: bool = False
    feature_importance: Optional[pd.Series] = None
    model_artifact_path: Optional[str] = None
    error: Optional[str] = None  # set instead of raising, if this model failed — keeps the run going for the rest

    @property
    def memory_kb(self) -> float:
        return round(self.memory_bytes / 1024, 2)

    def flat_row(self, primary_metric: str) -> dict:
        row = {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "task_type": self.task_type,
            "target_column": self.target_column,
            "was_tuned": self.was_tuned,
            "training_time_seconds": round(self.training_time_seconds, 4),
            "prediction_time_seconds": round(self.prediction_time_seconds, 6),
            "model_complexity": self.complexity_score,
            "memory_kb": self.memory_kb,
        }
        for split_name, metrics in (("train", self.train_metrics), ("validation", self.validation_metrics), ("test", self.test_metrics)):
            for metric_name, value in metrics.items():
                row[f"{split_name}_{metric_name}"] = value
        row["primary_metric"] = primary_metric
        row["primary_metric_value"] = self.test_metrics.get(primary_metric)
        return row


@dataclass
class ComparisonTable:
    task_type: str
    primary_metric: str
    primary_metric_direction: str  # "minimize" | "maximize"
    dataframe: pd.DataFrame
    narrative: str


class ModelComparator:
    """Ranks trained models and renders the sprint's reporting artifacts."""

    def __init__(self):
        self.logger = logger

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def build_comparison_table(
        self, results: list[ModelResult], primary_metric: str, primary_metric_direction: str,
    ) -> ComparisonTable:
        """Rank a list of same-task-type `ModelResult`s.

        Two ranks are reported, since "rank by accuracy/RMSE, training time,
        prediction speed, complexity, and memory" are six criteria that
        don't reduce to one number without an arbitrary weighting scheme:
          - `performance_rank`: sorted purely by `primary_metric` on the test
            split (industry-standard: correctness first).
          - `efficiency_rank`: an unweighted average of normalized
            performance + normalized (inverse) training time + complexity +
            memory — a secondary, clearly-labeled view for when "good enough
            and cheap" beats "best and expensive" (e.g. resource-constrained
            deployment).
        """
        usable = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]
        if failed:
            self.logger.warning("build_comparison_table: %d model(s) failed and are excluded from ranking: %s",
                                 len(failed), [r.model_name for r in failed])
        if not usable:
            return ComparisonTable(
                task_type=results[0].task_type if results else "unknown",
                primary_metric=primary_metric, primary_metric_direction=primary_metric_direction,
                dataframe=pd.DataFrame(), narrative="No model trained successfully.",
            )

        rows = [r.flat_row(primary_metric) for r in usable]
        df = pd.DataFrame(rows)

        ascending = primary_metric_direction == "minimize"
        df = df.sort_values("primary_metric_value", ascending=ascending).reset_index(drop=True)
        df.insert(0, "performance_rank", range(1, len(df) + 1))

        perf_norm = self._normalize(df["primary_metric_value"], higher_is_better=not ascending)
        time_norm = self._normalize(df["training_time_seconds"], higher_is_better=False)
        complexity_norm = self._normalize(df["model_complexity"], higher_is_better=False)
        memory_norm = self._normalize(df["memory_kb"], higher_is_better=False)
        df["efficiency_score"] = ((perf_norm + time_norm + complexity_norm + memory_norm) / 4).round(4)
        df = df.sort_values("efficiency_score", ascending=False)
        df["efficiency_rank"] = range(1, len(df) + 1)
        df = df.sort_values("performance_rank").reset_index(drop=True)

        narrative = self.explain_ranking(df, usable, primary_metric, primary_metric_direction)

        return ComparisonTable(
            task_type=usable[0].task_type, primary_metric=primary_metric,
            primary_metric_direction=primary_metric_direction, dataframe=df, narrative=narrative,
        )

    @staticmethod
    def _normalize(series: pd.Series, higher_is_better: bool) -> pd.Series:
        s = series.astype(float)
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(1.0, index=s.index)
        norm = (s - lo) / (hi - lo)
        return norm if higher_is_better else 1.0 - norm

    def explain_ranking(
        self, ranked_df: pd.DataFrame, results: list[ModelResult], primary_metric: str, direction: str,
    ) -> str:
        """A short, templated "why one model performs better" narrative —
        the Sprint 3 spec's Model Comparison section asks for this
        explicitly, generated from the actual numbers rather than boilerplate.
        """
        if ranked_df.empty:
            return "No model trained successfully; no comparison is available."

        best = ranked_df.iloc[0]
        worst = ranked_df.iloc[-1]
        median_value = ranked_df["primary_metric_value"].median()
        best_value = best["primary_metric_value"]

        better_word = "lower" if direction == "minimize" else "higher"
        pct_vs_median = abs(best_value - median_value) / (abs(median_value) + 1e-12) * 100

        fastest = ranked_df.loc[ranked_df["training_time_seconds"].idxmin()]
        speed_multiple = best["training_time_seconds"] / max(fastest["training_time_seconds"], 1e-9)

        lines = [
            f"**{best['display_name']}** ranks #1 on {primary_metric} with a test-set value of "
            f"{best_value:.6g} — {better_word} (better) than the median model's {median_value:.6g} "
            f"by roughly {pct_vs_median:.1f}%.",
        ]
        if best["model_name"] != fastest["model_name"]:
            lines.append(
                f"It trains in {best['training_time_seconds']:.3f}s, {speed_multiple:.1f}x slower than the "
                f"fastest model in this comparison ({fastest['display_name']}, "
                f"{fastest['training_time_seconds']:.3f}s) — a reasonable trade if the accuracy gain matters "
                f"more than training cost for this use case."
            )
        else:
            lines.append(f"It is also the fastest model to train in this comparison ({best['training_time_seconds']:.3f}s).")

        top_efficiency = ranked_df.loc[ranked_df["efficiency_rank"] == 1].iloc[0]
        if top_efficiency["model_name"] != best["model_name"]:
            lines.append(
                f"When training time, model complexity, and memory footprint are weighed alongside accuracy "
                f"(the 'efficiency_rank' column), **{top_efficiency['display_name']}** is the better-balanced "
                f"choice — worth considering if inference cost or interpretability outweighs the last few "
                f"percentage points of {primary_metric}."
            )
        lines.append(
            f"**{worst['display_name']}** ranks last on {primary_metric} "
            f"({worst['primary_metric_value']:.6g}) in this comparison."
        )
        return " ".join(lines)

    # ------------------------------------------------------------------
    # Report rendering
    # ------------------------------------------------------------------

    def render_evaluation_report_markdown(
        self, tables: list[ComparisonTable], ticker: str, version: str, dataset_info: dict,
    ) -> str:
        lines = [f"# Evaluation Report — {ticker} ({version})", ""]
        lines.append("## Dataset")
        for key, value in dataset_info.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")

        for table in tables:
            lines.append(f"## {table.task_type.title()} Models")
            lines.append(f"Primary ranking metric: `{table.primary_metric}` "
                          f"({'lower is better' if table.primary_metric_direction == 'minimize' else 'higher is better'})")
            lines.append("")
            if table.dataframe.empty:
                lines.append("_No model trained successfully for this task._")
                lines.append("")
                continue

            display_cols = [c for c in [
                "performance_rank", "display_name", "primary_metric_value", "training_time_seconds",
                "prediction_time_seconds", "model_complexity", "memory_kb", "efficiency_rank", "was_tuned",
            ] if c in table.dataframe.columns]
            lines.append(table.dataframe[display_cols].to_markdown(index=False))
            lines.append("")
            lines.append("### Why the top model performs better")
            lines.append(table.narrative)
            lines.append("")

            lines.append("### Per-model detail")
            for _, row in table.dataframe.iterrows():
                lines.append(f"#### `{row['model_name']}` — {row['display_name']} (performance rank #{int(row['performance_rank'])})")
                metric_cols = [c for c in row.index if c.startswith(("train_", "validation_", "test_"))]
                for c in metric_cols:
                    value = row[c]
                    if isinstance(value, list):
                        lines.append(f"- `{c}`: {value}")
                    elif pd.notna(value):
                        lines.append(f"- `{c}`: {value}")
                lines.append("")

        return "\n".join(lines)

    def render_model_documentation_markdown(self, model_infos: list[ModelInfo]) -> str:
        lines = ["# Model Documentation", "",
                 "Auto-generated from each model's `ModelInfo` (see "
                 "`ai/models/ml/models/*.py`) — the single source of truth, so this "
                 "document can never drift from the actual model implementations.", ""]

        by_task: dict[str, list[ModelInfo]] = {}
        for info in model_infos:
            by_task.setdefault(info.task_type, []).append(info)

        for task, infos in by_task.items():
            lines.append(f"## {task.title()} Models")
            for info in infos:
                optional_tag = " _(optional dependency)_" if info.is_optional_dependency else ""
                lines += [
                    f"### {info.display_name}{optional_tag}",
                    f"- **Registry name:** `{info.name}`",
                    f"- **Family:** {info.family}",
                    f"- **Purpose:** {info.purpose}",
                    f"- **Advantages:** {info.advantages}",
                    f"- **Limitations:** {info.limitations}",
                    f"- **Best use cases:** {info.best_use_cases}",
                    f"- **Recommended for:** {', '.join(info.recommended_for) or 'N/A'}",
                    "",
                ]
        return "\n".join(lines)