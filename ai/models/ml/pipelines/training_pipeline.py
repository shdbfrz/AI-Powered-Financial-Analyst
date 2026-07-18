"""
Training pipeline orchestration for `ai/models/ml/`.

Mirrors the Pipeline pattern established by
`ai.feature_engineering.pipeline.FeatureEngineeringPipeline`: one class,
one `run()` method, a dataclass result carrying every artifact path so
callers (a demo script, a test, a future Sprint 8 backend job) never have
to know the internal step order — they just call `.run()` and get back
everything that was produced.
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ai.models.ml.config import DEFAULT_ML_CONFIG, MLConfig
from ai.models.ml.data_loader import build_feature_matrix, load_processed_dataset
from ai.models.ml.evaluation.comparator import ComparisonTable, ModelComparator, ModelResult
from ai.models.ml.evaluation.metrics import compute_classification_metrics, compute_regression_metrics
from ai.models.ml.exceptions import MLPipelineError
from ai.models.ml.models.base import BaseMLModel
from ai.models.ml.models.registry import ModelFactory
from ai.models.ml.persistence.model_storage import ModelStorage, SavedModelInfo, estimate_memory_bytes
from ai.models.ml.selection.feature_selector import FeatureSelectionResult, FeatureSelector
from ai.models.ml.splitting import DataSplit, TimeSeriesSplitter
from ai.models.ml.tuning.param_grids import get_param_grid
from ai.models.ml.tuning.tuner import HyperparameterTuner
from ai.models.ml.visualization.plots import PlotGenerator
from ai.utils.config import settings
from ai.utils.logger import get_logger

logger = get_logger(__name__)

TaskType = Literal["regression", "classification"]
TuningMethod = Literal["grid", "random"]

_SCORING_FOR_TASK = {"regression": "neg_root_mean_squared_error", "classification": "f1"}


@dataclass
class MLTrainingResult:
    ticker: str
    version: str
    target_column: str
    task_type: TaskType
    dataset_info: dict
    model_results: list[ModelResult]
    comparison_table: ComparisonTable
    saved_models: list[SavedModelInfo]
    preprocessor_path: Path
    best_model_name: Optional[str]
    evaluation_report_path: Path
    comparison_csv_path: Path
    model_documentation_path: Path
    metadata_json_path: Path
    plot_paths: list = field(default_factory=list)
    feature_selection_result: Optional[FeatureSelectionResult] = None


class MLTrainingPipeline:
    """Trains, evaluates, compares, persists, and visualizes every
    registered model for one target column in one call.
    """

    def __init__(self, config: MLConfig = DEFAULT_ML_CONFIG):
        self.config = config
        self.splitter = TimeSeriesSplitter(config)
        self.selector = FeatureSelector(config)
        self.tuner = HyperparameterTuner(config)
        self.comparator = ModelComparator()
        self.storage = ModelStorage()
        self.plots = PlotGenerator()
        self.logger = logger

    def run(
        self,
        ticker: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        task: TaskType = "regression",
        target_column: Optional[str] = None,
        model_names: Optional[list[str]] = None,
        use_feature_selection: bool = False,
        tune_hyperparameters: bool = False,
        tuning_method: TuningMethod = "random",
        generate_plots: bool = True,
        version: Optional[str] = None,
    ) -> MLTrainingResult:
        """Train and compare every model in `model_names` (default: every
        registered model for `task`) on `target_column` (default: the
        task's configured default target).

        Args:
            ticker: looked up in `datasets/processed/` via `data_loader` if
                `df` isn't given directly.
            df: an already-loaded processed DataFrame (bypasses disk —
                used by tests and when Sprint 2 -> Sprint 3 is chained
                in-memory).
            use_feature_selection: run `FeatureSelector.recommend_feature_subset`
                on the training split and train only on the recommended
                columns, instead of all ~200+ engineered features.
            tune_hyperparameters: run `HyperparameterTuner` for every model
                that has a registered search space before its final fit.
            version: run identifier embedded in every output filename;
                defaults to a UTC timestamp (`vYYYYMMDDTHHMMSSZ`, matching
                Sprint 2's `storage.py` versioning convention).

        Raises:
            Any `MLPipelineError` subclass raised while loading/splitting
            the data (a single model failing to train does *not* abort the
            run — see `ModelResult.error`).
        """
        version = version or datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
        target_column = target_column or (
            self.config.default_regression_target if task == "regression" else self.config.default_classification_target
        )

        loaded_df = load_processed_dataset(ticker=ticker, df=df, config=self.config)
        resolved_ticker = (ticker or (loaded_df["ticker"].iloc[0] if "ticker" in loaded_df.columns else "UNKNOWN")).upper()

        self.logger.info("=== MLTrainingPipeline.run: ticker=%s target=%s task=%s version=%s ===",
                          resolved_ticker, target_column, task, version)

        X_raw, y, dates = build_feature_matrix(loaded_df, target_column, self.config)
        split, preprocessor = self.splitter.split_and_preprocess(X_raw, y, dates)

        selection_result = None
        if use_feature_selection:
            selection_result = self.selector.recommend_feature_subset(split.X_train, split.y_train, task)
            split = self._restrict_to_features(split, selection_result.recommended_features)

        model_names = model_names or ModelFactory.available_models(task)
        model_results: list[ModelResult] = []
        fitted_models: dict[str, BaseMLModel] = {}
        saved_models: list[SavedModelInfo] = []

        for name in model_names:
            result, fitted_model, saved_info = self._train_one_model(
                name=name, task=task, target_column=target_column, resolved_ticker=resolved_ticker,
                version=version, split=split, tune_hyperparameters=tune_hyperparameters, tuning_method=tuning_method,
            )
            model_results.append(result)
            if fitted_model is not None:
                fitted_models[name] = fitted_model
            if saved_info is not None:
                saved_models.append(saved_info)

        preprocessor_path = self.storage.save_preprocessor(
            preprocessor, ticker=resolved_ticker, target_column=target_column, version=version
        )
        metadata_json_path = self.storage.save_metadata_json(saved_models, resolved_ticker, version, target_column)

        primary_metric = self.config.regression_primary_metric if task == "regression" else self.config.classification_primary_metric
        direction = self.config.regression_primary_metric_direction if task == "regression" else self.config.classification_primary_metric_direction
        comparison_table = self.comparator.build_comparison_table(model_results, primary_metric, direction)
        best_model_name = (
            comparison_table.dataframe.iloc[0]["model_name"] if not comparison_table.dataframe.empty else None
        )

        dataset_info = {
            "ticker": resolved_ticker, "version": version, "target_column": target_column, "task": task,
            "rows_loaded": len(loaded_df), "rows_labeled": len(X_raw),
            "train_rows": len(split.X_train), "validation_rows": len(split.X_val), "test_rows": len(split.X_test),
            "n_features": split.X_train.shape[1], "feature_selection_used": use_feature_selection,
            "hyperparameter_tuning_used": tune_hyperparameters, "models_attempted": len(model_names),
            "models_succeeded": sum(1 for r in model_results if r.error is None),
        }

        reports_dir = settings.resolve(settings.ml_reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        comparison_csv_path = reports_dir / f"{resolved_ticker}_{target_column}_{version}_Model_Comparison.csv"
        comparison_table.dataframe.to_csv(comparison_csv_path, index=False)

        evaluation_report_path = reports_dir / f"{resolved_ticker}_{target_column}_{version}_Evaluation_Report.md"
        evaluation_report_path.write_text(
            self.comparator.render_evaluation_report_markdown([comparison_table], resolved_ticker, version, dataset_info),
            encoding="utf-8",
        )

        model_documentation_path = reports_dir / "Model_Documentation.md"
        model_documentation_path.write_text(
            self.comparator.render_model_documentation_markdown(ModelFactory.all_model_info()), encoding="utf-8",
        )

        plot_paths: list[Path] = []
        if generate_plots and not comparison_table.dataframe.empty:
            plot_paths = self._generate_plots(
                comparison_table=comparison_table, model_results=model_results, fitted_models=fitted_models,
                split=split, task=task, primary_metric=primary_metric, resolved_ticker=resolved_ticker,
                target_column=target_column, version=version, reports_dir=reports_dir,
            )

        self.logger.info(
            "=== MLTrainingPipeline.run complete: %d/%d model(s) succeeded, best=%s (%s=%s) ===",
            dataset_info["models_succeeded"], dataset_info["models_attempted"], best_model_name,
            primary_metric, comparison_table.dataframe.iloc[0]["primary_metric_value"] if not comparison_table.dataframe.empty else "n/a",
        )

        return MLTrainingResult(
            ticker=resolved_ticker, version=version, target_column=target_column, task_type=task,
            dataset_info=dataset_info, model_results=model_results, comparison_table=comparison_table,
            saved_models=saved_models, preprocessor_path=preprocessor_path, best_model_name=best_model_name,
            evaluation_report_path=evaluation_report_path, comparison_csv_path=comparison_csv_path,
            model_documentation_path=model_documentation_path, metadata_json_path=metadata_json_path,
            plot_paths=plot_paths, feature_selection_result=selection_result,
        )

    def run_all_targets(
        self, ticker: Optional[str] = None, df: Optional[pd.DataFrame] = None,
        include_regression: bool = True, include_classification: bool = True, **run_kwargs,
    ) -> dict[str, MLTrainingResult]:
        """Convenience wrapper: run every regression target
        (`target_1_day`/`target_3_day`/`target_5_day`/`future_return_*_day`)
        and/or every classification target
        (`target_direction_*_day`) in one call — satisfies the Sprint 3
        spec's "predict Target_1_Day / Target_3_Day / Target_5_Day /
        Future_Return" / "predict Target_Direction" requirement in full,
        one `MLTrainingPipeline.run()` per target.

        Runs `len(regression_targets) + len(classification_targets)` full
        training passes — expect this to take proportionally longer than a
        single `run()` call; prefer calling `run()` directly for one target
        during iteration/development.
        """
        results: dict[str, MLTrainingResult] = {}
        loaded_df = df if df is not None else load_processed_dataset(ticker=ticker, config=self.config)
        if include_regression:
            for target in self.config.regression_targets:
                results[target] = self.run(ticker=ticker, df=loaded_df, task="regression", target_column=target, **run_kwargs)
        if include_classification:
            for target in self.config.classification_targets:
                results[target] = self.run(ticker=ticker, df=loaded_df, task="classification", target_column=target, **run_kwargs)
        return results

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    @staticmethod
    def _restrict_to_features(split: DataSplit, feature_columns: list[str]) -> DataSplit:
        return DataSplit(
            X_train=split.X_train[feature_columns], y_train=split.y_train, dates_train=split.dates_train,
            X_val=split.X_val[feature_columns], y_val=split.y_val, dates_val=split.dates_val,
            X_test=split.X_test[feature_columns], y_test=split.y_test, dates_test=split.dates_test,
        )

    def _train_one_model(
        self, *, name: str, task: TaskType, target_column: str, resolved_ticker: str, version: str,
        split: DataSplit, tune_hyperparameters: bool, tuning_method: TuningMethod,
    ) -> tuple[ModelResult, Optional[BaseMLModel], Optional[SavedModelInfo]]:
        """One model's full lifecycle: [tune] -> fit -> evaluate -> persist.
        Never raises — a failure becomes a `ModelResult.error`, so one bad
        model (e.g. an estimator that fails to converge) never aborts the
        whole comparison run.
        """
        try:
            model = ModelFactory.create(name)
            was_tuned = False
            if tune_hyperparameters:
                model, tuning_result = self.tuner.tune(model, split.X_train, split.y_train, method=tuning_method)
                was_tuned = tuning_result.method != "none"

            model.fit(split.X_train, split.y_train)

            train_metrics, val_metrics, test_metrics = self._evaluate(model, task, split)

            complexity = model.estimate_complexity()
            memory_bytes = estimate_memory_bytes(model.estimator)

            result = ModelResult(
                model_name=name, display_name=model.info.display_name, task_type=task, target_column=target_column,
                hyperparameters=model.hyperparameters, train_metrics=train_metrics, validation_metrics=val_metrics,
                test_metrics=test_metrics, training_time_seconds=model.last_fit_seconds,
                prediction_time_seconds=model.last_predict_seconds, complexity_score=complexity,
                memory_bytes=memory_bytes, was_tuned=was_tuned, feature_importance=model.get_feature_importance(),
            )

            saved_info = self.storage.save(
                model, ticker=resolved_ticker, target_column=target_column, version=version, test_metrics=test_metrics,
            )
            result.model_artifact_path = str(saved_info.path)
            return result, model, saved_info

        except MLPipelineError as e:
            self.logger.error("Model '%s' failed and is excluded from the comparison: %s", name, e)
            failed = ModelResult(
                model_name=name, display_name=getattr(ModelFactory.get_model_info(name), "display_name", name),
                task_type=task, target_column=target_column, hyperparameters={}, train_metrics={},
                validation_metrics={}, test_metrics={}, training_time_seconds=0.0, prediction_time_seconds=0.0,
                complexity_score=0, memory_bytes=0, error=str(e),
            )
            return failed, None, None

    def _evaluate(self, model: BaseMLModel, task: TaskType, split: DataSplit) -> tuple[dict, dict, dict]:
        if task == "regression":
            n_features = split.X_train.shape[1]
            return (
                compute_regression_metrics(split.y_train, model.predict(split.X_train), n_features).as_dict(),
                compute_regression_metrics(split.y_val, model.predict(split.X_val), n_features).as_dict(),
                compute_regression_metrics(split.y_test, model.predict(split.X_test), n_features).as_dict(),
            )

        def _clf_metrics(X, y):
            preds = model.predict(X)
            proba = model.predict_proba(X)[:, 1] if hasattr(model.estimator, "predict_proba") or \
                (hasattr(model.estimator, "steps") and hasattr(model.estimator[-1], "predict_proba")) else None
            return compute_classification_metrics(y, preds, proba).as_dict()

        return _clf_metrics(split.X_train, split.y_train), _clf_metrics(split.X_val, split.y_val), _clf_metrics(split.X_test, split.y_test)

    def _generate_plots(
        self, *, comparison_table: ComparisonTable, model_results: list[ModelResult],
        fitted_models: dict[str, BaseMLModel], split: DataSplit, task: TaskType, primary_metric: str,
        resolved_ticker: str, target_column: str, version: str, reports_dir: Path,
    ) -> list[Path]:
        prefix = f"{resolved_ticker}_{target_column}_{version}"
        paths: list[Path] = []

        paths.append(self.plots.model_comparison_chart(
            comparison_table.dataframe, primary_metric,
            f"{task.title()} Model Comparison — {resolved_ticker} ({target_column})",
            reports_dir / f"{prefix}_model_comparison_chart.png",
        ))

        results_by_name = {r.model_name: r for r in model_results}
        top_names = comparison_table.dataframe.head(self.config.top_n_models_to_plot)["model_name"].tolist()

        for rank, name in enumerate(top_names, start=1):
            model = fitted_models.get(name)
            result = results_by_name.get(name)
            if model is None or result is None:
                continue
            label = f"{result.display_name} (rank #{rank})"

            if task == "regression":
                test_pred = model.predict(split.X_test)
                paths.append(self.plots.prediction_vs_actual(
                    split.y_test, test_pred, label, reports_dir / f"{prefix}_{name}_prediction_vs_actual.png",
                ))
                paths.append(self.plots.residual_plot(
                    split.y_test, test_pred, label, reports_dir / f"{prefix}_{name}_residuals.png",
                ))

            if result.feature_importance is not None and not result.feature_importance.empty:
                paths.append(self.plots.feature_importance_plot(
                    result.feature_importance, label, reports_dir / f"{prefix}_{name}_feature_importance.png",
                ))

        # Learning curve + validation curve: expensive (each refits the
        # model several times), so generated only for the single best model.
        if top_names:
            best_name = top_names[0]
            best_model = fitted_models.get(best_name)
            if best_model is not None:
                paths.extend(self._generate_diagnostic_curves(
                    best_model, split, task, prefix, reports_dir, results_by_name[best_name].display_name,
                ))

        return paths

    def _generate_diagnostic_curves(
        self, model: BaseMLModel, split: DataSplit, task: TaskType, prefix: str, reports_dir: Path, display_name: str,
    ) -> list[Path]:
        paths: list[Path] = []
        scoring = _SCORING_FOR_TASK[task]
        light_cv = TimeSeriesSplit(n_splits=3)  # lighter than the tuning default — this step alone refits several times

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                paths.append(self.plots.learning_curve_plot(
                    model.estimator, split.X_train, split.y_train, display_name,
                    reports_dir / f"{prefix}_{model.info.name}_learning_curve.png",
                    cv=light_cv, scoring=scoring, train_sizes=self.config.learning_curve_train_sizes,
                ))
        except Exception as e:  # noqa: BLE001 - a diagnostic plot failing must never abort the run
            self.logger.warning("Learning curve generation failed for '%s': %s", model.info.name, e)

        param_grid = get_param_grid(model.info.name)
        if param_grid:
            param_name, param_range = next(iter(param_grid.items()))
            is_pipeline = hasattr(model.estimator, "steps")
            prefixed_param_name = f"model__{param_name}" if is_pipeline else param_name
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    paths.append(self.plots.validation_curve_plot(
                        model.estimator, split.X_train, split.y_train, prefixed_param_name, param_range, display_name,
                        reports_dir / f"{prefix}_{model.info.name}_validation_curve.png",
                        cv=light_cv, scoring=scoring,
                    ))
            except Exception as e:  # noqa: BLE001
                self.logger.warning("Validation curve generation failed for '%s': %s", model.info.name, e)

        return paths