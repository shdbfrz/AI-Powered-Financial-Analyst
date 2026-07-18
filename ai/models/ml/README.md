# `ai/models/ml/` — Sprint 3: Baseline Machine Learning Pipeline

Status: **Complete**, on `feature/ai`
Depends on: Sprint 1 (`ai/data_collection/`), Sprint 2 (`ai/feature_engineering/`)
Feeds into: Sprint 4 (Time Series), Sprint 5 (Deep Learning), Sprint 7 (Decision Support Engine)

## 1. Sprint Goal

Train, tune, evaluate, compare, persist, and visualize baseline ML models on
Sprint 2's engineered features — a reusable, config-driven pipeline, not a
one-off notebook — so Sprints 4/5 can plug in Time Series/Deep Learning
models against the exact same contract, and Sprint 7 can consume whichever
model wins the comparison without knowing anything about how it was trained.

## 2. Business Problem

A trained model is only useful if it's reproducible, comparable against
alternatives, and safe to hand to a downstream system. This sprint answers
four concrete questions for every ticker: which of 15 candidate models
predicts price movement best, how much of that is worth the extra training
cost, which features actually drive the prediction, and can the winning
model be reloaded and used for inference without re-running the whole
pipeline.

## 3. Why This Sprint Comes Before The Next

Sprint 4 (Time Series) and Sprint 5 (Deep Learning) need a working
comparison harness to be judged against — "is an LSTM worth its extra
complexity?" is only answerable if there's already a baseline RMSE from
Sprint 3 to beat. Sprint 7 (Decision Support Engine) needs a stable
`InferenceService` contract to call; building that contract now, against
simple models, is far cheaper than discovering its gaps once three more
model families depend on it.

## 4. Architecture

See `docs/architecture/ML_PIPELINE_ARCHITECTURE.md` for the full diagram.
In short:

```
datasets/processed/{ticker}_{version}_features.csv   (Sprint 2 output)
        │
        ▼
data_loader.build_feature_matrix()   — exclude non-feature/target-family columns, drop unlabeled rows
        │
        ▼
splitting.TimeSeriesSplitter.split_and_preprocess()
        │  ├─ split()        — contiguous, chronological train/val/test (no shuffling)
        │  └─ FeaturePreprocessor.fit(train) → transform(train/val/test)   — one-hot encode + impute, fit on TRAIN ONLY
        ▼
[optional] selection.FeatureSelector.recommend_feature_subset()   — correlation / variance / RFE / permutation / tree importance
        ▼
for each model in ModelFactory.available_models(task):
        │  ├─ [optional] tuning.HyperparameterTuner.tune()   — GridSearchCV/RandomizedSearchCV + TimeSeriesSplit CV
        │  ├─ model.fit(X_train, y_train)
        │  ├─ evaluation.metrics.compute_*_metrics()   — train/validation/test
        │  └─ persistence.ModelStorage.save()   — storage/models/ml/*.joblib
        ▼
evaluation.comparator.ModelComparator   — Model_Comparison.csv, Evaluation_Report.md, Model_Documentation.md
        ▼
visualization.plots.PlotGenerator   — storage/reports/ml/*.png
```

## 5. Workflow Diagram

See `docs/architecture/ML_PIPELINE_WORKFLOW.md` for the Mermaid sequence
diagram covering a full `MLTrainingPipeline.run()` call.

## 6. Folder Structure

```
ai/models/ml/
├── __init__.py                  # public API: MLTrainingPipeline, ModelFactory, InferenceService, ...
├── config.py                    # MLConfig — split ratios, target columns, thresholds, tuning defaults
├── exceptions.py                # typed exception hierarchy
├── data_loader.py                # locate/load Sprint 2 output; build raw (unencoded) feature matrices
├── preprocessing.py              # FeaturePreprocessor — encoding + imputation, fit on train only
├── splitting.py                  # TimeSeriesSplitter — contiguous time-based split + CV helper
├── models/
│   ├── base.py                   # BaseMLModel (Template Method) + ModelInfo
│   ├── linear_models.py          # Linear/Ridge/Lasso/ElasticNet/Logistic (StandardScaler-wrapped)
│   ├── tree_ensemble_models.py   # DecisionTree/RandomForest/ExtraTrees/GradientBoosting/AdaBoost/RFClassifier
│   ├── boosting_models.py        # XGBoost (required) + LightGBM/CatBoost (optional)
│   └── registry.py               # ModelFactory — name -> class registry
├── selection/feature_selector.py # correlation / variance / RFE / permutation / tree importance
├── tuning/
│   ├── param_grids.py            # per-model GridSearchCV/RandomizedSearchCV search spaces
│   └── tuner.py                  # HyperparameterTuner (TimeSeriesSplit CV)
├── evaluation/
│   ├── metrics.py                # MAE/MSE/RMSE/MAPE/R2/Adj-R2, Accuracy/Precision/Recall/F1/ROC-AUC/ConfusionMatrix
│   └── comparator.py             # ModelResult, ModelComparator — ranking + Markdown/CSV report rendering
├── persistence/model_storage.py  # joblib save/load (atomic writes) + Model_Metadata.json
├── prediction/inference.py       # Predictor (in-memory) + InferenceService (loads from disk, facade)
├── pipelines/training_pipeline.py# MLTrainingPipeline — the orchestrator described above
└── visualization/plots.py        # prediction-vs-actual, residuals, feature importance, learning/validation curves, comparison chart

storage/models/ml/          # trained model + preprocessor artifacts (gitignored)
storage/reports/ml/         # Model_Comparison.csv, Evaluation_Report.md, Model_Documentation.md, *.png (gitignored)
docs/sprint3_sample_output/ # committed snapshot of the above, since the real outputs are gitignored
ai/tests/test_ml_models.py, test_ml_data_and_selection.py, test_ml_pipeline.py
scripts/run_ml_pipeline_demo.py
```

## 7. File Structure

See the folder tree above — every file has a single, named responsibility
(Single Responsibility Principle); nothing in `models/` knows about
persistence, nothing in `evaluation/` knows how a model was trained.

## 8. Class Design

- **`BaseMLModel`** (Template Method, mirrors `BaseDataProvider` from Sprint 1)
  — `fit`/`predict`/`predict_proba`/`get_feature_importance`/`estimate_complexity`
  are shared; each concrete model only implements `_build_estimator()` and
  `default_hyperparameters()`.
- **`ModelFactory`** (Factory/Registry, mirrors `DataCollectionManager`'s
  `_PROVIDER_REGISTRY`) — every model is instantiated by name; the pipeline
  never imports a concrete model class directly.
- **`FeaturePreprocessor`** — `fit()`/`transform()` split, identical in
  spirit to `sklearn.preprocessing.StandardScaler`: statistics come only
  from what `fit()` was called on.
- **`MLTrainingPipeline`** (Pipeline pattern, mirrors
  `FeatureEngineeringPipeline`) — one `run()` method, one dataclass result
  (`MLTrainingResult`) carrying every artifact path.
- **`ModelComparator`** — turns a list of `ModelResult` into a ranked table
  + rendered reports; has no idea how any model was trained.
- **`InferenceService`** (Facade, mirrors `DataCollectionManager`) — the
  single entry point Sprint 7/8 will call; hides model loading, preprocessor
  loading, and column alignment behind one `predict_latest()` call.

## 9. Design Patterns Used

Template Method (`BaseMLModel`), Factory/Registry (`ModelFactory`),
Pipeline (`MLTrainingPipeline`, mirrors Sprint 2), Facade
(`InferenceService`), Strategy (interchangeable models behind one
interface), fit/transform (`FeaturePreprocessor`, mirrors scikit-learn's
own convention), Configuration-Driven Design (`MLConfig` — nothing about
which targets/thresholds/ratios to use is hardcoded in pipeline logic).

## 10. External Dependencies

`scikit-learn==1.5.2`, `xgboost==2.1.1`, `lightgbm==4.5.0` (optional),
`catboost==1.2.7` (optional), `joblib==1.4.2`, `matplotlib==3.9.2`,
`seaborn==0.13.2`, `tabulate==0.9.0` (for `DataFrame.to_markdown()`) — see
`ai/requirements-core.txt`.

## 11. Configuration Files

`ai/models/ml/config.py::MLConfig` — a frozen dataclass (same two-tier split
as Sprint 2: `ai/utils/config.py::settings` for storage paths,
`MLConfig` for computational tunables). Two new `settings` fields were
added: `models_dir` (`storage/models`) and `ml_reports_dir`
(`storage/reports/ml`) — the canonical locations `ARCHITECTURE.md` §4
already specifies, not new directories invented for this sprint.

## 12. Complete Production Code

See the folder tree above; every file ships with full docstrings, type
hints, logging, and typed exception handling per the project's coding
standards.

## 13. Unit Tests

`ai/tests/test_ml_models.py` (registry + all 15 models' fit/predict/
importance/complexity contract), `test_ml_data_and_selection.py` (data
loading, time-based splitting, leakage-free preprocessing — including a
dedicated regression test that a column's train-only statistic is never
influenced by validation/test-period values — and feature selection),
`test_ml_pipeline.py` (metrics correctness, comparator ranking, persistence
round-trips, tuning, full pipeline integration including a simulated
mid-run model failure, and inference). **142/142 tests pass** (61 from
Sprints 1-2 + 81 new), verified against pandas 2.2.3.

## 14. Logging

Every module logs through `ai.utils.logger.get_logger(__name__)`: dataset
loaded, split sizes, preprocessing statistics, training start/complete per
model, tuning results, models saved, reports written, and every caught
exception — nothing fails silently (SRS NFR-7).

## 15. Exception Handling

`ai/models/ml/exceptions.py` — a typed hierarchy under `MLPipelineError`
(`ProcessedDatasetNotFoundError`, `InvalidTargetColumnError`,
`InsufficientTrainingDataError`, `UnknownModelError`, `ModelNotFittedError`,
`ModelTrainingError`, `PredictionError`, `ModelPersistenceError`,
`FeatureSelectionError`, `HyperparameterTuningError`). One model failing to
train raises `ModelTrainingError` internally, but
`MLTrainingPipeline._train_one_model` catches it and records
`ModelResult.error` instead of aborting the run — the other 14 models still
get compared.

## 16. README

This file.

## 17. Common Mistakes (and how this sprint avoids them)

- **Requiring every feature to be non-NaN before training.** Several Sprint
  2 indicators (`demand_zone_strength`, `supply_zone_strength`,
  `zone_width`) are legitimately NaN on most rows — a "drop any row with any
  NaN feature" strategy empirically drops **all** rows on the real 205-
  feature schema. `MLConfig.nan_strategy` defaults to per-column median
  imputation instead.
- **Imputing/encoding before splitting.** Computing a fill value (or a
  one-hot vocabulary) from the whole dataset and applying it everywhere —
  including back onto training rows — leaks validation/test-period
  information backward. `FeaturePreprocessor` is fit on the training
  partition only, exactly like fitting a `StandardScaler`.
- **Shuffling financial time series before splitting.** `TimeSeriesSplitter`
  never shuffles; train is strictly earlier than validation, which is
  strictly earlier than test.
- **Trusting MAPE on return-scale targets.** `future_return_5_day` values
  near zero make raw percentage error diverge; `_safe_mape` excludes
  near-zero true values and the evaluation report notes RMSE/R² are more
  reliable for these targets.
- **Running RFE with a tree-ensemble estimator.** RFE refits its estimator
  `n_features / step` times — using a 300-tree forest there (rather than a
  fast linear reference model) would make feature selection the pipeline's
  bottleneck by orders of magnitude.
- **Regularized linear models on unscaled features.** `volume` (millions)
  vs. `rsi` (0-100) vs. `future_return_*` (~0.01) on the same feature
  matrix would let Ridge/Lasso/ElasticNet/Logistic's penalty term be
  dominated by units, not predictive value — every linear model here is
  wrapped in a `StandardScaler` pipeline.

## 18. Improvements (tracked, not implemented this sprint — out of scope per `project_guide`)

- Native categorical handling for LightGBM/CatBoost (currently every model
  receives the same one-hot-encoded matrix for a fair, apples-to-apples
  comparison; both libraries can accept raw categoricals directly).
- A composite, weighted single ranking score across all six comparison
  criteria, if a future sprint's stakeholders want one number instead of
  the `performance_rank` / `efficiency_rank` pair this sprint reports.
- Verifying pandas 3.0 compatibility (checked for Sprints 1-2 per the
  project's established convention; not yet re-verified against Sprint 3's
  new dependency set).
- Multi-ticker training in one pipeline call (`run()` is single-ticker by
  design this sprint, matching Sprint 2's scope).

## 19. How This Sprint Connects To The Next

Sprint 4 (Time Series) and Sprint 5 (Deep Learning) will each add a sibling
package (`ai/models/time_series/`, `ai/models/deep_learning/`) implementing
the same evaluation contract (`compute_regression_metrics`,
`ModelComparator`) so all three model families can be ranked in one
combined comparison table before Sprint 7's Prediction Engine ensembles
them. `InferenceService` is the stable contract Sprint 7 (Decision Support
Engine) and Sprint 8 (Backend) will call — its interface should not need to
change as new model families are added, only the registry each one draws
from.

## 20-21. Git Commit Message / Pull Request Description

See the sprint delivery notes in the PR this README ships with.