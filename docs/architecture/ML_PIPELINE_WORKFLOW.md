# Sprint 3 — Machine Learning Pipeline — Workflow & Class Diagram

## Sprint-Level Workflow

```
Sprint 3
  |
  v
Data Loading (build_feature_matrix)
  |
  v
Time-Based Split + Leakage-Free Preprocessing
  |
  v
[Optional] Feature Selection
  |
  v
Per-Model: [Optional Tuning] -> Fit -> Evaluate -> Persist   (x 15 models)
  |
  v
Comparison + Reporting + Visualization
```

## Run-Level Workflow (what happens inside `MLTrainingPipeline.run()`)

```mermaid
flowchart TD
    A["Load processed dataset<br/>data_loader.load_processed_dataset"] --> B["Build raw feature matrix<br/>build_feature_matrix"]
    B --> C["Time-based split<br/>TimeSeriesSplitter.split"]
    C --> D["Fit preprocessor on TRAIN only<br/>FeaturePreprocessor.fit"]
    D --> E["Transform train/val/test<br/>FeaturePreprocessor.transform"]
    E --> F{"use_feature_selection?"}
    F -- yes --> G["FeatureSelector.recommend_feature_subset<br/>correlation + variance + RFE + permutation + tree importance"]
    G --> H["Restrict X_train/X_val/X_test to recommended columns"]
    F -- no --> H
    H --> I["For each registered model (ModelFactory)"]
    I --> J{"tune_hyperparameters?"}
    J -- yes --> K["HyperparameterTuner.tune<br/>GridSearchCV/RandomizedSearchCV + TimeSeriesSplit CV"]
    K --> L["model.fit(X_train, y_train)"]
    J -- no --> L
    L --> M["compute_regression_metrics /<br/>compute_classification_metrics<br/>(train, validation, test)"]
    M --> N["ModelStorage.save<br/>storage/models/ml/*.joblib"]
    N --> O{"more models?"}
    O -- yes --> I
    O -- no --> P["ModelStorage.save_preprocessor +<br/>save_metadata_json"]
    P --> Q["ModelComparator.build_comparison_table<br/>performance_rank + efficiency_rank + narrative"]
    Q --> R["Write Model_Comparison.csv,<br/>Evaluation_Report.md, Model_Documentation.md"]
    R --> S["PlotGenerator: comparison chart +<br/>top-N prediction/residual/importance +<br/>best-model learning/validation curve"]
    S --> T["Return MLTrainingResult"]
```

## Class Diagram

```mermaid
classDiagram
    class BaseMLModel {
        <<abstract>>
        +ModelInfo info
        +fit(X, y) BaseMLModel
        +predict(X) ndarray
        +predict_proba(X) ndarray
        +get_feature_importance() Series
        +estimate_complexity() int
        #_build_estimator(params)
        #default_hyperparameters()$
    }
    class ModelFactory {
        +available_models(task_type) list
        +create(name, **overrides) BaseMLModel
        +create_all(task_type, names) dict
        +get_model_info(name) ModelInfo
    }
    class FeaturePreprocessor {
        +fit(X_train) FeaturePreprocessor
        +transform(X) DataFrame
        +transform_and_align(X, y, dates) tuple
    }
    class TimeSeriesSplitter {
        +split(X, y, dates) DataSplit
        +split_and_preprocess(X, y, dates) tuple
        +time_series_cv(n_splits) TimeSeriesSplit
    }
    class HyperparameterTuner {
        +tune(model, X, y, method) tuple
    }
    class ModelComparator {
        +build_comparison_table(results, metric, direction) ComparisonTable
        +render_evaluation_report_markdown(...) str
        +render_model_documentation_markdown(...) str
    }
    class ModelStorage {
        +save(model, ...) SavedModelInfo
        +load(path) BaseMLModel
        +save_preprocessor(...) Path
        +save_metadata_json(...) Path
    }
    class InferenceService {
        +predict_latest(ticker, target_column, model_name) InferenceResult
    }
    class MLTrainingPipeline {
        +run(...) MLTrainingResult
        +run_all_targets(...) dict
    }

    ModelFactory --> BaseMLModel : creates
    MLTrainingPipeline --> TimeSeriesSplitter
    MLTrainingPipeline --> HyperparameterTuner
    MLTrainingPipeline --> ModelComparator
    MLTrainingPipeline --> ModelStorage
    MLTrainingPipeline --> ModelFactory
    TimeSeriesSplitter --> FeaturePreprocessor : fits + applies
    InferenceService --> ModelStorage : loads
    InferenceService --> FeaturePreprocessor : loads + applies
```