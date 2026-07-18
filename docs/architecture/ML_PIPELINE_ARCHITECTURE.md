# ML Pipeline Architecture — Sprint 3

Status: **Complete**
Owner: shdbfrz
Module: `ai/models/ml/`

---

## 1. Purpose

Defines the internal architecture of the baseline Machine Learning pipeline:
how a Sprint 2 processed dataset becomes a ranked, persisted, visualized set
of trained models. Supersedes nothing in `ARCHITECTURE.md` — this document
zooms into "Module 3 (Machine Learning)" from that file's component table.

## 2. Position In The System

```
ai/feature_engineering/  (Sprint 2)
        │  datasets/processed/{ticker}_{version}_features.csv
        ▼
ai/models/ml/             (Sprint 3 — this document)
        │  storage/models/ml/*.joblib  +  storage/reports/ml/*
        ▼
ai/models/time_series/, ai/models/deep_learning/   (Sprint 4, 5 — not yet built)
        │
        ▼
ai/decision_engine/       (Sprint 7 — ensembles all three model families)
```

## 3. Data Flow Through The Pipeline

```
                    ┌───────────────────────────────┐
                    │  datasets/processed/*.csv       │  Sprint 2 output
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  data_loader.build_feature_matrix│  exclude non-feature/target columns
                    └───────────────┬─────────────────┘   drop rows with no known target
                                    ▼
                    ┌───────────────────────────────┐
                    │  TimeSeriesSplitter.split()      │  contiguous, chronological, no shuffling
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ FeaturePreprocessor.fit(train)   │  categorical encoding + NaN imputation,
                    │  .transform(train/val/test)      │  statistics from TRAIN ONLY
                    └───────────────┬─────────────────┘
                        ┌───────────┼────────────┐
                        ▼ (optional)              ▼
        ┌───────────────────────┐   ┌─────────────────────────────┐
        │ FeatureSelector          │   │ for each of 15 models:         │
        │ (correlation/variance/   │   │  [HyperparameterTuner] -> fit  │
        │  RFE/permutation/tree)   │──▶│  -> evaluate -> persist         │
        └───────────────────────┘   └───────────────┬─────────────────┘
                                                     ▼
                                    ┌───────────────────────────────┐
                                    │  ModelComparator                 │  rank, narrative,
                                    │  (Model_Comparison.csv,          │  Markdown/CSV reports
                                    │   Evaluation_Report.md)          │
                                    └───────────────┬─────────────────┘
                                                     ▼
                                    ┌───────────────────────────────┐
                                    │  PlotGenerator                   │  storage/reports/ml/*.png
                                    └───────────────────────────────┘
```

## 4. Component Responsibilities

| Component | Responsibility | Talks to |
|---|---|---|
| `data_loader` | Locate + load Sprint 2 output; build raw (unencoded) `(X, y, dates)` | `datasets/processed/` |
| `splitting` | Contiguous chronological train/val/test partitioning | `data_loader` output |
| `preprocessing` | One-hot encode + impute, fit on train only | `splitting` output |
| `selection` | Target-aware feature ranking (RFE/permutation/tree importance) | preprocessed train split |
| `models` | 15 model implementations behind one `BaseMLModel` contract | preprocessed splits |
| `tuning` | GridSearchCV/RandomizedSearchCV + TimeSeriesSplit CV | one model + train split |
| `evaluation` | Metrics + ranking + Markdown/CSV report rendering | fitted models' predictions |
| `persistence` | Atomic joblib save/load + `Model_Metadata.json` | fitted models, `storage/models/ml/` |
| `prediction` | In-memory prediction wrapper + load-from-disk facade | `persistence`, new feature rows |
| `pipelines` | Orchestrates every component above via one `run()` call | all of the above |
| `visualization` | Matplotlib plot generation | `evaluation` results, fitted models |

## 5. Key Architectural Decisions

1. **Split-then-preprocess, never preprocess-then-split.** Encoding/imputation
   statistics are data-derived; computing them from the full dataset and
   applying them everywhere (including training rows) leaks
   validation/test-period information backward. `FeaturePreprocessor.fit()`
   only ever sees the training partition — see `ai/models/ml/preprocessing.py`'s
   module docstring and `test_ml_data_and_selection.py::test_imputation_statistics_come_from_train_only`.
2. **Per-column median imputation, not row-dropping, as the default NaN
   strategy.** Several Sprint 2 indicators are legitimately NaN on most
   rows (no zone/pattern is active that day); requiring every feature
   simultaneously non-NaN empirically drops 100% of rows on the real
   205-feature schema.
3. **Storage locations are the ones `ARCHITECTURE.md`/`storage/README.md`
   already define** (`storage/models/ml/`, `storage/reports/ml/`) — not new
   directories invented per-sprint, so Sprint 4/5's model artifacts land
   next to Sprint 3's under one predictable tree.
4. **A model that fails to train doesn't abort the run.** `ModelResult.error`
   captures the failure; `ModelComparator` excludes it from ranking but the
   other fourteen models still get compared (SRS NFR-7: graceful
   degradation, not silent or total failure).
5. **Two ranking angles, not one composite score.** "Rank by accuracy, RMSE,
   training time, prediction speed, complexity, and memory" doesn't reduce
   to one number without an arbitrary weighting scheme — `performance_rank`
   (pure predictive performance) and `efficiency_rank` (performance +
   resource cost, equally weighted) are reported side by side instead.

## 6. Open Decisions (tracked for Sprint 4/5)

- Whether Sprint 4/5's model families reuse `ModelComparator` directly or
  need a combined cross-family comparator once ensembling begins in
  Sprint 7.
- Whether `InferenceService` needs a batch (`predict_range`) mode once the
  backend (Sprint 8) has concrete latency requirements.