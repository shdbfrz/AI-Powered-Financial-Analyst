# Sprint 3 — Sample Output (for review without running the pipeline)

`storage/models/ml/` and `storage/reports/ml/` are gitignored (see
`storage/README.md`) — real pipeline output isn't committed, by design,
same as Sprint 2's `datasets/processed/`. That's normal for a training
pipeline, but it means a reviewer (professor, teammate) can't see what it
actually produces without running it themselves.

This folder is the fix: a **committed, static snapshot** of one real
pipeline run (all 15 models actually trained, no hand-edited numbers), so
the generated reports and comparison output are visible directly on GitHub.

## What's here

| File | What it is |
|---|---|
| `Model_Comparison_regression.csv` / `_classification.csv` | The ranked comparison table (`performance_rank`, `efficiency_rank`, every train/validation/test metric, training time, complexity, memory) for all 12 regression models and all 3 classification models. |
| `Evaluation_Report_regression.md` / `_classification.md` | The full rendered report: dataset summary, ranked table, the auto-generated "why the top model performs better" narrative, and per-model metric detail. |
| `Model_Documentation.md` | Purpose / advantages / limitations / best-use-cases for all 15 models, generated from each model's `ModelInfo` — the sprint's "explain every model" deliverable (mirrors Sprint 2's `Feature_Report.md`). |
| `Model_Metadata_regression.json` / `_classification.json` | Machine-readable metadata for every saved `.joblib` artifact: model name, training date, features used, test metrics, version. |
| `plots/` | Both tasks' model-comparison charts, plus the winning model's full plot set (prediction-vs-actual, residuals, feature importance, learning curve, validation curve) for regression (Ridge Regression) and the applicable subset for classification (Logistic Regression). |

## Reading the results honestly

The regularized linear models (Ridge/Lasso/ElasticNet) edge out every tree
ensemble and boosting model on this run, and every regression model's test
R² is small or negative. That's expected, not a bug: the sample data is
synthetic, close to a random walk with a mild sinusoidal regime added, so
there is very little genuine signal for 205 engineered features to find —
exactly the situation where unregularized/complex models overfit and simple
regularized ones generalize better. Predicting real stock returns is hard
for the same underlying reason (SRS §2.4's disclaimer requirement exists
precisely because of this); a suspiciously high R² here would be more
concerning than these honest, modest numbers.

## How this was generated

Synthetic OHLCV data (2,000 trading days, ticker `MLDEMO`) was run through
the real Sprint 2 pipeline, then the real Sprint 3 pipeline, exactly as
`scripts/run_ml_pipeline_demo.py` does:

```python
from ai.models.ml import MLTrainingPipeline

pipeline = MLTrainingPipeline()
regression_result = pipeline.run(ticker="MLDEMO", task="regression", version="demo")
classification_result = pipeline.run(ticker="MLDEMO", task="classification", version="demo")
```

Synthetic data was used (not a real ticker) specifically so this sample is
reproducible by anyone, with no dependency on Sprint 1's live data
providers or `.env` API keys.

## To regenerate with real data

Once Sprint 1 has pulled a real ticker into `datasets/raw/` and Sprint 2 has
processed it:

```powershell
python scripts\run_ml_pipeline_demo.py AAPL
```

Real output lands in `storage/models/ml/` and `storage/reports/ml/`
(gitignored, local-only) — this `docs/` folder is not automatically updated
and should be refreshed manually if it needs to reflect a newer version of
the pipeline.