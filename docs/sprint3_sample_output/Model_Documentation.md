# Model Documentation

Auto-generated from each model's `ModelInfo` (see `ai/models/ml/models/*.py`) — the single source of truth, so this document can never drift from the actual model implementations.

## Regression Models
### Linear Regression
- **Registry name:** `linear_regression`
- **Family:** linear
- **Purpose:** Fits a straight-line (hyperplane) relationship between the engineered features and the target with no regularization — the simplest possible baseline every other model must beat to justify its extra complexity.
- **Advantages:** Fast to train and predict, fully interpretable coefficients, no hyperparameters to tune, deterministic.
- **Limitations:** Assumes a linear relationship; cannot capture the non-linear interactions (e.g. RSI x volatility regime) that drive real price moves. Sensitive to multicollinearity among the 205 engineered features (many indicators are derived from the same close series).
- **Best use cases:** Sanity-check baseline and as an interpretable reference point when explaining *why* a more complex model's extra accuracy is worth its extra opacity (relevant for the LLM Explanation Layer in Sprint 8).
- **Recommended for:** baseline, interpretability

### Ridge Regression
- **Registry name:** `ridge_regression`
- **Family:** linear
- **Purpose:** Linear regression with an L2 penalty that shrinks all coefficients toward zero, trading a small amount of bias for a large reduction in variance when features are correlated.
- **Advantages:** Handles the multicollinearity that plain Linear Regression struggles with (e.g. sma_5/sma_10/sma_20 are highly correlated by construction); still closed-form and fast; every coefficient stays in the model (no hard feature selection), so nothing is discarded silently.
- **Limitations:** Does not perform feature selection — with 205 candidate features, a Ridge model keeps (and must be interpreted alongside) all of them. The regularization strength `alpha` needs tuning per dataset.
- **Best use cases:** Preferred over plain Linear Regression whenever the feature set has known collinear groups, which is the norm here (Bollinger/SMA/EMA families all move together).
- **Recommended for:** regularization, collinear features

### Lasso Regression
- **Registry name:** `lasso_regression`
- **Family:** linear
- **Purpose:** Linear regression with an L1 penalty that can shrink coefficients exactly to zero, performing embedded feature selection while fitting.
- **Advantages:** Produces a sparse, more interpretable model by automatically dropping features it finds uninformative — useful as a cheap complement to the explicit Recursive Feature Elimination step in this sprint's feature selection module.
- **Limitations:** Arbitrarily picks one feature from a correlated group and zeroes out the rest, which can be unstable (a slightly different data window may keep a different member of the same indicator family). Requires tuning `alpha`; too large a value can zero out every coefficient.
- **Best use cases:** Exploratory feature-importance signal ('which of these 205 indicators does a sparse linear model keep?') more than a final production model on its own.
- **Recommended for:** feature selection, sparse models

### ElasticNet
- **Registry name:** `elastic_net`
- **Family:** linear
- **Purpose:** Linear regression with a weighted mix of L1 and L2 penalties (`l1_ratio`), combining Lasso's sparsity with Ridge's stability on correlated feature groups.
- **Advantages:** More stable than pure Lasso when features are highly correlated (the common case here), while still capable of zeroing out irrelevant features; two knobs (`alpha`, `l1_ratio`) give finer control than either Ridge or Lasso alone.
- **Limitations:** Two hyperparameters instead of one means a larger tuning search space; still fundamentally a linear model and cannot capture non-linear interactions.
- **Best use cases:** A middle-ground default when it's unclear upfront whether Ridge- or Lasso-style regularization suits the feature set better.
- **Recommended for:** regularization, collinear features, feature selection

### Decision Tree Regressor
- **Registry name:** `decision_tree`
- **Family:** tree
- **Purpose:** Learns a piecewise-constant prediction by recursively splitting the feature space on the single most informative feature/threshold at each node — the building block every ensemble in this sprint (Random Forest, Extra Trees, Gradient Boosting, AdaBoost) is made of.
- **Advantages:** Captures non-linear relationships and feature interactions with zero preprocessing; trivially interpretable for shallow depths (can be printed/plotted as an actual flowchart); fast to train.
- **Limitations:** A single tree overfits easily — deep enough to fit training noise, it memorizes rather than generalizes, which is precisely why it's used as a component of ensembles rather than deployed alone.
- **Best use cases:** Diagnostic tool (visualize *why* a specific split matters) and as the weak learner inside AdaBoost; rarely the final production model.
- **Recommended for:** interpretability, ensemble building block

### Random Forest
- **Registry name:** `random_forest`
- **Family:** ensemble
- **Purpose:** Averages many regression trees, each trained on a bootstrap sample with a random feature subset per split, to reduce the variance/overfitting of any single tree while keeping its ability to model non-linear interactions.
- **Advantages:** Strong out-of-the-box accuracy with minimal tuning, robust to outliers and irrelevant features (of which 205 candidates surely include some), provides feature importances for free, trains in parallel (`n_jobs=-1`).
- **Limitations:** Less interpretable than a single tree or a linear model; can still overfit with very deep trees on a small dataset; prediction requires walking every tree, so it's slower at inference than a linear model.
- **Best use cases:** Strong default choice for tabular financial features — this project's ARCHITECTURE.md names it explicitly as one of the two required core models alongside XGBoost.
- **Recommended for:** strong default, feature importance

### Extra Trees
- **Registry name:** `extra_trees`
- **Family:** ensemble
- **Purpose:** Like Random Forest, but each tree also picks its split *thresholds* randomly (rather than searching for the optimal one), then averages across many such trees — trading a bit of per-tree accuracy for a further reduction in variance and faster training.
- **Advantages:** Typically trains faster than Random Forest (no per-split threshold search) and can generalize slightly better on noisy tabular data such as engineered technical indicators, which are inherently noisy.
- **Limitations:** The extra randomization can underfit if the true signal is subtle and the dataset is small; like Random Forest, still an ensemble of black boxes rather than a single interpretable model.
- **Best use cases:** Direct A/B comparison against Random Forest on the same features — this sprint's Model Comparison table is exactly the mechanism for deciding which one wins on a given ticker.
- **Recommended for:** variance reduction, fast training

### Gradient Boosting
- **Registry name:** `gradient_boosting`
- **Family:** boosting
- **Purpose:** Builds trees sequentially, each one fit to the *residual errors* of the ensemble so far, gradually reducing bias — scikit-learn's native gradient boosting implementation (as distinct from the external XGBoost library).
- **Advantages:** Often reaches lower bias than bagged ensembles (Random Forest/Extra Trees) on structured tabular data; `learning_rate` gives fine control over the bias/variance trade-off.
- **Limitations:** Trains sequentially (trees depend on prior trees), so it cannot parallelize across estimators the way Random Forest can, making it noticeably slower to train; more sensitive to hyperparameters (`learning_rate` x `n_estimators` x `max_depth` all interact) and to overfitting if boosted too long.
- **Best use cases:** When XGBoost isn't available/desired and a boosted (bias-reducing) rather than bagged (variance-reducing) tree ensemble is wanted, or as a second, dependency-light boosting reference point next to XGBoost in the comparison table.
- **Recommended for:** bias reduction, boosting baseline

### AdaBoost
- **Registry name:** `adaboost`
- **Family:** boosting
- **Purpose:** Fits a sequence of weak learners (shallow decision trees), each one reweighting the training samples the previous learner predicted worst, then combines all of them via a weighted vote/average.
- **Advantages:** Simple, few hyperparameters to tune, tends to be robust when the base learner is kept intentionally weak (shallow trees), fast to train relative to Gradient Boosting.
- **Limitations:** Sensitive to noisy targets/outliers — a mislabeled or extreme-outlier sample gets increasingly upweighted across rounds, which is a real risk on raw financial return data (single-day price shocks). Generally underperforms Gradient Boosting/XGBoost on complex tabular data.
- **Best use cases:** A lightweight boosting reference point in the comparison table, particularly useful for illustrating the bias/variance trade-off against the bagged ensembles (Random Forest/Extra Trees) in the sprint's model-comparison writeup.
- **Recommended for:** boosting baseline, low-noise targets

### XGBoost
- **Registry name:** `xgboost`
- **Family:** boosting
- **Purpose:** A highly optimized, regularized gradient-boosted tree ensemble — adds L1/L2 regularization on leaf weights and second-order gradient information on top of scikit-learn's Gradient Boosting, generally reaching lower error for the same training budget.
- **Advantages:** State-of-the-art accuracy on structured/tabular data in most published benchmarks; built-in handling of missing values; trains fast via histogram-based split finding; explicit regularization (`reg_alpha`/`reg_lambda`) reduces overfitting risk versus plain Gradient Boosting.
- **Limitations:** More hyperparameters to tune than Random Forest for full performance; an external dependency (not part of scikit-learn) that must be version-pinned (`ai/requirements.txt` pins `xgboost==2.1.1`); like every tree ensemble here, a black box relative to the linear family.
- **Best use cases:** Named explicitly in this project's ARCHITECTURE.md and PDR as one of the two required core ML models (alongside Random Forest) — the expected top performer in the regression comparison table for most tickers.
- **Recommended for:** strong default, production candidate

### LightGBM _(optional dependency)_
- **Registry name:** `lightgbm`
- **Family:** boosting
- **Purpose:** A histogram-based gradient boosting framework using leaf-wise (best-first) tree growth instead of XGBoost's level-wise growth, generally trading a bit of overfitting risk for faster training on larger datasets.
- **Advantages:** Typically the fastest-training boosted ensemble of the three (native XGBoost, LightGBM, CatBoost) here, with competitive accuracy; native categorical-feature support (relevant for Sprint 2's `price_action_label`/`trend_label`/`market_bias` columns without one-hot expansion, though this pipeline currently one-hot-encodes them upstream for consistency across all models).
- **Limitations:** Leaf-wise growth can overfit small datasets more readily than level-wise growth (XGBoost's default) unless `num_leaves`/`min_child_samples` are constrained; optional dependency — the pipeline must run correctly without it installed.
- **Best use cases:** Useful third boosting data point in the comparison table, particularly valuable once the dataset grows large enough (multi-ticker, multi-year) that XGBoost's training time becomes a bottleneck.
- **Recommended for:** fast training, large datasets

### CatBoost _(optional dependency)_
- **Registry name:** `catboost`
- **Family:** boosting
- **Purpose:** A gradient boosting framework built around ordered boosting and native categorical-feature handling, designed to reduce the prediction shift (target leakage) that naive gradient boosting can suffer on categorical columns.
- **Advantages:** Strong out-of-the-box defaults (often competitive without any tuning), native handling of Sprint 2's categorical Phase-4 columns (`price_action_label`, `trend_label`, `breakout_label`, `market_bias`, `market_structure`), generally robust to overfitting via ordered boosting.
- **Limitations:** Slower to train than LightGBM in most benchmarks; larger install footprint; optional dependency — the pipeline must run correctly without it installed; verbose output must be explicitly silenced.
- **Best use cases:** Fourth boosting reference point, particularly worth including once Sprint 2's categorical columns are passed in natively (rather than one-hot encoded) to see whether native categorical handling measurably helps on this feature set.
- **Recommended for:** categorical features, robust defaults

## Classification Models
### Logistic Regression
- **Registry name:** `logistic_regression`
- **Family:** linear
- **Purpose:** Linear decision boundary over the engineered features, predicting the probability that `target_direction_{h}_day` is True (price rises over the next h day(s)).
- **Advantages:** Fast, well-calibrated probabilities (useful downstream for the Decision Support Engine's confidence_score in Sprint 7), interpretable coefficients, a strong and cheap baseline for a binary signal.
- **Limitations:** Linear decision boundary — cannot represent the non-linear regime changes (e.g. a bullish RSI crossover only mattering when volatility is also low) that tree-based classifiers pick up naturally.
- **Best use cases:** Baseline for the Buy/Sell direction classification task and as the reference point for whether Random Forest/XGBoost's extra complexity is actually earning better accuracy on this ticker.
- **Recommended for:** baseline, probability calibration

### Random Forest Classifier
- **Registry name:** `random_forest_classifier`
- **Family:** ensemble
- **Purpose:** The classification counterpart of Random Forest Regressor: averages many classification trees' votes to predict `target_direction_{h}_day` (up/down) with a well-behaved probability estimate from `predict_proba`.
- **Advantages:** Handles non-linear feature interactions naturally (e.g. RSI is only predictive conditional on trend regime — exactly the kind of interaction a linear classifier misses); robust to the class-imbalance and outliers common in direction-prediction tasks; feature importances for free.
- **Limitations:** Predicted probabilities from tree ensembles are less well-calibrated out of the box than Logistic Regression's (worth checking before feeding `confidence_score` in Sprint 7's Decision Support Engine); less interpretable than the linear baseline.
- **Best use cases:** Primary classification model for the Buy/Sell direction signal — named explicitly alongside XGBoost Classifier as a required model in this sprint's specification.
- **Recommended for:** strong default, direction classification

### XGBoost Classifier
- **Registry name:** `xgboost_classifier`
- **Family:** boosting
- **Purpose:** The classification counterpart of XGBoost Regressor: a regularized, boosted tree ensemble predicting `target_direction_{h}_day`, typically the strongest of the three classification models on structured features.
- **Advantages:** Same regularization/accuracy advantages as the regressor variant, plus `scale_pos_weight` for handling any class imbalance between up/down days; well-calibrated `predict_proba` relative to bagged forests.
- **Limitations:** Same as the regressor: more hyperparameters, external dependency, black-box relative to Logistic Regression.
- **Best use cases:** Expected top performer for the Buy/Sell direction signal that feeds the Decision Support Engine's `confidence_score` (Sprint 7) — named explicitly as a required classification model in this sprint's specification.
- **Recommended for:** strong default, direction classification, production candidate
