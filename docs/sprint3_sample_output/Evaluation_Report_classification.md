# Evaluation Report — MLDEMO (demo)

## Dataset
- **ticker:** MLDEMO
- **version:** demo
- **target_column:** target_direction_5_day
- **task:** classification
- **rows_loaded:** 2000
- **rows_labeled:** 1995
- **train_rows:** 1396
- **validation_rows:** 299
- **test_rows:** 300
- **n_features:** 230
- **feature_selection_used:** False
- **hyperparameter_tuning_used:** False
- **models_attempted:** 3
- **models_succeeded:** 3

## Classification Models
Primary ranking metric: `f1_score` (higher is better)

|   performance_rank | display_name             |   primary_metric_value |   training_time_seconds |   prediction_time_seconds |   model_complexity |   memory_kb |   efficiency_rank | was_tuned   |
|-------------------:|:-------------------------|-----------------------:|------------------------:|--------------------------:|-------------------:|------------:|------------------:|:------------|
|                  1 | Logistic Regression      |               0.587361 |                  0.1016 |                  0.002268 |                231 |       13.67 |                 1 | False       |
|                  2 | XGBoost Classifier       |               0.566038 |                  4.6564 |                  0.040028 |              12800 |      773.64 |                 2 | False       |
|                  3 | Random Forest Classifier |               0.555556 |                  2.5005 |                  0.018265 |              76800 |     1866.93 |                 3 | False       |

### Why the top model performs better
**Logistic Regression** ranks #1 on f1_score with a test-set value of 0.587361 — higher (better) than the median model's 0.566038 by roughly 3.8%. It is also the fastest model to train in this comparison (0.102s). **Random Forest Classifier** ranks last on f1_score (0.555556) in this comparison.

### Per-model detail
#### `logistic_regression` — Logistic Regression (performance rank #1)
- `train_accuracy`: 0.782951
- `train_precision`: 0.776712
- `train_recall`: 0.80198
- `train_f1_score`: 0.789144
- `train_roc_auc`: 0.862443
- `train_confusion_matrix`: [[526, 163], [140, 567]]
- `train_n_samples`: 1396
- `validation_accuracy`: 0.575251
- `validation_precision`: 0.586957
- `validation_recall`: 0.377622
- `validation_f1_score`: 0.459574
- `validation_roc_auc`: 0.651784
- `validation_confusion_matrix`: [[118, 38], [89, 54]]
- `validation_n_samples`: 299
- `test_accuracy`: 0.63
- `test_precision`: 0.484663
- `test_recall`: 0.745283
- `test_f1_score`: 0.587361
- `test_roc_auc`: 0.710903
- `test_confusion_matrix`: [[110, 84], [27, 79]]
- `test_n_samples`: 300

#### `xgboost_classifier` — XGBoost Classifier (performance rank #2)
- `train_accuracy`: 1.0
- `train_precision`: 1.0
- `train_recall`: 1.0
- `train_f1_score`: 1.0
- `train_roc_auc`: 1.0
- `train_confusion_matrix`: [[689, 0], [0, 707]]
- `train_n_samples`: 1396
- `validation_accuracy`: 0.628763
- `validation_precision`: 0.568376
- `validation_recall`: 0.93007
- `validation_f1_score`: 0.70557
- `validation_roc_auc`: 0.780482
- `validation_confusion_matrix`: [[55, 101], [10, 133]]
- `validation_n_samples`: 299
- `test_accuracy`: 0.463333
- `test_precision`: 0.396226
- `test_recall`: 0.990566
- `test_f1_score`: 0.566038
- `test_roc_auc`: 0.704873
- `test_confusion_matrix`: [[34, 160], [1, 105]]
- `test_n_samples`: 300

#### `random_forest_classifier` — Random Forest Classifier (performance rank #3)
- `train_accuracy`: 0.889685
- `train_precision`: 0.879287
- `train_recall`: 0.906648
- `train_f1_score`: 0.892758
- `train_roc_auc`: 0.959111
- `train_confusion_matrix`: [[601, 88], [66, 641]]
- `train_n_samples`: 1396
- `validation_accuracy`: 0.558528
- `validation_precision`: 0.521073
- `validation_recall`: 0.951049
- `validation_f1_score`: 0.673267
- `validation_roc_auc`: 0.768155
- `validation_confusion_matrix`: [[31, 125], [7, 136]]
- `validation_n_samples`: 299
- `test_accuracy`: 0.44
- `test_precision`: 0.386029
- `test_recall`: 0.990566
- `test_f1_score`: 0.555556
- `test_roc_auc`: 0.725394
- `test_confusion_matrix`: [[27, 167], [1, 105]]
- `test_n_samples`: 300
