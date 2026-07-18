# Evaluation Report — MLDEMO (demo)

## Dataset
- **ticker:** MLDEMO
- **version:** demo
- **target_column:** future_return_5_day
- **task:** regression
- **rows_loaded:** 2000
- **rows_labeled:** 1995
- **train_rows:** 1396
- **validation_rows:** 299
- **test_rows:** 300
- **n_features:** 230
- **feature_selection_used:** False
- **hyperparameter_tuning_used:** False
- **models_attempted:** 12
- **models_succeeded:** 12

## Regression Models
Primary ranking metric: `rmse` (lower is better)

|   performance_rank | display_name            |   primary_metric_value |   training_time_seconds |   prediction_time_seconds |   model_complexity |   memory_kb |   efficiency_rank | was_tuned   |
|-------------------:|:------------------------|-----------------------:|------------------------:|--------------------------:|-------------------:|------------:|------------------:|:------------|
|                  1 | Ridge Regression        |               0.032351 |                  0.0142 |                  0.002293 |                231 |       13.37 |                 1 | False       |
|                  2 | Lasso Regression        |               0.032506 |                  0.0338 |                  0.00231  |                231 |       13.45 |                 2 | False       |
|                  3 | ElasticNet              |               0.032963 |                  0.0949 |                  0.00232  |                231 |       13.45 |                 3 | False       |
|                  4 | Extra Trees             |               0.037605 |                  5.0502 |                  0.019252 |             307200 |     3632.24 |                11 | False       |
|                  5 | Random Forest           |               0.037805 |                 30.3016 |                  0.018731 |             307200 |     2746.7  |                12 | False       |
|                  6 | Linear Regression       |               0.038765 |                  0.0319 |                  0.00239  |                231 |       15.15 |                 4 | False       |
|                  7 | XGBoost                 |               0.039689 |                  6.7731 |                  0.04048  |              12800 |      883.72 |                 7 | False       |
|                  8 | Decision Tree Regressor |               0.040075 |                  0.1323 |                  0.002095 |                 79 |       11.89 |                 5 | False       |
|                  9 | CatBoost                |               0.041133 |                 14.9677 |                  0.009403 |              25600 |      461.81 |                 8 | False       |
|                 10 | AdaBoost                |               0.041562 |                  5.4241 |                  0.028634 |              38400 |      210.39 |                 6 | False       |
|                 11 | LightGBM                |               0.046643 |                  4.2644 |                  0.006817 |             102400 |     1198.04 |                10 | False       |
|                 12 | Gradient Boosting       |               0.048649 |                 12.2565 |                  0.003345 |               1600 |      364.75 |                 9 | False       |

### Why the top model performs better
**Ridge Regression** ranks #1 on rmse with a test-set value of 0.032351 — lower (better) than the median model's 0.039227 by roughly 17.5%. It is also the fastest model to train in this comparison (0.014s). **Gradient Boosting** ranks last on rmse (0.048649) in this comparison.

### Per-model detail
#### `ridge_regression` — Ridge Regression (performance rank #1)
- `train_mae`: 0.013487
- `train_mse`: 0.000294
- `train_rmse`: 0.017139
- `train_mape_pct`: 312.6245
- `train_r2`: 0.460661
- `train_adjusted_r2`: 0.354183
- `train_n_samples`: 1396
- `train_n_features`: 230
- `validation_mae`: 0.023865
- `validation_mse`: 0.000856
- `validation_rmse`: 0.029253
- `validation_mape_pct`: 325.2697
- `validation_r2`: -0.339551
- `validation_adjusted_r2`: -4.870386
- `validation_n_samples`: 299
- `validation_n_features`: 230
- `test_mae`: 0.026325
- `test_mse`: 0.001047
- `test_rmse`: 0.032351
- `test_mape_pct`: 377.8464
- `test_r2`: 0.034093
- `test_adjusted_r2`: -3.185599
- `test_n_samples`: 300
- `test_n_features`: 230

#### `lasso_regression` — Lasso Regression (performance rank #2)
- `train_mae`: 0.015079
- `train_mse`: 0.000373
- `train_rmse`: 0.019319
- `train_mape_pct`: 243.3043
- `train_r2`: 0.314745
- `train_adjusted_r2`: 0.179459
- `train_n_samples`: 1396
- `train_n_features`: 230
- `validation_mae`: 0.018124
- `validation_mse`: 0.00051
- `validation_rmse`: 0.022585
- `validation_mape_pct`: 162.6825
- `validation_r2`: 0.201565
- `validation_adjusted_r2`: -2.499023
- `validation_n_samples`: 299
- `validation_n_features`: 230
- `test_mae`: 0.026145
- `test_mse`: 0.001057
- `test_rmse`: 0.032506
- `test_mape_pct`: 258.4246
- `test_r2`: 0.024801
- `test_adjusted_r2`: -3.225863
- `test_n_samples`: 300
- `test_n_features`: 230

#### `elastic_net` — ElasticNet (performance rank #3)
- `train_mae`: 0.014581
- `train_mse`: 0.000348
- `train_rmse`: 0.01866
- `train_mape_pct`: 250.318
- `train_r2`: 0.360722
- `train_adjusted_r2`: 0.234512
- `train_n_samples`: 1396
- `train_n_features`: 230
- `validation_mae`: 0.018205
- `validation_mse`: 0.000513
- `validation_rmse`: 0.022639
- `validation_mape_pct`: 173.5473
- `validation_r2`: 0.197718
- `validation_adjusted_r2`: -2.515881
- `validation_n_samples`: 299
- `validation_n_features`: 230
- `test_mae`: 0.026558
- `test_mse`: 0.001087
- `test_rmse`: 0.032963
- `test_mape_pct`: 315.4188
- `test_r2`: -0.002765
- `test_adjusted_r2`: -3.345317
- `test_n_samples`: 300
- `test_n_features`: 230

#### `extra_trees` — Extra Trees (performance rank #4)
- `train_mae`: 0.009215
- `train_mse`: 0.000143
- `train_rmse`: 0.011944
- `train_mape_pct`: 195.9956
- `train_r2`: 0.738058
- `train_adjusted_r2`: 0.686345
- `train_n_samples`: 1396
- `train_n_features`: 230
- `validation_mae`: 0.018407
- `validation_mse`: 0.000549
- `validation_rmse`: 0.023431
- `validation_mape_pct`: 215.3013
- `validation_r2`: 0.140641
- `validation_adjusted_r2`: -2.766013
- `validation_n_samples`: 299
- `validation_n_features`: 230
- `test_mae`: 0.030389
- `test_mse`: 0.001414
- `test_rmse`: 0.037605
- `test_mape_pct`: 316.3214
- `test_r2`: -0.305122
- `test_adjusted_r2`: -4.655529
- `test_n_samples`: 300
- `test_n_features`: 230

#### `random_forest` — Random Forest (performance rank #5)
- `train_mae`: 0.008939
- `train_mse`: 0.000145
- `train_rmse`: 0.012043
- `train_mape_pct`: 181.2289
- `train_r2`: 0.733715
- `train_adjusted_r2`: 0.681144
- `train_n_samples`: 1396
- `train_n_features`: 230
- `validation_mae`: 0.019644
- `validation_mse`: 0.000614
- `validation_rmse`: 0.024782
- `validation_mape_pct`: 236.7741
- `validation_r2`: 0.038665
- `validation_adjusted_r2`: -3.212911
- `validation_n_samples`: 299
- `validation_n_features`: 230
- `test_mae`: 0.030337
- `test_mse`: 0.001429
- `test_rmse`: 0.037805
- `test_mape_pct`: 291.2357
- `test_r2`: -0.319021
- `test_adjusted_r2`: -4.715757
- `test_n_samples`: 300
- `test_n_features`: 230

#### `linear_regression` — Linear Regression (performance rank #6)
- `train_mae`: 0.013225
- `train_mse`: 0.000282
- `train_rmse`: 0.016805
- `train_mape_pct`: 280.4595
- `train_r2`: 0.481519
- `train_adjusted_r2`: 0.379158
- `train_n_samples`: 1396
- `train_n_features`: 230
- `validation_mae`: 0.026742
- `validation_mse`: 0.001057
- `validation_rmse`: 0.032504
- `validation_mape_pct`: 388.369
- `validation_r2`: -0.653778
- `validation_adjusted_r2`: -6.247438
- `validation_n_samples`: 299
- `validation_n_features`: 230
- `test_mae`: 0.031493
- `test_mse`: 0.001503
- `test_rmse`: 0.038765
- `test_mape_pct`: 518.968
- `test_r2`: -0.38688
- `test_adjusted_r2`: -5.009813
- `test_n_samples`: 300
- `test_n_features`: 230

#### `xgboost` — XGBoost (performance rank #7)
- `train_mae`: 0.004709
- `train_mse`: 3.8e-05
- `train_rmse`: 0.006194
- `train_mape_pct`: 150.947
- `train_r2`: 0.92957
- `train_adjusted_r2`: 0.915665
- `train_n_samples`: 1396
- `train_n_features`: 230
- `validation_mae`: 0.019589
- `validation_mse`: 0.000598
- `validation_rmse`: 0.024461
- `validation_mape_pct`: 249.2079
- `validation_r2`: 0.063419
- `validation_adjusted_r2`: -3.104427
- `validation_n_samples`: 299
- `validation_n_features`: 230
- `test_mae`: 0.032392
- `test_mse`: 0.001575
- `test_rmse`: 0.039689
- `test_mape_pct`: 347.4266
- `test_r2`: -0.453785
- `test_adjusted_r2`: -5.299733
- `test_n_samples`: 300
- `test_n_features`: 230

#### `decision_tree` — Decision Tree Regressor (performance rank #8)
- `train_mae`: 0.012446
- `train_mse`: 0.000266
- `train_rmse`: 0.016322
- `train_mape_pct`: 204.9652
- `train_r2`: 0.510881
- `train_adjusted_r2`: 0.414317
- `train_n_samples`: 1396
- `train_n_features`: 230
- `validation_mae`: 0.02219
- `validation_mse`: 0.000788
- `validation_rmse`: 0.028063
- `validation_mape_pct`: 240.0513
- `validation_r2`: -0.232767
- `validation_adjusted_r2`: -4.40242
- `validation_n_samples`: 299
- `validation_n_features`: 230
- `test_mae`: 0.031681
- `test_mse`: 0.001606
- `test_rmse`: 0.040075
- `test_mape_pct`: 279.209
- `test_r2`: -0.482178
- `test_adjusted_r2`: -5.422771
- `test_n_samples`: 300
- `test_n_features`: 230

#### `catboost` — CatBoost (performance rank #9)
- `train_mae`: 0.007849
- `train_mse`: 9.7e-05
- `train_rmse`: 0.009846
- `train_mape_pct`: 203.6216
- `train_r2`: 0.822009
- `train_adjusted_r2`: 0.786869
- `train_n_samples`: 1396
- `train_n_features`: 230
- `validation_mae`: 0.01898
- `validation_mse`: 0.000569
- `validation_rmse`: 0.023857
- `validation_mape_pct`: 244.8013
- `validation_r2`: 0.109066
- `validation_adjusted_r2`: -2.904387
- `validation_n_samples`: 299
- `validation_n_features`: 230
- `test_mae`: 0.033654
- `test_mse`: 0.001692
- `test_rmse`: 0.041133
- `test_mape_pct`: 428.9956
- `test_r2`: -0.561496
- `test_adjusted_r2`: -5.766484
- `test_n_samples`: 300
- `test_n_features`: 230

#### `adaboost` — AdaBoost (performance rank #10)
- `train_mae`: 0.013201
- `train_mse`: 0.000256
- `train_rmse`: 0.016013
- `train_mape_pct`: 307.6777
- `train_r2`: 0.529199
- `train_adjusted_r2`: 0.436251
- `train_n_samples`: 1396
- `train_n_features`: 230
- `validation_mae`: 0.019015
- `validation_mse`: 0.000566
- `validation_rmse`: 0.02379
- `validation_mape_pct`: 222.8902
- `validation_r2`: 0.11409
- `validation_adjusted_r2`: -2.882372
- `validation_n_samples`: 299
- `validation_n_features`: 230
- `test_mae`: 0.033638
- `test_mse`: 0.001727
- `test_rmse`: 0.041562
- `test_mape_pct`: 490.1153
- `test_r2`: -0.594236
- `test_adjusted_r2`: -5.908354
- `test_n_samples`: 300
- `test_n_features`: 230

#### `lightgbm` — LightGBM (performance rank #11)
- `train_mae`: 0.001702
- `train_mse`: 5e-06
- `train_rmse`: 0.002337
- `train_mape_pct`: 55.7717
- `train_r2`: 0.989976
- `train_adjusted_r2`: 0.987998
- `train_n_samples`: 1396
- `train_n_features`: 230
- `validation_mae`: 0.025176
- `validation_mse`: 0.000975
- `validation_rmse`: 0.031219
- `validation_mape_pct`: 399.6996
- `validation_r2`: -0.525608
- `validation_adjusted_r2`: -5.685751
- `validation_n_samples`: 299
- `validation_n_features`: 230
- `test_mae`: 0.038307
- `test_mse`: 0.002176
- `test_rmse`: 0.046643
- `test_mape_pct`: 500.2985
- `test_r2`: -1.007801
- `test_adjusted_r2`: -7.700473
- `test_n_samples`: 300
- `test_n_features`: 230

#### `gradient_boosting` — Gradient Boosting (performance rank #12)
- `train_mae`: 0.008681
- `train_mse`: 0.000117
- `train_rmse`: 0.010837
- `train_mape_pct`: 212.9288
- `train_r2`: 0.784356
- `train_adjusted_r2`: 0.741783
- `train_n_samples`: 1396
- `train_n_features`: 230
- `validation_mae`: 0.020512
- `validation_mse`: 0.000682
- `validation_rmse`: 0.026119
- `validation_mape_pct`: 274.9314
- `validation_r2`: -0.067845
- `validation_adjusted_r2`: -3.679676
- `validation_n_samples`: 299
- `validation_n_features`: 230
- `test_mae`: 0.040361
- `test_mse`: 0.002367
- `test_rmse`: 0.048649
- `test_mape_pct`: 671.7183
- `test_r2`: -1.184244
- `test_adjusted_r2`: -8.465057
- `test_n_samples`: 300
- `test_n_features`: 230
