# `ai/models/time_series/` — Sprint 4: Time Series Forecasting

Classical statistical forecasting of a ticker's Close price, built on the
`datasets/processed/{ticker}_{version}_features.csv` files Sprint 1/2
already produce. This module forecasts the raw price series directly — it
does **not** consume Sprint 2's 205 engineered features (those remain
Sprint 3's ML input); ARIMA/SARIMA/Prophet-style models work off the
univariate series itself.

## 1. Purpose

Provide a second, structurally different family of forecasts (classical
statistical time series, as opposed to Sprint 3's feature-driven ML models)
so that:

1. The Decision Support Engine (Sprint 7) can ensemble across genuinely
   different modeling assumptions rather than several ML variants that all
   see the same 205 features.
2. `ModelComparator`-style side-by-side evaluation (`evaluation/comparator.py`)
   can honestly answer "does ARIMA beat Random Forest on this ticker?"
   instead of assuming one approach is always better.

## 2. Workflow

```
Sprint 2 processed CSV (date, close, ...)
        │
        ▼
  data_loader.load_price_series()  → chronological pd.Series
        │
        ▼
  splitting.TimeSeriesSplitter     → train / validation / test (contiguous, no shuffle)
        │
        ▼
  stationarity.analyze_stationarity() ── ADF + KPSS → differencing order d
  analysis.detect_seasonal_period()   ── ACF-based   → seasonal period m
        │
        ▼
  models/registry.TimeSeriesModelFactory
        │  ┌── ArimaModel(p,d,q)
        │  ├── SarimaModel(p,d,q)(P,D,Q,m)
        │  ├── AutoArimaModel        (optional — pmdarima)
        │  ├── ProphetModel          (optional — prophet)
        │  └── ExponentialSmoothingModel
        ▼
  pipelines.TimeSeriesTrainingPipeline.run(ticker)
        │  fit each model → forecast validation/test → evaluate → persist
        ▼
  evaluation.TimeSeriesModelComparator  → Model_Comparison.csv + narrative
        │
        ▼
  prediction.ForecastService.forecast(model, ticker, horizon)
        │
        ▼
  Sprint 5 (Deep Learning) / Sprint 7 (Decision Engine) / Sprint 8 (Backend API)
```

## 3. Mathematical Background

### 3.1 Stationarity (ADF + KPSS)

ARIMA-family models assume the series has constant mean/variance over time.
Two complementary tests are run (see `stationarity.py`'s module docstring
for the full 2×2 verdict table):

- **ADF**: `Δy_t = α + βt + γy_{t-1} + Σδᵢ Δy_{t-i} + ε_t`. Tests
  `H0: γ = 0` (unit root ⇒ non-stationary).
- **KPSS**: decomposes `y_t = ξt + r_t + ε_t` (deterministic trend + random
  walk + stationary error) and tests `H0: Var(random walk innovations) = 0`
  (⇒ stationary).

If non-stationary, the series is **differenced** (`y_t - y_{t-1}`) and
re-tested, up to `TimeSeriesConfig.max_differencing_order` (default 2).

### 3.2 ARIMA(p, d, q)

`(1 - Σφᵢ Lⁱ)(1 - L)ᵈ y_t = (1 + Σθⱼ Lʲ) ε_t` — AR order `p` (past values),
differencing order `d` (from §3.1), MA order `q` (past forecast errors).
`p, q` are chosen by AIC grid search (`arima_model.py`) rather than manual
ACF/PACF eyeballing, though ACF/PACF are still plotted for the required
visualizations and to sanity-check the AIC choice.

### 3.3 SARIMA(p,d,q)(P,D,Q,m)

Adds a seasonal AR/I/MA block at lag `m` (seasonal period, ACF-detected —
`analysis.detect_seasonal_period`, defaulting to 5 for weekly-cycle daily
equity data). `sarima_model.py` grid-searches a deliberately small seasonal
space (`P, Q ∈ {0,1}`) since SARIMAX fit cost grows quickly.

### 3.4 Auto ARIMA

`pmdarima.auto_arima`'s stepwise Hyndman-Khandakar algorithm — a smarter
search than brute-force grid search, useful as a cross-check against the
manual AIC search above.

### 3.5 Prophet

Additive decomposition `y(t) = g(t) + s(t) + h(t) + ε_t` — piecewise-linear
trend `g`, Fourier-series seasonality `s`, holiday effects `h` (unused here).
Robust to missing data and outliers; changepoint detection handles trend
shifts an ARIMA order search won't.

### 3.6 Exponential Smoothing (Holt-Winters)

`level, trend, seasonal` components updated by exponentially-decaying
weights. Included as a fast, low-assumption baseline.

## 4. Advantages / Limitations / Best Use Cases

| Model | Advantages | Limitations | Best use case |
|---|---|---|---|
| ARIMA | Simple, interpretable, fast | Assumes linearity, no seasonality | Short-horizon, non-seasonal series |
| SARIMA | Captures seasonal cycles | Slower to fit, more hyperparameters | Series with a clear weekly/monthly cycle |
| Auto ARIMA | No manual order search | Still linear/Gaussian assumptions | Quick baseline across many tickers |
| Prophet | Handles trend shifts, missing data, holidays | Can over-smooth sharp moves | Longer-horizon, business-calendar-driven series |
| Exp. Smoothing | Very fast, few assumptions | No confidence-interval theory (approximated here) | Fast baseline / sanity check |

**Financial interpretation:** none of these models see order flow,
fundamentals, or news — they extrapolate the price series' own statistical
structure. A high `directional_accuracy` on a validation split for a
near-random-walk price series should be read with skepticism (see Sprint
3's own honest finding that regularized linear models beat ensembles on
synthetic near-random-walk data — the same caution applies here). Every
`ForecastService` response carries `educational_disclaimer` for this reason
(ARCHITECTURE.md §2.4, SRS NFR-11).

## 5. Model Comparison: ML (Sprint 3) vs Time Series (Sprint 4)

`evaluation/comparator.py::TimeSeriesModelComparator.merge_ml_results()`
lets a Sprint 3 `ModelComparator` comparison table's flattened rows be
folded into the same ranked table as this sprint's models, so both families
are compared on `test_rmse` / `test_mae` / training & prediction time side
by side, rather than in two separate, un-comparable reports.

## 6. Known Issues

- **pmdarima / prophet build failures on Python 3.13**: both ship compiled
  extensions (Cython / Stan via `cmdstanpy`) that don't always have
  pre-built wheels for the newest Python versions yet — the same class of
  problem Sprint 3 hit with CatBoost. `TimeSeriesModelFactory.available_models()`
  detects this at runtime and simply omits the affected model; the pipeline
  and `ForecastService` both continue to work with whatever models did
  install.
- **Holt-Winters confidence intervals are approximated** (residual-std,
  horizon-widened) — statsmodels' `ExponentialSmoothing` has no native
  interval; this is documented on `ForecastResult.params["confidence_interval_method"]`
  rather than silently presented as an exact interval.
- **Walk-forward validation cost**: `pipelines/forecasting_pipeline.py::walk_forward_validate`
  re-fits a fresh model per fold — expensive for SARIMA/Prophet on long
  series. Use a larger `step` to keep runs practical.

## 7. How This Sprint Connects to the Next

- **Sprint 5 (Deep Learning)**: LSTM/GRU models can be benchmarked against
  this sprint's ARIMA/SARIMA/Prophet results using the same
  `TimeSeriesModelResult` / `TimeSeriesModelComparator` shapes.
- **Sprint 7 (Decision Support Engine)**: calls
  `ForecastService.forecast_all_models(ticker, horizon)` to get every
  available model's forecast in one call for ensembling.
- **Sprint 8 (Backend API)**: wraps `ForecastService.forecast(...)` behind
  `/api/v1/forecast/{ticker}`.