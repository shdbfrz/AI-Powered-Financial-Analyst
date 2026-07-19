"""
Configuration for `ai/models/time_series/` (Sprint 4 — Time Series Forecasting).

Same two-tier split established in `ai/models/ml/config.py`:

- `ai/utils/config.py::settings` — environment-variable-backed settings
  (storage paths, log level). This module reads from there for I/O.
- `TimeSeriesConfig` (this file) — non-secret, purely computational tunables
  (split ratios, ADF/KPSS significance level, ARIMA search bounds, seasonal
  period, forecast horizons). Frozen dataclass, passed explicitly through the
  pipeline instead of living in `.env`, so a run's exact configuration is
  reproducible and loggable (ARCHITECTURE.md §2.3).
"""

from dataclasses import dataclass, field
from typing import Literal

TrendMode = Literal["additive", "multiplicative"]


@dataclass(frozen=True)
class TimeSeriesConfig:
    """All tunable parameters for the Sprint 4 time series pipeline."""

    # --- Target series ------------------------------------------------------
    # Sprint 4 spec: target is Close price, forecast on the raw price level
    # (not the Sprint 2/3 engineered future_return_* columns — those are
    # ML-only targets). Column names as written by Sprint 1/2 to
    # datasets/processed/{ticker}_{version}_features.csv.
    date_column: str = "date"
    price_column: str = "close"

    # --- Forecast horizons (trading days) ------------------------------------
    forecast_horizons: tuple[int, ...] = (1, 3, 5)

    # --- Data splitting (contiguous, chronological — never shuffled) --------
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    min_rows_required: int = 100  # below this, statistical models are unreliable

    # --- Stationarity testing -------------------------------------------------
    adf_significance_level: float = 0.05
    kpss_significance_level: float = 0.05
    max_differencing_order: int = 2  # d rarely needs to exceed 2 for daily price data
    log_transform_skew_threshold: float = 1.0  # |skew| above this triggers a log-transform recommendation

    # --- Seasonality --------------------------------------------------------
    # 5-trading-day (weekly) seasonality is the standard default for daily
    # equity data; SARIMA's seasonal_period is exposed separately so it can
    # be overridden per-ticker if a different cycle is detected.
    default_seasonal_period: int = 5
    seasonality_detection_max_lag: int = 60

    # --- ACF / PACF ------------------------------------------------------------
    acf_pacf_max_lags: int = 40
    rolling_window: int = 20  # ~1 trading month, matches Sprint 2's rolling-feature convention

    # --- ARIMA / SARIMA search bounds (used by both manual grid search and
    # as sanity bounds around pmdarima's auto_arima result) ------------------
    arima_max_p: int = 5
    arima_max_d: int = 2
    arima_max_q: int = 5
    sarima_max_P: int = 2
    sarima_max_D: int = 1
    sarima_max_Q: int = 2

    # --- Prophet --------------------------------------------------------------
    prophet_seasonality_mode: TrendMode = "additive"
    prophet_weekly_seasonality: bool = True
    prophet_yearly_seasonality: bool = True
    prophet_daily_seasonality: bool = False
    prophet_changepoint_prior_scale: float = 0.05

    # --- Confidence intervals ---------------------------------------------------
    confidence_level: float = 0.95

    # --- Walk-forward validation ------------------------------------------------
    walk_forward_min_train_size: int = 100
    walk_forward_step: int = 5

    # --- Reproducibility ----------------------------------------------------------
    random_state: int = 42

    # --- Model ranking (mirrors ai/models/ml/config.py's ModelComparator convention) ---
    primary_metric: str = "rmse"
    primary_metric_direction: Literal["minimize", "maximize"] = "minimize"

    # --- Visualization --------------------------------------------------------------
    plot_confidence_band_alpha: float = 0.2
    top_n_forecast_points_to_annotate: int = 5
    learning_curve_train_sizes: tuple[float, ...] = field(
        default_factory=lambda: (0.2, 0.4, 0.6, 0.8, 1.0)
    )

    def __post_init__(self) -> None:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + validation_ratio + test_ratio must sum to 1.0, got {total:.4f}"
            )
        if not (0 < self.train_ratio < 1) or not (0 < self.test_ratio < 1):
            raise ValueError("train_ratio and test_ratio must both be in (0, 1)")
        if not (0 < self.confidence_level < 1):
            raise ValueError("confidence_level must be in (0, 1)")


DEFAULT_TS_CONFIG = TimeSeriesConfig()