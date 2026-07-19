"""
End-to-end Sprint 4 training/evaluation orchestrator.

Mirrors `ai.models.ml.pipelines.training_pipeline.MLTrainingPipeline`'s
shape: one `.run(ticker)` call that loads data, splits it, analyzes
stationarity, trains every available model, evaluates each on
validation/test, ranks them, and persists artifacts — with each model's
failure isolated (`TimeSeriesModelResult.error` set, not raised) so one
model failing to converge never aborts the whole run.
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import pandas as pd

from ai.models.time_series.analysis import (
    DecompositionResult,
    compute_acf_pacf,
    decompose_series,
    detect_seasonal_period,
    detect_trend_direction,
    rolling_statistics,
)
from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.models.time_series.data_loader import load_price_series
from ai.models.time_series.evaluation.comparator import (
    TimeSeriesComparisonTable,
    TimeSeriesModelComparator,
    TimeSeriesModelResult,
)
from ai.models.time_series.evaluation.metrics import evaluate_forecast
from ai.models.time_series.models.registry import TimeSeriesModelFactory
from ai.models.time_series.persistence.model_storage import TimeSeriesModelStorage
from ai.models.time_series.splitting import SeriesSplit, TimeSeriesSplitter
from ai.models.time_series.stationarity import StationarityReport, analyze_stationarity
from ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TimeSeriesRunResult:
    ticker: str
    series: pd.Series
    split: SeriesSplit
    stationarity_report: StationarityReport
    decomposition: DecompositionResult
    seasonal_period: int
    model_results: dict[int, list[TimeSeriesModelResult]] = field(default_factory=dict)  # horizon -> results
    comparison_tables: dict[int, TimeSeriesComparisonTable] = field(default_factory=dict)  # horizon -> table
    saved_artifact_paths: list[str] = field(default_factory=list)


class TimeSeriesTrainingPipeline:
    """Pipeline pattern orchestrator for Sprint 4, run per-ticker."""

    def __init__(self, config: TimeSeriesConfig = DEFAULT_TS_CONFIG, persist: bool = True):
        self.config = config
        self.persist = persist
        self.comparator = TimeSeriesModelComparator(
            primary_metric=config.primary_metric,
            primary_metric_direction=config.primary_metric_direction,
        )
        self.storage = TimeSeriesModelStorage()
        self.logger = logger

    def run(
        self,
        ticker: str,
        *,
        model_names: Optional[list[str]] = None,
        horizons: Optional[tuple[int, ...]] = None,
    ) -> TimeSeriesRunResult:
        horizons = horizons or self.config.forecast_horizons
        self.logger.info("time series pipeline started", extra={"ticker": ticker, "horizons": horizons})

        series = load_price_series(ticker, config=self.config)
        split = TimeSeriesSplitter(self.config).split(series)

        stationarity_report = analyze_stationarity(split.train, config=self.config)
        seasonal_period, _ = detect_seasonal_period(split.train, config=self.config)
        decomposition = decompose_series(split.train, seasonal_period=seasonal_period)

        # Logged/available for the required exploratory artifacts even
        # though the pipeline itself only consumes stationarity/seasonality.
        compute_acf_pacf(split.train, config=self.config)
        rolling_statistics(split.train, window=self.config.rolling_window)
        detect_trend_direction(split.train)

        available_models = model_names or TimeSeriesModelFactory.available_models(self.config)
        self.logger.info("models available for this run", extra={"models": available_models})

        result = TimeSeriesRunResult(
            ticker=ticker,
            series=series,
            split=split,
            stationarity_report=stationarity_report,
            decomposition=decomposition,
            seasonal_period=seasonal_period,
        )

        train_plus_val = pd.concat([split.train, split.validation])

        for horizon in horizons:
            horizon_results: list[TimeSeriesModelResult] = []
            for model_name in available_models:
                model_result = self._train_and_evaluate_one(
                    model_name=model_name,
                    train=split.train,
                    validation=split.validation,
                    train_plus_val=train_plus_val,
                    test=split.test,
                    horizon=horizon,
                    ticker=ticker,
                )
                horizon_results.append(model_result)

            result.model_results[horizon] = horizon_results
            successful = [r for r in horizon_results if r.error is None]
            if successful:
                result.comparison_tables[horizon] = self.comparator.compare(successful, horizon_days=horizon)

        self.logger.info("time series pipeline complete", extra={"ticker": ticker})
        return result

    def _train_and_evaluate_one(
        self,
        *,
        model_name: str,
        train: pd.Series,
        validation: pd.Series,
        train_plus_val: pd.Series,
        test: pd.Series,
        horizon: int,
        ticker: str,
    ) -> TimeSeriesModelResult:
        try:
            # --- Validation phase: fit on train, forecast into validation ---
            val_model = TimeSeriesModelFactory.create(model_name, config=self.config)
            val_model.fit(train)
            val_steps = min(len(validation), max(horizon, self.config.rolling_window))
            val_forecast = val_model.forecast(val_steps)
            val_metrics = evaluate_forecast(
                validation.iloc[:horizon],
                val_forecast.forecast.iloc[:horizon],
                previous_value=float(train.iloc[-1]),
            )

            # --- Test phase: refit on train+validation, forecast into test ---
            test_model = TimeSeriesModelFactory.create(model_name, config=self.config)
            start = perf_counter()
            test_model.fit(train_plus_val)
            fit_time = perf_counter() - start
            test_steps = min(len(test), max(horizon, self.config.rolling_window))
            start = perf_counter()
            test_forecast = test_model.forecast(test_steps)
            predict_time = perf_counter() - start
            test_metrics = evaluate_forecast(
                test.iloc[:horizon],
                test_forecast.forecast.iloc[:horizon],
                previous_value=float(train_plus_val.iloc[-1]),
            )

            if self.persist:
                info = self.storage.save(
                    test_model, ticker=ticker, horizon_days=horizon, test_metrics=test_metrics.to_dict()
                )
                self._last_saved_path = str(info.path)

            return TimeSeriesModelResult(
                model_name=model_name,
                horizon_days=horizon,
                hyperparameters=test_model.get_params(),
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
                training_time_seconds=fit_time,
                prediction_time_seconds=predict_time,
            )
        except Exception as exc:  # noqa: BLE001 — isolate one model's failure from the rest of the run
            self.logger.info(
                "model failed during training/evaluation, skipping",
                extra={"model": model_name, "horizon": horizon, "reason": str(exc)},
            )
            return TimeSeriesModelResult(
                model_name=model_name,
                horizon_days=horizon,
                hyperparameters={},
                validation_metrics=None,
                test_metrics=None,
                training_time_seconds=0.0,
                prediction_time_seconds=0.0,
                error=str(exc),
            )