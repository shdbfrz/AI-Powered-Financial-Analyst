"""
`ForecastService` — the single entry point Sprint 5 (Deep Learning), Sprint 7
(Decision Support Engine), and Sprint 8 (Backend API) are expected to call
instead of touching any individual model class directly (Facade pattern,
same role Sprint 3's `ai.models.ml.prediction.inference.InferenceService`
plays for the ML pipeline).

    from ai.models.time_series.prediction.forecast_service import ForecastService

    response = ForecastService().forecast(model="arima", ticker="AAPL", horizon=5)

Design decision — always fit fresh by default: unlike Sprint 3's
`InferenceService` (which loads a persisted model trained once on a fixed
split), a forecast consumed by a live decision must reflect the *current*
close price, not whatever the training split's `test` cutoff happened to be
weeks ago. So `forecast()` fits each model on the full available series by
default (`use_cached_artifact=False`); set `use_cached_artifact=True` to
instead load the most recent artifact `ai.models.time_series.pipelines
.training_pipeline.TimeSeriesTrainingPipeline` already saved (fast path, at
the cost of staleness) — the two modes are documented on the flag itself,
not hidden behind a guess.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.models.time_series.data_loader import load_price_series
from ai.models.time_series.exceptions import ModelPersistenceError
from ai.models.time_series.models.base import ForecastResult
from ai.models.time_series.models.registry import TimeSeriesModelFactory
from ai.models.time_series.persistence.model_storage import TimeSeriesModelStorage
from ai.utils.logger import get_logger

logger = get_logger(__name__)

# ARCHITECTURE.md §2.4 / SRS NFR-11: every prediction-bearing response must
# carry this disclaimer at the schema level, not just in the UI.
EDUCATIONAL_DISCLAIMER = (
    "This forecast is generated for educational and research purposes only. "
    "It is not financial advice and must not be used as the sole basis for "
    "any investment decision."
)


@dataclass
class ForecastPoint:
    date: str
    value: float
    lower_bound: Optional[float]
    upper_bound: Optional[float]


@dataclass
class StandardForecastResponse:
    """The one shape every consumer (Decision Support Engine, backend API,
    Deep Learning ensembling) receives, regardless of which underlying model
    produced it."""

    ticker: str
    model_name: str
    horizon_days: int
    generated_at: str
    confidence_level: float
    points: list[ForecastPoint]
    hyperparameters: dict
    fit_time_seconds: float
    forecast_time_seconds: float
    educational_disclaimer: str = EDUCATIONAL_DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "model_name": self.model_name,
            "horizon_days": self.horizon_days,
            "generated_at": self.generated_at,
            "confidence_level": self.confidence_level,
            "points": [p.__dict__ for p in self.points],
            "hyperparameters": self.hyperparameters,
            "fit_time_seconds": round(self.fit_time_seconds, 4),
            "forecast_time_seconds": round(self.forecast_time_seconds, 4),
            "educational_disclaimer": self.educational_disclaimer,
        }


class ForecastService:
    """Unified forecasting facade. One call, one standardized response,
    regardless of ARIMA/SARIMA/Auto-ARIMA/Prophet/Exponential-Smoothing
    underneath.
    """

    def __init__(self, config: TimeSeriesConfig = DEFAULT_TS_CONFIG):
        self.config = config
        self.storage = TimeSeriesModelStorage()
        self.logger = logger

    def forecast(
        self,
        model: str,
        ticker: str,
        horizon: int,
        *,
        use_cached_artifact: bool = False,
        persist: bool = False,
        model_kwargs: Optional[dict] = None,
    ) -> StandardForecastResponse:
        """Generate a `horizon`-day-ahead forecast for `ticker` using
        `model` (one of `TimeSeriesModelFactory.available_models()`).

        Args:
            use_cached_artifact: load the latest persisted artifact instead
                of fitting fresh (faster, may be stale — see module docstring).
            persist: save the freshly-fitted model artifact after forecasting
                (ignored if `use_cached_artifact=True`).
            model_kwargs: extra constructor kwargs forwarded to the model
                (e.g. an explicit ARIMA `order=(2, 1, 2)`).
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

        series = load_price_series(ticker, config=self.config)

        if use_cached_artifact:
            fitted_model = self._load_latest_artifact(ticker, model, horizon)
        else:
            fitted_model = TimeSeriesModelFactory.create(model, config=self.config, **(model_kwargs or {}))
            fitted_model.fit(series)
            if persist:
                # Test metrics aren't available for a fresh production fit
                # (there's no held-out future to score against yet) — the
                # artifact is saved for reuse, not for comparison ranking.
                self.storage.save(fitted_model, ticker=ticker, horizon_days=horizon, test_metrics={})

        forecast_result = fitted_model.forecast(horizon)
        return self._to_standard_response(ticker, forecast_result)

    def forecast_all_models(
        self, ticker: str, horizon: int, *, models: Optional[list[str]] = None
    ) -> dict[str, StandardForecastResponse]:
        """Convenience for the Decision Support Engine / ensembling: forecast
        with every available model and return `{model_name: response}`,
        skipping (and logging) any model that fails rather than aborting the
        whole batch.
        """
        model_names = models or TimeSeriesModelFactory.available_models(self.config)
        responses = {}
        for name in model_names:
            try:
                responses[name] = self.forecast(name, ticker, horizon)
            except Exception as exc:  # noqa: BLE001
                self.logger.info(
                    "model failed in forecast_all_models, skipping",
                    extra={"model": name, "ticker": ticker, "reason": str(exc)},
                )
        return responses

    def _load_latest_artifact(self, ticker: str, model_name: str, horizon: int):
        import json

        from ai.utils.config import settings

        metadata_path = settings.resolve(settings.models_dir, "time_series", "Model_Metadata.json")
        if not metadata_path.exists():
            raise ModelPersistenceError(
                f"no cached artifacts found — Model_Metadata.json does not exist at '{metadata_path}'"
            )
        entries = json.loads(metadata_path.read_text())
        matches = [
            e for e in entries
            if e["ticker"] == ticker and e["model_name"] == model_name and e["horizon_days"] == horizon
        ]
        if not matches:
            raise ModelPersistenceError(
                f"no cached artifact for ticker='{ticker}', model='{model_name}', horizon={horizon}"
            )
        latest = max(matches, key=lambda e: e["version"])
        from pathlib import Path

        return self.storage.load(Path(latest["path"]))

    @staticmethod
    def _to_standard_response(ticker: str, result: ForecastResult) -> StandardForecastResponse:
        points = []
        for date, value in result.forecast.items():
            lower = float(result.lower_bound.loc[date]) if result.lower_bound is not None else None
            upper = float(result.upper_bound.loc[date]) if result.upper_bound is not None else None
            points.append(ForecastPoint(date=str(date.date()) if hasattr(date, "date") else str(date), value=float(value), lower_bound=lower, upper_bound=upper))

        return StandardForecastResponse(
            ticker=ticker,
            model_name=result.model_name,
            horizon_days=len(points),
            generated_at=datetime.now(timezone.utc).isoformat(),
            confidence_level=result.confidence_level,
            points=points,
            hyperparameters=result.params,
            fit_time_seconds=result.fit_time_seconds,
            forecast_time_seconds=result.forecast_time_seconds,
        )