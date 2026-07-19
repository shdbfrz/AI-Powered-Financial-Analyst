"""
Factory / Registry for `ai/models/time_series/` models.

Mirrors `ai.models.ml.models.registry.ModelFactory`: a name -> constructor
mapping, plus `available_models()` which probes optional-dependency models
(auto_arima, prophet) and silently excludes any that raise
`ModelUnavailableError` — this is how the pipeline stays runnable on a
machine where `pmdarima` or `prophet` failed to install (same graceful-skip
contract Sprint 3 established for LightGBM/CatBoost).
"""

from typing import Callable

from ai.models.time_series.config import DEFAULT_TS_CONFIG, TimeSeriesConfig
from ai.models.time_series.exceptions import ModelUnavailableError, UnknownModelError
from ai.models.time_series.models.arima_model import ArimaModel
from ai.models.time_series.models.auto_arima_model import AutoArimaModel
from ai.models.time_series.models.base import BaseTimeSeriesModel
from ai.models.time_series.models.exponential_smoothing_model import ExponentialSmoothingModel
from ai.models.time_series.models.prophet_model import ProphetModel
from ai.models.time_series.models.sarima_model import SarimaModel
from ai.utils.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_MODELS: dict[str, Callable[..., BaseTimeSeriesModel]] = {
    "arima": ArimaModel,
    "sarima": SarimaModel,
    "exponential_smoothing": ExponentialSmoothingModel,
}

# Constructors that may raise ModelUnavailableError if their optional
# package isn't installed.
_OPTIONAL_MODELS: dict[str, Callable[..., BaseTimeSeriesModel]] = {
    "auto_arima": AutoArimaModel,
    "prophet": ProphetModel,
}

_ALL_MODELS: dict[str, Callable[..., BaseTimeSeriesModel]] = {**_REQUIRED_MODELS, **_OPTIONAL_MODELS}


class TimeSeriesModelFactory:
    """Creates model instances by name; reports which optional models are
    actually usable in the current environment."""

    @staticmethod
    def create(model_name: str, config: TimeSeriesConfig = DEFAULT_TS_CONFIG, **kwargs) -> BaseTimeSeriesModel:
        if model_name not in _ALL_MODELS:
            raise UnknownModelError(model_name, available=list(_ALL_MODELS))
        return _ALL_MODELS[model_name](config=config, **kwargs)

    @staticmethod
    def available_models(config: TimeSeriesConfig = DEFAULT_TS_CONFIG) -> list[str]:
        """Every model name that can actually be instantiated right now —
        required models always included, optional models included only if
        their dependency imported successfully."""
        available = list(_REQUIRED_MODELS)
        for name, constructor in _OPTIONAL_MODELS.items():
            try:
                constructor(config=config)
                available.append(name)
            except ModelUnavailableError as exc:
                logger.info(
                    "optional model unavailable, skipping",
                    extra={"model": name, "missing_package": exc.package},
                )
        return available

    @staticmethod
    def registered_model_names() -> list[str]:
        """Every name the factory knows about, regardless of availability."""
        return list(_ALL_MODELS)