"""
`ai/models/time_series/` — Sprint 4: classical statistical forecasting
(ARIMA, SARIMA, Auto ARIMA, Prophet, Exponential Smoothing) over the Close
price series produced by Sprint 2's feature engineering pipeline.

Primary entry points:
    - `ai.models.time_series.pipelines.training_pipeline.TimeSeriesTrainingPipeline`
      — train + evaluate every available model for a ticker.
    - `ai.models.time_series.prediction.forecast_service.ForecastService`
      — unified `forecast(model, ticker, horizon)` facade for Sprint 5/7/8.
"""