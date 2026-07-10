"""
Adapter interface for market-data providers.

Every concrete provider (Yahoo Finance, Alpha Vantage, Polygon, Twelve Data)
implements this interface so the rest of the pipeline never depends on a
specific vendor's API shape. Default provider is Yahoo Finance (free tier,
no API key required) — set DATA_PROVIDER in .env to switch.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseDataProvider(ABC):
    @abstractmethod
    def get_historical_ohlcv(self, ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        """Return a DataFrame with columns: date, open, high, low, close, volume."""
        raise NotImplementedError

    @abstractmethod
    def get_company_info(self, ticker: str) -> dict:
        """Return basic company/fundamental info for the ticker."""
        raise NotImplementedError
