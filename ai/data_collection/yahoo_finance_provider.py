"""Default data provider: Yahoo Finance via the `yfinance` package (free, no API key)."""

import pandas as pd
import yfinance as yf

from ai.data_collection.base_provider import BaseDataProvider


class YahooFinanceProvider(BaseDataProvider):
    def get_historical_ohlcv(self, ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        df = df.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        return df[["date", "open", "high", "low", "close", "volume"]]

    def get_company_info(self, ticker: str) -> dict:
        info = yf.Ticker(ticker).info
        return {
            "symbol": ticker,
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
        }
