"""Module to interact with Yahoo Finance API to fetch market data."""

import yfinance as yf

from tracker.config.decorators import log_api_calls
from tracker.config.logger import get_logger

logger = get_logger(__name__)


class YFinanceAPI:
    """Class to interact with Yahoo Finance API to fetch market data."""

    @log_api_calls("YFinance")
    def fetch_data(self, tickers, start_date, end_date, interval="1d", datatype=None):
        """Fetch market data from Yahoo Finance for a list of tickers over a specified date range.

        Returns a dict mapping ticker to list of model data.
        """
        logger.info(
            "Fetching data from Yahoo Finance from %s to %s for tickers: %s",
            start_date,
            end_date,
            tickers,
        )
        if isinstance(tickers, str):
            tickers = [tickers]

        results = {}
        for ticker in tickers:
            try:
                logger.info("Fetching data for ticker: %s", ticker)
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
                if data.empty:
                    logger.warning("No data found for ticker: %s", ticker)
                    results[ticker] = []
                    continue

                market_data_list = []
                if datatype.lower() == "close":
                    data = data[["Close"]]
                elif datatype is not None:
                    logger.warning("Unsupported datatype: %s", datatype)

                market_data_list = [
                    {
                        "instrument": ticker,
                        "date": row[0].tz_localize("UTC"),
                        "data_type": "close",
                        "data_source": "yfinance",
                        "quote": float(row[1]),
                    }
                    for col in data.columns
                    for row in data[col].to_frame().itertuples()
                ]

                results[ticker] = market_data_list
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error fetching data for %s: %s", ticker, e)
                results[ticker] = []
        return results

    @log_api_calls("YFinance")
    def fetch_period(self, tickers, period, interval="1d", datatype=None):
        """Fetch market data from Yahoo Finance for a list of tickers over a specified period.

        Returns a dict mapping ticker to list of model data.
        """
        logger.info(
            "Fetching data from Yahoo Finance over period %s for tickers: %s",
            period,
            tickers,
        )
        if isinstance(tickers, str):
            tickers = [tickers]

        results = {}
        for ticker in tickers:
            try:
                logger.info("Fetching data for ticker: %s", ticker)
                data = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
                if data.empty:
                    logger.warning("No data found for ticker: %s", ticker)
                    results[ticker] = []
                    continue

                market_data_list = []
                if datatype.lower() == "close":
                    data = data[["Close"]]
                elif datatype is not None:
                    logger.warning("Unsupported datatype: %s", datatype)

                market_data_list = [
                    {
                        "instrument": ticker,
                        "date": row[0].tz_localize("UTC"),
                        "data_type": "close",
                        "data_source": "yfinance",
                        "quote": float(row[1]),
                    }
                    for col in data.columns
                    for row in data[col].to_frame().itertuples()
                ]

                results[ticker] = market_data_list
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error fetching data for %s: %s", ticker, e)
                results[ticker] = []
        return results
