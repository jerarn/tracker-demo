"""Module to interact with Binance API for price data and transaction history."""

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
import time
from typing import Any
from urllib.parse import urlencode

from binance.client import BinanceAPIException, Client
from dotenv import load_dotenv
import pandas as pd
from pandas.tseries.offsets import DateOffset
import requests

from tracker.config.decorators import log_api_calls, log_calls, log_performance
from tracker.config.logger import get_logger

logger = get_logger(__name__)


class BinanceAPI:
    """Class to interact with Binance API for price data and transaction history."""

    COMMON_QUOTE_ASSETS = (
        "USDT",
        "EUR",
        "BTC",
        "BUSD",
        "ETH",
    )  # order matters - longer first

    def __init__(self, api_key: str | None = None, api_secret: str | None = None):
        """Initialize BinanceAPI with API key and secret. If not provided, read from env vars."""
        load_dotenv()
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        if not (self.api_key and self.api_secret):
            raise RuntimeError(
                "BINANCE_API_KEY and BINANCE_API_SECRET must be set in environment."
            )
        self.client = Client(self.api_key, self.api_secret)
        logger.info("Initialized BinanceManager")

    # ---------- Helpers ----------
    @log_api_calls("Binance")
    def _rate_limited_api_call(
        self, func, *args, max_attempts: int = 4, backoff: float = 1.0, **kwargs
    ):
        """Generic wrapper to retry Binance API calls on rate limit. Exponential backoff.

        Returns the function result or None on failure.
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                return func(*args, **kwargs)
            except BinanceAPIException as e:
                msg = str(e)
                # more advanced detection could inspect attributes
                if "429" in msg or "Too many requests" in msg or "Rate limit" in msg:
                    wait = backoff * (2**attempt)
                    logger.warning(
                        "Rate limit hit, sleeping %.1f seconds (attempt %d/%d)",
                        wait,
                        attempt + 1,
                        max_attempts,
                    )
                    time.sleep(wait)
                    attempt += 1
                    continue
                logger.warning("Binance API error (no retry): %s", e)
                return None
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Unexpected error calling Binance API: %s", e)
                return None
        logger.error(
            "Exceeded max attempts for API call %s",
            getattr(func, "__name__", str(func)),
        )
        return None

    @log_api_calls("Binance")
    def _get_klines(
        self, symbol: str, interval: str, start_ts: int, end_ts: int
    ) -> list | None:
        """Wrapper for client.get_historical_klines with consistent param names."""
        return self._rate_limited_api_call(
            self.client.get_historical_klines,
            symbol,
            interval,
            start_ts,
            end_ts,
        )

    @log_performance(warn_threshold=1.0, error_threshold=5.0)
    def _process_klines_to_df(self, klines: Iterable) -> pd.DataFrame:
        """Convert raw klines list to a DataFrame indexed by UTC timestamp.

        Expects Binance kline format.
        """
        if not klines:
            logger.debug("No klines data to process.")
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            ).astype(float)

        df = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.set_index("open_time", inplace=True)

        logger.debug("Processed %d klines into DataFrame", len(df))
        return df

    @log_calls()
    def _split_symbol(self, symbol: str) -> tuple[str, str]:
        """Split a Binance symbol into base and quote using known common quote assets.

        Falls back to first 3/ - but tries to be robust.
        """
        for q in self.COMMON_QUOTE_ASSETS:
            if symbol.endswith(q):
                base = symbol[: -len(q)]
                return base, q
        # fallback: naive split (may be wrong for exotic symbols)
        return symbol[:-3], symbol[-3:]

    # ---------- Price conversion / fetching ----------
    @log_performance(warn_threshold=2.0, error_threshold=10.0)
    def _calculate_price_klines(
        self,
        klines_pair: Iterable,
        pair_quote: str,
        klines_quote_to_eur: Iterable | None = None,
    ) -> pd.Series | None:
        """From pair klines (price quoted in pair_quote), compute price in EUR.

        pair_quote is like 'EUR', 'USDT' or 'BTC'.
        klines_quote_to_eur is klines converting quote -> EUR (e.g. EURUSDT or BTCEUR).
        Returns a pd.Series indexed by UTC timestamps with float prices (EUR).
        """
        try:
            df_pair = self._process_klines_to_df(klines_pair)
            if pair_quote == "EUR":
                prices = df_pair["close"].astype(float).rename("price_eur")
                logger.debug(f"Direct EUR price for pair_quote {pair_quote}")
                return prices

            if klines_quote_to_eur is None:
                logger.debug(
                    "No quote->EUR klines provided for pair_quote %s", pair_quote
                )
                return None

            df_quote = self._process_klines_to_df(klines_quote_to_eur)
            # align on timestamps using an inner join
            merged = df_pair[["close"]].join(
                df_quote[["close"]], how="inner", lsuffix="_pair", rsuffix="_quote"
            )

            if merged.empty:
                logger.warning(f"No overlapping timestamps for pair_quote {pair_quote}")
                return None

            if pair_quote == "USDT":
                # pair close is in USDT, quote close is EURUSDT (USDT per EUR)
                # => EUR_price = pair_close / EURUSDT
                prices = (
                    merged["close_pair"].astype(float)
                    / merged["close_quote"].astype(float)
                ).rename("price_eur")
            elif pair_quote == "BTC":
                # pair close in BTC, BTCEUR gives EUR per BTC => EUR_price = pair_close * BTCEUR
                prices = (
                    merged["close_pair"].astype(float)
                    * merged["close_quote"].astype(float)
                ).rename("price_eur")
            else:
                logger.debug("Unsupported pair_quote: %s", pair_quote)
                return None

            logger.debug(f"Calculated EUR prices for pair_quote {pair_quote}")
            return prices

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error calculating price klines: %s", e)
            return None

    @log_api_calls("Binance")
    @log_performance(warn_threshold=5.0, error_threshold=30.0)
    def fetch_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = Client.KLINE_INTERVAL_1DAY,
    ) -> dict[datetime, float]:
        """Fetch price history for a symbol from start_date to end_date using given kline interval.

        Returns dict mapping timezone-aware UTC datetime (kline open_time) -> price in EUR (float).
        Strategy: try EUR, then USDT (using EURUSDT), then BTC (using BTCEUR).
        """
        logger.info(
            "Fetching %s price data from %s to %s", symbol, start_date, end_date
        )

        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        # helper to convert pandas Series -> dict of datetimes
        def _series_to_dict(series: pd.Series) -> dict[datetime, float]:
            out: dict[datetime, float] = {}
            for ts, val in series.items():
                out[ts.to_pydatetime()] = float(val) if pd.notna(val) else None
            return out

        pairs_to_try = [
            (f"{symbol}EUR", "EUR"),
            (f"{symbol}USDT", "USDT"),
            (f"{symbol}BTC", "BTC"),
        ]

        for pair, quote in pairs_to_try:
            try:
                logger.info("Trying pair %s to get EUR price", pair)
                klines_pair = self._get_klines(pair, interval, start_ts, end_ts)
                if not klines_pair:
                    logger.debug("No klines returned for pair %s", pair)
                    continue

                if quote == "EUR":
                    series = self._calculate_price_klines(klines_pair, "EUR")
                    if series is not None and not series.empty:
                        result = _series_to_dict(series)
                        logger.info(
                            "Fetched %d price points for %s", len(result), symbol
                        )
                        return result

                elif quote == "USDT":
                    klines_eurusdt = self._get_klines(
                        "EURUSDT", interval, start_ts, end_ts
                    )
                    series = self._calculate_price_klines(
                        klines_pair, "USDT", klines_eurusdt
                    )
                    if series is not None and not series.empty:
                        result = _series_to_dict(series)
                        logger.info(
                            "Fetched %d price points for %s", len(result), symbol
                        )
                        return result

                elif quote == "BTC":
                    klines_btceur = self._get_klines(
                        "BTCEUR", interval, start_ts, end_ts
                    )
                    series = self._calculate_price_klines(
                        klines_pair, "BTC", klines_btceur
                    )
                    if series is not None and not series.empty:
                        result = _series_to_dict(series)
                        logger.info(
                            "Fetched %d price points for %s", len(result), symbol
                        )
                        return result

            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Error fetching price for %s: %s", pair, e)

        logger.warning(
            "Could not find price route for %s in %s - %s", symbol, start_date, end_date
        )
        # return empty dict when no data found
        return {}

    @log_performance(warn_threshold=5.0, error_threshold=30.0)
    def fetch_data_period(
        self,
        symbols: list[str],
        period: str,
        interval: str = Client.KLINE_INTERVAL_1DAY,
    ) -> dict[str, dict[datetime, float]]:
        """Fetch price data.

        Fetch historical price data for a list of symbols using a period string (e.g. '1mo',
        '7 days ago UTC'), and a kline interval. Returns dict mapping symbol -> dict[datetime, float].
        Uses exact timestamps returned by Binance klines.
        """
        logger.info("Fetching data for symbols %s over period %s", symbols, period)
        today = pd.Timestamp.now(tz=timezone.utc)
        if "mo" in period:
            n_months = int(period.replace("mo", ""))
            start_date = today - DateOffset(months=n_months)
        elif "days ago" in period:
            n_days = int(period.split()[0])
            start_date = today - pd.Timedelta(days=n_days)
        else:
            start_date = today

        results: dict[str, dict[datetime, float]] = {}
        for symbol in symbols:
            try:
                prices = self.fetch_data(
                    symbol,
                    start_date.to_pydatetime(),
                    today.to_pydatetime(),
                    interval=interval,
                )
                results[symbol] = prices
                logger.info("Fetched %d data points for %s", len(prices), symbol)

            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error fetching data for %s: %s", symbol, e)
                results[symbol] = []

        logger.info("Completed fetching data for all symbols.")
        return results

    # ---------- BETH staking ----------
    @log_api_calls("Binance")
    def _fetch_beth_staking_rewards(
        self,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch BETH staking rewards via Binance signed endpoint.

        Handles >10 day ranges by batching requests in 10-day intervals.
        Returns list of rows (may be empty).
        """
        logger.debug(
            "Fetching BETH staking rewards from %s to %s", start_time, end_time
        )

        base_url = "https://api.binance.com"
        endpoint = "/sapi/v1/eth-staking/eth/history/rewardsHistory"
        url = base_url + endpoint

        now = int(time.time() * 1000)
        # Default: last 10 days
        start = start_time if start_time is not None else now - 10 * 24 * 3600 * 1000
        end = end_time if end_time is not None else now
        max_days = 10
        ms_per_day = 24 * 3600 * 1000
        batch_size = max_days * ms_per_day

        all_rows = []
        batch_start = start
        while batch_start < end:
            batch_end = min(batch_start + batch_size - 1, end)
            params = {
                "startTime": batch_start,
                "endTime": batch_end,
                "limit": limit,
                "timestamp": now,
            }
            query_string = urlencode(params)
            signature = hmac.new(
                self.api_secret.encode(), query_string.encode(), hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
            headers = {"X-MBX-APIKEY": self.api_key}

            try:
                logger.debug("Sending request to %s with params: %s", url, params)
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                rows = data.get("rows", data if isinstance(data, list) else [])
                all_rows.extend(rows)
                logger.debug(
                    "Fetched %d BETH rewards for batch %s - %s",
                    len(rows),
                    batch_start,
                    batch_end,
                )

            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    "Error fetching BETH staking rewards for %s-%s: %s",
                    batch_start,
                    batch_end,
                    e,
                )
            batch_start = batch_end + 1

        logger.info("Fetched total %d BETH staking rewards", len(all_rows))
        return all_rows

    # ---------- Transactions ----------
    @log_performance(warn_threshold=30.0, error_threshold=60.0)
    def fetch_transactions_history(
        self,
        tx_type: str = "trade",
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch transaction history: 'trade', 'deposit', 'withdraw', 'beth_rewards'.

        Returns a list of model data.
        """
        logger.info(
            "Fetching %s transactions from %s to %s",
            tx_type,
            start_time,
            end_time,
        )

        results: list[dict[str, Any]] = []
        try:
            if tx_type == "trade":
                logger.debug("Fetching trade history")
                exchange_info = self._rate_limited_api_call(
                    self.client.get_exchange_info
                )
                if not exchange_info:
                    logger.error("Could not fetch exchange info for trades")
                    return results

                symbols = [s["symbol"] for s in exchange_info.get("symbols", [])]
                logger.info("Found %d symbols for trade history", len(symbols))

                for symbol in symbols:
                    trades = self._rate_limited_api_call(
                        self.client.get_my_trades,
                        symbol=symbol,
                        startTime=start_time,
                        endTime=end_time,
                    )
                    if not trades:
                        continue
                    for _ in trades:
                        raise NotImplementedError(
                            "Trade conversion not implemented yet"
                        )

            elif tx_type == "deposit":
                logger.debug("Fetching deposit history")
                deposits = self._rate_limited_api_call(
                    self.client.get_deposit_history,
                    startTime=start_time,
                    endTime=end_time,
                )
                if deposits:
                    logger.info("Found %d deposits", len(deposits))
                    for d in deposits:
                        transfer = self._convert_deposit(d)
                        if transfer:
                            results.append(transfer)

            elif tx_type == "withdraw":
                logger.debug("Fetching withdraw history")
                withdrawals = self._rate_limited_api_call(
                    self.client.get_withdraw_history,
                    startTime=start_time,
                    endTime=end_time,
                )
                if withdrawals:
                    logger.info("Found %d withdrawals", len(withdrawals))
                    for _ in withdrawals:
                        raise NotImplementedError(
                            "Withdraw conversion not implemented yet"
                        )

            elif tx_type == "beth_rewards":
                logger.debug("Fetching BETH staking rewards")
                rewards = self._fetch_beth_staking_rewards(
                    start_time=start_time, end_time=end_time
                )
                if rewards:
                    logger.info("Found %d BETH rewards", len(rewards))
                    for r in rewards:
                        cf = self._convert_beth_reward(r)
                        if cf:
                            results.append(cf)

            else:
                logger.warning("Unsupported transaction type: %s", tx_type)

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error fetching %s history: %s", tx_type, e)

        logger.info("Fetched %d %s transactions.", len(results), tx_type)
        return results

    # ---------- Converters ----------
    @log_calls(log_result=False)
    def _convert_deposit(self, deposit: dict) -> dict[str, Any] | None:
        try:
            date = datetime.fromtimestamp(deposit["insertTime"] / 1000, tz=timezone.utc)
            account_from = None
            portfolio_from = None
            account_to = "Binance"
            portfolio_to = "Spot"
            quantity = float(deposit.get("amount", 0))
            currency = "EUR"
            description = f"Deposit {deposit.get('txId')}"
            coin = deposit.get("coin")
            ticker = coin + "-EUR"

            logger.debug("Converting deposit for coin %s", coin)
            price = self.fetch_data(
                coin, date, date + timedelta(minutes=1), Client.KLINE_INTERVAL_1SECOND
            ).get(date)

            return {
                "date": date,
                "account_from": account_from,
                "portfolio_from": portfolio_from,
                "account_to": account_to,
                "portfolio_to": portfolio_to,
                "quantity": quantity,
                "currency": currency,
                "description": description,
                "instrument": ticker,
                "price": price,
            }
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error converting deposit: %s", e)
            return None

    @log_calls(log_result=False)
    def _convert_beth_reward(self, reward: dict) -> dict[str, Any] | None:
        try:
            # reward timestamp is in ms
            date = datetime.fromtimestamp(reward["time"] / 1000, tz=timezone.utc)
            account = "Binance"
            portfolio = "Spot"
            tx_type = "INTEREST"
            quantity = float(reward.get("amount", 0))
            ticker = "BETH-EUR"

            logger.debug("Converting BETH reward at %s", date)
            prices = self.fetch_data(
                "ETH", date, date + timedelta(minutes=1), Client.KLINE_INTERVAL_1SECOND
            )
            price = prices.get(date)
            description = "ETH 2.0 Staking Rewards"
            return {
                "date": date,
                "account": account,
                "portfolio": portfolio,
                "instrument": ticker,
                "tx_type": tx_type,
                "quantity": quantity,
                "price": price,
                "currency": "EUR",
                "description": description,
            }

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error converting BETH reward: %s", e)
            return None
