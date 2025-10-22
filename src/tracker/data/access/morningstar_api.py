"""Module to interact with the Morningstar API to fetch fund data."""

from datetime import datetime, timedelta, timezone
import json
import secrets
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tracker.config.decorators import log_api_calls, log_performance
from tracker.config.logger import get_logger

logger = get_logger(__name__)


class MorningstarAPI:
    """Class to interact with the Morningstar API to fetch fund data."""

    def __init__(self):
        """Initialize the MorningstarAPI with a requests session and retry strategy."""
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        logger.info("Initialized MorningstarManager with retry strategy")

    @log_api_calls("Morningstar")
    @log_performance(warn_threshold=5.0, error_threshold=20.0)
    def find_morningstar_id(self, isin: str) -> str | None:
        """Find the Morningstar ID for a given ISIN."""
        logger.info("Finding Morningstar ID for ISIN: %s", isin)
        url = f"https://www.morningstar.fr/fr/util/SecuritySearch.ashx?q={isin}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.morningstar.fr/fr/screener/fund.aspx",
        }
        try:
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            parts = response.text.split("|")
            if len(parts) < 3:
                return None
            data = json.loads(parts[1])
            morningstar_id = data.get("i")
            logger.info("Found Morningstar ID %s for ISIN %s", morningstar_id, isin)
            return morningstar_id
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error("Error finding Morningstar ID for %s: %s", isin, str(e))
            return None

    @log_api_calls("Morningstar")
    @log_performance(warn_threshold=10.0, error_threshold=30.0)
    def fetch_nav_data(
        self, ticker: str, isin: str, currency: str = "EUR", period: str = "1mo"
    ) -> list[dict[str, Any]]:
        """Fetch historical NAV data for a fund using ISIN and return a list of model data.

        Period is a string like '1mo', '1y', '30d', etc.
        """
        logger.info("Fetching NAV data over period %s for ISIN: %s", period, isin)
        morningstar_id = self.find_morningstar_id(isin)
        if not morningstar_id:
            logger.error("Could not find Morningstar ID for %s", isin)
            return []

        # Calculate start and end dates from period
        end_date = datetime.now(timezone.utc)
        if period.endswith("mo"):
            months = int(period[:-2])
            start_date = end_date - timedelta(days=months * 30)
        elif period.endswith("y"):
            years = int(period[:-1])
            start_date = end_date - timedelta(days=years * 365)
        elif period.endswith("d"):
            days = int(period[:-1])
            start_date = end_date - timedelta(days=days)
        else:
            start_date = end_date - timedelta(days=30)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        url = (
            "https://tools.morningstar.fr/api/rest.svc/timeseries_price/"
            f"ok91jeenoo?id={morningstar_id}%5D2%5D1%5D&currencyId={currency}&idtype=Morningstar"
            f"&frequency=daily&startDate={start_date_str}&endDate={end_date_str}&outputType=JSON"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.morningstar.fr/",
        }
        time.sleep(secrets.randbelow(30) / 10 + 2)
        try:
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            price_data = (
                data.get("TimeSeries", {})
                .get("Security", [{}])[0]
                .get("HistoryDetail", [])
            )
            if not price_data:
                logger.warning("No NAV data returned for %s", morningstar_id)
                return []

            logger.info("Fetched %d NAV data points for ISIN %s", len(price_data), isin)
            return [
                {
                    "instrument": ticker,
                    "date": datetime.strptime(item["EndDate"], "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    ),
                    "data_type": "close",
                    "data_source": "morning star",
                    "quote": float(item["Value"]),
                }
                for item in price_data
            ]
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error("Error fetching NAV data for %s: %s", isin, str(e))
            return []
