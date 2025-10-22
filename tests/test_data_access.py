"""Tests for external data access managers.

These tests verify integration with external APIs.
"""

from datetime import datetime
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from tracker import BinanceAPI, MorningstarAPI, YFinanceAPI


class TestYFinanceAPI(unittest.TestCase):
    """Test Yahoo Finance API functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.yf_api = YFinanceAPI()

    @patch("yfinance.download")
    def test_fetch_period_single_ticker(self, mock_download):
        """Test fetching data for a single ticker."""
        # Create sample data
        dates = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        data = pd.DataFrame({"Close": [150.0, 151.0]}, index=dates)
        mock_download.return_value = data

        # Test fetch
        result = self.yf_api.fetch_period("AAPL", "1mo", datatype="close")

        # Verify API was called correctly
        mock_download.assert_called_once_with(
            "AAPL", period="1mo", interval="1d", auto_adjust=True, progress=False
        )

        # Verify result structure
        self.assertIn("AAPL", result)
        self.assertIsInstance(result["AAPL"], list)
        self.assertEqual(len(result["AAPL"]), 2)

        # Verify result contents
        for item in result["AAPL"]:
            self.assertIsInstance(item, dict)
            self.assertEqual(item["instrument"], "AAPL")
            self.assertEqual(item["data_source"], "yfinance")
            self.assertEqual(item["data_type"], "close")

    @patch("yfinance.download")
    def test_fetch_period_multiple_tickers(self, mock_download):
        """Test fetching data for multiple tickers."""

        def mock_download_side_effect(ticker, _period=None, _interval=None):
            dates = [datetime(2024, 1, 1)]
            price = 150.0 if ticker == "AAPL" else 120.0
            return pd.DataFrame({"Close": [price]}, index=dates)

        mock_download.side_effect = mock_download_side_effect

        # Test fetch for multiple tickers
        result = self.yf_api.fetch_period(["AAPL", "GOOGL"], "1mo", datatype="close")

        # Verify both tickers in result
        self.assertIn("AAPL", result)
        self.assertIn("GOOGL", result)

        # Verify API was called for each ticker
        self.assertEqual(mock_download.call_count, 2)

    @patch("yfinance.download")
    def test_fetch_period_empty_data(self, mock_download):
        """Test handling of empty data response."""
        # Mock empty DataFrame
        mock_download.return_value = pd.DataFrame()

        # Test fetch
        result = self.yf_api.fetch_period("INVALID", "1mo")

        # Verify empty result
        self.assertIn("INVALID", result)
        self.assertEqual(len(result["INVALID"]), 0)

    @patch("yfinance.download")
    def test_fetch_period_exception_handling(self, mock_download):
        """Test exception handling during data fetch."""
        # Mock exception
        mock_download.side_effect = Exception("Network error")

        # Test fetch
        result = self.yf_api.fetch_period("AAPL", "1mo")

        # Verify error handling
        self.assertIn("AAPL", result)
        self.assertEqual(len(result["AAPL"]), 0)


class TestBinanceAPI(unittest.TestCase):
    """Test Binance API functionality."""

    @patch.dict(
        "os.environ",
        {"BINANCE_API_KEY": "test_key", "BINANCE_API_SECRET": "test_secret"},
    )
    @patch("tracker.data.access.binance_api.Client")
    def test_binance_api_initialization(self, mock_client):
        """Test Binance API initialization."""
        manager = BinanceAPI()

        # Verify client was created with correct credentials
        mock_client.assert_called_once_with("test_key", "test_secret")
        self.assertEqual(manager.api_key, "test_key")
        self.assertEqual(manager.api_secret, "test_secret")

    @patch("tracker.data.access.binance_api.load_dotenv")
    def test_binance_api_missing_credentials(self, mock_load_dotenv):
        """Test Binance API with missing credentials."""
        # Mock load_dotenv to do nothing
        mock_load_dotenv.return_value = None

        with patch.dict("os.environ", {}, clear=True), pytest.raises(RuntimeError):
            BinanceAPI()


class TestMorningstarAPI(unittest.TestCase):
    """Test Morningstar API functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.ms_api = MorningstarAPI()

    def test_find_morningstar_id_success(self):
        """Test successful Morningstar ID lookup."""
        # Mock the session get method on the instance
        with patch.object(self.ms_api.session, "get") as mock_get:
            # Mock successful response
            mock_response = Mock()
            mock_response.text = 'prefix|{"i":"test_id","name":"Test Fund"}|suffix'
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Test ID lookup
            result = self.ms_api.find_morningstar_id("FR0000000000")

            # Verify result
            self.assertEqual(result, "test_id")
            mock_get.assert_called_once()

    def test_find_morningstar_id_failure(self):
        """Test failed Morningstar ID lookup."""
        # Mock the session get method on the instance to raise a requests exception
        with patch.object(self.ms_api.session, "get") as mock_get:
            mock_get.side_effect = requests.RequestException("Network error")

            # Test ID lookup - should return None when exception occurs
            result = self.ms_api.find_morningstar_id("INVALID")

            # Verify failure handling
            self.assertIsNone(result)

    @patch.object(MorningstarAPI, "find_morningstar_id")
    def test_fetch_nav_data_success(self, mock_find_id):
        """Test successful NAV data fetch."""
        # Mock ID lookup
        mock_find_id.return_value = "test_morningstar_id"

        # Mock the session get method on the instance
        with patch.object(self.ms_api.session, "get") as mock_get:
            # Mock NAV data response
            mock_response = Mock()
            mock_response.json.return_value = {
                "TimeSeries": {
                    "Security": [
                        {
                            "HistoryDetail": [
                                {"EndDate": "2024-01-01", "Value": 100.0},
                                {"EndDate": "2024-01-02", "Value": 101.0},
                            ]
                        }
                    ]
                }
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Test data fetch
            result = self.ms_api.fetch_nav_data("TEST", "FR0000000000", "EUR", "1mo")

            # Verify result
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)

            # Verify result contents
            for item in result:
                self.assertIsInstance(item, dict)
                self.assertEqual(item["instrument"], "TEST")
                self.assertEqual(item["data_source"], "morning star")

    @patch.object(MorningstarAPI, "find_morningstar_id")
    def test_fetch_nav_data_no_id(self, mock_find_id):
        """Test NAV data fetch when ID lookup fails."""
        # Mock failed ID lookup
        mock_find_id.return_value = None

        # Test data fetch
        result = self.ms_api.fetch_nav_data("TEST", "INVALID", "EUR", "1mo")

        # Verify empty result
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
