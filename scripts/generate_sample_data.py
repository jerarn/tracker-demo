#!/usr/bin/env python3
"""Generate simple sample data for portfolio tracker demonstration."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import random

import numpy as np
import pandas as pd

from tracker import YFinanceAPI

# Set seed for reproducible sample data
random.seed(42)
np.random.seed(42)

# Minimal instrument set
INSTRUMENTS = [
    {
        "ticker": "SPY",
        "name": "SPDR S&P 500 ETF",
        "isin": "US78462F1030",
        "asset_class": "ETF",
        "currency": "USD",
        "description": "S&P 500",
    },
    {
        "ticker": "QQQ",
        "name": "Invesco QQQ Trust",
        "isin": "US46090E1038",
        "asset_class": "ETF",
        "currency": "USD",
        "description": "NASDAQ 100",
    },
    {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "isin": "US0378331005",
        "asset_class": "Equity",
        "currency": "USD",
        "description": "Apple Stock",
    },
    {
        "ticker": "BTC-USD",
        "name": "Bitcoin",
        "isin": "BTCUSD000000",
        "asset_class": "Crypto",
        "currency": "USD",
        "description": "Bitcoin",
    },
]

# Simple account structure
ACCOUNTS = [
    {"name": "Demo Account", "currency": "USD"},
]

# Two clear portfolio strategies
PORTFOLIOS = [
    {"name": "Conservative", "account": "Demo Account", "currency": "USD"},
    {"name": "Aggressive", "account": "Demo Account", "currency": "USD"},
]


def fetch_market_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch market data for the given date range."""
    api = YFinanceAPI()
    tickers = [inst["ticker"] for inst in INSTRUMENTS]
    data = api.fetch_data(
        tickers,
        start_date,
        end_date,
        interval="1d",
        datatype="close",
    )
    return pd.concat([pd.DataFrame(data[ticker]) for ticker in data], ignore_index=True)


def generate_simple_transactions(
    start_date: datetime, end_date: datetime, market_data: pd.DataFrame
) -> pd.DataFrame:
    """Generate simple, predictable transaction patterns."""
    transactions = []

    # Simple portfolio allocations
    conservative_portfolio = [
        ("SPY", 10_000),  # $10k in S&P 500
        ("QQQ", 5_000),  # $5k in Tech
    ]

    aggressive_portfolio = [
        ("AAPL", 8_000),  # $8k in Apple
        ("BTC-USD", 2_000),  # $2k in Bitcoin
    ]

    # Conservative Portfolio - Simple buy and hold
    for ticker, target_amount in conservative_portfolio:
        # Initial purchase on day 2
        purchase_date = start_date + timedelta(days=1)

        price_data = market_data[
            (market_data["instrument"] == ticker)
            & (market_data["date"] <= purchase_date)
        ].sort_values("date")

        if len(price_data) > 0:
            price = price_data.iloc[-1]["quote"]
            fee = 0.001 * target_amount
            quantity = (target_amount - fee) / price

            transactions.append(
                {
                    "date": purchase_date,
                    "account": "Demo Account",
                    "portfolio": "Conservative",
                    "tx_type": "BUY",
                    "instrument": ticker,
                    "quantity": quantity,
                    "price": price,
                    "currency": "USD",
                    "description": f"Initial {ticker} purchase",
                    "fee_amount": fee,
                    "fee_ticker": None,
                    "fee_price": None,
                }
            )

    # Aggressive Portfolio - Initial purchase + some trading
    for ticker, target_amount in aggressive_portfolio:
        # Initial purchase on day 3
        purchase_date = start_date + timedelta(days=2)

        price_data = market_data[
            (market_data["instrument"] == ticker)
            & (market_data["date"] <= purchase_date)
        ].sort_values("date")

        if len(price_data) > 0:
            price = price_data.iloc[-1]["quote"]
            fee = 0.001 * target_amount
            quantity = (target_amount - fee) / price

            transactions.append(
                {
                    "date": purchase_date,
                    "account": "Demo Account",
                    "portfolio": "Aggressive",
                    "tx_type": "BUY",
                    "instrument": ticker,
                    "quantity": quantity,
                    "price": price,
                    "currency": "USD",
                    "description": f"Initial {ticker} purchase",
                    "fee_amount": fee,
                    "fee_ticker": None,
                    "fee_price": None,
                }
            )

    # Add one rebalancing trade for aggressive portfolio
    mid_date = start_date + (end_date - start_date) / 2

    # Sell some AAPL, buy more BTC
    apple_price_data = market_data[
        (market_data["instrument"] == "AAPL") & (market_data["date"] <= mid_date)
    ].sort_values("date")

    btc_price_data = market_data[
        (market_data["instrument"] == "BTC-USD") & (market_data["date"] <= mid_date)
    ].sort_values("date")

    if len(apple_price_data) > 0 and len(btc_price_data) > 0:
        apple_price = apple_price_data.iloc[-1]["quote"]
        btc_price = btc_price_data.iloc[-1]["quote"]

        # Sell $1000 worth of AAPL
        fee = 0.001 * 1_000
        sell_quantity = (1000 - fee) / apple_price
        transactions.append(
            {
                "date": mid_date,
                "account": "Demo Account",
                "portfolio": "Aggressive",
                "tx_type": "SELL",
                "instrument": "AAPL",
                "quantity": sell_quantity,
                "price": apple_price,
                "currency": "USD",
                "description": "Rebalancing - reduce AAPL",
                "fee_amount": fee,
                "fee_ticker": None,
                "fee_price": None,
            }
        )

        # Buy $950 worth of BTC (after fees)
        fee = 0.001 * 950
        btc_quantity = (950 - fee) / btc_price
        transactions.append(
            {
                "date": mid_date,
                "account": "Demo Account",
                "portfolio": "Aggressive",
                "tx_type": "BUY",
                "instrument": "BTC-USD",
                "quantity": btc_quantity,
                "price": btc_price,
                "currency": "USD",
                "description": "Rebalancing - increase BTC",
                "fee_amount": fee,
                "fee_ticker": None,
                "fee_price": None,
            }
        )

    return pd.DataFrame(transactions)


def generate_simple_cash_flows(
    start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """Generate simple cash flows."""
    cash_flows = []

    # Initial deposits for each portfolio
    cash_flows.extend(
        [
            {
                "date": start_date,
                "account": "Demo Account",
                "portfolio": "Conservative",
                "cf_type": "DEPOSIT",
                "amount": 15_000,
                "currency": "USD",
                "description": "Initial conservative portfolio funding",
            },
            {
                "date": start_date + timedelta(days=1),
                "account": "Demo Account",
                "portfolio": "Aggressive",
                "cf_type": "DEPOSIT",
                "amount": 10_000,
                "currency": "USD",
                "description": "Initial aggressive portfolio funding",
            },
        ]
    )

    # Add a few dividends (quarterly)
    current_date = start_date + timedelta(days=90)  # First dividend after 3 months

    while current_date <= end_date:
        if (current_date - start_date).days >= 90:  # Only after 3 months
            cash_flows.append(
                {
                    "date": current_date,
                    "account": "Demo Account",
                    "portfolio": "Aggressive",
                    "cf_type": "DIVIDEND",
                    "amount": 0.005 / 4 * 7_000,
                    "currency": "USD",
                    "description": "Quarterly Apple dividends",
                }
            )
        current_date += timedelta(days=90)

    return pd.DataFrame(cash_flows)


def main():
    """Generate simplified sample data for demo."""
    print("Generating simplified sample data for portfolio tracker demo...")

    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 12, 31, tzinfo=timezone.utc)

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Generate simplified datasets
    print("Generating accounts...")
    accounts_df = pd.DataFrame(ACCOUNTS)
    accounts_df.to_csv(data_dir / "accounts.csv", index=False)

    print("Generating portfolios...")
    portfolios_df = pd.DataFrame(PORTFOLIOS)
    portfolios_df.to_csv(data_dir / "portfolios.csv", index=False)

    print("Generating instruments...")
    instruments_df = pd.DataFrame(INSTRUMENTS)
    instruments_df.to_csv(data_dir / "instruments.csv", index=False)

    print("Fetching market data...")
    market_data = fetch_market_data(start_date, end_date)
    market_data.to_csv(data_dir / "market_data_daily.csv", index=False)

    print("Generating transactions...")
    transactions = generate_simple_transactions(start_date, end_date, market_data)
    transactions.to_csv(data_dir / "transactions.csv", index=False)

    print("Generating cash flows...")
    cash_flows = generate_simple_cash_flows(start_date, end_date)
    cash_flows.to_csv(data_dir / "cash_flows.csv", index=False)

    # Summary with portfolio breakdown
    print(f"\n{'=' * 50}")
    print("DEMO DATA SUMMARY")
    print(f"{'=' * 50}")
    print(
        f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"Duration: {(end_date - start_date).days} days")
    print()
    print("📊 Data Generated:")
    print(f"  • {len(accounts_df)} account")
    print(f"  • {len(portfolios_df)} portfolios")
    print(f"  • {len(instruments_df)} instruments")
    print(f"  • {len(market_data):,} market data points")
    print(f"  • {len(transactions)} transactions")
    print(f"  • {len(cash_flows)} cash flows")
    print()
    print("💼 Portfolio Strategies:")
    print("  • Conservative: $15k → SPY ($10k) + QQQ ($5k) [Buy & Hold]")
    print("  • Aggressive: $10k → AAPL ($8k) + BTC ($2k) [+ Rebalancing]")
    print()
    print(f"📁 Data saved to: {data_dir.absolute()}")


if __name__ == "__main__":
    main()
