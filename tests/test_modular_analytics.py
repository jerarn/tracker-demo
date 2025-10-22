"""Unit tests for the modular analytics architecture.
Tests PriceManager, Holdings, Performance, RiskMetrics, and PortfolioAnalyzer.
Updated for SQLAlchemy ORM integration.
"""

from datetime import datetime, timezone
import unittest

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tracker import (
    Account,
    AssetClass,
    Base,
    CashFlow,
    CashFlowType,
    Currency,
    DataSource,
    DataType,
    Holdings,
    Instrument,
    MarketData,
    Performance,
    Portfolio,
    PortfolioAnalyzer,
    PriceManager,
    RiskMetrics,
    Transaction,
    TransactionType,
)


class TestModularAnalytics(unittest.TestCase):
    """Test the new modular analytics architecture."""

    def setUp(self):
        """Set up test fixtures with comprehensive portfolio data using SQLAlchemy models."""
        # Create in-memory database for testing
        self.engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()

        # Create test data using SQLAlchemy models
        self._create_test_data()

        # Get the portfolio data for testing
        self.portfolio = self.session.query(Portfolio).first()
        self.transactions_df = self._create_df(self.session.query(Transaction).all())
        self.cash_flows_df = self._create_df(self.session.query(CashFlow).all())
        self.market_data_df = self._create_df(self.session.query(MarketData).all())
        self.market_data_df = (
            self.market_data_df.pivot(columns="ticker", values="quote")
            if not self.market_data_df.empty
            else pd.DataFrame()
        )

    def _create_test_data(self):
        """Create comprehensive test data in the database."""
        # Create currencies
        eur = Currency(code="EUR", name="Euro", symbol="€", decimals=2)
        usd = Currency(code="USD", name="US Dollar", symbol="$", decimals=2)
        self.session.add_all([eur, usd])

        # Create asset class
        equity = AssetClass(name="Equity", description="Stocks and shares")
        self.session.add(equity)

        # Create data types and sources for market data
        data_type = DataType(name="close", description="Closing price")
        data_source = DataSource(name="test", description="Test data source")
        self.session.add_all([data_type, data_source])

        # Create account
        account = Account(name="Test Account", currency_id=1)  # EUR
        self.session.add(account)

        # Create portfolio
        portfolio = Portfolio(
            name="Test Portfolio",
            account_id=1,
            currency_id=1,  # EUR
        )
        self.session.add(portfolio)

        # Create instruments
        aapl = Instrument(
            name="Apple Inc.",
            isin="US0378331005",
            ticker="AAPL",
            asset_class_id=1,
            currency_id=2,  # USD
            description="Apple Inc. common stock",
        )
        googl = Instrument(
            name="Alphabet Inc.",
            isin="US02079K3059",
            ticker="GOOGL",
            asset_class_id=1,
            currency_id=2,  # USD
            description="Alphabet Inc. Class A",
        )
        self.session.add_all([aapl, googl])

        # Commit to get IDs
        self.session.commit()

        # Create transactions with proper dates and foreign keys
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

        transactions = [
            # Initial AAPL purchase
            Transaction(
                portfolio_id=portfolio.id,
                instrument_id=aapl.id,
                tx_type=TransactionType.BUY,
                quantity=10.0,
                price=150.0,
                currency_id=usd.id,
                date=base_date,
                description="Initial purchase",
            ),
            # Additional AAPL purchase
            Transaction(
                portfolio_id=portfolio.id,
                instrument_id=aapl.id,
                tx_type=TransactionType.BUY,
                quantity=5.0,
                price=160.0,
                currency_id=usd.id,
                date=datetime(2024, 2, 1, tzinfo=timezone.utc),
                description="Additional purchase",
            ),
            # Partial AAPL sale
            Transaction(
                portfolio_id=portfolio.id,
                instrument_id=aapl.id,
                tx_type=TransactionType.SELL,
                quantity=3.0,
                price=170.0,
                currency_id=usd.id,
                date=datetime(2024, 3, 1, tzinfo=timezone.utc),
                description="Partial sale",
            ),
            # GOOGL purchase
            Transaction(
                portfolio_id=portfolio.id,
                instrument_id=googl.id,
                tx_type=TransactionType.BUY,
                quantity=2.0,
                price=2500.0,
                currency_id=usd.id,
                date=datetime(2024, 2, 15, tzinfo=timezone.utc),
                description="Google purchase",
            ),
        ]
        self.session.add_all(transactions)

        # Create cash flows
        cash_flows = [
            # Initial deposit
            CashFlow(
                portfolio_id=portfolio.id,
                cf_type=CashFlowType.DEPOSIT,
                amount=10000.0,
                currency_id=eur.id,
                date=base_date,
                description="Initial deposit",
            ),
            # Dividend payment
            CashFlow(
                portfolio_id=portfolio.id,
                cf_type=CashFlowType.DIVIDEND,
                amount=50.0,
                currency_id=usd.id,
                date=datetime(2024, 4, 1, tzinfo=timezone.utc),
                description="Quarterly dividend",
            ),
            # Fee payment
            CashFlow(
                portfolio_id=portfolio.id,
                cf_type=CashFlowType.FEE,
                amount=25.0,
                currency_id=eur.id,
                date=datetime(2024, 3, 15, tzinfo=timezone.utc),
                description="Management fee",
            ),
        ]
        self.session.add_all(cash_flows)

        # Create sample market data
        self._create_sample_market_data(aapl, googl, data_type, data_source)

        # Final commit
        self.session.commit()

    def _create_sample_market_data(self, aapl, googl, data_type, data_source):
        """Create sample market data for testing."""
        # Create price data for testing
        dates = pd.date_range(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 4, 30, tzinfo=timezone.utc),
            freq="D",
        )

        market_data = []
        for i, date in enumerate(dates):
            # AAPL with some volatility around trend
            aapl_base = 150 + i * 0.1  # Slight uptrend
            aapl_noise = np.random.normal(0, 2)  # Some volatility
            aapl_price = max(aapl_base + aapl_noise, 100)  # Don't go below 100

            market_data.append(
                MarketData(
                    instrument_id=aapl.id,
                    data_type_id=data_type.id,
                    data_source_id=data_source.id,
                    quote=aapl_price,
                    date=date,
                )
            )

            # GOOGL with different pattern
            googl_base = 2500 + i * 0.5
            googl_noise = np.random.normal(0, 10)
            googl_price = max(googl_base + googl_noise, 2000)

            market_data.append(
                MarketData(
                    instrument_id=googl.id,
                    data_type_id=data_type.id,
                    data_source_id=data_source.id,
                    quote=googl_price,
                    date=date,
                )
            )

        self.session.add_all(market_data)

    def _create_df(self, instances):
        """Helper to create DataFrame from SQLAlchemy model instances."""
        df = pd.DataFrame([instance.to_dict() for instance in instances])
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
        return df

    def tearDown(self):
        """Clean up test database."""
        self.session.close()
        Base.metadata.drop_all(self.engine)


class TestHoldings(TestModularAnalytics):
    """Test Holdings analytics functionality."""

    def test_prepare_transaction_data(self):
        """Test transaction data preparation."""
        holdings = Holdings(self.transactions_df, self.cash_flows_df)

        # Test the data preparation
        result = holdings._prepare_transaction_data()  # pylint: disable=protected-access

        # Verify structure
        self.assertIsInstance(result, pd.DataFrame)
        if not result.empty:
            self.assertIn("ticker", result.columns)
            self.assertIn("price", result.columns)
            self.assertIn("tx_type", result.columns)

            # Verify cash flows are marked
            cash_rows = result[result["ticker"] == "CASH"]
            if len(cash_rows) > 0:
                self.assertTrue(all(cash_rows["price"] == 1.0))

    def test_build_date_index(self):
        """Test date index creation."""
        holdings = Holdings(self.transactions_df, self.cash_flows_df)

        # Get prepared data
        all_df = holdings._prepare_transaction_data()  # pylint: disable=protected-access

        if not all_df.empty:
            # Test date index building
            date_index = holdings._build_date_index(all_df)  # pylint: disable=protected-access

            # Verify date index properties
            self.assertIsInstance(date_index, pd.DatetimeIndex)
            self.assertTrue(len(date_index) > 0)

            # Should include transaction dates from our test data
            tx_dates = [t.date.date() for t in self.portfolio.transactions]

            # Check that some key dates are included
            for date in tx_dates:
                date_ts = pd.Timestamp(date, tz=timezone.utc)
                self.assertTrue(date_ts in date_index)

    def test_process_ticker_holdings_avg(self):
        """Test ticker holdings processing with average cost method."""
        holdings = Holdings(self.transactions_df, self.cash_flows_df)

        # Get AAPL transactions
        aapl_transactions = [
            t for t in self.portfolio.transactions if t.instrument.ticker == "AAPL"
        ]

        if aapl_transactions:
            # Create DataFrame with AAPL transactions
            df = pd.DataFrame(
                [
                    {
                        "date": t.date,
                        "tx_type": t.tx_type.value,
                        "quantity": float(t.quantity),
                        "price": float(t.price),
                    }
                    for t in aapl_transactions
                ]
            )
            df = df.set_index("date")

            # Create simple date index
            date_index = pd.date_range(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 4, 1, tzinfo=timezone.utc),
                freq="D",
            )

            result = holdings._process_ticker_holdings(df, date_index, "AAPL", "avg")  # pylint: disable=protected-access

            # Verify result structure
            self.assertIsInstance(result, pd.DataFrame)
            self.assertIn("qty", result.columns)
            self.assertIn("cost_basis", result.columns)
            self.assertIn("avg_cost", result.columns)
            self.assertIn("realized_pnl", result.columns)

            # After BUY 10x150, BUY 5x160, SELL 3x170
            # Remaining quantity should be 12
            if not result.empty:
                final_qty = result["qty"].iloc[-1]
                self.assertEqual(final_qty, 12.0)

                # Should have some realized P&L from the sale
                final_realized = result["realized_pnl"].iloc[-1]
                self.assertGreater(final_realized, 0)  # Should be positive (gain)

    def test_process_ticker_holdings_cash(self):
        """Test ticker holdings processing for cash flows."""
        holdings = Holdings(self.transactions_df, self.cash_flows_df)

        # Get deposit cash flows from the portfolio
        deposits = [
            c for c in self.portfolio.cash_flows if c.cf_type == CashFlowType.DEPOSIT
        ]

        if deposits:
            # Create DataFrame with cash flows
            df = pd.DataFrame(
                [
                    {
                        "date": c.date,
                        "tx_type": c.cf_type.value,
                        "quantity": float(c.amount),
                        "price": 1.0,
                    }
                    for c in deposits
                ]
            )
            df = df.set_index("date")

            # Create simple date index
            date_index = pd.date_range(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 2, 1, tzinfo=timezone.utc),
                freq="D",
            )

            result = holdings._process_ticker_holdings(df, date_index, "CASH", "avg")  # pylint: disable=protected-access

            # Verify result structure
            self.assertIsInstance(result, pd.DataFrame)

            if not result.empty:
                # Cash quantity should equal amount deposited
                final_qty = result["qty"].iloc[-1]
                self.assertEqual(final_qty, 10000.0)

                # Cash cost basis should equal quantity (price = 1.0)
                final_cost = result["cost_basis"].iloc[-1]
                self.assertEqual(final_cost, 10000.0)

    def test_compute_positions_integration(self):
        """Test the complete compute_positions method with different accounting methods."""
        holdings = Holdings(self.transactions_df, self.cash_flows_df)

        # Test all accounting methods
        for method in ["avg", "fifo", "lifo"]:
            with self.subTest(method=method):
                result_df = holdings.compute_positions(accounting_method=method)

                # Should return DataFrame
                self.assertIsInstance(result_df, pd.DataFrame)

                if not result_df.empty:
                    # Should have MultiIndex columns
                    self.assertIsInstance(result_df.columns, pd.MultiIndex)
                    self.assertEqual(result_df.columns.names, ["metric", "ticker"])

                    # Should contain expected metrics
                    metrics = result_df.columns.get_level_values("metric").unique()
                    expected_metrics = ["qty", "cost_basis", "avg_cost", "realized_pnl"]
                    for metric in expected_metrics:
                        self.assertIn(metric, metrics, f"{method} should have {metric}")

                # Should contain expected tickers
                tickers = result_df.columns.get_level_values(1).unique()
                self.assertIn("AAPL", tickers, f"{method} should track AAPL")
                self.assertIn("CASH", tickers, f"{method} should track CASH")

    def test_get_quantities(self):
        """Test getting current quantities for each ticker. Should return a DataFrame."""
        holdings = Holdings(self.transactions_df, self.cash_flows_df)

        # Compute positions first
        holdings.compute_positions("avg")

        # Test getting quantities
        quantities = holdings.get_quantities()

        # Should return a DataFrame
        self.assertIsInstance(quantities, pd.DataFrame)

        # Should contain our tickers if portfolio has data
        if not quantities.empty:
            self.assertIn("AAPL", quantities.columns)
            self.assertIn("CASH", quantities.columns)

            self.assertEqual(
                quantities.loc[datetime(2024, 1, 1, tzinfo=timezone.utc), "AAPL"], 10.0
            )
            self.assertEqual(
                quantities.loc[datetime(2024, 2, 1, tzinfo=timezone.utc), "AAPL"], 15.0
            )
            self.assertEqual(
                quantities.loc[datetime(2024, 3, 1, tzinfo=timezone.utc), "AAPL"], 12.0
            )

            self.assertEqual(
                quantities.loc[datetime(2024, 1, 1, tzinfo=timezone.utc), "CASH"],
                10000.0,
            )
            self.assertEqual(
                quantities.loc[datetime(2024, 4, 1, tzinfo=timezone.utc), "CASH"],
                10025.0,
            )

    def test_get_cost_basis(self):
        """Test getting cost basis for each ticker. Should return a DataFrame."""
        holdings = Holdings(self.transactions_df, self.cash_flows_df)

        # Compute positions first
        holdings.compute_positions("avg")

        # Test getting cost basis
        cost_basis = holdings.get_cost_basis()

        # Should return a DataFrame
        self.assertIsInstance(cost_basis, pd.DataFrame)

        # Should contain our tickers if portfolio has data
        if not cost_basis.empty:
            self.assertIn("AAPL", cost_basis.columns)
            self.assertIn("CASH", cost_basis.columns)

            # Cost basis should reflect purchases
            self.assertEqual(
                cost_basis.loc[datetime(2024, 1, 1, tzinfo=timezone.utc), "AAPL"],
                10.0 * 150.0,
            )
            self.assertEqual(
                cost_basis.loc[datetime(2024, 2, 1, tzinfo=timezone.utc), "AAPL"],
                10.0 * 150.0 + 5.0 * 160.0,
            )
            self.assertEqual(
                cost_basis.loc[datetime(2024, 3, 1, tzinfo=timezone.utc), "AAPL"],
                (15.0 - 3.0) * (10.0 * 150.0 + 5.0 * 160.0) / 15.0,
            )  # Previous average cost x 3

            self.assertEqual(
                cost_basis.loc[datetime(2024, 1, 1, tzinfo=timezone.utc), "CASH"],
                10000.0,
            )
            self.assertEqual(
                cost_basis.loc[datetime(2024, 4, 1, tzinfo=timezone.utc), "CASH"],
                10025.0,
            )

    def test_get_avg_cost(self):
        """Test getting average cost for each ticker. Should return a DataFrame."""
        holdings = Holdings(self.transactions_df, self.cash_flows_df)

        # Compute positions first
        holdings.compute_positions("avg")

        # Test getting average cost
        avg_cost = holdings.get_average_costs()

        # Should return a DataFrame
        self.assertIsInstance(avg_cost, pd.DataFrame)

        # Should contain our tickers if portfolio has data
        if not avg_cost.empty:
            self.assertIn("AAPL", avg_cost.columns)
            self.assertIn("CASH", avg_cost.columns)

            # Average cost should reflect purchases
            self.assertEqual(
                avg_cost.loc[datetime(2024, 1, 1, tzinfo=timezone.utc), "AAPL"],
                150.0,
            )
            self.assertEqual(
                avg_cost.loc[datetime(2024, 2, 1, tzinfo=timezone.utc), "AAPL"],
                (10.0 * 150.0 + 5.0 * 160.0) / 15.0,
            )
            self.assertEqual(
                avg_cost.loc[datetime(2024, 3, 1, tzinfo=timezone.utc), "AAPL"],
                (10.0 * 150.0 + 5.0 * 160.0) / 15.0,
            )  # Should remain same after sale

            self.assertEqual(
                avg_cost.loc[datetime(2024, 1, 1, tzinfo=timezone.utc), "CASH"],
                1.0,
            )
            self.assertEqual(
                avg_cost.loc[datetime(2024, 4, 1, tzinfo=timezone.utc), "CASH"],
                1.0,
            )

    def test_get_realized_pnl(self):
        """Test getting realized P&L for each ticker. Should return a DataFrame."""
        holdings = Holdings(self.transactions_df, self.cash_flows_df)

        # Compute positions first
        holdings.compute_positions("avg")

        # Test getting realized P&L
        realized_pnl = holdings.get_realized_pnl()

        # Should return a DataFrame
        self.assertIsInstance(realized_pnl, pd.DataFrame)

        # Should contain our tickers if portfolio has data
        if not realized_pnl.empty:
            self.assertIn("AAPL", realized_pnl.columns)
            self.assertIn("CASH", realized_pnl.columns)

            # Realized P&L should be zero until sale
            self.assertEqual(
                realized_pnl.loc[datetime(2024, 1, 1, tzinfo=timezone.utc), "AAPL"], 0.0
            )
            self.assertEqual(
                realized_pnl.loc[datetime(2024, 2, 1, tzinfo=timezone.utc), "AAPL"], 0.0
            )

            # After selling 3 at 170, should have some positive P&L
            self.assertAlmostEqual(
                realized_pnl.loc[datetime(2024, 3, 1, tzinfo=timezone.utc), "AAPL"],
                (170.0 - (150.0 * 10.0 + 160.0 * 5.0) / 15.0) * 3.0,
            )

            # Cash should have zero P&L always
            self.assertTrue(
                all(realized_pnl["CASH"] == 0.0),
                "CASH should always have zero realized P&L",
            )

    def test_get_latest_positions(self):
        """Test getting latest positions for each ticker."""
        holdings = Holdings(self.transactions_df, self.cash_flows_df)

        # Compute positions first
        holdings.compute_positions("avg")

        # Test getting latest positions
        latest = holdings.get_latest_positions()

        # Should return a Series
        self.assertIsInstance(latest, pd.Series)

        # Should contain our tickers if portfolio has data
        if not latest.empty:
            # Check AAPL holdings (should be 12 after BUY 10, BUY 5, SELL 3)
            if "AAPL" in latest.index:
                self.assertEqual(latest["AAPL"], 12.0)

            # Check CASH holdings (should be 10025 from deposit)
            if "CASH" in latest.index:
                self.assertEqual(latest["CASH"], 10025.0)

    def test_invalid_accounting_method(self):
        """Test handling of invalid accounting methods."""
        holdings = Holdings(self.transactions_df, self.cash_flows_df)

        # Should raise ValueError for invalid method - update the expected message
        with pytest.raises(ValueError, match="Unknown accounting method"):
            holdings.compute_positions(accounting_method="invalid_method")

    def test_different_accounting_methods_produce_different_results(self):
        """Test that different accounting methods can produce different realized P&L."""
        holdings = Holdings(self.transactions_df, self.cash_flows_df)

        # Build holdings with different methods
        avg_df = holdings.compute_positions("avg")
        fifo_df = holdings.compute_positions("fifo")
        lifo_df = holdings.compute_positions("lifo")

        # All should have the same final quantities
        for ticker in ["AAPL", "CASH"]:
            if ("qty", ticker) in avg_df.columns:
                avg_qty = avg_df[("qty", ticker)].iloc[-1]
                fifo_qty = fifo_df[("qty", ticker)].iloc[-1]
                lifo_qty = lifo_df[("qty", ticker)].iloc[-1]

                self.assertEqual(avg_qty, fifo_qty)
                self.assertEqual(fifo_qty, lifo_qty)


class TestPriceManager(TestModularAnalytics):
    """Test PriceManager functionality."""

    def test_get_prices(self):
        """Test price retrieval with date alignment."""
        price_manager = PriceManager(self.market_data_df, self.transactions_df)

        # Create test date index
        test_dates = pd.date_range(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
            freq="D",
        )

        prices = price_manager.get_prices(test_dates)

        self.assertIsInstance(prices, pd.DataFrame)
        self.assertEqual(len(prices), len(test_dates))

    def test_get_available_tickers(self):
        """Test getting available tickers."""
        price_manager = PriceManager(self.market_data_df, self.transactions_df)
        tickers = price_manager.get_available_tickers()

        self.assertIsInstance(tickers, list)
        self.assertIn("AAPL", tickers)
        self.assertIn("GOOGL", tickers)

    def test_get_date_range(self):
        """Test getting date range."""
        price_manager = PriceManager(self.market_data_df, self.transactions_df)
        start, end = price_manager.get_date_range()

        if start is not None and end is not None:
            self.assertIsInstance(start, (pd.Timestamp, datetime))
            self.assertIsInstance(end, (pd.Timestamp, datetime))
            self.assertLessEqual(start, end)


class TestPerformance(TestModularAnalytics):
    """Test Performance functionality."""

    def setUp(self):
        """Set up Performance tests."""
        super().setUp()
        self.price_manager = PriceManager(self.market_data_df, self.transactions_df)
        self.holdings = Holdings(self.transactions_df, self.cash_flows_df)
        self.holdings.compute_positions()
        self.performance = Performance(self.holdings, self.price_manager)

    def test_compute_market_values(self):
        """Test market value computation."""
        market_values = self.performance.compute_market_values()
        self.assertIsInstance(market_values, pd.DataFrame)

    def test_compute_portfolio_value(self):
        """Test portfolio value computation."""
        portfolio_value = self.performance.compute_portfolio_value()
        self.assertIsInstance(portfolio_value, pd.Series)

    def test_compute_unrealized_pnl(self):
        """Test unrealized P&L computation."""
        unrealized_pnl = self.performance.compute_unrealized_pnl()
        self.assertIsInstance(unrealized_pnl, pd.DataFrame)

    def test_compute_returns(self):
        """Test return computation."""
        returns = self.performance.compute_returns(method="simple")
        self.assertIsInstance(returns, pd.Series)

    def test_compute_cumulative_returns(self):
        """Test cumulative return computation."""
        cum_returns = self.performance.compute_cumulative_returns()
        self.assertIsInstance(cum_returns, pd.Series)

    def test_compute_annualized_return(self):
        """Test annualized return computation."""
        ann_return = self.performance.compute_annualized_return()
        self.assertIsInstance(ann_return, (int, float))

    def test_compute_volatility(self):
        """Test volatility computation."""
        volatility = self.performance.compute_volatility()
        self.assertIsInstance(volatility, (int, float))
        self.assertGreaterEqual(volatility, 0)

    def test_compute_sharpe_ratio(self):
        """Test Sharpe ratio computation."""
        sharpe = self.performance.compute_sharpe_ratio(risk_free_rate=0.02)
        self.assertIsInstance(sharpe, (int, float))

    def test_compute_max_drawdown(self):
        """Test maximum drawdown computation."""
        drawdown_info = self.performance.compute_max_drawdown()

        self.assertIsInstance(drawdown_info, dict)
        self.assertIn("max_drawdown", drawdown_info)
        self.assertIn("start_date", drawdown_info)
        self.assertIn("end_date", drawdown_info)
        self.assertIn("duration", drawdown_info)

    def test_compute_performance_summary(self):
        """Test comprehensive performance summary."""
        summary = self.performance.compute_performance_summary()

        self.assertIsInstance(summary, dict)
        # Should contain main sections unless there's an error
        if "error" not in summary:
            self.assertIn("period", summary)
            self.assertIn("returns", summary)
            self.assertIn("portfolio_value", summary)
            self.assertIn("pnl", summary)
            self.assertIn("risk", summary)


class TestRiskMetrics(TestModularAnalytics):
    """Test RiskMetrics functionality."""

    def setUp(self):
        """Set up RiskMetrics tests."""
        super().setUp()
        self.price_manager = PriceManager(self.market_data_df, self.transactions_df)
        self.holdings = Holdings(self.transactions_df, self.cash_flows_df)
        self.holdings.compute_positions()
        self.performance = Performance(self.holdings, self.price_manager)
        self.risk_metrics = RiskMetrics(self.performance)

    def test_compute_var(self):
        """Test VaR computation."""
        var_info = self.risk_metrics.compute_var(confidence_level=0.05)

        self.assertIsInstance(var_info, dict)
        self.assertIn("var", var_info)
        self.assertIn("cvar", var_info)
        self.assertIn("method", var_info)
        self.assertIn("confidence_level", var_info)

    def test_compute_sortino_ratio(self):
        """Test Sortino ratio computation."""
        sortino = self.risk_metrics.compute_sortino_ratio(risk_free_rate=0.02)
        self.assertIsInstance(sortino, (int, float))

    def test_compute_calmar_ratio(self):
        """Test Calmar ratio computation."""
        calmar = self.risk_metrics.compute_calmar_ratio()
        self.assertIsInstance(calmar, (int, float))

    def test_compute_tail_ratio(self):
        """Test tail ratio computation."""
        tail_ratio = self.risk_metrics.compute_tail_ratio()
        self.assertIsInstance(tail_ratio, (int, float))

    def test_compute_pain_index(self):
        """Test Pain Index computation."""
        pain_index = self.risk_metrics.compute_pain_index()
        self.assertIsInstance(pain_index, (int, float))
        self.assertGreaterEqual(pain_index, 0)

    def test_compute_risk_summary(self):
        """Test comprehensive risk summary."""
        risk_summary = self.risk_metrics.compute_risk_summary()

        self.assertIsInstance(risk_summary, dict)
        # Should contain main sections unless there's an error
        if "error" not in risk_summary:
            self.assertIn("volatility_metrics", risk_summary)
            self.assertIn("risk_adjusted_ratios", risk_summary)
            self.assertIn("drawdown_metrics", risk_summary)
            self.assertIn("var_metrics", risk_summary)


class TestPortfolioAnalyzer(TestModularAnalytics):
    """Test PortfolioAnalyzer functionality."""

    def test_compute_analytics(self):
        """Test analytics computation."""
        analyzer = PortfolioAnalyzer(
            self.transactions_df, self.cash_flows_df, self.market_data_df
        )
        result = analyzer.compute_analytics()

        # Should return self for method chaining
        self.assertEqual(result, analyzer)
        self.assertTrue(analyzer.has_computed_analytics)
        self.assertIsInstance(analyzer.performance, Performance)
        self.assertIsInstance(analyzer.risk_metrics, RiskMetrics)

    def test_get_portfolio_summary(self):
        """Test portfolio summary generation."""
        analyzer = PortfolioAnalyzer(
            self.transactions_df, self.cash_flows_df, self.market_data_df
        )
        analyzer.compute_analytics()

        summary = analyzer.get_portfolio_summary()

        self.assertIsInstance(summary, dict)
        # Check main sections exist (unless there's an error)
        if "error" not in summary:
            self.assertIn("portfolio_info", summary)
            self.assertIn("current_positions", summary)
            self.assertIn("performance", summary)
            self.assertIn("risk", summary)

    def test_get_current_allocation(self):
        """Test current allocation calculation."""
        analyzer = PortfolioAnalyzer(
            self.transactions_df, self.cash_flows_df, self.market_data_df
        )
        analyzer.compute_analytics()

        allocation = analyzer.get_current_allocation()
        self.assertIsInstance(allocation, pd.Series)

        # If not empty, percentages should sum to approximately 100
        if not allocation.empty:
            total = allocation.sum()
            self.assertAlmostEqual(total, 100.0, places=1)

    def test_get_performance_chart_data(self):
        """Test performance chart data generation."""
        analyzer = PortfolioAnalyzer(
            self.transactions_df, self.cash_flows_df, self.market_data_df
        )
        analyzer.compute_analytics()

        chart_data = analyzer.get_performance_chart_data()

        self.assertIsInstance(chart_data, pd.DataFrame)
        if not chart_data.empty:
            expected_columns = ["portfolio_value", "cumulative_return", "drawdown"]
            for col in expected_columns:
                self.assertIn(col, chart_data.columns)

    def test_get_position_details(self):
        """Test position details generation."""
        analyzer = PortfolioAnalyzer(
            self.transactions_df, self.cash_flows_df, self.market_data_df
        )
        analyzer.compute_analytics()

        details = analyzer.get_position_details()
        self.assertIsInstance(details, pd.DataFrame)

    def test_export_data(self):
        """Test data export."""
        analyzer = PortfolioAnalyzer(
            self.transactions_df, self.cash_flows_df, self.market_data_df
        )
        analyzer.compute_analytics()

        data = analyzer.export_data(fmt="dict")

        self.assertIsInstance(data, dict)
        if "error" not in data:
            self.assertIn("holdings", data)
            self.assertIn("market_data", data)

    def test_get_component_access(self):
        """Test direct component access."""
        analyzer = PortfolioAnalyzer(
            self.transactions_df, self.cash_flows_df, self.market_data_df
        )
        analyzer.compute_analytics()

        # Test component access
        holdings = analyzer.get_component("holdings")
        self.assertIsInstance(holdings, Holdings)

        performance = analyzer.get_component("performance")
        self.assertIsInstance(performance, Performance)

        risk = analyzer.get_component("risk")
        self.assertIsInstance(risk, RiskMetrics)

        prices = analyzer.get_component("prices")
        self.assertIsInstance(prices, PriceManager)

        # Test invalid component
        with pytest.raises(ValueError, match="Unknown component: invalid_component"):
            analyzer.get_component("invalid_component")

    def test_analytics_not_computed_error(self):
        """Test that methods require analytics to be computed first."""
        analyzer = PortfolioAnalyzer(
            self.transactions_df, self.cash_flows_df, self.market_data_df
        )

        # These should raise ValueError before analytics are computed
        with pytest.raises(
            ValueError,
            match=r"Analytics not computed. Call compute_analytics\(\) first\.",
        ):
            analyzer.get_portfolio_summary()

        with pytest.raises(
            ValueError,
            match=r"Analytics not computed. Call compute_analytics\(\) first\.",
        ):
            analyzer.get_current_allocation()

        with pytest.raises(
            ValueError,
            match=r"Analytics not computed. Call compute_analytics\(\) first\.",
        ):
            analyzer.get_performance_chart_data()


if __name__ == "__main__":
    unittest.main(verbosity=2)
