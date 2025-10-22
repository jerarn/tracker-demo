"""High-level portfolio analyzer orchestrating all components.

Provides simple interface while maintaining component access for advanced users.
"""

import json
from typing import Any

import numpy as np
import pandas as pd

from tracker.config.logger import get_logger

from .core.holdings import Holdings
from .core.performance import Performance
from .core.price_manager import PriceManager
from .core.risk_metrics import RiskMetrics

logger = get_logger(__name__)


class PortfolioAnalyzer:
    """High-level portfolio analyzer orchestrating all components.

    Provides simple interface for common tasks while allowing direct component access.
    """

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        cash_flows_df: pd.DataFrame,
        market_data_df: pd.DataFrame,
    ):
        """Initialize analyzer with dataframes."""
        # Initialize core components
        self.price_manager = PriceManager(market_data_df, transactions_df)
        self.holdings = Holdings(transactions_df, cash_flows_df)
        self.performance = None  # Initialized after holdings computation
        self.risk_metrics = None  # Initialized after performance computation

        # State tracking
        self._holdings_computed = False
        self._performance_computed = False

    def compute_analytics(
        self,
        accounting_method: str = "avg",
        end_date: pd.Timestamp | None = None,
    ) -> "PortfolioAnalyzer":
        """Compute all analytics components.

        Args:
            accounting_method: Accounting method for holdings
            end_date: End date for analysis
            load_data: Whether to load portfolios data

        Returns:
            Self for method chaining
        """
        # Compute holdings
        self.holdings.compute_positions(accounting_method, end_date)
        self._holdings_computed = True

        # Initialize performance and risk components
        self.performance = Performance(self.holdings, self.price_manager)
        self.risk_metrics = RiskMetrics(self.performance)
        self._performance_computed = True

        return self

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get comprehensive portfolio summary.

        Returns:
            Dictionary with portfolio overview
        """
        if not self._holdings_computed:
            raise ValueError("Analytics not computed. Call compute_analytics() first.")

        try:
            # Portfolio basics
            portfolio_ids = self.holdings.get_portfolio_ids()
            start_dates = self.holdings.get_start_dates()

            # Current positions
            latest_positions = self.holdings.get_latest_positions()
            portfolio_value = self.performance.compute_portfolio_value()
            current_val = portfolio_value.iloc[-1] if not portfolio_value.empty else 0.0

            # Performance metrics
            perf_summary = self.performance.compute_performance_summary()
            risk_summary = self.risk_metrics.compute_risk_summary()

            return {
                "portfolio_info": {
                    "portfolio_ids": portfolio_ids,
                    "start_date": min(start_dates) if start_dates else None,
                    "end_date": portfolio_value.index[-1]
                    if not portfolio_value.empty
                    else None,
                    "current_value": current_val,
                    "position_count": len(latest_positions[latest_positions != 0]),
                },
                "current_positions": latest_positions.to_dict(),
                "performance": perf_summary,
                "risk": risk_summary,
            }

        except Exception as e:  # pylint: disable=broad-except
            return {"error": f"Summary generation failed: {e}"}

    def get_current_allocation(self) -> pd.Series:
        """Get current portfolio allocation as percentages.

        Returns:
            Series with allocation percentages by ticker
        """
        if not self._performance_computed:
            raise ValueError("Analytics not computed. Call compute_analytics() first.")

        market_values = self.performance.compute_market_values()
        if market_values.empty:
            return pd.Series(dtype=float)

        latest_values = market_values.iloc[-1]
        total_value = latest_values.sum()

        if total_value == 0:
            return pd.Series(dtype=float)

        return (latest_values / total_value * 100).round(2)

    def get_performance_chart_data(self) -> pd.DataFrame:
        """Get data for performance charting.

        Returns:
            DataFrame with portfolio value, returns, and drawdown over time
        """
        if not self._performance_computed:
            raise ValueError("Analytics not computed. Call compute_analytics() first.")

        portfolio_value = self.performance.compute_portfolio_value()
        cumulative_returns = self.performance.compute_cumulative_returns()

        # Calculate drawdown
        if not cumulative_returns.empty:
            running_max = cumulative_returns.cummax()
            drawdown = cumulative_returns - running_max
        else:
            drawdown = pd.Series(dtype=float)

        # Combine data
        chart_data = pd.DataFrame(
            {
                "portfolio_value": portfolio_value,
                "cumulative_return": cumulative_returns,
                "drawdown": drawdown,
            }
        )

        return chart_data.fillna(0.0)

    def get_position_details(self) -> pd.DataFrame:
        """Get detailed position information.

        Returns:
            DataFrame with position details by ticker
        """
        if not self._performance_computed:
            raise ValueError("Analytics not computed. Call compute_analytics() first.")

        try:
            # Get latest data
            quantities = self.holdings.get_quantities()
            cost_basis = self.holdings.get_cost_basis()
            avg_costs = self.holdings.get_average_costs()
            market_values = self.performance.compute_market_values()
            unrealized_pnl = self.performance.compute_unrealized_pnl()

            if quantities.empty:
                return pd.DataFrame()

            # Get latest values for each metric
            latest_qty = quantities.iloc[-1] if not quantities.empty else pd.Series()
            latest_cost = cost_basis.iloc[-1] if not cost_basis.empty else pd.Series()
            latest_avg = avg_costs.iloc[-1] if not avg_costs.empty else pd.Series()
            latest_market = (
                market_values.iloc[-1] if not market_values.empty else pd.Series()
            )
            latest_unrealized = (
                unrealized_pnl.iloc[-1] if not unrealized_pnl.empty else pd.Series()
            )

            # Get current prices
            prices = self.price_manager.get_prices(self.holdings.positions_df.index)
            latest_prices = prices.iloc[-1] if not prices.empty else pd.Series()

            # Combine into details DataFrame
            tickers = latest_qty[latest_qty != 0].index
            details = []

            for ticker in tickers:
                qty = latest_qty.get(ticker, 0.0)
                if qty == 0:
                    continue

                cost = latest_cost.get(ticker, 0.0)
                avg_cost = latest_avg.get(ticker, 0.0)
                market_val = latest_market.get(ticker, 0.0)
                unrealized = latest_unrealized.get(ticker, 0.0)
                current_price = latest_prices.get(ticker, 0.0)

                details.append(
                    {
                        "ticker": ticker,
                        "quantity": qty,
                        "avg_cost": avg_cost,
                        "current_price": current_price,
                        "cost_basis": cost,
                        "market_value": market_val,
                        "unrealized_pnl": unrealized,
                        "pnl_percent": (unrealized / cost * 100) if cost != 0 else 0.0,
                    }
                )

            if details:
                return pd.DataFrame(details).set_index("ticker")
            return pd.DataFrame()

        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Error getting position details: {e}")
            return pd.DataFrame()

    def compare_with_benchmark(
        self, benchmark_returns: pd.Series, return_method: str = "simple"
    ) -> dict[str, Any]:
        """Compare portfolio performance with benchmark.

        Args:
            benchmark_returns: Benchmark return series
            return_method: Return calculation method

        Returns:
            Dictionary with comparison metrics
        """
        if not self._performance_computed:
            raise ValueError("Analytics not computed. Call compute_analytics() first.")

        try:
            # Portfolio metrics
            portfolio_returns = self.performance.compute_returns(return_method)
            portfolio_summary = self.performance.compute_performance_summary(
                return_method
            )

            # Benchmark metrics
            if benchmark_returns.empty:
                return {"error": "Empty benchmark data"}

            # Align dates
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if common_dates.empty:
                return {"error": "No overlapping dates with benchmark"}

            benchmark_aligned = benchmark_returns.loc[common_dates]

            # Benchmark performance
            benchmark_cumret = (1 + benchmark_aligned).cumprod() - 1
            benchmark_ann_ret = (1 + benchmark_aligned).prod() ** (
                252 / len(benchmark_aligned)
            ) - 1
            benchmark_vol = benchmark_aligned.std() * np.sqrt(252)

            # Relative metrics
            risk_summary = self.risk_metrics.compute_risk_summary(
                return_method=return_method, benchmark_returns=benchmark_returns
            )

            return {
                "period": {
                    "start_date": common_dates[0],
                    "end_date": common_dates[-1],
                    "days": len(common_dates),
                },
                "portfolio": {
                    "total_return": portfolio_summary["returns"]["total_return"],
                    "annualized_return": portfolio_summary["returns"][
                        "annualized_return"
                    ],
                    "volatility": portfolio_summary["returns"]["volatility"],
                    "sharpe_ratio": portfolio_summary["returns"]["sharpe_ratio"],
                },
                "benchmark": {
                    "total_return": benchmark_cumret.iloc[-1],
                    "annualized_return": benchmark_ann_ret,
                    "volatility": benchmark_vol,
                    "sharpe_ratio": benchmark_ann_ret / benchmark_vol
                    if benchmark_vol != 0
                    else 0.0,
                },
                "relative_metrics": risk_summary.get("benchmark_metrics", {}),
            }

        except Exception as e:  # pylint: disable=broad-except
            return {"error": f"Benchmark comparison failed: {e}"}

    def export_data(self, fmt: str = "dict") -> dict[str, Any]:
        """Export all computed data.

        Args:
            fmt: Export format ('dict', 'json')

        Returns:
            Dictionary with all data or JSON string
        """
        if not self._holdings_computed:
            raise ValueError("Analytics not computed. Call compute_analytics() first.")

        try:
            data = {
                "holdings": {
                    "quantities": self.holdings.get_quantities()
                    .reset_index()
                    .T.to_dict(),
                    "cost_basis": self.holdings.get_cost_basis()
                    .reset_index()
                    .T.to_dict(),
                    "average_costs": self.holdings.get_average_costs()
                    .reset_index()
                    .T.to_dict(),
                    "realized_pnl": self.holdings.get_realized_pnl()
                    .reset_index()
                    .T.to_dict(),
                },
                "market_data": {
                    "prices": self.price_manager.get_prices(
                        self.holdings.positions_df.index
                    )
                    .reset_index()
                    .T.to_dict()
                },
            }

            if self._performance_computed:
                data["performance"] = {
                    "portfolio_value": self.performance.compute_portfolio_value()
                    .reset_index()
                    .T.to_dict(),
                    "market_values": self.performance.compute_market_values()
                    .reset_index()
                    .T.to_dict(),
                    "returns": self.performance.compute_returns()
                    .reset_index()
                    .T.to_dict(),
                    "cumulative_returns": self.performance.compute_cumulative_returns()
                    .reset_index()
                    .T.to_dict(),
                }

            if fmt == "json":
                return json.dumps(data, indent=2, default=str)
            return data

        except Exception as e:  # pylint: disable=broad-except
            return {"error": f"Export failed: {e}"}

    # Direct component access for advanced users
    @property
    def has_computed_analytics(self) -> bool:
        """Check if analytics have been computed."""
        return self._holdings_computed and self._performance_computed

    def get_component(self, component: str):
        """Get direct access to components for advanced usage.

        Args:
            component: Component name ('holdings', 'performance', 'risk', 'prices')

        Returns:
            Component instance
        """
        if component == "holdings":
            return self.holdings
        if component == "performance":
            if not self._performance_computed:
                raise ValueError(
                    "Performance not computed. Call compute_analytics() first."
                )
            return self.performance
        if component == "risk":
            if not self._performance_computed:
                raise ValueError(
                    "Risk metrics not computed. Call compute_analytics() first."
                )
            return self.risk_metrics
        if component == "prices":
            return self.price_manager
        raise ValueError(f"Unknown component: {component}")

    def __repr__(self) -> str:
        """String representation."""
        portfolio_ids = [p.id for p in self.portfolios]
        status = "computed" if self._holdings_computed else "not computed"
        return f"PortfolioAnalyzer(portfolios={portfolio_ids}, analytics={status})"
