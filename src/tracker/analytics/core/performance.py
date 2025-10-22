"""Clean performance metrics calculation using Holdings and PriceManager."""

from typing import Any

import numpy as np
import pandas as pd

from .holdings import Holdings
from .price_manager import PriceManager


class Performance:
    """Performance metrics calculation using Holdings and PriceManager.

    Separates market valuation from position tracking.
    """

    def __init__(self, holdings: Holdings, price_manager: PriceManager):
        """Initialize with Holdings and PriceManager instances."""
        self.holdings = holdings
        self.price_manager = price_manager

    def compute_market_values(self) -> pd.DataFrame:
        """Compute market values by combining positions and prices.

        Returns:
            DataFrame with market values for each ticker over time
        """
        quantities = self.holdings.get_quantities()
        if quantities.empty:
            return pd.DataFrame()

        prices = self.price_manager.get_prices(self.holdings.positions_df.index)
        if prices.empty:
            return pd.DataFrame()

        # Align indices and columns
        common_dates = quantities.index.intersection(prices.index)
        common_tickers = quantities.columns.intersection(prices.columns)

        if common_dates.empty or common_tickers.empty:
            return pd.DataFrame()

        # Calculate market values
        aligned_quantities = quantities.loc[common_dates, common_tickers]
        aligned_prices = prices.loc[common_dates, common_tickers]

        market_values = aligned_quantities * aligned_prices
        return market_values.fillna(0.0)

    def compute_portfolio_value(self) -> pd.Series:
        """Compute total portfolio value over time.

        Returns:
            Series with total portfolio value by date
        """
        market_values = self.compute_market_values()
        if market_values.empty:
            return pd.Series(dtype=float)

        return market_values.sum(axis=1)

    def compute_unrealized_pnl(self) -> pd.DataFrame:
        """Compute unrealized P&L by ticker.

        Returns:
            DataFrame with unrealized P&L for each ticker over time
        """
        market_values = self.compute_market_values()
        cost_basis = self.holdings.get_cost_basis()

        if market_values.empty or cost_basis.empty:
            return pd.DataFrame()

        # Align data
        common_dates = market_values.index.intersection(cost_basis.index)
        common_tickers = market_values.columns.intersection(cost_basis.columns)

        if common_dates.empty or common_tickers.empty:
            return pd.DataFrame()

        aligned_market = market_values.loc[common_dates, common_tickers]
        aligned_cost = cost_basis.loc[common_dates, common_tickers]

        unrealized_pnl = aligned_market - aligned_cost
        return unrealized_pnl.fillna(0.0)

    def compute_total_pnl(self) -> pd.Series:
        """Compute total P&L (realized + unrealized) over time.

        Returns:
            Series with total P&L by date
        """
        unrealized_pnl = self.compute_unrealized_pnl()
        realized_pnl = self.holdings.get_realized_pnl()

        if unrealized_pnl.empty and realized_pnl.empty:
            return pd.Series(dtype=float)

        total_unrealized = (
            unrealized_pnl.sum(axis=1)
            if not unrealized_pnl.empty
            else pd.Series(0.0, index=unrealized_pnl.index)
        )
        total_realized = (
            realized_pnl.sum(axis=1)
            if not realized_pnl.empty
            else pd.Series(0.0, index=realized_pnl.index)
        )

        # Align indices
        if total_unrealized.empty:
            return total_realized
        if total_realized.empty:
            return total_unrealized
        common_dates = total_unrealized.index.intersection(total_realized.index)
        if common_dates.empty:
            return pd.Series(dtype=float)

        return total_unrealized.loc[common_dates] + total_realized.loc[common_dates]

    def compute_returns(self, method: str = "simple") -> pd.Series:
        """Compute portfolio returns.

        Args:
            method: Return calculation method ('simple', 'log', 'money_weighted')

        Returns:
            Series with returns by date
        """
        cost_basis = self.holdings.get_cost_basis().sum(axis=1)
        portfolio_value = self.compute_portfolio_value()
        if portfolio_value.empty or len(portfolio_value) < 2:
            return pd.Series(dtype=float)

        daily_cost = cost_basis.diff().fillna(0.0)
        previous_value = portfolio_value.shift(1).fillna(0.0)
        daily_change = portfolio_value.diff().fillna(0.0)
        pnl = daily_change - daily_cost
        returns = pd.Series(
            np.where(
                previous_value > 0,
                pnl / previous_value,
                0.0,
            ),
            index=portfolio_value.index,
        )

        if method == "log":
            returns = np.log1p(returns)
        elif method == "money_weighted":
            # Money-weighted returns require cash flow data
            returns = self._compute_money_weighted_returns(portfolio_value)
        elif method != "simple":
            raise ValueError(f"Unknown return method: {method}")

        return returns.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    def compute_cumulative_returns(self, method: str = "simple") -> pd.Series:
        """Compute cumulative returns.

        Args:
            method: Return calculation method

        Returns:
            Series with cumulative returns by date
        """
        returns = self.compute_returns(method)
        if returns.empty:
            return pd.Series(dtype=float)

        if method == "log":
            return returns.cumsum()
        return (1 + returns).cumprod() - 1

    def compute_annualized_return(self, method: str = "simple") -> float:
        """Compute annualized return.

        Args:
            method: Return calculation method

        Returns:
            Annualized return as float
        """
        returns = self.compute_returns(method)
        if returns.empty or len(returns) < 2:
            return 0.0

        # Calculate number of periods per year
        time_diff = (returns.index[-1] - returns.index[0]).days
        if time_diff <= 0:
            return 0.0

        years = time_diff / 365.25

        if method == "log":
            total_return = returns.sum()
            return total_return / years

        total_return = (1 + returns).prod() - 1
        return (1 + total_return) ** (1 / years) - 1

    def compute_volatility(
        self, method: str = "simple", annualized: bool = True
    ) -> float:
        """Compute return volatility.

        Args:
            method: Return calculation method
            annualized: Whether to annualize volatility

        Returns:
            Volatility as float
        """
        returns = self.compute_returns(method)
        if returns.empty:
            return 0.0

        vol = returns.std(ddof=1)
        if not annualized:
            return vol

        total_days = (returns.index[-1] - returns.index[0]).days
        if total_days <= 0:
            return 0.0

        periods_per_year = 365.25 / (total_days / len(returns))
        return vol * np.sqrt(periods_per_year)

    def compute_sharpe_ratio(
        self, risk_free_rate: float = 0.0, method: str = "simple"
    ) -> float:
        """Compute Sharpe ratio.

        Args:
            risk_free_rate: Risk-free rate (annualized)
            method: Return calculation method

        Returns:
            Sharpe ratio as float
        """
        ann_return = self.compute_annualized_return(method)
        volatility = self.compute_volatility(method, annualized=True)

        if volatility == 0:
            return 0.0

        return (ann_return - risk_free_rate) / volatility

    def compute_max_drawdown(self, method: str = "simple") -> dict[str, Any]:
        """Compute maximum drawdown.

        Args:
            method: Return calculation method

        Returns:
            Dictionary with max drawdown info
        """
        cumulative_returns = self.compute_cumulative_returns(method)
        if cumulative_returns.empty:
            return {
                "max_drawdown": 0.0,
                "start_date": None,
                "end_date": None,
                "duration": 0,
            }

        # Calculate running maximum
        wealth = 1 + cumulative_returns
        running_max = wealth.cummax()
        drawdown = wealth / running_max - 1

        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()

        # Find start of drawdown period
        start_date = running_max[running_max.index <= max_dd_date].idxmax()

        # Recovery date: first time wealth recovers to the previous peak
        post_trough = wealth[wealth.index > max_dd_date]
        recovered = post_trough[post_trough >= running_max.loc[max_dd_date]]
        end_date = recovered.index[0] if not recovered.empty else wealth.index[-1]
        duration = (end_date - start_date).days

        return {
            "max_drawdown": max_dd,
            "start_date": start_date,
            "end_date": end_date,
            "duration": duration,
        }

    def compute_performance_summary(
        self, method: str = "simple", risk_free_rate: float = 0.0
    ) -> dict[str, Any]:
        """Compute comprehensive performance summary.

        Args:
            method: Return calculation method
            risk_free_rate: Risk-free rate for Sharpe ratio

        Returns:
            Dictionary with performance metrics
        """
        try:
            portfolio_value = self.compute_portfolio_value()

            if portfolio_value.empty:
                return self._empty_performance_summary()

            # Basic metrics
            total_return = self.compute_cumulative_returns(method)
            final_return = total_return.iloc[-1] if not total_return.empty else 0.0

            ann_return = self.compute_annualized_return(method)
            volatility = self.compute_volatility(method, annualized=True)
            sharpe = self.compute_sharpe_ratio(risk_free_rate, method)

            # Drawdown analysis
            drawdown_info = self.compute_max_drawdown(method)

            # Portfolio value stats
            start_value = portfolio_value.iloc[0] if len(portfolio_value) > 0 else 0.0
            end_value = portfolio_value.iloc[-1] if len(portfolio_value) > 0 else 0.0

            # P&L breakdown
            total_pnl = self.compute_total_pnl()
            final_pnl = total_pnl.iloc[-1] if not total_pnl.empty else 0.0

            realized_pnl = self.holdings.get_realized_pnl()
            total_realized = (
                realized_pnl.sum(axis=1).iloc[-1] if not realized_pnl.empty else 0.0
            )

            unrealized_pnl = self.compute_unrealized_pnl()
            total_unrealized = (
                unrealized_pnl.sum(axis=1).iloc[-1] if not unrealized_pnl.empty else 0.0
            )

            return {
                "period": {
                    "start_date": portfolio_value.index[0],
                    "end_date": portfolio_value.index[-1],
                    "days": (portfolio_value.index[-1] - portfolio_value.index[0]).days,
                },
                "returns": {
                    "total_return": final_return,
                    "annualized_return": ann_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe,
                },
                "portfolio_value": {
                    "start_value": start_value,
                    "end_value": end_value,
                    "max_value": portfolio_value.max(),
                    "min_value": portfolio_value.min(),
                },
                "pnl": {
                    "total_pnl": final_pnl,
                    "realized_pnl": total_realized,
                    "unrealized_pnl": total_unrealized,
                },
                "risk": {
                    "max_drawdown": drawdown_info["max_drawdown"],
                    "max_drawdown_start": drawdown_info["start_date"],
                    "max_drawdown_end": drawdown_info["end_date"],
                    "max_drawdown_duration": drawdown_info["duration"],
                },
            }

        except Exception as e:  # pylint: disable=broad-except
            return {"error": f"Performance calculation failed: {e}"}

    def _compute_money_weighted_returns(self, portfolio_value: pd.Series) -> pd.Series:
        """Compute money-weighted returns considering cash flows."""
        # Simplified implementation - would need actual cash flow integration
        # For now, fall back to simple returns
        return portfolio_value.pct_change()

    def _empty_performance_summary(self) -> dict[str, Any]:
        """Return empty performance summary structure."""
        return {
            "period": {"start_date": None, "end_date": None, "days": 0},
            "returns": {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
            },
            "portfolio_value": {
                "start_value": 0.0,
                "end_value": 0.0,
                "max_value": 0.0,
                "min_value": 0.0,
            },
            "pnl": {"total_pnl": 0.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0},
            "risk": {
                "max_drawdown": 0.0,
                "max_drawdown_start": None,
                "max_drawdown_end": None,
                "max_drawdown_duration": 0,
            },
        }
