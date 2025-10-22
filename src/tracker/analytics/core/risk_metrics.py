"""Specialized risk calculations - VaR, conditional VaR, Sortino, etc."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .performance import Performance


class RiskMetrics:
    """Specialized risk calculations using Performance data.

    Focuses on advanced risk metrics beyond basic volatility.
    """

    def __init__(self, performance: Performance):
        """Initialize with a Performance instance."""
        self.performance = performance

    def compute_var(
        self,
        confidence_level: float = 0.05,
        method: str = "historical",
        return_method: str = "simple",
    ) -> dict[str, float]:
        """Compute Value at Risk (VaR).

        Args:
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')
            return_method: Return calculation method

        Returns:
            Dictionary with VaR metrics
        """
        returns = self.performance.compute_returns(return_method)
        if returns.empty:
            return {"var": 0.0, "cvar": 0.0, "method": method}

        if method == "historical":
            var = np.quantile(returns, confidence_level)
            cvar = returns[returns <= var].mean()
        elif method == "parametric":
            mu = returns.mean()
            sigma = returns.std()
            var = stats.norm.ppf(confidence_level, mu, sigma)
            # Conditional VaR for normal distribution
            cvar = (
                mu
                - sigma
                * stats.norm.pdf(stats.norm.ppf(confidence_level))
                / confidence_level
            )
        elif method == "monte_carlo":
            # Simplified Monte Carlo - could be enhanced
            mu = returns.mean()
            sigma = returns.std()
            simulated = np.random.normal(mu, sigma, 10000)
            var = np.quantile(simulated, confidence_level)
            cvar = simulated[simulated <= var].mean()
        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return {
            "var": var,
            "cvar": cvar,  # Conditional VaR (Expected Shortfall)
            "method": method,
            "confidence_level": confidence_level,
        }

    def compute_sortino_ratio(
        self,
        risk_free_rate: float = 0.0,
        return_method: str = "simple",
        target_return: float = 0.0,
    ) -> float:
        """Compute Sortino ratio (downside-focused Sharpe ratio).

        Args:
            risk_free_rate: Risk-free rate (annualized)
            return_method: Return calculation method
            target_return: Target return for downside deviation calculation

        Returns:
            Sortino ratio as float
        """
        returns = self.performance.compute_returns(return_method)
        if returns.empty:
            return 0.0

        # Annualized return
        ann_return = self.performance.compute_annualized_return(return_method)

        total_days = (returns.index[-1] - returns.index[0]).days
        if total_days <= 0:
            return 0.0
        periods_per_year = 365.25 / (total_days / len(returns))

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < target_return]
        if downside_returns.empty:
            return float("inf") if ann_return > risk_free_rate else 0.0

        downside_std = downside_returns.std(ddof=1) * np.sqrt(periods_per_year)
        if downside_std == 0:
            return float("inf") if ann_return > risk_free_rate else 0.0

        return (ann_return - risk_free_rate) / downside_std

    def compute_calmar_ratio(self, return_method: str = "simple") -> float:
        """Compute Calmar ratio (return vs max drawdown).

        Args:
            return_method: Return calculation method

        Returns:
            Calmar ratio as float
        """
        ann_return = self.performance.compute_annualized_return(return_method)
        drawdown_info = self.performance.compute_max_drawdown(return_method)

        max_dd = abs(drawdown_info["max_drawdown"])
        if max_dd == 0:
            return float("inf") if ann_return > 0 else 0.0

        return ann_return / max_dd

    def compute_information_ratio(
        self, benchmark_returns: pd.Series, return_method: str = "simple"
    ) -> float:
        """Compute Information ratio vs benchmark.

        Args:
            benchmark_returns: Benchmark return series
            return_method: Return calculation method

        Returns:
            Information ratio as float
        """
        portfolio_returns = self.performance.compute_returns(return_method)
        if portfolio_returns.empty or benchmark_returns.empty:
            return 0.0

        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if common_dates.empty:
            return 0.0

        portfolio_aligned = portfolio_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]

        # Active returns
        active_returns = portfolio_aligned - benchmark_aligned

        if active_returns.std() == 0:
            return 0.0

        return active_returns.mean() / active_returns.std() * np.sqrt(252)

    def compute_beta(
        self, benchmark_returns: pd.Series, return_method: str = "simple"
    ) -> dict[str, float]:
        """Compute beta and alpha vs benchmark.

        Args:
            benchmark_returns: Benchmark return series
            return_method: Return calculation method

        Returns:
            Dictionary with beta and alpha
        """
        portfolio_returns = self.performance.compute_returns(return_method)
        if portfolio_returns.empty or benchmark_returns.empty:
            return {"beta": 0.0, "alpha": 0.0, "r_squared": 0.0}

        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if common_dates.empty or len(common_dates) < 2:
            return {"beta": 0.0, "alpha": 0.0, "r_squared": 0.0}

        portfolio_aligned = portfolio_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]

        # Linear regression
        try:
            slope, intercept, r_value, _, _ = stats.linregress(
                benchmark_aligned, portfolio_aligned
            )
            total_days = (common_dates[-1] - common_dates[0]).days
            periods_per_year = 365.25 / (total_days / len(common_dates))

            return {
                "beta": slope,
                "alpha": intercept * periods_per_year,  # Annualized alpha
                "r_squared": r_value**2,
            }
        except Exception:  # pylint: disable=broad-except
            return {"beta": 0.0, "alpha": 0.0, "r_squared": 0.0}

    def compute_tracking_error(
        self, benchmark_returns: pd.Series, return_method: str = "simple"
    ) -> float:
        """Compute tracking error vs benchmark.

        Args:
            benchmark_returns: Benchmark return series
            return_method: Return calculation method

        Returns:
            Tracking error (annualized) as float
        """
        portfolio_returns = self.performance.compute_returns(return_method)
        if portfolio_returns.empty or benchmark_returns.empty:
            return 0.0

        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if common_dates.empty:
            return 0.0

        portfolio_aligned = portfolio_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]

        # Active returns standard deviation
        active_returns = portfolio_aligned - benchmark_aligned
        return active_returns.std() * np.sqrt(252)

    def compute_downside_capture(
        self, benchmark_returns: pd.Series, return_method: str = "simple"
    ) -> dict[str, float]:
        """Compute upside and downside capture ratios.

        Args:
            benchmark_returns: Benchmark return series
            return_method: Return calculation method

        Returns:
            Dictionary with capture ratios
        """
        portfolio_returns = self.performance.compute_returns(return_method)
        if portfolio_returns.empty or benchmark_returns.empty:
            return {"upside_capture": 0.0, "downside_capture": 0.0}

        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if common_dates.empty:
            return {"upside_capture": 0.0, "downside_capture": 0.0}

        portfolio_aligned = portfolio_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]

        # Upside periods
        up_periods = benchmark_aligned > 0
        if up_periods.sum() > 0:
            portfolio_up = portfolio_aligned[up_periods].mean()
            benchmark_up = benchmark_aligned[up_periods].mean()
            upside_capture = portfolio_up / benchmark_up if benchmark_up != 0 else 0.0
        else:
            upside_capture = 0.0

        # Downside periods
        down_periods = benchmark_aligned < 0
        if down_periods.sum() > 0:
            portfolio_down = portfolio_aligned[down_periods].mean()
            benchmark_down = benchmark_aligned[down_periods].mean()
            downside_capture = (
                portfolio_down / benchmark_down if benchmark_down != 0 else 0.0
            )
        else:
            downside_capture = 0.0

        return {"upside_capture": upside_capture, "downside_capture": downside_capture}

    def compute_tail_ratio(self, return_method: str = "simple") -> float:
        """Compute tail ratio (95th percentile / 5th percentile).

        Args:
            return_method: Return calculation method

        Returns:
            Tail ratio as float
        """
        returns = self.performance.compute_returns(return_method)
        if returns.empty:
            return 0.0

        p95 = np.quantile(returns, 0.95)
        p5 = np.quantile(returns, 0.05)

        if p5 == 0:
            return float("inf") if p95 > 0 else 0.0

        return abs(p95 / p5)

    def compute_pain_index(self, return_method: str = "simple") -> float:
        """Compute Pain Index (average drawdown over time).

        Args:
            return_method: Return calculation method

        Returns:
            Pain Index as float
        """
        cumulative_returns = self.performance.compute_cumulative_returns(return_method)
        if cumulative_returns.empty:
            return 0.0

        # Calculate running maximum
        running_max = cumulative_returns.cummax()

        # Calculate drawdown for each period
        drawdown = cumulative_returns - running_max

        # Pain Index is the mean of all negative drawdowns
        return abs(drawdown.mean())

    def compute_risk_summary(
        self,
        return_method: str = "simple",
        benchmark_returns: pd.Series | None = None,
        risk_free_rate: float = 0.0,
        var_confidence: float = 0.05,
    ) -> dict[str, Any]:
        """Compute comprehensive risk summary.

        Args:
            return_method: Return calculation method
            benchmark_returns: Optional benchmark for relative metrics
            risk_free_rate: Risk-free rate for ratios
            var_confidence: VaR confidence level

        Returns:
            Dictionary with risk metrics
        """
        try:
            # Basic performance metrics
            volatility = self.performance.compute_volatility(
                return_method, annualized=True
            )
            sharpe = self.performance.compute_sharpe_ratio(
                risk_free_rate, return_method
            )

            # Risk-adjusted ratios
            sortino = self.compute_sortino_ratio(risk_free_rate, return_method)
            calmar = self.compute_calmar_ratio(return_method)

            # Drawdown metrics
            drawdown_info = self.performance.compute_max_drawdown(return_method)
            pain_index = self.compute_pain_index(return_method)

            # VaR metrics
            var_info = self.compute_var(var_confidence, "historical", return_method)

            # Tail risk
            tail_ratio = self.compute_tail_ratio(return_method)

            summary = {
                "volatility_metrics": {
                    "volatility": volatility,
                    "downside_deviation": self._compute_downside_deviation(
                        return_method
                    ),
                },
                "risk_adjusted_ratios": {
                    "sharpe_ratio": sharpe,
                    "sortino_ratio": sortino,
                    "calmar_ratio": calmar,
                },
                "drawdown_metrics": {
                    "max_drawdown": drawdown_info["max_drawdown"],
                    "max_drawdown_duration": drawdown_info["duration"],
                    "pain_index": pain_index,
                },
                "var_metrics": {"var_95": var_info["var"], "cvar_95": var_info["cvar"]},
                "tail_metrics": {"tail_ratio": tail_ratio},
            }

            # Add benchmark-relative metrics if benchmark provided
            if benchmark_returns is not None and not benchmark_returns.empty:
                beta_info = self.compute_beta(benchmark_returns, return_method)
                info_ratio = self.compute_information_ratio(
                    benchmark_returns, return_method
                )
                tracking_error = self.compute_tracking_error(
                    benchmark_returns, return_method
                )
                capture_ratios = self.compute_downside_capture(
                    benchmark_returns, return_method
                )

                summary["benchmark_metrics"] = {
                    "beta": beta_info["beta"],
                    "alpha": beta_info["alpha"],
                    "r_squared": beta_info["r_squared"],
                    "information_ratio": info_ratio,
                    "tracking_error": tracking_error,
                    "upside_capture": capture_ratios["upside_capture"],
                    "downside_capture": capture_ratios["downside_capture"],
                }

            return summary

        except Exception as e:  # pylint: disable=broad-except
            return {"error": f"Risk calculation failed: {e}"}

    def _compute_downside_deviation(self, return_method: str = "simple") -> float:
        """Compute downside deviation (annualized)."""
        returns = self.performance.compute_returns(return_method)
        if returns.empty:
            return 0.0

        downside_returns = returns[returns < 0]
        if downside_returns.empty:
            return 0.0

        return downside_returns.std() * np.sqrt(252)
