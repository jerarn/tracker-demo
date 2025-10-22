#!/usr/bin/env python3
"""Portfolio Analytics Display."""

from datetime import datetime, timezone
import math

import pandas as pd

from tracker import AnalyticsService, DBManager, PortfolioAnalyzer, YFinanceAPI


def pretty_pct(x):
    """Format percentage."""
    return f"{x:.2%}" if x is not None and not math.isnan(x) else "n/a"


def pretty_currency(x, currency_sym="$"):
    """Format currency values with thousands separators."""
    return f"{currency_sym}{x:,.2f}" if x is not None and not math.isnan(x) else "n/a"


def print_header(title):
    """Print a major section header."""
    print(f"\n{'=' * 60}")
    print(f"📊 {title.upper()}")
    print(f"{'=' * 60}")


def print_section(title, emoji=""):
    """Print a section header with optional emoji."""
    print(f"\n{emoji} {title}")
    print("-" * max(len(title) + len(emoji) + 1, 30))


def detect_and_convert_var(summary):
    """Ensure VaR/CVaR printing uses consistent units."""
    pv = summary["portfolio_info"].get("current_value", None)
    var95 = summary["risk"]["var_metrics"].get("var_95", None)
    cvar95 = summary["risk"]["var_metrics"].get("cvar_95", None)

    converted = {}
    if pv is not None:
        for name, val in (("var_95", var95), ("cvar_95", cvar95)):
            if val is None:
                converted[name] = None
            elif abs(val) < 1:  # assume percent/fraction
                converted[name] = -float(val) * float(pv)  # make positive loss
            else:
                converted[name] = float(val)  # already monetary
    else:
        converted["var_95"], converted["cvar_95"] = var95, cvar95
    return converted


def main():
    """Main analytics display function."""
    print_header("Portfolio Tracker Analytics Results")

    try:
        # Fetch data and compute analytics
        db = DBManager()
        analytics_service = AnalyticsService(db)
        data_dfs = analytics_service.fetch_data_for_portfolios()

        analyzer = PortfolioAnalyzer(
            data_dfs["transactions"], data_dfs["cash_flows"], data_dfs["market_data"]
        )

        end_date = datetime(2024, 12, 31, tzinfo=timezone.utc)
        analyzer.compute_analytics(accounting_method="avg", end_date=end_date)
        summary = analyzer.get_portfolio_summary()

        # Data overview
        print_section("Data Overview", "📋")
        print(f"   Transactions:     {len(data_dfs['transactions'])} records")
        print(f"   Market Data:      {len(data_dfs['market_data'])} price points")
        print(f"   Cash Flows:       {len(data_dfs['cash_flows'])} entries")
        date_range = pd.to_datetime(data_dfs["market_data"].index)
        print(
            f"   Period:           {date_range.min().strftime('%Y-%m-%d')} to {date_range.max().strftime('%Y-%m-%d')}"
        )
        print(f"   Duration:         {(date_range.max() - date_range.min()).days} days")

        # Performance Overview
        print_section("Portfolio Performance", "💰")
        cv = summary["portfolio_info"].get("current_value", float("nan"))
        returns = summary["performance"]["returns"]

        print(f"   Current Value:    {pretty_currency(cv)}")
        print(f"   Cumul. Return:    {pretty_pct(returns.get('total_return'))}")
        print(f"   Annual. Return:   {pretty_pct(returns.get('annualized_return'))}")
        print(f"   Volatility:       {pretty_pct(returns.get('volatility'))}")
        print(f"   Sharpe Ratio:     {returns.get('sharpe_ratio', 0):.3f}")

        # Risk Analysis
        print_section("Risk Metrics", "⚠️")
        risk_metrics = summary["risk"]["drawdown_metrics"]
        print(f"   Max Drawdown:     {pretty_pct(risk_metrics.get('max_drawdown'))}")
        print(
            f"   Drawdown Days:    {risk_metrics.get('max_drawdown_duration', 'n/a')} days"
        )

        var95 = summary["risk"]["var_metrics"].get("var_95", None)
        cvar95 = summary["risk"]["var_metrics"].get("cvar_95", None)
        if var95 is not None:
            print(f"   VaR (95%):        {pretty_currency(-float(cv) * float(var95))}")
        if cvar95 is not None:
            print(f"   CVaR (95%):       {pretty_currency(-float(cv) * float(cvar95))}")

        # Additional risk metrics
        vol_metrics = summary["risk"]["volatility_metrics"]
        risk_ratios = summary["risk"]["risk_adjusted_ratios"]
        print(
            f"   Downside Dev:     {pretty_pct(vol_metrics.get('downside_deviation'))}"
        )
        print(f"   Sortino Ratio:    {risk_ratios.get('sortino_ratio', 0):.3f}")
        print(f"   Calmar Ratio:     {risk_ratios.get('calmar_ratio', 0):.3f}")

        # Portfolio Allocation
        print_section("Current Allocation", "🥧")
        try:
            alloc = analyzer.get_current_allocation()
            sorted_alloc = alloc.sort_values(ascending=False).head(10)

            for asset, weight in sorted_alloc.items():
                if weight > 0.5:  # Only show >0.5%
                    print(f"   {asset:<12}: {weight:>8}%")
        except Exception as e:
            print(f"   Allocation unavailable: {e}")

        # Benchmark Comparison
        print_section("Benchmark Analysis", "📈")
        try:
            # Use market data date range for benchmark
            md = data_dfs["market_data"]
            dates = pd.to_datetime(md.index)
            t0, t1 = dates.min(), dates.max()

            # Fetch benchmark data
            ticker = "ACWI"  # Global market benchmark
            yf = YFinanceAPI()
            raw = yf.fetch_data([ticker], start_date=t0, end_date=t1, datatype="close")
            bench_df = pd.DataFrame(raw[ticker])
            bench_df.index = pd.to_datetime(bench_df.index)
            bench_df.set_index("date", inplace=True)
            bench_returns = bench_df["quote"].pct_change().dropna()

            comparison = analyzer.compare_with_benchmark(bench_returns)

            print(
                f"   Portfolio Return: {pretty_pct(comparison['portfolio']['annualized_return'])}"
            )
            print(
                f"   Benchmark (ACWI): {pretty_pct(comparison['benchmark']['annualized_return'])}"
            )
            print(
                f"   Alpha:            {pretty_pct(comparison['relative_metrics']['alpha'])}"
            )
            print(f"   Beta:             {comparison['relative_metrics']['beta']:.3f}")

            # Outperformance
            portfolio_ret = comparison["portfolio"]["annualized_return"]
            benchmark_ret = comparison["benchmark"]["annualized_return"]
            if portfolio_ret and benchmark_ret:
                outperformance = portfolio_ret - benchmark_ret
                print(f"   Outperformance:   {pretty_pct(outperformance)}")

        except Exception as e:
            print(f"   Benchmark comparison failed: {str(e)[:50]}...")

    except Exception as e:
        print(f"❌ Error in analytics: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
