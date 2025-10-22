#!/usr/bin/env python3
"""Portfolio Analytics Dashboard using Streamlit.

Provides comprehensive visualization and analysis of portfolio performance,
risk metrics, allocations, and benchmarking in an interactive web interface.
"""

from datetime import datetime, timezone
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from tracker import AnalyticsService, DBManager, PortfolioAnalyzer, YFinanceAPI

# Page configuration
st.set_page_config(
    page_title="Portfolio Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def format_currency(value, currency="USD"):
    """Format currency values."""
    if value is None or math.isnan(value):
        return "N/A"
    symbol = "$" if currency == "USD" else currency
    return f"{symbol}{value:,.2f}"


def format_percentage(value):
    """Format percentage values."""
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.2%}"


def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a styled metric card."""
    st.metric(title, value, delta=delta, delta_color=delta_color)


@st.cache_data
def load_portfolio_data():
    """Load and cache portfolio data."""
    try:
        db = DBManager()
        analytics_service = AnalyticsService(db)

        # Fetch all portfolio data
        data_dfs = analytics_service.fetch_data_for_portfolios()

        return data_dfs, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def compute_analytics(data_dfs, accounting_method="avg"):
    """Compute and cache analytics."""
    try:
        analyzer = PortfolioAnalyzer(
            data_dfs["transactions"], data_dfs["cash_flows"], data_dfs["market_data"]
        )

        end_date = datetime(2024, 12, 31, tzinfo=timezone.utc)
        analyzer.compute_analytics(
            accounting_method=accounting_method, end_date=end_date
        )

        # Get summary and chart data
        summary = analyzer.get_portfolio_summary()
        chart_data = analyzer.get_performance_chart_data()
        allocation = analyzer.get_current_allocation()
        position_details = analyzer.get_position_details()

        return analyzer, summary, chart_data, allocation, position_details, None
    except Exception as e:
        return None, None, None, None, None, str(e)


@st.cache_data
def get_benchmark_data(start_date, end_date, ticker="ACWI"):
    """Get benchmark data for comparison."""
    try:
        yf = YFinanceAPI()
        raw = yf.fetch_data(
            [ticker], start_date=start_date, end_date=end_date, datatype="close"
        )
        bench_df = pd.DataFrame(raw[ticker])
        bench_df.index = pd.to_datetime(bench_df.index)
        bench_df = bench_df.set_index("date")
        bench_returns = bench_df["quote"].pct_change().dropna()
        return bench_returns, None
    except Exception as e:
        return None, str(e)


def plot_performance_chart(chart_data, data_dfs=None, benchmark_returns=None):
    """Create performance chart with multiple metrics."""
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Portfolio Value Over Time",
            "Cumulative Returns (%)",
            "Drawdown (%)",
        ),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.3, 0.2],
        shared_xaxes=True,  # This ensures x-axes are aligned
    )

    # Portfolio value - removed fill
    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["portfolio_value"],
            name="Portfolio Value",
            line={"color": "#1f77b4", "width": 3},
            mode="lines",
        ),
        row=1,
        col=1,
    )

    # Add cash flows and transactions markers if data is provided
    if data_dfs is not None:
        # Add cash flow markers (deposits/withdrawals)
        if not data_dfs["cash_flows"].empty:
            cf_df = data_dfs["cash_flows"].copy()
            # Date is in the index, reset it to make it a column
            cf_df = cf_df.reset_index()
            cf_df["date"] = pd.to_datetime(cf_df["date"])

            # Separate deposits and withdrawals
            deposits = cf_df[cf_df["cf_type"] == "DEPOSIT"]
            withdrawals = cf_df[cf_df["cf_type"] == "WITHDRAWAL"]
            dividends = cf_df[cf_df["cf_type"] == "DIVIDEND"]

            # Get portfolio values at cash flow dates
            for cf_type, cf_data, color, symbol in [
                ("Deposits", deposits, "#00ff00", "triangle-up"),
                ("Withdrawals", withdrawals, "#ff0000", "triangle-down"),
                ("Dividends", dividends, "#ffa500", "diamond"),
            ]:
                if not cf_data.empty:
                    # Match dates with portfolio values
                    cf_values = []
                    cf_dates = []
                    for _, row in cf_data.iterrows():
                        # Find closest portfolio value date
                        closest_date = chart_data.index[chart_data.index >= row["date"]]
                        if len(closest_date) > 0:
                            portfolio_val = chart_data.loc[
                                closest_date[0], "portfolio_value"
                            ]
                            cf_values.append(portfolio_val)
                            cf_dates.append(row["date"])

                    if cf_values:
                        fig.add_trace(
                            go.Scatter(
                                x=cf_dates,
                                y=cf_values,
                                mode="markers",
                                marker={
                                    "color": color,
                                    "size": 10,
                                    "symbol": symbol,
                                    "line": {"width": 2, "color": "white"},
                                },
                                name=cf_type,
                                hovertemplate=f"<b>{cf_type}</b><br>"
                                + "Date: %{x}<br>"
                                + "Portfolio Value: $%{y:,.2f}<extra></extra>",
                            ),
                            row=1,
                            col=1,
                        )

        # Add transaction markers (buys/sells)
        if not data_dfs["transactions"].empty:
            tx_df = data_dfs["transactions"].copy()
            # Date is in the index, reset it to make it a column
            tx_df = tx_df.reset_index()
            tx_df["date"] = pd.to_datetime(tx_df["date"])

            # Separate buys and sells
            buys = tx_df[tx_df["tx_type"] == "BUY"]
            sells = tx_df[tx_df["tx_type"] == "SELL"]

            for tx_type, tx_data, color, symbol in [
                ("Buys", buys, "#32cd32", "circle"),
                ("Sells", sells, "#dc143c", "x"),
            ]:
                if not tx_data.empty:
                    # Match dates with portfolio values
                    tx_values = []
                    tx_dates = []
                    tx_instruments = []
                    tx_amounts = []

                    for _, row in tx_data.iterrows():
                        # Find closest portfolio value date
                        closest_date = chart_data.index[chart_data.index >= row["date"]]
                        if len(closest_date) > 0:
                            portfolio_val = chart_data.loc[
                                closest_date[0], "portfolio_value"
                            ]
                            tx_values.append(portfolio_val)
                            tx_dates.append(row["date"])
                            # Use ticker if available, otherwise use instrument_id
                            instrument = row.get(
                                "ticker", row.get("instrument_id", "Unknown")
                            )
                            tx_instruments.append(instrument)
                            tx_amounts.append(row["quantity"] * row["price"])

                    if tx_values:
                        fig.add_trace(
                            go.Scatter(
                                x=tx_dates,
                                y=tx_values,
                                mode="markers",
                                marker={
                                    "color": color,
                                    "size": 8,
                                    "symbol": symbol,
                                    "line": {"width": 1, "color": "white"},
                                },
                                name=tx_type,
                                customdata=list(
                                    zip(tx_instruments, tx_amounts, strict=True)
                                ),
                                hovertemplate=f"<b>{tx_type[:-1]}</b><br>"
                                + "Date: %{x}<br>"
                                + "Instrument: %{customdata[0]}<br>"
                                + "Amount: $%{customdata[1]:,.2f}<br>"
                                + "Portfolio Value: $%{y:,.2f}<extra></extra>",
                            ),
                            row=1,
                            col=1,
                        )

    # Cumulative returns - Portfolio
    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["cumulative_return"] * 100,
            name="Portfolio (%)",
            line={"color": "#2ca02c", "width": 2},
        ),
        row=2,
        col=1,
    )

    # Add benchmark cumulative returns if provided
    if benchmark_returns is not None and not benchmark_returns.empty:
        # Calculate benchmark cumulative returns
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1

        # Align dates with portfolio data
        common_dates = chart_data.index.intersection(benchmark_cumulative.index)
        if not common_dates.empty:
            benchmark_aligned = benchmark_cumulative.loc[common_dates]

            fig.add_trace(
                go.Scatter(
                    x=benchmark_aligned.index,
                    y=benchmark_aligned * 100,
                    name="Benchmark (%)",
                    line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
                ),
                row=2,
                col=1,
            )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["drawdown"] * 100,
            name="Drawdown (%)",
            line={"color": "#d62728", "width": 2},
            fill="tonexty",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=800,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "center",
            "x": 0.5,
        },
    )

    # Update x-axes to ensure alignment
    fig.update_xaxes(showgrid=True, matches="x")
    fig.update_yaxes(showgrid=True)

    # Only show x-axis labels on bottom chart
    fig.update_xaxes(showticklabels=False, row=1)
    fig.update_xaxes(showticklabels=False, row=2)
    fig.update_xaxes(showticklabels=True, row=3)

    return fig


def plot_allocation_pie(allocation):
    """Create allocation pie chart."""
    # Filter out very small allocations
    allocation_filtered = allocation[allocation > 0.5]

    fig = px.pie(
        values=allocation_filtered.values,
        names=allocation_filtered.index,
        title="Current Portfolio Allocation",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{percent}<br>%{value:.2f}%<extra></extra>",
    )

    fig.update_layout(showlegend=True, height=400)

    return fig


def display_risk_metrics(summary):
    """Display risk metrics in a structured way."""
    risk_data = summary.get("risk", {})

    # Create a more responsive layout with better organization
    st.subheader("⚠️ Risk Analysis")

    # Primary risk metrics in a 2x2 grid
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📉 Drawdown & Volatility**")
        drawdown_metrics = risk_data.get("drawdown_metrics", {})
        vol_metrics = risk_data.get("volatility_metrics", {})

        # Drawdown metrics
        max_dd = drawdown_metrics.get("max_drawdown")
        dd_duration = drawdown_metrics.get("max_drawdown_duration", "N/A")
        volatility = vol_metrics.get("volatility")
        downside_vol = vol_metrics.get("downside_deviation")

        # Display in a clean format
        st.metric(
            "Max Drawdown",
            format_percentage(max_dd),
            help="Largest peak-to-trough decline. Lower is better (less risk)",
        )
        st.metric(
            "Volatility",
            format_percentage(volatility),
            help="Standard deviation of returns. Measures price fluctuation intensity",
        )
        st.metric(
            "Downside Deviation",
            format_percentage(downside_vol),
            help="Volatility of negative returns only. Focus on downside risk",
        )
        st.caption(f"Max Drawdown Duration: {dd_duration} days")
        if dd_duration != "N/A" and isinstance(dd_duration, (int, float)):
            dd_years = dd_duration / 252
            st.caption(f"≈ {dd_years:.1f} years to recover from worst loss")

    with col2:
        st.markdown("**📊 Risk-Adjusted Performance**")
        risk_ratios = risk_data.get("risk_adjusted_ratios", {})
        var_metrics = risk_data.get("var_metrics", {})

        # Risk ratios
        sharpe = risk_ratios.get("sharpe_ratio", 0)
        sortino = risk_ratios.get("sortino_ratio", 0)
        calmar = risk_ratios.get("calmar_ratio", 0)

        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.3f}",
            help="Return per unit of total risk. >1 good, >2 excellent, >3 outstanding",
        )
        st.metric(
            "Sortino Ratio",
            f"{sortino:.3f}",
            help="Return per unit of downside risk. Higher is better (focuses on bad volatility)",
        )
        st.metric(
            "Calmar Ratio",
            f"{calmar:.3f}",
            help="Annual return divided by max drawdown. Measures return vs worst loss",
        )

    # Additional risk metrics in expandable section
    with st.expander("📋 Additional Risk Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Value at Risk (95%)**")
            var_95 = var_metrics.get("var_95")
            cvar_95 = var_metrics.get("cvar_95")

            # Convert VaR to monetary terms if needed
            portfolio_value = summary.get("portfolio_info", {}).get("current_value", 0)
            if var_95 is not None and portfolio_value > 0:
                if abs(var_95) < 1:  # Percentage form
                    var_dollar = abs(var_95) * portfolio_value
                    cvar_dollar = abs(cvar_95) * portfolio_value if cvar_95 else None
                else:  # Already in dollar form
                    var_dollar = abs(var_95)
                    cvar_dollar = abs(cvar_95) if cvar_95 else None

                st.metric(
                    "VaR (95%)",
                    f"${var_dollar:,.0f}",
                    help="Maximum expected loss over 1 day with 95% confidence",
                )
                if cvar_dollar:
                    st.metric(
                        "CVaR (95%)",
                        f"${cvar_dollar:,.0f}",
                        help="Average loss when VaR threshold is exceeded (worst 5% of days)",
                    )
            else:
                st.metric("VaR (95%)", "N/A")
                st.metric("CVaR (95%)", "N/A")

        with col2:
            st.markdown("**Drawdown Details**")
            pain_index = drawdown_metrics.get("pain_index")
            st.metric(
                "Pain Index",
                f"{pain_index:.4f}" if pain_index else "N/A",
                help="Average drawdown severity over time. Lower is better",
            )

            # Show percentage form of max drawdown duration
            if dd_duration != "N/A" and isinstance(dd_duration, (int, float)):
                # Assuming roughly 252 trading days per year
                dd_years = dd_duration / 252
                st.caption(f"≈ {dd_years:.1f} years")

        with col3:
            st.markdown("**Distribution Metrics**")
            tail_metrics = risk_data.get("tail_metrics", {})
            tail_ratio = tail_metrics.get("tail_ratio")
            st.metric(
                "Tail Ratio",
                f"{tail_ratio:.3f}" if tail_ratio else "N/A",
                help="Ratio of upside to downside tail risk. >1 means more upside potential",
            )

            # Additional context
            if tail_ratio and tail_ratio > 1:
                st.caption("✅ More upside than downside tail risk")
            elif tail_ratio and tail_ratio < 1:
                st.caption("⚠️ More downside than upside tail risk")
            else:
                st.caption("Balanced tail risk")


def display_position_details(position_details):
    """Display detailed position information."""
    if position_details.empty:
        st.warning("No position data available")
        return

    # Add styling for positive/negative values
    def style_pnl(val):
        if isinstance(val, (int, float)):
            color = "green" if val >= 0 else "red"
            return f"color: {color}"
        return ""

    # Format the dataframe for display
    display_df = position_details.copy()
    display_df["quantity"] = display_df["quantity"].apply(lambda x: f"{x:.4f}")
    display_df["avg_cost"] = display_df["avg_cost"].apply(lambda x: format_currency(x))
    display_df["current_price"] = display_df["current_price"].apply(
        lambda x: format_currency(x)
    )
    display_df["cost_basis"] = display_df["cost_basis"].apply(
        lambda x: format_currency(x)
    )
    display_df["market_value"] = display_df["market_value"].apply(
        lambda x: format_currency(x)
    )
    display_df["unrealized_pnl"] = display_df["unrealized_pnl"].apply(
        lambda x: format_currency(x)
    )
    display_df["pnl_percent"] = display_df["pnl_percent"].apply(
        lambda x: format_percentage(x / 100)
    )

    # Rename columns for better display
    display_df.columns = [
        "Quantity",
        "Avg Cost",
        "Current Price",
        "Cost Basis",
        "Market Value",
        "Unrealized P&L",
        "P&L %",
    ]

    st.dataframe(
        display_df.style.applymap(
            lambda x: style_pnl(
                float(x.replace("$", "").replace(",", "")) if "$" in str(x) else x
            ),
            subset=["Unrealized P&L", "P&L %"],
        ),
        use_container_width=True,
    )


def main():
    """Main dashboard application."""
    st.title("📊 Portfolio Analytics Dashboard")
    st.markdown(
        "**Real-time portfolio performance analysis with comprehensive risk metrics and visualizations**"
    )
    st.markdown("---")

    # Load data
    with st.spinner("Loading portfolio data..."):
        data_dfs, data_error = load_portfolio_data()

    if data_error:
        st.error(f"Error loading data: {data_error}")
        return

    if data_dfs is None:
        st.error("No data available")
        return

    # Data overview in sidebar
    st.sidebar.header("Data Overview")
    st.sidebar.metric("Transactions", len(data_dfs["transactions"]))
    st.sidebar.metric("Cash Flows", len(data_dfs["cash_flows"]))
    st.sidebar.metric("Market Data Points", len(data_dfs["market_data"]))

    # Compute analytics
    with st.spinner("Computing analytics..."):
        _, summary, chart_data, allocation, position_details, analytics_error = (
            compute_analytics(data_dfs, "avg")
        )

    if analytics_error:
        st.error(f"Error computing analytics: {analytics_error}")
        return

    # Main dashboard content
    if summary is None:
        st.error("No analytics summary available")
        return

    # Portfolio Overview
    st.header("📋 Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)

    portfolio_info = summary.get("portfolio_info", {})
    performance = summary.get("performance", {}).get("returns", {})

    with col1:
        st.metric(
            "Current Value",
            format_currency(portfolio_info.get("current_value")),
            help="Total current market value of all positions including cash",
        )

    with col2:
        st.metric(
            "Total Return",
            format_percentage(performance.get("total_return")),
            help="Total cumulative return since inception (capital gains + dividends)",
        )

    with col3:
        st.metric(
            "Annualized Return",
            format_percentage(performance.get("annualized_return")),
            help="Compound annual growth rate (CAGR) - what you'd earn per year on average",
        )

    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{performance.get('sharpe_ratio', 0):.3f}",
            help="Risk-adjusted return measure. >1 is good, >2 is excellent (excess return per unit of risk)",
        )

    st.markdown("---")

    # Get benchmark data early for use in performance charts
    market_data = data_dfs["market_data"]
    if hasattr(market_data.index, "min"):
        start_date = market_data.index.min()
        end_date = market_data.index.max()
    else:
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

    benchmark_returns, bench_error = get_benchmark_data(start_date, end_date)
    if bench_error:
        st.warning(f"Benchmark data unavailable: {bench_error}")
        benchmark_returns = None

    # Performance Charts
    st.header("📈 Performance Analysis")
    st.markdown(
        "Track portfolio value, returns, and drawdowns over time. Markers show cash flows and transactions."
    )

    if not chart_data.empty:
        perf_fig = plot_performance_chart(chart_data, data_dfs, benchmark_returns)
        st.plotly_chart(perf_fig, use_container_width=True)
    else:
        st.warning("No chart data available")

    # Two-column layout for allocation and risk
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🥧 Current Allocation")
        if not allocation.empty and allocation.sum() > 0:
            alloc_fig = plot_allocation_pie(allocation)
            st.plotly_chart(alloc_fig, use_container_width=True)
        else:
            st.warning("No allocation data available")

    with col2:
        display_risk_metrics(summary)

    st.markdown("---")

    # Position Details
    st.header("📋 Position Details")
    st.markdown(
        "Current holdings with cost basis, market values, and unrealized gains/losses."
    )

    if not position_details.empty:
        display_position_details(position_details)
    else:
        st.warning("No position details available")


if __name__ == "__main__":
    main()
