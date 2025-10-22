"""Pure holdings tracking - positions and cost basis only.

No performance calculations.
"""

from typing import ClassVar

import pandas as pd

from tracker.analytics.accounting.avg_cost import AverageCost
from tracker.analytics.accounting.cash import Cash
from tracker.analytics.accounting.fifo import FIFO
from tracker.analytics.accounting.lifo import LIFO
from tracker.analytics.accounting.strategy import AccountingStrategy


class Holdings:
    """Pure holdings tracking - positions and cost basis only.

    No performance calculations, just position accounting.
    """

    # Class-level strategy cache to avoid repeated instantiation
    _STRATEGIES: ClassVar[dict[str, AccountingStrategy]] = {
        "avg": AverageCost(),
        "fifo": FIFO(),
        "lifo": LIFO(),
        "cash": Cash(),
    }

    def __init__(self, transactions_df: pd.DataFrame, cash_flows_df: pd.DataFrame):
        """Initialize with transactions and cash flows DataFrames.

        Args:
            transactions_df: DataFrame of transactions
            cash_flows_df: DataFrame of cash flows
        """
        self.transactions_df = transactions_df.copy()
        self.cash_flows_df = cash_flows_df.copy()
        self.positions_df = None

    def compute_positions(
        self, accounting_method: str = "avg", end_date: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Compute positions using specified accounting method.

        Args:
            accounting_method: Accounting method ('avg', 'fifo', 'lifo')
            end_date: End date for calculation (default: today)

        Returns:
            DataFrame with MultiIndex columns (metric, ticker)
            Metrics: qty, cost_basis, avg_cost, realized_pnl
        """
        if accounting_method not in self._STRATEGIES:
            raise ValueError(
                f"Unknown accounting method: {accounting_method}. "
                f"Available methods: {list(self._STRATEGIES.keys())}"
            )

        # Get consolidated transaction data
        all_data = self._prepare_transaction_data()
        if all_data.empty:
            self.positions_df = pd.DataFrame()
            return self.positions_df

        # Build date index
        date_index = self._build_date_index(all_data, end_date)

        # Process each ticker
        tickers = all_data["ticker"].dropna().unique()
        results = {}

        for ticker in tickers:
            ticker_data = all_data[all_data["ticker"] == ticker].sort_values(
                ["date", "order"]
            )
            ticker_results = self._process_ticker_holdings(
                ticker_data, date_index, ticker, accounting_method
            )

            for metric in ["qty", "cost_basis", "avg_cost", "realized_pnl"]:
                results[(metric, ticker)] = ticker_results[metric]

        # Combine results
        if results:
            self.positions_df = pd.concat(results.values(), axis=1)
            self.positions_df.columns = pd.MultiIndex.from_tuples(
                results.keys(), names=("metric", "ticker")
            )
            self.positions_df.index.name = "date"
        else:
            self.positions_df = pd.DataFrame(index=date_index)

        return self.positions_df

    def get_portfolio_ids(self) -> list[int]:
        """Get list of all portfolio IDs in holdings."""
        portfolio_ids = set()
        if not self.transactions_df.empty:
            portfolio_ids.update(self.transactions_df["portfolio_id"].unique())
        if not self.cash_flows_df.empty:
            portfolio_ids.update(self.cash_flows_df["portfolio_id"].unique())

        return sorted(portfolio_ids)

    def get_start_dates(self) -> list[pd.Timestamp]:
        """Get list of all portfolio start dates in holdings."""
        start_dates = set()
        if not self.transactions_df.empty:
            start_dates.update(
                self.transactions_df.groupby("portfolio_id").apply(
                    lambda df: df.index.min(), include_groups=False
                )
            )
        if not self.cash_flows_df.empty:
            start_dates.update(
                self.cash_flows_df.groupby("portfolio_id").apply(
                    lambda df: df.index.min(), include_groups=False
                )
            )

        return sorted(start_dates)

    def get_instruments(self) -> list[str]:
        """Get list of all instruments (tickers) in holdings."""
        if self.positions_df is None:
            raise ValueError("Positions not computed. Call compute_positions() first.")

        if self.positions_df.empty:
            return []

        return sorted(
            self.positions_df.columns.get_level_values("ticker")
            .unique()
            .drop("CASH", errors="ignore")
        )

    def get_quantities(self) -> pd.DataFrame:
        """Get quantity data only."""
        if self.positions_df is None:
            raise ValueError("Positions not computed. Call compute_positions() first.")

        if self.positions_df.empty:
            return pd.DataFrame()

        return self.positions_df.xs("qty", level="metric", axis=1)

    def get_cost_basis(self) -> pd.DataFrame:
        """Get cost basis data only."""
        if self.positions_df is None:
            raise ValueError("Positions not computed. Call compute_positions() first.")

        if self.positions_df.empty:
            return pd.DataFrame()

        return self.positions_df.xs("cost_basis", level="metric", axis=1)

    def get_average_costs(self) -> pd.DataFrame:
        """Get average cost data only."""
        if self.positions_df is None:
            raise ValueError("Positions not computed. Call compute_positions() first.")

        if self.positions_df.empty:
            return pd.DataFrame()

        return self.positions_df.xs("avg_cost", level="metric", axis=1)

    def get_realized_pnl(self) -> pd.DataFrame:
        """Get realized P&L data only."""
        if self.positions_df is None:
            raise ValueError("Positions not computed. Call compute_positions() first.")

        if self.positions_df.empty:
            return pd.DataFrame()

        return self.positions_df.xs("realized_pnl", level="metric", axis=1)

    def get_latest_positions(self) -> pd.Series:
        """Get the most recent positions for each ticker."""
        quantities = self.get_quantities()
        if quantities.empty:
            return pd.Series(dtype=float)

        return quantities.iloc[-1].dropna()

    def _prepare_transaction_data(self) -> pd.DataFrame:
        """Prepare and normalize all transaction data."""
        # Prepare cash flow data
        tx_dfs = self.transactions_df
        cf_dfs = self.cash_flows_df
        if not cf_dfs.empty:
            cf_dfs = cf_dfs.rename(columns={"amount": "quantity", "cf_type": "tx_type"})
            cf_dfs["price"] = 1.0
            cf_dfs["ticker"] = "CASH"

        if tx_dfs.empty and cf_dfs.empty:
            return pd.DataFrame()
        if tx_dfs.empty:
            all_data = cf_dfs
        elif cf_dfs.empty:
            all_data = tx_dfs
        else:
            # Combine all data
            all_data = pd.concat([tx_dfs, cf_dfs], sort=False)

        # Filter internal transfers
        all_data = self._filter_internal_transfers(all_data)

        # Add transaction ordering
        return self._add_transaction_order(all_data)

    def _filter_internal_transfers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out internal transfers."""
        if df.empty:
            return df

        df = df.reset_index()
        transfers = df[df["tx_type"].isin(["TRANSFER_IN", "TRANSFER_OUT"])].copy()
        if transfers.empty:
            return df.set_index("date")

        # Create matching keys and find internal transfer pairs
        transfers["match_key"] = (
            transfers["ticker"].astype(str)
            + "_"
            + transfers["date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
            + "_"
            + transfers["quantity"].abs().astype(str)
        )

        transfer_counts = transfers.groupby("match_key")["tx_type"].apply(set)
        internal_keys = transfer_counts[
            transfer_counts.apply({"TRANSFER_IN", "TRANSFER_OUT"}.issubset)
        ].index

        internal_indices = transfers[transfers["match_key"].isin(internal_keys)].index
        df = df.drop(internal_indices)
        return df.set_index("date")

    def _add_transaction_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transaction ordering for same-day transactions."""
        order = [
            "DEPOSIT",
            "WITHDRAWAL",
            "SELL",
            "BUY",
            "DIVIDEND",
            "INTEREST",
            "FEE",
            "TAX",
            "TRANSFER_OUT",
            "TRANSFER_IN",
        ]
        df["order"] = pd.Categorical(df["tx_type"], categories=order, ordered=True)
        return df

    def _build_date_index(
        self,
        df: pd.DataFrame,
        end_date: pd.Timestamp | None = None,
        use_transaction_dates: bool = False,
    ) -> pd.DatetimeIndex:
        """Build comprehensive date index."""
        # Get start dates from portfolios
        if df.empty:
            return pd.DatetimeIndex([])

        start = df.index.min().normalize()
        end = end_date or pd.Timestamp.utcnow().normalize()

        # Create daily range
        daily_range = pd.date_range(start=start, end=end, freq="D")

        # Add transaction dates
        transaction_dates = (
            df[
                (
                    df["tx_type"].isin(
                        [
                            "BUY",
                            "SELL",
                            "DIVIDEND",
                            "INTEREST",
                            "FEE",
                            "TAX",
                            "TRANSFER_IN",
                            "TRANSFER_OUT",
                        ]
                    )
                )
                & (df.index <= end)
            ].index.unique()
            if not df.empty
            else []
        )

        # Combine and sort
        all_dates = (
            set(daily_range.tolist() + list(transaction_dates))
            if use_transaction_dates
            else set(daily_range.tolist())
        )
        return pd.DatetimeIndex(sorted(all_dates))

    def _process_ticker_holdings(
        self,
        df: pd.DataFrame,
        date_index: pd.DatetimeIndex,
        ticker: str,
        accounting_method: str,
    ) -> pd.DataFrame:
        """Process holdings for a single ticker."""
        if df.empty:
            return pd.DataFrame(
                {"qty": 0.0, "cost_basis": 0.0, "avg_cost": 0.0, "realized_pnl": 0.0},
                index=date_index,
            )

        # Select appropriate strategy
        strategy = self._STRATEGIES.get(
            "cash" if ticker == "CASH" else accounting_method
        )
        if strategy is None:
            raise ValueError(f"Unknown accounting method: {accounting_method}")

        # Process transactions sequentially
        state = {}
        cumulative_realized = 0.0
        results = []

        for date, row in df.iterrows():
            state, realized = strategy.process_transaction(
                state, row["tx_type"], row["quantity"], row["price"]
            )
            cumulative_realized += realized
            metrics = strategy.get_current_metrics(state)

            results.append(
                {
                    "date": date.normalize(),
                    "qty": metrics["qty"],
                    "cost_basis": metrics["cost_basis"],
                    "avg_cost": metrics["avg_cost"],
                    "realized_pnl": cumulative_realized,
                }
            )

        # Create DataFrame and reindex
        if results:
            events_df = pd.DataFrame(results).set_index("date")
            events_df = events_df.groupby("date").last()
            return events_df.reindex(date_index).ffill().fillna(0.0)
        return pd.DataFrame(
            {"qty": 0.0, "cost_basis": 0.0, "avg_cost": 0.0, "realized_pnl": 0.0},
            index=date_index,
        )
