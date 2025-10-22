"""Centralized price data management.

Handles market data + transaction price overlays.
"""

import pandas as pd


class PriceManager:
    """Centralized price data management.

    Handles market data + transaction price overlays.
    """

    def __init__(self, market_data_df: pd.DataFrame, transaction_data_df: pd.DataFrame):
        """Initialize with market data and transaction DataFrames."""
        self.market_data_df = market_data_df
        self.transaction_data_df = transaction_data_df

    def get_prices(
        self, date_index: pd.DatetimeIndex, use_transaction_prices: bool = False
    ) -> pd.DataFrame:
        """Get price data aligned to date index.

        Args:
            date_index: Target date index
            use_transaction_prices: Whether to overlay transaction prices on transaction dates

        Returns:
            DataFrame with price data aligned to date_index
        """
        if self.market_data_df.empty:
            return pd.DataFrame(index=date_index)

        # Start with market data aligned to date index
        aligned_prices = self.market_data_df.reindex(date_index).ffill().bfill()

        if not use_transaction_prices:
            return aligned_prices

        # Overlay transaction prices on transaction dates
        tx_data = self.transaction_data_df

        if not tx_data.empty:
            # Filter to only dates in our index
            tx_data = tx_data[tx_data.index.isin(date_index)]

            for date, row in tx_data.iterrows():
                ticker = row["ticker"]
                price = row["price"]
                tx_type = row["tx_type"]

                # Only override if we have this ticker in our price data
                if (
                    ticker in aligned_prices.columns
                    and date in aligned_prices.index
                    and tx_type not in ["TRANSFER_IN", "TRANSFER_OUT"]
                ):
                    aligned_prices.loc[date, ticker] = price

        return aligned_prices

    def get_available_tickers(self) -> list[str]:
        """Get list of all available tickers."""
        tickers = set()

        if not self.market_data_df.empty:
            tickers.update(self.market_data_df.columns)

        if not self.transaction_data_df.empty:
            tickers.update(self.transaction_data_df["ticker"].unique())

        return sorted(tickers)

    def get_date_range(self) -> tuple:
        """Get the available date range (start, end)."""
        start_dates = []
        end_dates = []

        if not self.market_data_df.empty:
            start_dates.append(self.market_data_df.index.min())
            end_dates.append(self.market_data_df.index.max())

        if not self.transaction_data_df.empty:
            start_dates.append(self.transaction_data_df.index.min())
            end_dates.append(self.transaction_data_df.index.max())

        if start_dates and end_dates:
            return min(start_dates), max(end_dates)

        return None, None
