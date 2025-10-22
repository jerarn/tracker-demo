"""Service for portfolio analytics operations."""

from typing import Any

import pandas as pd

from tracker.data.managers.db_manager import DBManager
from tracker.data.orm.cash_flow import CashFlow
from tracker.data.orm.instrument import Instrument
from tracker.data.orm.market_data import MarketData
from tracker.data.orm.portfolio import Portfolio
from tracker.data.orm.transaction import Transaction
from tracker.data.repos.cf_repo import CashFlowRepository
from tracker.data.repos.mkt_repo import MarketDataRepository
from tracker.data.repos.repository import Repository
from tracker.data.repos.tx_repo import TransactionRepository

from .service import Service


class AnalyticsService(Service):
    """Service for portfolio analytics operations."""

    def __init__(self, db_manager: DBManager | None = None):
        """Initialize the analytics service."""
        super().__init__(db_manager)
        self.tx_repo: TransactionRepository = self.get_repository(Transaction)
        self.cf_repo: CashFlowRepository = self.get_repository(CashFlow)
        self.mkt_repo: MarketDataRepository = self.get_repository(MarketData)

    def _consolidate_market_data_df(
        self, mkt_df: pd.DataFrame, instruments: list[Instrument]
    ) -> pd.DataFrame:
        """Market price data pivoted with dates as index and tickers as columns."""
        if mkt_df.empty:
            return pd.DataFrame(
                {"CASH": 1.0},
                index=pd.DatetimeIndex([pd.Timestamp.now(tz="UTC")]).normalize(),
            )

        mkt_df = mkt_df[mkt_df["data_type_name"] == "close"]

        priority = {"yfinance": 1}
        mkt_df["priority"] = mkt_df["data_source_name"].map(priority).fillna(99)
        mkt_df = (
            mkt_df.sort_values("priority")
            .groupby(["date", "ticker"])
            .first()
            .reset_index()
        )
        mkt_df.drop("priority", axis=1, inplace=True)
        mkt_df.set_index("date", inplace=True)

        pivot_df = (
            mkt_df.pivot(columns="ticker", values="quote")
            if not mkt_df.empty
            else pd.DataFrame()
        )

        pivot_df["CASH"] = 1.0

        tickers = [instrument.ticker for instrument in instruments]
        if "BETH-EUR" in tickers:
            if "ETH-EUR" not in pivot_df.columns:
                raise ValueError(
                    "BETH-EUR found but ETH-EUR missing. Cannot compute BETH-EUR prices."
                )

            pivot_df["BETH-EUR"] = pivot_df["ETH-EUR"]

        for instrument in instruments:
            if (
                instrument.asset_class_name == "Cash"
                and instrument.ticker not in pivot_df.columns
            ):
                pivot_df[instrument.ticker] = 1.0

        return pivot_df

    def fetch_data_for_portfolios(
        self, filters: dict[str, Any] | None = None, strategy: str = "optimized"
    ) -> dict[str, pd.DataFrame]:
        """Fetch all necessary data for analytics on the given portfolios."""
        repo = Repository(Portfolio, self.db_manager)

        with self.db_manager.get_session() as session:
            portfolios = repo.get(
                filters=filters, load=False, method="all", session=session
            )
            portfolio_ids = [p.id for p in portfolios if p.id is not None]

            tx_df = self.tx_repo.fetch_from_portfolio_ids(
                portfolio_ids, strategy=strategy, session=session
            )
            cf_df = self.cf_repo.fetch_from_portfolio_ids(
                portfolio_ids, strategy=strategy, session=session
            )

            instrument_ids = (
                tx_df["instrument_id"].dropna().unique().tolist()
                if not tx_df.empty and "instrument_id" in tx_df.columns
                else []
            )
            mkt_df = self.mkt_repo.fetch_from_instrument_ids(
                instrument_ids, strategy=strategy, session=session
            )

            instruments = [session.query(Instrument).get(iid) for iid in instrument_ids]
            mkt_df = self._consolidate_market_data_df(mkt_df, instruments)

        return {
            "transactions": tx_df,
            "cash_flows": cf_df,
            "market_data": mkt_df,
        }
