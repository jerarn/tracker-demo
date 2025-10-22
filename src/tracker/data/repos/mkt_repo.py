"""Repository for MarketData entities."""

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from tracker.config.logger import get_logger
from tracker.data.managers.db_manager import DBManager
from tracker.data.orm.market_data import MarketData

from .repository import Repository

logger = get_logger(__name__)


class MarketDataRepository(Repository[MarketData]):
    """Repository for managing MarketData entities in the database."""

    def __init__(self, db_manager: DBManager | None = None):
        """Initialize MarketDataRepository with DBManager."""
        super().__init__(MarketData, db_manager)

    def _postprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            df["quote"] = df["quote"].astype(float)

        return df

    def fetch_from_instrument_ids(
        self,
        instrument_ids: list[int],
        strategy: str,
        session: Session | None = None,
    ) -> pd.DataFrame:
        """Fetch all market data for the given instruments."""
        if not instrument_ids:
            return pd.DataFrame()

        with self.get_session(session) as session:
            if strategy == "normal":
                market_data = (
                    session.query(MarketData)
                    .filter(MarketData.instrument_id.in_(instrument_ids))
                    .order_by(MarketData.date, MarketData.id)
                    .all()
                )

                df = self.to_df(market_data)
                return self._postprocess_df(df)

            if strategy == "optimized":
                stmt = (
                    select(
                        MarketData.id,
                        MarketData.date,
                        MarketData.instrument_id,
                        MarketData.ticker,
                        MarketData.quote,
                        MarketData.data_source_name,
                        MarketData.data_type_name,
                    )
                    .where(MarketData.instrument_id.in_(instrument_ids))
                    .order_by(MarketData.date, MarketData.id)
                )
                results = session.execute(stmt)

                df = pd.DataFrame(
                    results.fetchall(), columns=results.keys() if results else []
                )
                return self._postprocess_df(df)

            if strategy == "sql":
                stmt = """
                SELECT md.id, md.date, md.instrument_id, i.ticker, md.quote, ds.name AS data_source_name, dt.name AS data_type_name
                FROM market_data md
                JOIN instrument i ON md.instrument_id = i.id
                JOIN data_source ds ON md.data_source_id = ds.id
                JOIN data_type dt ON md.data_type_id = dt.id
                WHERE md.instrument_id IN :instrument_ids
                ORDER BY md.date, md.id
                """
                df = pd.read_sql_query(
                    text(stmt),
                    session.bind,
                    params={"instrument_ids": tuple(instrument_ids)},
                )
                return self._postprocess_df(df)

            raise ValueError(f"Unknown strategy: {strategy}")
