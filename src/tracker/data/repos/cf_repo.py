"""Repository for managing CashFlow entities in the database."""

from datetime import datetime
from decimal import Decimal

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from tracker.config.logger import get_logger
from tracker.data.managers.db_manager import DBManager
from tracker.data.orm.cash_flow import CashFlow, CashFlowType

from .repository import Repository

logger = get_logger(__name__)


class CashFlowRepository(Repository[CashFlow]):
    """Repository for managing CashFlow entities in the database."""

    def __init__(self, db_manager: DBManager | None = None):
        """Initialize CashFlowRepository with DBManager."""
        super().__init__(CashFlow, db_manager)

    def _create(
        self,
        session: Session,
        date: datetime,
        portfolio_id: int,
        cf_type: CashFlowType,
        amount: Decimal,
        currency_id: int,
        description: str,
        tx_id: int | None = None,
    ):
        cf = CashFlow(
            date=date,
            portfolio_id=portfolio_id,
            cf_type=cf_type,
            amount=amount,
            currency_id=currency_id,
            description=description,
            transaction_id=tx_id,
        )
        session.add(cf)
        logger.debug("Inserted linked cash flow %s", cf)
        return cf

    def _postprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            df["cf_type"] = df["cf_type"].map(
                lambda x: x.value if isinstance(x, CashFlowType) else x
            )
            df["amount"] = df["amount"].astype(float)

        return df

    def fetch_from_portfolio_ids(
        self,
        portfolio_ids: list[int],
        strategy: str = "normal",
        session: Session | None = None,
    ) -> pd.DataFrame:
        """Fetch all cash flows for the given portfolios."""
        if not portfolio_ids:
            return pd.DataFrame()

        with self.get_session(session) as session:
            if strategy == "normal":
                cash_flows = (
                    session.query(CashFlow)
                    .filter(CashFlow.portfolio_id.in_(portfolio_ids))
                    .order_by(CashFlow.date, CashFlow.id)
                    .all()
                )

                df = self.to_df(cash_flows)
                return self._postprocess_df(df)

            if strategy == "optimized":
                stmt = (
                    select(
                        CashFlow.id,
                        CashFlow.portfolio_id,
                        CashFlow.date,
                        CashFlow.amount,
                        CashFlow.cf_type,
                    )
                    .where(CashFlow.portfolio_id.in_(portfolio_ids))
                    .order_by(CashFlow.date, CashFlow.id)
                )
                results = session.execute(stmt)

                df = pd.DataFrame(
                    results.fetchall(),
                    columns=results.keys() if results else [],
                )
                return self._postprocess_df(df)

            if strategy == "sql":
                stmt = """
                SELECT cf.id, cf.portfolio_id, cf.date, cf.amount, cf.cf_type
                FROM cash_flow cf
                WHERE cf.portfolio_id IN :portfolio_ids
                ORDER BY cf.date, cf.id
                """
                df = pd.read_sql(
                    text(stmt),
                    session.bind,
                    params={"portfolio_ids": tuple(portfolio_ids)},
                )
                return self._postprocess_df(df)

            raise ValueError(f"Unknown strategy: {strategy}")
