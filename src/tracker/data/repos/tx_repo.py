"""Repository for managing Transaction entities in the database."""

from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from sqlalchemy import Column, Integer, Numeric, select, text
from sqlalchemy.orm import Session

from tracker.config.logger import get_logger
from tracker.data.managers.db_manager import DBManager
from tracker.data.orm.cash_flow import CashFlow, CashFlowType
from tracker.data.orm.transaction import (
    Transaction,
    TransactionMethod,
    TransactionType,
)

from .repository import Repository

logger = get_logger(__name__)


class TransactionRepository(Repository[Transaction]):
    """Repository for managing Transaction entities in the database."""

    def __init__(self, db_manager: DBManager | None = None):
        """Initialize TransactionRepository with DBManager."""
        super().__init__(Transaction, db_manager)

    @property
    def columns(self) -> dict[str, Column]:
        """Get the columns of the Transaction table, including fee-related columns."""
        tx_columns = super().columns
        tx_columns.update(
            {
                "fee_ticker_id": Column("fee_ticker_id", Integer),
                "fee_amount": Column("fee_amount", Numeric(precision=24, scale=12)),
                "fee_price": Column("fee_price", Numeric(precision=24, scale=12)),
                "base_ticker_id": Column("base_ticker_id", Integer),
                "quote_ticker_id": Column("quote_ticker_id", Integer),
                "base_quantity": Column(
                    "base_quantity", Numeric(precision=24, scale=12)
                ),
                "quote_quantity": Column(
                    "quote_quantity", Numeric(precision=24, scale=12)
                ),
                "base_price": Column("base_price", Numeric(precision=24, scale=12)),
                "quote_price": Column("quote_price", Numeric(precision=24, scale=12)),
                "portfolio_from_id": Column("portfolio_from_id", Integer),
                "portfolio_to_id": Column("portfolio_to_id", Integer),
            }
        )
        return tx_columns

    def _create(
        self,
        session: Session,
        date: datetime,
        portfolio_id: int,
        tx_type: TransactionType,
        quantity: Decimal,
        price: Decimal,
        currency_id: int,
        instrument_id: int,
        description: str,
        transaction_id: int | None = None,
    ) -> list[Transaction | CashFlow]:
        tx = Transaction(
            date=date,
            portfolio_id=portfolio_id,
            tx_type=tx_type,
            quantity=quantity,
            price=price,
            currency_id=currency_id,
            instrument_id=instrument_id,
            description=description,
            transaction_id=transaction_id,
        )
        session.add(tx)
        logger.debug("Inserted main transaction %s", tx)
        return tx

    def _create_cash_flow(
        self,
        session: Session,
        tx: Transaction,
        cf_type: CashFlowType,
        description: str,
        amount: Decimal | None = None,
    ) -> CashFlow:
        cf = CashFlow(
            date=tx.date,
            portfolio_id=tx.portfolio_id,
            cf_type=cf_type,
            amount=amount or (tx.quantity * tx.price),
            currency_id=tx.currency_id,
            description=description,
            transaction_id=tx.id,
        )
        session.add(cf)
        logger.debug("Inserted cash flow %s", cf)
        return cf

    def _create_fee(
        self,
        session: Session,
        tx: Transaction,
        fee_amount: Decimal,
        fee_price: Decimal | None = None,
        fee_ticker_id: int | None = None,
    ):
        """Create a fee transaction or cash flow."""
        fee_is_cash = fee_ticker_id is None
        if fee_is_cash:
            fee = self._create_cash_flow(
                session,
                tx,
                CashFlowType.FEE,
                description=f"[auto] fee: {tx.description}",
                amount=fee_amount,
            )

        else:
            fee = self._create(
                session,
                date=tx.date,
                portfolio_id=tx.portfolio_id,
                quantity=fee_amount,
                price=fee_price,
                tx_type=TransactionType.FEE,
                currency_id=tx.currency_id,
                instrument_id=fee_ticker_id,
                description=f"[auto] fee: {tx.description}",
                transaction_id=tx.id,
            )

        return fee

    def _create_transaction(
        self,
        session: Session,
        date: datetime,
        portfolio_id: int,
        tx_type: TransactionType,
        quantity: Decimal,
        price: Decimal,
        currency_id: int,
        instrument_id: int | None = None,
        description: str | None = None,
        fee_amount: Decimal | None = None,
        fee_price: Decimal | None = None,
        fee_ticker_id: int | None = None,
        **kwargs,  # noqa: ARG002
    ) -> list[Transaction | CashFlow]:
        """Create a new transaction record."""
        main_tx = self._create(
            session,
            date,
            portfolio_id,
            tx_type,
            quantity,
            price,
            currency_id,
            instrument_id,
            description,
        )
        session.flush()

        result = [main_tx]

        if tx_type in (TransactionType.BUY, TransactionType.SELL):
            cf = self._create_cash_flow(
                session,
                main_tx,
                CashFlowType.DEPOSIT
                if tx_type == TransactionType.SELL
                else CashFlowType.WITHDRAWAL,
                description=f"[auto] cash flow: {description}",
            )
            result.append(cf)

        if float(fee_amount or 0) > 0:
            fee = self._create_fee(
                session,
                main_tx,
                fee_amount,
                fee_price,
                fee_ticker_id,
            )
            result.append(fee)

        return result

    def _create_transfer(
        self,
        session: Session,
        date: datetime,
        portfolio_from_id: int,
        portfolio_to_id: int,
        quantity: Decimal,
        price: Decimal,
        currency_id: int,
        instrument_id: int | None = None,
        description: str | None = None,
        fee_amount: Decimal | None = None,
        **kwargs,  # noqa: ARG002
    ) -> list[Transaction]:
        """Create a transfer between two portfolios."""
        tx_from = self._create(
            session,
            date=date,
            portfolio_id=portfolio_from_id,
            instrument_id=instrument_id,
            quantity=quantity,
            price=price,
            tx_type=TransactionType.TRANSFER_OUT,
            currency_id=currency_id,
            description=description,
        )
        tx_to = self._create(
            session,
            date=date,
            portfolio_id=portfolio_to_id,
            instrument_id=instrument_id,
            quantity=quantity,
            price=price,
            tx_type=TransactionType.TRANSFER_IN,
            currency_id=currency_id,
            description=f"[auto] transfer: {description}",
        )

        session.flush()
        tx_from.transaction_id = tx_to.id
        tx_to.transaction_id = tx_from.id

        result = [tx_from, tx_to]

        if float(fee_amount or 0) > 0:
            fee_tx = self._create_fee(
                session=session,
                tx=tx_from,
                fee_amount=fee_amount,
                fee_price=price,
                fee_ticker_id=instrument_id,
            )
            result.append(fee_tx)

        return result

    def _create_trade(
        self,
        session: Session,
        date: datetime,
        portfolio_id: int,
        base_ticker_id: int,
        quote_ticker_id: int,
        base_quantity: Decimal,
        quote_quantity: Decimal,
        base_price: Decimal,
        quote_price: Decimal,
        currency_id: int,
        description: str,
        fee_amount: Decimal | None = None,
        fee_price: Decimal | None = None,
        fee_ticker_id: int | None = None,
        adjust_price_on: str = "sell",
        **kwargs,  # noqa: ARG002
    ) -> list[Transaction]:
        """Create a trade (buy and sell) transaction pair."""
        if adjust_price_on == "sell":
            cash_amount = quote_quantity * quote_price
        elif adjust_price_on == "buy":
            cash_amount = base_quantity * base_price
        elif adjust_price_on == "mid":
            cash_amount = (
                base_quantity * base_price + quote_quantity * quote_price
            ) / 2
        else:
            raise ValueError("adjust_price_on must be 'buy', 'sell', or 'mid'")

        tx_buy = self._create(
            session=session,
            date=date,
            portfolio_id=portfolio_id,
            instrument_id=base_ticker_id,
            quantity=base_quantity,
            price=cash_amount / base_quantity,
            tx_type=TransactionType.BUY,
            currency_id=currency_id,
            description=description,
        )
        tx_sell = self._create(
            session=session,
            date=date,
            portfolio_id=portfolio_id,
            instrument_id=quote_ticker_id,
            quantity=quote_quantity,
            price=cash_amount / quote_quantity,
            tx_type=TransactionType.SELL,
            currency_id=currency_id,
            description=f"[auto] trade: {description}",
        )

        session.flush()
        tx_sell.transaction_id = tx_buy.id
        tx_buy.transaction_id = tx_sell.id

        result = [tx_buy, tx_sell]

        if float(fee_amount or 0) > 0:
            if fee_ticker_id is None:
                raise ValueError("fee_ticker_id must be provided if fee_amount > 0")

            if fee_ticker_id == base_ticker_id:
                fee_price = cash_amount / base_quantity
            elif fee_ticker_id == quote_ticker_id:
                fee_price = cash_amount / quote_quantity

            fee_tx = self._create_fee(
                session=session,
                tx=tx_buy,
                fee_amount=fee_amount,
                fee_price=fee_price,
                fee_ticker_id=fee_ticker_id,
            )
            result.append(fee_tx)

        return result

    def create(
        self, method: TransactionMethod, session: Session | None = None, **kwargs
    ) -> list[Transaction | CashFlow]:
        """Create a new transaction record."""
        mapping = {
            TransactionMethod.BASIC: self._create_transaction,
            TransactionMethod.TRANSFER: self._create_transfer,
            TransactionMethod.TRADE: self._create_trade,
        }
        with self.get_session(session) as session:
            creator = mapping.get(method)
            if creator is None:
                raise ValueError(f"Unsupported transaction method: {method}")

            return creator(session=session, **kwargs)

    def create_many(
        self,
        data_list: list[dict[str, Any]],
        strategy: str = "add",
        batch_size: int = 1,
        return_instances: bool = False,
        session: Session | None = None,
        method: TransactionMethod | None = None,
    ) -> list[Transaction] | int:
        """Create multiple transaction records."""
        if method is None:
            raise ValueError("method must be provided for creating transactions")
        if not data_list:
            logger.warning("No records provided for processing.")
            return []
        if strategy != "add":
            logger.warning(
                "Only single record creation is supported for transactions. Ignoring."
            )
        if batch_size != 1:
            logger.warning("Batch size is not applicable for transactions. Ignoring.")

        creators = {
            TransactionMethod.BASIC: self._create_transaction,
            TransactionMethod.TRANSFER: self._create_transfer,
            TransactionMethod.TRADE: self._create_trade,
        }

        instances = []
        with self.get_session(session) as session:
            creator = creators.get(method)
            if creator is None:
                raise ValueError(f"Unsupported transaction method: {method}")

            for record in data_list:
                instances.extend(creator(session=session, **record))

        return instances if return_instances else len(instances)

    def _postprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process the DataFrame after fetching from the database."""
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            df["tx_type"] = df["tx_type"].map(
                lambda x: x.value if isinstance(x, TransactionType) else x
            )
            df["quantity"] = df["quantity"].astype(float)
            df["price"] = df["price"].astype(float)

        return df

    def fetch_from_portfolio_ids(
        self,
        portfolio_ids: list[int],
        strategy: str = "sql",
        session: Session | None = None,
    ) -> pd.DataFrame:
        """Fetch all transactions for the given portfolios."""
        if not portfolio_ids:
            return pd.DataFrame()

        with self.get_session(session) as session:
            if strategy == "normal":
                transactions = (
                    session.query(Transaction)
                    .filter(Transaction.portfolio_id.in_(portfolio_ids))
                    .order_by(Transaction.date, Transaction.id)
                    .all()
                )

                df = self.to_df(transactions)
                return self._postprocess_df(df)

            if strategy == "optimized":
                stmt = (
                    select(
                        Transaction.id,
                        Transaction.instrument_id,
                        Transaction.portfolio_id,
                        Transaction.date,
                        Transaction.ticker,
                        Transaction.quantity,
                        Transaction.price,
                        Transaction.tx_type,
                    )
                    .where(Transaction.portfolio_id.in_(portfolio_ids))
                    .order_by(Transaction.date, Transaction.id)
                )
                results = session.execute(stmt)

                df = pd.DataFrame(
                    results.fetchall(),
                    columns=results.keys() if results else [],
                )
                return self._postprocess_df(df)

            if strategy == "sql":
                stmt = """
                SELECT t.id, t.date, t.portfolio_id, t.instrument_id, i.ticker, t.tx_type, t.quantity, t.price
                FROM transaction t
                JOIN instrument i ON t.instrument_id = i.id
                WHERE t.portfolio_id IN :portfolio_ids
                ORDER BY t.date, t.id
                """
                df = pd.read_sql_query(
                    sql=text(stmt),
                    con=session.bind,
                    params={"portfolio_ids": tuple(portfolio_ids)},
                )
                return self._postprocess_df(df)

            raise ValueError(f"Unknown strategy: {strategy}")
