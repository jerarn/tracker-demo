"""Portfolio ORM model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, UniqueConstraint, select
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    object_session,
    relationship,
    selectinload,
)
from sqlalchemy.orm.exc import DetachedInstanceError
from sqlalchemy.sql import func

from .base import Base

if TYPE_CHECKING:
    from .account import Account
    from .cash_flow import CashFlow
    from .currency import Currency
    from .instrument import Instrument
    from .transaction import Transaction


class Portfolio(Base):
    """Portfolio table."""

    __tablename__ = "portfolio"

    # Columns
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    currency_id: Mapped[int] = mapped_column(ForeignKey("currency.id"), nullable=False)
    account_id: Mapped[int] = mapped_column(ForeignKey("account.id"), nullable=False)

    # Relationships
    currency: Mapped[Currency] = relationship(
        back_populates="portfolios", lazy="joined"
    )
    account: Mapped[Account] = relationship(back_populates="portfolios", lazy="joined")
    transactions: Mapped[list[Transaction]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
        order_by="Transaction.date",
    )
    cash_flows: Mapped[list[CashFlow]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
        order_by="CashFlow.date",
    )

    __table_args__ = (
        UniqueConstraint("name", "account_id", name="uq_portfolio_account"),
    )

    @hybrid_property
    def currency_code(self) -> str | None:
        """Get currency code."""
        return self.currency.code if self.currency else None

    @currency_code.expression
    def currency_code(cls):  # noqa: N805
        """Get currency code expression."""
        from .currency import Currency

        return (
            select(Currency.code)
            .where(Currency.id == cls.currency_id)
            .scalar_subquery()
        )

    @hybrid_property
    def account_name(self) -> str | None:
        """Get account name."""
        return self.account.name if self.account else None

    @account_name.expression
    def account_name(cls):  # noqa: N805
        """Get account name expression."""
        from .account import Account

        return (
            select(Account.name).where(Account.id == cls.account_id).scalar_subquery()
        )

    @property
    def start_date(self) -> datetime | None:
        """Query the earliest transaction/cash-flow date efficiently."""
        from .cash_flow import CashFlow
        from .transaction import Transaction

        session = object_session(self)
        if session is None:
            try:
                dates = [
                    *(t.date for t in self.transactions),
                    *(c.date for c in self.cash_flows),
                ]
            except DetachedInstanceError:
                return None
            return min(dates, default=None)

        # SQL-level computation (faster)
        txn_min = select(func.min(Transaction.date)).where(
            Transaction.portfolio_id == self.id
        )
        cf_min = (
            select(func.min(CashFlow.date))
            .where(CashFlow.portfolio_id == self.id)
            .where(
                CashFlow.date != None  # noqa: E711
            )
        )
        result = session.scalar(
            select(
                func.min(
                    func.coalesce(txn_min.scalar_subquery(), cf_min.scalar_subquery())
                )
            )
        )
        return result or None

    @property
    def latest_date(self) -> datetime | None:
        """Query the latest transaction/cash-flow date efficiently."""
        from .cash_flow import CashFlow
        from .transaction import Transaction

        session = object_session(self)
        if session is None:
            try:
                dates = [
                    *(t.date for t in self.transactions),
                    *(c.date for c in self.cash_flows),
                ]
            except DetachedInstanceError:
                return None
            return max(dates, default=None)

        txn_max = select(func.max(Transaction.date)).where(
            Transaction.portfolio_id == self.id
        )
        cf_max = (
            select(func.max(CashFlow.date))
            .where(CashFlow.portfolio_id == self.id)
            .where(CashFlow.date != None)  # noqa: E711
        )
        result = session.scalar(
            select(
                func.max(
                    func.coalesce(txn_max.scalar_subquery(), cf_max.scalar_subquery())
                )
            )
        )
        return result or None

    @property
    def instruments(self) -> list[Instrument]:
        """Return a list of unique instruments, or empty list if none."""
        from .instrument import Instrument
        from .transaction import Transaction

        session = object_session(self)
        if session is None:
            try:
                return list({t.instrument for t in self.transactions if t.instrument})
            except DetachedInstanceError:
                return []

        stmt = (
            select(Instrument)
            .join(Transaction)
            .where(Transaction.portfolio_id == self.id)
            .distinct()
        )
        result = session.execute(stmt).scalars().all()
        return result or []

    @staticmethod
    def load_options():
        """Return list of SQLAlchemy loading options for eager loading."""
        from .instrument import Instrument
        from .transaction import Transaction

        return [
            # Eager load transactions and their instruments with market data
            selectinload(Portfolio.transactions)
            .joinedload(Transaction.instrument)
            .selectinload(Instrument.market_data),
            # Eager load cash flows
            selectinload(Portfolio.cash_flows),
        ]

    @classmethod
    def conflict_fields(cls):
        """Get fields to check for conflicts."""
        return ["name", "account_id"]

    def __str__(self):
        return f"Portfolio({self.name}, {self.account_name}, {self.currency_code})"

    def __repr__(self):
        return (
            f"<Portfolio(id={self.id}, name='{self.name}', "
            f"account='{self.account_name}', currency='{self.currency_code}')>"
        )
