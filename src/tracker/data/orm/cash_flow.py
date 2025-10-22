"""Cash flow ORM model."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import enum
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    select,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UTCDateTime

if TYPE_CHECKING:
    from .currency import Currency
    from .portfolio import Portfolio
    from .transaction import Transaction


class CashFlowType(enum.Enum):
    """Enum of cash flow types."""

    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    DIVIDEND = "DIVIDEND"
    INTEREST = "INTEREST"
    FEE = "FEE"
    TAX = "TAX"
    TRANSFER_IN = "TRANSFER_IN"
    TRANSFER_OUT = "TRANSFER_OUT"


class CashFlow(Base):
    """Cash flow table for deposits, withdrawals, etc."""

    __tablename__ = "cash_flow"

    # Columns
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolio.id", ondelete="CASCADE"), nullable=False
    )
    cf_type: Mapped[CashFlowType] = mapped_column(
        Enum(CashFlowType, name="cash_flow_type"), nullable=False
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    currency_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("currency.id"), nullable=False
    )
    cash_flow_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("cash_flow.id", ondelete="CASCADE")
    )
    transaction_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("transaction.id", ondelete="CASCADE")
    )
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(Text, default=None)
    sub_category: Mapped[str | None] = mapped_column(Text, default=None)
    operation: Mapped[str | None] = mapped_column(Text, default=None)

    # Relationships
    portfolio: Mapped[Portfolio] = relationship(
        back_populates="cash_flows", lazy="joined"
    )
    currency: Mapped[Currency] = relationship(
        back_populates="cash_flows", lazy="joined"
    )
    parent_cash_flow: Mapped[CashFlow] = relationship(
        remote_side=[id], back_populates="child_cash_flows"
    )
    child_cash_flows: Mapped[list[CashFlow]] = relationship(
        remote_side=[cash_flow_id],
        back_populates="parent_cash_flow",
        cascade="all, delete-orphan",
        single_parent=True,
    )
    transaction: Mapped[Transaction] = relationship(back_populates="cash_flows")

    __table_args__ = (
        CheckConstraint("amount >= 0", name="chk_cash_flow_amount_positive"),
        Index("ix_cash_flow_portfolio", "portfolio_id"),
        Index("ix_cash_flow_date", "date"),
        Index("ix_cash_flow_portfolio_date_type", "portfolio_id", "date", "cf_type"),
    )

    @hybrid_property
    def account(self) -> str | None:
        """Get account name."""
        return (
            self.portfolio.account.name
            if self.portfolio and self.portfolio.account
            else None
        )

    @account.expression
    def account(cls):  # noqa: N805
        """Get account name expression."""
        from .account import Account
        from .portfolio import Portfolio

        return (
            select(Account.name)
            .join(Portfolio, Portfolio.account_id == Account.id)
            .where(Portfolio.id == cls.portfolio_id)
            .scalar_subquery()
        )

    @hybrid_property
    def portfolio_name(self) -> str | None:
        """Get portfolio name."""
        return self.portfolio.name if self.portfolio else None

    @portfolio_name.expression
    def portfolio_name(cls):  # noqa: N805
        """Get portfolio name expression."""
        from .portfolio import Portfolio

        return (
            select(Portfolio.name)
            .where(Portfolio.id == cls.portfolio_id)
            .scalar_subquery()
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

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update(
            {
                "portfolio_name": self.portfolio_name,
                "account": self.account,
                "currency_code": self.currency_code,
            }
        )
        return d

    def __str__(self):
        return (
            f"CashFlow({self.date}, {self.account}-{self.portfolio_name}, "
            f"{self.cf_type.value}, {self.amount} {self.currency_code})"
        )

    def __repr__(self):
        return (
            f"<CashFlow(id={self.id}, date={self.date}, "
            f"portfolio={self.account}-{self.portfolio_name}, "
            f"cf_type='{self.cf_type.value}', amount={self.amount}, currency='{self.currency_code}')>"
        )
