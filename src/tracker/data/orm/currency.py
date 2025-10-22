"""Currency ORM model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .account import Account
    from .cash_flow import CashFlow
    from .instrument import Instrument
    from .portfolio import Portfolio
    from .transaction import Transaction


class Currency(Base):
    """Currency lookup table."""

    __tablename__ = "currency"

    # Columns
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(3), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(64))
    symbol: Mapped[str] = mapped_column(String(3))
    decimals: Mapped[int] = mapped_column(Integer, default=2, server_default=text("2"))

    # Relationships
    accounts: Mapped[list[Account]] = relationship(
        back_populates="currency", order_by="Account.name"
    )
    portfolios: Mapped[list[Portfolio]] = relationship(
        back_populates="currency", order_by="Portfolio.name"
    )
    instruments: Mapped[list[Instrument]] = relationship(
        back_populates="currency", order_by="Instrument.ticker"
    )
    transactions: Mapped[list[Transaction]] = relationship(
        back_populates="currency", order_by="Transaction.date"
    )
    cash_flows: Mapped[list[CashFlow]] = relationship(
        back_populates="currency", order_by="CashFlow.date"
    )

    __table_args__ = (
        CheckConstraint("length(code) = 3", name="chk_currency_code_length"),
        CheckConstraint("decimals BETWEEN 0 AND 8", name="chk_currency_decimals"),
    )

    def __str__(self):
        return f"Currency({self.code})"

    def __repr__(self):
        return f"<Currency(id={self.id}, code='{self.code}')>"
