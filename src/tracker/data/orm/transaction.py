"""Transaction ORM model."""

from datetime import datetime
import enum
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Enum, ForeignKey, Index, Numeric, Text, select
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UTCDateTime

if TYPE_CHECKING:
    from .cash_flow import CashFlow
    from .currency import Currency
    from .instrument import Instrument
    from .portfolio import Portfolio


class TransactionMethod(enum.Enum):
    """Enum of transaction methods."""

    BASIC = "BASIC"
    TRADE = "TRADE"
    TRANSFER = "TRANSFER"


class TransactionType(enum.Enum):
    """Enum of transaction types."""

    BUY = "BUY"
    SELL = "SELL"
    DIVIDEND = "DIVIDEND"
    INTEREST = "INTEREST"
    FEE = "FEE"
    TAX = "TAX"
    TRANSFER_IN = "TRANSFER_IN"
    TRANSFER_OUT = "TRANSFER_OUT"


class Transaction(Base):
    """Transaction table for BUY/SELL/DIVIDEND events."""

    __tablename__ = "transaction"

    # Columns
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolio.id", ondelete="CASCADE"), nullable=False
    )
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instrument.id"), nullable=False
    )
    tx_type: Mapped[TransactionType] = mapped_column(
        Enum(TransactionType, name="transaction_type"), nullable=False
    )
    quantity: Mapped[float] = mapped_column(Numeric(24, 12))
    price: Mapped[float] = mapped_column(Numeric(24, 12))
    currency_id: Mapped[int] = mapped_column(ForeignKey("currency.id"), nullable=False)
    transaction_id: Mapped[int | None] = mapped_column(
        ForeignKey("transaction.id", ondelete="CASCADE")
    )
    description: Mapped[str | None] = mapped_column(Text)

    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(
        back_populates="transactions", lazy="joined"
    )
    instrument: Mapped["Instrument"] = relationship(
        back_populates="transactions", lazy="joined"
    )
    currency: Mapped["Currency"] = relationship(
        back_populates="transactions", lazy="joined"
    )
    parent_transaction: Mapped["Transaction"] = relationship(
        remote_side=[id], back_populates="child_transactions"
    )
    child_transactions: Mapped[list["Transaction"]] = relationship(
        remote_side=[transaction_id],
        back_populates="parent_transaction",
        cascade="all, delete-orphan",
        single_parent=True,
    )
    cash_flows: Mapped[list["CashFlow"]] = relationship(back_populates="transaction")

    __table_args__ = (
        CheckConstraint("quantity >= 0", name="transaction_quantity_positive"),
        CheckConstraint("price >= 0", name="transaction_price_positive"),
        Index("idx_transaction_date", "date"),
        Index("idx_transaction_portfolio", "portfolio_id"),
        Index("idx_transaction_instrument", "instrument_id"),
        Index(
            "idx_transaction_date_portfolio_instrument",
            "date",
            "portfolio_id",
            "instrument_id",
        ),
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
    def ticker(self) -> str | None:
        """Get instrument ticker."""
        return self.instrument.ticker if self.instrument else None

    @ticker.expression
    def ticker(cls):  # noqa: N805
        """Get instrument ticker expression."""
        from .instrument import Instrument

        return (
            select(Instrument.ticker)
            .where(Instrument.id == cls.instrument_id)
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

    def to_dict(self):
        """Convert Transaction to dictionary."""
        d = super().to_dict()
        d.update(
            {
                "account": self.account,
                "portfolio_name": self.portfolio_name,
                "ticker": self.ticker,
                "currency_code": self.currency_code,
            }
        )
        return d

    def __str__(self):
        return (
            f"Transaction({self.date}, {self.account}-{self.portfolio_name}, {self.tx_type.value}, "
            f"{self.ticker}, {self.quantity}x{self.price} {self.currency_code})"
        )

    def __repr__(self):
        return (
            f"<Transaction(id={self.id}, date={self.date}, "
            f"portfolio={self.account}-{self.portfolio_name}, tx_type='{self.tx_type.value}', "
            f"ticker='{self.ticker}', amount={self.quantity}x{self.price} {self.currency_code})>"
        )
