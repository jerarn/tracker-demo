"""Instrument ORM model."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Numeric,
    String,
    Text,
    select,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from .base import Base

if TYPE_CHECKING:
    from .currency import Currency
    from .lookups import AssetClass
    from .market_data import MarketData
    from .transaction import Transaction


class Instrument(Base):
    """Instrument table for financial instruments."""

    __tablename__ = "instrument"

    # Columns
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    isin: Mapped[str] = mapped_column(String(12), unique=True, nullable=False)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, nullable=False)
    asset_class_id: Mapped[int] = mapped_column(
        ForeignKey("asset_class.id"), nullable=False
    )
    currency_id: Mapped[int] = mapped_column(ForeignKey("currency.id"), nullable=False)
    fee_pa: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    description: Mapped[str | None] = mapped_column(Text)

    # Relationships
    asset_class: Mapped[AssetClass] = relationship(
        back_populates="instruments", lazy="joined"
    )
    currency: Mapped[Currency] = relationship(
        back_populates="instruments", lazy="joined"
    )
    transactions: Mapped[list[Transaction]] = relationship(
        back_populates="instrument",
        cascade="all, delete-orphan",
        order_by="Transaction.date",
    )
    market_data: Mapped[list[MarketData]] = relationship(
        back_populates="instrument",
        cascade="all, delete-orphan",
        order_by="MarketData.date",
    )

    __table_args__ = (
        CheckConstraint("length(isin) = 12", name="instrument_isin_length"),
        CheckConstraint("fee_pa >= 0", name="instrument_fee_pa_positive"),
    )

    @validates("isin", "ticker")
    def validate_uppercase(self, _key, value: str | None) -> str | None:
        """Ensure ISIN and ticker are uppercase and stripped."""
        if value is None:
            return None
        return value.strip().upper()

    @hybrid_property
    def asset_class_name(self) -> str | None:
        """Get asset class name."""
        return self.asset_class.name if self.asset_class else None

    @asset_class_name.expression
    def asset_class_name(cls):  # noqa: N805
        """Get asset class name expression."""
        from .lookups import AssetClass

        return (
            select(AssetClass.name)
            .where(AssetClass.id == cls.asset_class_id)
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

    @classmethod
    def conflict_fields(cls):
        """Get fields to check for conflicts."""
        return ["ticker"]

    def __str__(self):
        return (
            f"Instrument({self.name}, {self.isin}, {self.ticker}, "
            f"{self.asset_class_name}, {self.currency_code})"
        )

    def __repr__(self):
        return (
            f"<Instrument(id={self.id}, name='{self.name}', isin='{self.isin}', "
            f"ticker='{self.ticker}', asset_class='{self.asset_class_name}', "
            f"currency='{self.currency_code}')>"
        )
