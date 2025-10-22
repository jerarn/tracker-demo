"""Account ORM model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, select
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .currency import Currency
    from .portfolio import Portfolio


class Account(Base):
    """Account table for financial accounts."""

    __tablename__ = "account"

    # Columns
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    currency_id: Mapped[int] = mapped_column(ForeignKey("currency.id"), nullable=False)

    # Relationships
    currency: Mapped[Currency] = relationship(back_populates="accounts", lazy="joined")
    portfolios: Mapped[list[Portfolio]] = relationship(back_populates="account")

    @hybrid_property
    def currency_code(self) -> str | None:
        """Get currency code."""
        return self.currency.code

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
        return ["name"]

    def __str__(self):
        return f"Account({self.name}, {self.currency_code})"

    def __repr__(self):
        return f"<Account(id={self.id}, name='{self.name}', currency='{self.currency_code}')>"
