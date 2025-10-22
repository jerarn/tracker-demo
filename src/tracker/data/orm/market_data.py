"""Market data ORM model."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Numeric, UniqueConstraint, select
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UTCDateTime

if TYPE_CHECKING:
    from .instrument import Instrument
    from .lookups import DataSource, DataType


class MarketData(Base):
    """Market data table for historical price data."""

    __tablename__ = "market_data"

    # Columns
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instrument.id"), nullable=False
    )
    data_type_id: Mapped[int] = mapped_column(
        ForeignKey("data_type.id"), nullable=False
    )
    data_source_id: Mapped[int] = mapped_column(
        ForeignKey("data_source.id"), nullable=False
    )
    quote: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False)

    # Relationships
    instrument: Mapped[Instrument] = relationship(
        back_populates="market_data", lazy="joined"
    )
    data_type: Mapped[DataType] = relationship(
        back_populates="market_data", lazy="joined"
    )
    data_source: Mapped[DataSource] = relationship(
        back_populates="market_data", lazy="joined"
    )

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "data_type_id",
            "data_source_id",
            "date",
            name="uq_market_data",
        ),
        Index("idx_market_data_date", "date"),
        Index("idx_market_data_instrument_date", "instrument_id", "date"),
        Index(
            "idx_market_data_latest", "instrument_id", "date", postgresql_using="btree"
        ),
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
    def data_type_name(self) -> str | None:
        """Get data type name."""
        return self.data_type.name if self.data_type else None

    @data_type_name.expression
    def data_type_name(cls):  # noqa: N805
        """Get data type name expression."""
        from .lookups import DataType

        return (
            select(DataType.name)
            .where(DataType.id == cls.data_type_id)
            .scalar_subquery()
        )

    @hybrid_property
    def data_source_name(self) -> str | None:
        """Get data source name."""
        return self.data_source.name if self.data_source else None

    @data_source_name.expression
    def data_source_name(cls):  # noqa: N805
        """Get data source name expression."""
        from .lookups import DataSource

        return (
            select(DataSource.name)
            .where(DataSource.id == cls.data_source_id)
            .scalar_subquery()
        )

    def to_dict(self):
        """Convert MarketData to dictionary."""
        d = super().to_dict()
        d.update(
            {
                "ticker": self.ticker,
                "data_type_name": self.data_type_name,
                "data_source_name": self.data_source_name,
            }
        )
        return d

    @classmethod
    def conflict_fields(cls):
        """Get fields to check for conflicts."""
        return [
            "instrument_id",
            "data_type_id",
            "data_source_id",
            "date",
        ]

    def __str__(self):
        return (
            f"MarketData({self.ticker}, {self.date}, {self.quote}, "
            f"{self.data_type_name}, {self.data_source_name})"
        )

    def __repr__(self):
        return (
            f"<MarketData(id={self.id}, ticker='{self.ticker}', date={self.date}, "
            f"quote={self.quote}, data_type='{self.data_type_name}', "
            f"data_source='{self.data_source_name}')>"
        )
