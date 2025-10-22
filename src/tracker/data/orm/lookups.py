"""Lookup tables for asset classes, data types, and data sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .instrument import Instrument
    from .market_data import MarketData


class AssetClass(Base):
    """Asset class lookup table."""

    __tablename__ = "asset_class"

    # Columns
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    # Relationships
    instruments: Mapped[list[Instrument]] = relationship(
        back_populates="asset_class",
        order_by="Instrument.ticker",
    )

    def __str__(self):
        return f"AssetClass({self.name})"

    def __repr__(self):
        return f"<AssetClass(id={self.id}, name='{self.name}')>"


class DataType(Base):
    """Data type lookup table for market data."""

    __tablename__ = "data_type"

    # Columns
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    # Relationships
    market_data: Mapped[list[MarketData]] = relationship(
        back_populates="data_type",
        order_by="MarketData.date",
    )

    def __str__(self):
        return f"DataType({self.name})"

    def __repr__(self):
        return f"<DataType(id={self.id}, name='{self.name}')>"


class DataSource(Base):
    """Data source lookup table for market data."""

    __tablename__ = "data_source"

    # Columns
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    # Relationships
    market_data: Mapped[list[MarketData]] = relationship(
        back_populates="data_source",
        order_by="MarketData.date",
    )

    def __str__(self):
        return f"DataSource({self.name})"

    def __repr__(self):
        return f"<DataSource(id={self.id}, name='{self.name}')>"
