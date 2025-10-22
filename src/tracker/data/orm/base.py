"""SQLAlchemy ORM models for the portfolio tracker database.

This module defines SQLAlchemy ORM classes that map to the database tables
defined in portfoliodb_schema.sql. These models replace the existing model
classes for better database interaction and app development.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from sqlalchemy import DateTime, String, TypeDecorator, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class UTCDateTime(TypeDecorator):
    """Store as TIMESTAMP (Postgres) or ISO string (SQLite). Always return tz-aware UTC datetimes."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        """Use DateTime for Postgres, String for SQLite."""
        if dialect.name == "sqlite":
            return dialect.type_descriptor(String())
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(self, value, dialect):
        """Convert datetime to appropriate format for DB storage."""
        if value is None:
            return None

        if not isinstance(value, datetime):
            raise TypeError("UTCDateTime expects a datetime")

        # normalize to UTC aware
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)

        if dialect.name == "sqlite":
            # SQLite: bind as ISO string
            return value.isoformat(timespec="seconds")
        # Postgres: bind as datetime object
        return value

    def process_result_value(self, value, dialect):
        """Convert DB value back to tz-aware UTC datetime."""
        if value is None:
            return None

        if dialect.name == "sqlite":
            # value will be a string produced by SQLite (or by server DEFAULT)
            # datetime.fromisoformat handles "YYYY-MM-DD HH:MM:SS" and "YYYY-MM-DDTHH:MM:SS+00:00"
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        # Postgres: driver returns datetime; ensure tz-aware UTC
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        return value

    def process_literal_param(self, value, dialect):
        """Handle literal binds (for SQL compilation)."""
        return self.process_bind_param(value, dialect)

    @property
    def python_type(self):
        """Return the Python type handled by this TypeDecorator."""
        return datetime


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(timezone.utc),
        server_default=text("(CURRENT_TIMESTAMP)"),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=text("(CURRENT_TIMESTAMP)"),
        nullable=False,
    )

    def to_dict(self):
        """Convert model instance to dictionary with native Python types."""
        result = {}
        for c in self.__table__.columns:
            value = getattr(self, c.name)

            # Type-safe conversions
            if isinstance(value, Decimal):
                value = float(value)
            elif isinstance(value, Enum):
                value = value.value

            result[c.name] = value

        return result

    @classmethod
    def from_dict(cls, data: dict) -> Base:
        """Create model instance from dictionary."""
        return cls(**data)

    @classmethod
    def load_options(cls):
        """Return list of SQLAlchemy loading options for eager loading."""
        return []

    @classmethod
    def conflict_fields(cls) -> list[str]:
        """Return list of field names that define uniqueness for upsert operations."""
        return []

    def __str__(self):
        """Return a string representation of the model instance."""
        return f"{self.__class__.__name__}(id={getattr(self, 'id', 'None')})"

    def __repr__(self):
        """Return a string representation of the model instance."""
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"<{self.__class__.__name__}({attrs})>"
