"""Tests for the DBManager database manager.

This module tests the database manager functionality including
session handling, connection pooling, and CRUD operations.
"""

import contextlib
import os
from pathlib import Path
import tempfile

import pytest
from sqlalchemy.exc import SQLAlchemyError

from tracker import Currency, DBManager


@pytest.fixture(name="db_mgr")
def fixture_db_mgr():
    """Create a database manager instance for testing."""
    # Create a temporary file for SQLite database
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)  # Close the file descriptor, SQLite will handle the file

    # Set environment variables for test database
    original_env = {}
    test_env = {
        "DB_DRIVER": "sqlite",
        "DB_NAME": db_path,
        "DB_ECHO": "false",
        "DB_POOL_SIZE": "1",
        "DB_POOL_MAX_OVERFLOW": "0",
    }

    # Store original environment and set test values
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    # Reset the singleton instance
    DBManager._instance = None  # pylint: disable=protected-access

    mgr = DBManager()
    mgr.create_schema()
    yield mgr

    # Clean up
    mgr.close()
    DBManager._instance = None  # pylint: disable=protected-access

    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    # Clean up the temporary database file
    with contextlib.suppress(OSError):
        Path(db_path).unlink()


class TestDBManager:
    """Test database manager."""

    def test_singleton_pattern(self, db_mgr):
        """Test that DBManager follows singleton pattern."""
        mgr = DBManager()

        assert db_mgr is mgr
        assert id(db_mgr) == id(mgr)

    def test_get_session_context_manager(self, db_mgr):
        """Test session context manager."""
        with db_mgr.get_session() as session:
            assert session is not None
            # Session should be active
            assert session.is_active

    def test_session_commit_on_success(self, db_mgr):
        """Test that session commits on successful operations."""
        currency = Currency(code="USD", name="US Dollar", symbol="$")

        with db_mgr.get_session() as session:
            session.add(currency)
            # No explicit commit needed, context manager handles it

        # Verify the currency was committed
        with db_mgr.get_session() as session:
            retrieved = session.query(Currency).filter_by(code="USD").first()
            assert retrieved is not None
            assert retrieved.name == "US Dollar"

    def test_session_rollback_on_error(self, db_mgr):
        """Test that session rolls back on errors."""
        currency1 = Currency(code="EUR", name="Euro", symbol="€")
        currency2 = Currency(
            code="EUR", name="Another Euro", symbol="€"
        )  # Duplicate code

        # First currency should be added successfully
        with db_mgr.get_session() as session:
            session.add(currency1)

        # Second currency with duplicate code should cause rollback
        with pytest.raises(SQLAlchemyError), db_mgr.get_session() as session:
            session.add(currency2)
            session.commit()

        # Verify only the first currency exists (rollback worked)
        with db_mgr.get_session() as session:
            currencies = session.query(Currency).filter_by(code="EUR").all()
            assert len(currencies) == 1
            assert currencies[0].name == "Euro"

    def test_readonly_session_parameter(self, db_mgr):
        """Test readonly session parameter."""
        with db_mgr.get_session(readonly=True) as session:
            currency = Currency(code="GBP", name="British Pound", symbol="£")
            session.add(currency)

        # Verify the currency was not added
        with db_mgr.get_session() as session:
            retrieved = session.query(Currency).filter_by(code="GBP").first()
            assert retrieved is None

    def test_drop_schema_standard(self, db_mgr):
        """Test dropping schema using SQLAlchemy metadata."""
        # Add some data first
        currency = Currency(code="JPY", name="Japanese Yen", symbol="¥")
        with db_mgr.get_session() as session:
            session.add(currency)

        # Drop and recreate schema
        db_mgr.drop_schema()
        db_mgr.create_schema()

        # Data should be gone
        with db_mgr.get_session() as session:
            count = session.query(Currency).count()
            assert count == 0

    def test_execute_query_no_fetch(self, db_mgr):
        """Test executing query without fetching results."""
        result = db_mgr.execute("SELECT 1", fetch=False)
        assert result is None

    def test_execute_query_with_fetch(self, db_mgr):
        """Test executing query with fetching results."""
        result = db_mgr.execute("SELECT 1 as test_col", fetch=True)

        assert result is not None
        assert len(result) == 1
        assert result[0]["test_col"] == 1

    def test_execute_query_with_params(self, db_mgr):
        """Test executing query with parameters."""
        # Add a currency first
        currency = Currency(code="THB", name="Thai Baht", symbol="฿")
        with db_mgr.get_session() as session:
            session.add(currency)

        result = db_mgr.execute(
            "SELECT * FROM currency WHERE code = :code",
            params={"code": "THB"},
            fetch=True,
        )

        assert len(result) == 1
        assert result[0]["code"] == "THB"

    def test_execute_many(self, db_mgr):
        """Test executing query with multiple parameter sets."""
        query = (
            "INSERT INTO currency (code, name, symbol) VALUES (:code, :name, :symbol)"
        )
        params = [
            {"code": "MXN", "name": "Mexican Peso", "symbol": "$"},
            {"code": "BRL", "name": "Brazilian Real", "symbol": "R$"},
        ]

        db_mgr.execute_many(query, params)

        # Verify both currencies were inserted
        with db_mgr.get_session() as session:
            count = (
                session.query(Currency)
                .filter(Currency.code.in_(["MXN", "BRL"]))
                .count()
            )
            assert count == 2

    def test_execute_file_not_found(self, db_mgr):
        """Test executing non-existent SQL file."""
        # Test without mocking to see actual behavior
        with pytest.raises((FileNotFoundError, RuntimeError)):
            db_mgr.execute_file("nonexistent.sql")

    def test_execute_file_sql_error(self, db_mgr, tmp_path):
        """Test executing SQL file with syntax error."""
        sql_file = tmp_path / "bad.sql"
        sql_file.write_text("INVALID SQL SYNTAX;")

        with pytest.raises(SQLAlchemyError):
            db_mgr.execute_file(str(sql_file))

    def test_get_pool_status(self, db_mgr):
        """Test getting pool status."""
        status = db_mgr.get_pool_status()

        assert status["status"] in ["active", "not_initialized"]
        if status["status"] == "active":
            assert "size" in status
            assert "checked_in" in status
            assert "checked_out" in status

    def test_engine_not_initialized_error(self):
        """Test operations when engine is not initialized."""
        mgr = object.__new__(DBManager)
        mgr._engine = None  # pylint: disable=protected-access

        with pytest.raises(RuntimeError, match="Database engine not initialized"):
            mgr.create_schema()

        with pytest.raises(RuntimeError, match="Database engine not initialized"):
            mgr.drop_schema()


if __name__ == "__main__":
    pytest.main([__file__])
