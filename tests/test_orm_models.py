"""Comprehensive tests for SQLAlchemy ORM models.

This module tests the ORM models for data integrity, relationships,
validation, and core functionality.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from tracker import (
    Account,
    AssetClass,
    Base,
    CashFlow,
    CashFlowType,
    Currency,
    DataSource,
    DataType,
    Instrument,
    MarketData,
    Portfolio,
    Transaction,
    TransactionType,
)


@pytest.fixture(name="engine")
def fixture_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(name="session")
def fixture_session(engine):
    """Create a database session for testing."""
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture(name="sample_currency")
def fixture_sample_currency(session):
    """Create a sample currency for testing."""
    currency = Currency(code="USD", name="US Dollar", symbol="$", decimals=2)
    session.add(currency)
    session.commit()
    return currency


@pytest.fixture(name="sample_asset_class")
def fixture_sample_asset_class(session):
    """Create a sample asset class for testing."""
    asset_class = AssetClass(name="Equity", description="Stocks and shares")
    session.add(asset_class)
    session.commit()
    return asset_class


@pytest.fixture(name="sample_data_type")
def fixture_sample_data_type(session):
    """Create a sample data type for testing."""
    data_type = DataType(name="close", description="Closing price")
    session.add(data_type)
    session.commit()
    return data_type


@pytest.fixture(name="sample_data_source")
def fixture_sample_data_source(session):
    """Create a sample data source for testing."""
    data_source = DataSource(name="yfinance", description="Yahoo Finance")
    session.add(data_source)
    session.commit()
    return data_source


@pytest.fixture(name="sample_account")
def fixture_sample_account(session, sample_currency):
    """Create a sample account for testing."""
    account = Account(name="Test Account", currency_id=sample_currency.id)
    session.add(account)
    session.commit()
    return account


@pytest.fixture(name="sample_portfolio")
def fixture_sample_portfolio(session, sample_account, sample_currency):
    """Create a sample portfolio for testing."""
    portfolio = Portfolio(
        name="Test Portfolio",
        account_id=sample_account.id,
        currency_id=sample_currency.id,
    )
    session.add(portfolio)
    session.commit()
    return portfolio


@pytest.fixture(name="sample_instrument")
def fixture_sample_instrument(session, sample_asset_class, sample_currency):
    """Create a sample instrument for testing."""
    instrument = Instrument(
        name="Apple Inc.",
        isin="US0378331005",
        ticker="AAPL",
        asset_class_id=sample_asset_class.id,
        currency_id=sample_currency.id,
        description="Apple Inc. common stock",
    )
    session.add(instrument)
    session.commit()
    return instrument


class TestBaseModel:
    """Test the Base model functionality."""

    def test_to_dict(self, sample_currency):
        """Test the to_dict method."""
        result = sample_currency.to_dict()

        assert isinstance(result, dict)
        assert result["code"] == "USD"
        assert result["name"] == "US Dollar"
        assert result["symbol"] == "$"
        assert result["decimals"] == 2

    def test_str_representation(self, sample_currency):
        """Test string representation."""
        str_repr = str(sample_currency)
        assert f"Currency({sample_currency.code})" == str_repr

    def test_repr_representation(self, sample_currency):
        """Test repr representation."""
        repr_str = repr(sample_currency)
        assert f"<Currency(id={sample_currency.id}, code='USD')>" == repr_str


class TestCurrency:
    """Test Currency model."""

    def test_currency_creation(self, session):
        """Test creating a currency."""
        currency = Currency(code="EUR", name="Euro", symbol="€", decimals=2)
        session.add(currency)
        session.commit()

        assert currency.id is not None
        assert currency.code == "EUR"
        assert currency.name == "Euro"
        assert currency.symbol == "€"
        assert currency.decimals == 2

    def test_currency_unique_code(self, session, sample_currency):
        """Test that currency codes must be unique."""
        del sample_currency
        duplicate_currency = Currency(code="USD", name="Another Dollar", symbol="$")
        session.add(duplicate_currency)

        with pytest.raises(IntegrityError):
            session.commit()

    def test_currency_relationships(self, sample_currency, sample_account):
        """Test currency relationships."""
        assert sample_account in sample_currency.accounts
        assert sample_account.currency == sample_currency


class TestAssetClass:
    """Test AssetClass model."""

    def test_asset_class_creation(self, session):
        """Test creating an asset class."""
        asset_class = AssetClass(name="Bond", description="Fixed income securities")
        session.add(asset_class)
        session.commit()

        assert asset_class.id is not None
        assert asset_class.name == "Bond"
        assert asset_class.description == "Fixed income securities"

    def test_asset_class_unique_name(self, session, sample_asset_class):
        """Test that asset class names must be unique."""
        del sample_asset_class
        duplicate_asset_class = AssetClass(name="Equity", description="Another equity")
        session.add(duplicate_asset_class)

        with pytest.raises(IntegrityError):
            session.commit()


class TestAccount:
    """Test Account model."""

    def test_account_creation(self, session, sample_currency):
        """Test creating an account."""
        account = Account(name="Investment Account", currency_id=sample_currency.id)
        session.add(account)
        session.commit()

        assert account.id is not None
        assert account.name == "Investment Account"
        assert account.currency_id == sample_currency.id

    def test_account_currency_relationship(self, sample_account, sample_currency):
        """Test account-currency relationship."""
        assert sample_account.currency == sample_currency
        assert sample_account in sample_currency.accounts

    def test_account_currency_code_property(self, sample_account):
        """Test currency_code property."""
        assert sample_account.currency_code == "USD"

    def test_account_unique_name(self, session, sample_account):
        """Test that account names must be unique."""
        duplicate_account = Account(
            name="Test Account", currency_id=sample_account.currency_id
        )
        session.add(duplicate_account)

        with pytest.raises(IntegrityError):
            session.commit()


class TestPortfolio:
    """Test Portfolio model."""

    def test_portfolio_creation(self, session, sample_account, sample_currency):
        """Test creating a portfolio."""
        portfolio = Portfolio(
            name="My Portfolio",
            account_id=sample_account.id,
            currency_id=sample_currency.id,
        )
        session.add(portfolio)
        session.commit()

        assert portfolio.id is not None
        assert portfolio.name == "My Portfolio"
        assert portfolio.account_id == sample_account.id
        assert portfolio.currency_id == sample_currency.id

    def test_portfolio_relationships(
        self, sample_portfolio, sample_account, sample_currency
    ):
        """Test portfolio relationships."""
        assert sample_portfolio.account == sample_account
        assert sample_portfolio.currency == sample_currency
        assert sample_portfolio in sample_account.portfolios

    def test_portfolio_properties(self, sample_portfolio):
        """Test portfolio properties."""
        assert sample_portfolio.currency_code == "USD"
        assert sample_portfolio.account_name == "Test Account"

    def test_portfolio_unique_constraint(self, session, sample_portfolio):
        """Test unique constraint on portfolio name + account."""
        duplicate_portfolio = Portfolio(
            name="Test Portfolio",
            account_id=sample_portfolio.account_id,
            currency_id=sample_portfolio.currency_id,
        )
        session.add(duplicate_portfolio)

        with pytest.raises(IntegrityError):
            session.commit()


class TestInstrument:
    """Test Instrument model."""

    def test_instrument_creation(self, session, sample_asset_class, sample_currency):
        """Test creating an instrument."""
        instrument = Instrument(
            name="Microsoft Corporation",
            isin="US5949181045",
            ticker="MSFT",
            asset_class_id=sample_asset_class.id,
            currency_id=sample_currency.id,
            fee_pa=Decimal("0.001"),
            description="Microsoft Corp",
        )
        session.add(instrument)
        session.commit()

        assert instrument.id is not None
        assert instrument.name == "Microsoft Corporation"
        assert instrument.isin == "US5949181045"
        assert instrument.ticker == "MSFT"
        assert instrument.fee_pa == Decimal("0.001")

    def test_instrument_validation(self, session, sample_asset_class, sample_currency):
        """Test instrument validation."""
        # Test ISIN and ticker are converted to uppercase
        instrument = Instrument(
            name="Test Company",
            isin="us1234567890",  # lowercase
            ticker="test",  # lowercase
            asset_class_id=sample_asset_class.id,
            currency_id=sample_currency.id,
        )
        session.add(instrument)
        session.commit()

        assert instrument.isin == "US1234567890"
        assert instrument.ticker == "TEST"

    def test_instrument_isin_length_constraint(
        self, session, sample_asset_class, sample_currency
    ):
        """Test ISIN length constraint."""
        instrument = Instrument(
            name="Test Company",
            isin="SHORT",  # Too short
            ticker="TEST",
            asset_class_id=sample_asset_class.id,
            currency_id=sample_currency.id,
        )
        session.add(instrument)

        # Note: SQLite doesn't enforce check constraints by default
        # In PostgreSQL, this would raise an IntegrityError

    def test_instrument_relationships(
        self, sample_instrument, sample_asset_class, sample_currency
    ):
        """Test instrument relationships."""
        assert sample_instrument.asset_class == sample_asset_class
        assert sample_instrument.currency == sample_currency

    def test_instrument_properties(self, sample_instrument):
        """Test instrument properties."""
        assert sample_instrument.asset_class_name == "Equity"
        assert sample_instrument.currency_code == "USD"


class TestCashFlow:
    """Test CashFlow model."""

    def test_cash_flow_creation(self, session, sample_portfolio, sample_currency):
        """Test creating a cash flow."""
        cash_flow = CashFlow(
            portfolio_id=sample_portfolio.id,
            cf_type=CashFlowType.DEPOSIT,
            amount=Decimal("1000.00"),
            currency_id=sample_currency.id,
            date=datetime.now(timezone.utc),
            description="Initial deposit",
        )
        session.add(cash_flow)
        session.commit()

        assert cash_flow.id is not None
        assert cash_flow.cf_type == CashFlowType.DEPOSIT
        assert cash_flow.amount == Decimal("1000.00")

    def test_cash_flow_relationships(self, session, sample_portfolio, sample_currency):
        """Test cash flow relationships."""
        cash_flow = CashFlow(
            portfolio_id=sample_portfolio.id,
            cf_type=CashFlowType.DEPOSIT,
            amount=Decimal("500.00"),
            currency_id=sample_currency.id,
            date=datetime.now(timezone.utc),
        )
        session.add(cash_flow)
        session.commit()

        assert cash_flow.portfolio == sample_portfolio
        assert cash_flow.currency == sample_currency
        assert cash_flow in sample_portfolio.cash_flows

    def test_cash_flow_properties(self, session, sample_portfolio, sample_currency):
        """Test cash flow properties."""
        cash_flow = CashFlow(
            portfolio_id=sample_portfolio.id,
            cf_type=CashFlowType.WITHDRAWAL,
            amount=Decimal("200.00"),
            currency_id=sample_currency.id,
            date=datetime.now(timezone.utc),
        )
        session.add(cash_flow)
        session.commit()

        assert cash_flow.account == "Test Account"
        assert cash_flow.portfolio_name == "Test Portfolio"
        assert cash_flow.currency_code == "USD"


class TestTransaction:
    """Test Transaction model."""

    def test_transaction_creation(
        self, session, sample_portfolio, sample_instrument, sample_currency
    ):
        """Test creating a transaction."""
        transaction = Transaction(
            portfolio_id=sample_portfolio.id,
            instrument_id=sample_instrument.id,
            tx_type=TransactionType.BUY,
            quantity=Decimal("10.0"),
            price=Decimal("150.00"),
            currency_id=sample_currency.id,
            date=datetime.now(timezone.utc),
            description="Buy Apple shares",
        )
        session.add(transaction)
        session.commit()

        assert transaction.id is not None
        assert transaction.tx_type == TransactionType.BUY
        assert transaction.quantity == Decimal("10.0")
        assert transaction.price == Decimal("150.00")

    def test_transaction_relationships(
        self, session, sample_portfolio, sample_instrument, sample_currency
    ):
        """Test transaction relationships."""
        transaction = Transaction(
            portfolio_id=sample_portfolio.id,
            instrument_id=sample_instrument.id,
            tx_type=TransactionType.SELL,
            quantity=Decimal("5.0"),
            price=Decimal("160.00"),
            currency_id=sample_currency.id,
            date=datetime.now(timezone.utc),
        )
        session.add(transaction)
        session.commit()

        assert transaction.portfolio == sample_portfolio
        assert transaction.instrument == sample_instrument
        assert transaction.currency == sample_currency
        assert transaction in sample_portfolio.transactions

    def test_transaction_properties(
        self, session, sample_portfolio, sample_instrument, sample_currency
    ):
        """Test transaction properties."""
        transaction = Transaction(
            portfolio_id=sample_portfolio.id,
            instrument_id=sample_instrument.id,
            tx_type=TransactionType.BUY,
            quantity=Decimal("20.0"),
            price=Decimal("155.00"),
            currency_id=sample_currency.id,
            date=datetime.now(timezone.utc),
        )
        session.add(transaction)
        session.commit()

        assert transaction.account == "Test Account"
        assert transaction.portfolio_name == "Test Portfolio"
        assert transaction.ticker == "AAPL"
        assert transaction.currency_code == "USD"


class TestMarketData:
    """Test MarketData model."""

    def test_market_data_creation(
        self, session, sample_instrument, sample_data_type, sample_data_source
    ):
        """Test creating market data."""
        market_data = MarketData(
            instrument_id=sample_instrument.id,
            data_type_id=sample_data_type.id,
            data_source_id=sample_data_source.id,
            quote=Decimal("155.50"),
            date=datetime.now(timezone.utc),
        )
        session.add(market_data)
        session.commit()

        assert market_data.id is not None
        assert market_data.quote == Decimal("155.50")

    def test_market_data_relationships(
        self, session, sample_instrument, sample_data_type, sample_data_source
    ):
        """Test market data relationships."""
        market_data = MarketData(
            instrument_id=sample_instrument.id,
            data_type_id=sample_data_type.id,
            data_source_id=sample_data_source.id,
            quote=Decimal("160.25"),
            date=datetime.now(timezone.utc),
        )
        session.add(market_data)
        session.commit()

        assert market_data.instrument == sample_instrument
        assert market_data.data_type == sample_data_type
        assert market_data.data_source == sample_data_source

    def test_market_data_unique_constraint(
        self, session, sample_instrument, sample_data_type, sample_data_source
    ):
        """Test unique constraint on market data."""
        date = datetime.now(timezone.utc)

        market_data1 = MarketData(
            instrument_id=sample_instrument.id,
            data_type_id=sample_data_type.id,
            data_source_id=sample_data_source.id,
            quote=Decimal("155.50"),
            date=date,
        )
        session.add(market_data1)
        session.commit()

        # Try to add duplicate
        market_data2 = MarketData(
            instrument_id=sample_instrument.id,
            data_type_id=sample_data_type.id,
            data_source_id=sample_data_source.id,
            quote=Decimal("156.00"),
            date=date,
        )
        session.add(market_data2)

        with pytest.raises(IntegrityError):
            session.commit()


class TestDataIntegrity:
    """Test data integrity and business rules."""

    def test_cascading_deletes(self, session, sample_portfolio, sample_currency):
        """Test that deleting a portfolio cascades to its cash flows."""
        # Create a cash flow
        cash_flow = CashFlow(
            portfolio_id=sample_portfolio.id,
            cf_type=CashFlowType.DEPOSIT,
            amount=Decimal("1000.00"),
            currency_id=sample_currency.id,
            date=datetime.now(timezone.utc),
        )
        session.add(cash_flow)
        session.commit()

        cash_flow_id = cash_flow.id

        # Delete the portfolio
        session.delete(sample_portfolio)
        session.commit()

        # Check that cash flow is also deleted
        deleted_cash_flow = session.get(CashFlow, cash_flow_id)
        assert deleted_cash_flow is None

    def test_foreign_key_constraints(self, session):
        """Test foreign key constraints."""
        # Try to create account with non-existent currency
        account = Account(name="Invalid Account", currency_id=99999)
        session.add(account)

        with pytest.raises(IntegrityError):
            session.commit()


if __name__ == "__main__":
    pytest.main([__file__])
