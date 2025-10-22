#!/usr/bin/env python3
"""Database migration management script.

Provides convenient command-line interface for Alembic operations.
"""

import argparse
import os
import sys

from alembic.config import Config

from alembic import command
from tracker import (
    AssetClass,
    Currency,
    DataSource,
    DataType,
    DBManager,
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)


def init_lookup_data():
    """Initialize lookup data in the database."""
    logger.info("Initializing lookup data...")
    db = DBManager()
    data = [
        # Currencies
        Currency(code="USD", name="US Dollar", symbol="$", decimals=2),
        Currency(code="EUR", name="Euro", symbol="€", decimals=2),
        Currency(code="GBP", name="British Pound", symbol="£", decimals=2),
        Currency(code="CHF", name="Swiss Franc", symbol="CHF", decimals=2),
        Currency(code="JPY", name="Japanese Yen", symbol="¥", decimals=0),
        # Asset Classes
        AssetClass(name="Cash", description="Cash and cash equivalents"),
        AssetClass(name="Forex", description="Foreign exchange"),
        AssetClass(name="Equity", description="Stocks and shares"),
        AssetClass(name="Bond", description="Fixed income securities"),
        AssetClass(name="Commodity", description="Physical goods"),
        AssetClass(name="Real Estate", description="Property investments"),
        AssetClass(name="Crypto", description="Digital currencies"),
        AssetClass(name="Alternative", description="Alternative investments"),
        AssetClass(name="Derivative", description="Financial derivatives"),
        AssetClass(name="Fund", description="Mutual funds"),
        AssetClass(name="ETF", description="Exchange-traded funds"),
        AssetClass(name="Other", description="Other asset types"),
        # Data Types
        DataType(name="open", description="Opening price"),
        DataType(name="close", description="Closing price"),
        DataType(name="high", description="Highest price"),
        DataType(name="low", description="Lowest price"),
        DataType(name="volume", description="Trading volume"),
        # Data Sources
        DataSource(name="yfinance", description="Yahoo Finance"),
        DataSource(name="bloomberg", description="Bloomberg Terminal"),
        DataSource(name="coinbase", description="Coinbase Exchange"),
        DataSource(name="kraken", description="Kraken Exchange"),
        DataSource(name="binance", description="Binance Exchange"),
        DataSource(name="ft", description="Financial Times"),
        DataSource(name="morning star", description="Morning Star"),
        DataSource(name="boursorama", description="Boursorama"),
        DataSource(name="alpha vantage", description="Alpha Vantage API"),
        DataSource(name="comdirect", description="Comdirect Bank"),
        DataSource(name="manual", description="Manual Entry"),
    ]

    with db.get_session() as session:
        for item in data:
            session.add(item)


def migrate_up(args):
    """Run migrations to upgrade database."""
    logger.info("Running database migrations...")
    db = DBManager()
    try:
        db.run_migrations(args.revision)
        logger.info("Database migrations completed successfully!")
    except Exception as e:  # pylint: disable=broad-except
        logger.info(f"Migration failed: {e}")
        sys.exit(1)


def migrate_down(args):
    """Rollback migrations."""
    db = DBManager()

    # First check current status
    try:
        current = db.get_current_revision()
        if not current:
            logger.info(" No migrations to rollback - database not initialized")
            return
        logger.info(f"Current revision: {current}")
    except Exception as e:  # pylint: disable=broad-except
        logger.info(f" Could not check current revision: {e}")

    # Choose rollback method
    if args.revision:
        logger.info(f"Rolling back to revision: {args.revision}")
        try:
            db.rollback_to_revision(args.revision)
            logger.info("Migration rollback completed successfully!")
        except Exception as e:  # pylint: disable=broad-except
            logger.info(f"Rollback failed: {e}")
            logger.info("\nTroubleshooting tips:")
            logger.info(
                "   - Check available revisions: python scripts/migrate.py status -H"
            )
            logger.info("   - Use 'base' to rollback to initial state")
            sys.exit(1)
    else:
        logger.info(f"Rolling back {args.steps} migration(s)...")
        try:
            db.rollback_migration(args.steps)
            logger.info("Migration rollback completed successfully!")
        except Exception as e:  # pylint: disable=broad-except
            logger.info(f"Rollback failed: {e}")
            logger.info("\nTroubleshooting tips:")
            logger.info(
                "   - Check current status: python scripts/migrate.py status -H"
            )
            logger.info(
                "   - Try rolling back to a specific revision: "
                "python scripts/migrate.py down -r <revision>"
            )
            logger.info(
                "   - Use 'base' to rollback all: python scripts/migrate.py down -r base"
            )
            sys.exit(1)

    # Show new status
    try:
        new_current = db.get_current_revision()
        if new_current:
            logger.info(f"New revision: {new_current}")
        else:
            logger.info("Database rolled back to initial state")
    except Exception as e:  # pylint: disable=broad-except
        logger.info(f"Failed to show new status: {e}")  # Log the exception


def create_migration(args):
    """Create a new migration."""
    logger.info(f"Creating migration: {args.message}")
    db = DBManager()
    try:
        revision = db.create_migration(args.message, autogenerate=args.autogenerate)
        if revision:
            logger.info(f"Created migration {revision}: {args.message}")
        else:
            logger.info(" No changes detected - no migration created")
    except Exception as e:  # pylint: disable=broad-except
        logger.info(f"Migration creation failed: {e}")
        sys.exit(1)


def show_status(args):
    """Show current migration status."""
    db = DBManager()
    try:
        current = db.get_current_revision()
        if current:
            logger.info(f"Current database revision: {current}")
        else:
            logger.info("No migrations have been applied")

        if args.history:
            logger.info("\nMigration history:")
            history = db.get_migration_history()
            if history:
                for line in history:
                    if line.strip():
                        logger.info(f"   {line}")
            else:
                logger.info("   No migration history found")
    except Exception as e:  # pylint: disable=broad-except
        logger.info(f"Failed to get status: {e}")
        sys.exit(1)


def init_db(args):
    """Initialize database with current schema."""
    logger.info("Initializing database...")
    db = DBManager()
    try:
        # Option 1: Create schema directly (bypassing migrations)
        if args.direct:
            logger.info("Creating database schema directly from models...")
            db.create_schema()
            init_lookup_data()
            logger.info("Database schema created successfully!")
        else:
            # Option 2: Run migrations
            logger.info("Running migrations to initialize database...")
            db.run_migrations("head")
            logger.info("Database initialized with migrations!")
    except Exception as e:  # pylint: disable=broad-except
        logger.info(f"Database initialization failed: {e}")
        sys.exit(1)


def stamp_current(args):
    """Mark current database schema as up-to-date without running migrations."""
    logger.info(f"Marking database as current revision: {args.revision}")

    try:
        config = Config("alembic.ini")
        command.stamp(config, args.revision)
        logger.info("Database marked as current successfully!")
        logger.info(f"Database is now at revision: {args.revision}")
    except Exception as e:  # pylint: disable=broad-except
        logger.info(f"Failed to stamp database: {e}")
        sys.exit(1)


def reset_db(args):
    """Reset database (drop and recreate)."""
    if not args.force:
        response = input(" This will DELETE ALL DATA. Are you sure? (yes/no): ")
        if response.lower() != "yes":
            logger.info("Operation cancelled")
            return

    logger.info("Resetting database...")
    db = DBManager()
    try:
        logger.info(" Dropping existing schema...")
        db.drop_schema(raw=True)
        logger.info("Creating new schema...")
        db.create_schema()
        logger.info("Adding lookup data, procedures, functions, and views...")
        init_lookup_data()

        logger.info("Database reset completed successfully!")
    except Exception as e:  # pylint: disable=broad-except
        logger.info(f"Database reset failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Database migration management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init                     # Initialize database
  %(prog)s up                       # Run all pending migrations
  %(prog)s down -s 1                # Rollback 1 migration
  %(prog)s down -r base             # Rollback to initial state
  %(prog)s create "add user table"  # Create new migration
  %(prog)s status -H                # Show status and history
  %(prog)s stamp head               # Mark database as current
  %(prog)s reset --force            # Reset database (danger!)
        """,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Initialize database
    init_parser = subparsers.add_parser("init", help="Initialize database")
    init_parser.add_argument(
        "--direct",
        action="store_true",
        help="Create schema directly instead of using migrations",
    )
    init_parser.set_defaults(func=init_db)

    # Migrate up
    up_parser = subparsers.add_parser("up", help="Run migrations")
    up_parser.add_argument(
        "-r", "--revision", default="head", help="Target revision (default: head)"
    )
    up_parser.set_defaults(func=migrate_up)

    # Migrate down
    down_parser = subparsers.add_parser("down", help="Rollback migrations")
    down_parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=1,
        help="Number of migrations to rollback (default: 1)",
    )
    down_parser.add_argument(
        "-r",
        "--revision",
        help="Specific revision to rollback to (alternative to --steps)",
    )
    down_parser.set_defaults(func=migrate_down)

    # Create migration
    create_parser = subparsers.add_parser("create", help="Create new migration")
    create_parser.add_argument("message", help="Migration description")
    create_parser.add_argument(
        "--no-autogenerate",
        dest="autogenerate",
        action="store_false",
        help="Don't auto-generate migration from model changes",
    )
    create_parser.set_defaults(func=create_migration)

    # Show status
    status_parser = subparsers.add_parser("status", help="Show migration status")
    status_parser.add_argument(
        "-H", "--history", action="store_true", help="Show migration history"
    )
    status_parser.set_defaults(func=show_status)

    # Stamp database as current
    stamp_parser = subparsers.add_parser(
        "stamp", help="Mark database as current revision"
    )
    stamp_parser.add_argument(
        "revision",
        default="head",
        nargs="?",
        help="Revision to mark as current (default: head)",
    )
    stamp_parser.set_defaults(func=stamp_current)

    # Reset database
    reset_parser = subparsers.add_parser("reset", help="Reset database (DANGER!)")
    reset_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )
    reset_parser.set_defaults(func=reset_db)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
    setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()
