"""Database Manager for handling DB connections, sessions, and migrations."""

from contextlib import contextmanager
from io import StringIO
import os
from pathlib import Path
import sys
import threading

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from dotenv import load_dotenv
from sqlalchemy import Engine, QueuePool, create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import DropTable

from alembic import command
from tracker.config.decorators import (
    log_calls,
    log_database_operations,
    log_performance,
)
from tracker.config.logger import get_logger
from tracker.data.orm.base import Base

logger = get_logger(__name__)


class DBManager:
    """Database Manager for handling DB connections, sessions, and migrations."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Implement singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the DBManager with configuration from environment variables."""
        if getattr(self, "_initialized", False):
            return

        load_dotenv()
        self._engine = None
        self._session_factory = None

        # Database configuration
        self.db_config = {
            "driver": os.getenv("DB_DRIVER", "postgresql+psycopg2"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "database": os.getenv("DB_NAME", "portfolio"),
            "username": os.getenv("DB_USER", "portfolio"),
            "password": os.getenv("DB_PASSWORD", "portfolio"),
            "echo": os.getenv("DB_ECHO", "false").lower() == "true",
        }

        # Connection pool configuration
        self.pool_config = {
            "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
            "max_overflow": int(os.getenv("DB_POOL_MAX_OVERFLOW", "10")),
            "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
            "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "3600")),
            "pool_pre_ping": os.getenv("DB_POOL_PRE_PING", "true").lower() == "true",
        }

        self._initialize_engine()
        self._initialized = True

    def _initialize_engine(self):
        if self.db_config["driver"].startswith("sqlite"):
            path = f"/{self.db_config['database']}"
        elif self.db_config["driver"].startswith("postgresql"):
            path = (
                f"{self.db_config['username']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}"
                f"/{self.db_config['database']}"
            )
        else:
            raise ValueError(f"Unsupported DB_DRIVER: {self.db_config['driver']}")

        db_url = f"{self.db_config['driver']}://{path}"

        try:
            self._engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=self.pool_config["pool_size"],
                max_overflow=self.pool_config["max_overflow"],
                pool_timeout=self.pool_config["pool_timeout"],
                pool_recycle=self.pool_config["pool_recycle"],
                pool_pre_ping=self.pool_config["pool_pre_ping"],
                echo=self.db_config["echo"],
                future=True,
            )

            self._session_factory = sessionmaker(
                bind=self._engine, expire_on_commit=False
            )

            self._setup_event_listeners()

            logger.info(
                "Database engine initialized: %s@%s:%s/%s (pool_size=%d, max_overflow=%d)",
                self.db_config["username"],
                self.db_config["host"],
                self.db_config["port"],
                self.db_config["database"],
                self.pool_config["pool_size"],
                self.pool_config["max_overflow"],
            )

        except SQLAlchemyError as e:
            logger.error("Failed to initialize database engine: %s", e)
            raise

    def _setup_event_listeners(self):
        """Set up event listeners for connection pool events."""
        if self._engine is None:
            raise RuntimeError("Database engine not initialized")

        if self._engine.url.get_backend_name() == "sqlite":

            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(dbapi_connection, _connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        @event.listens_for(self._engine, "connect")
        def receive_connect(_dbapi_connection, _connection_record):
            """Log when a new connection is established."""
            logger.debug("New database connection established")

        @event.listens_for(self._engine, "checkout")
        def receive_checkout(_dbapi_connection, _connection_record, _connection_proxy):
            """Log when a connection is checked out from the pool."""
            logger.debug("Connection checked out from pool")

        @event.listens_for(self._engine, "checkin")
        def receive_checkin(_dbapi_connection, _connection_record):
            """Log when a connection is returned to the pool."""
            logger.debug("Connection returned to pool")

    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine."""
        if self._engine is None:
            raise RuntimeError("Database engine not initialized")
        return self._engine

    @contextmanager
    def get_session(self, readonly=False):
        """Provide a transactional scope around a series of operations."""
        if self._session_factory is None:
            raise RuntimeError("Session factory not initialized")

        session = self._session_factory()
        try:
            yield session
            if readonly:
                session.rollback()
                logger.debug("Session rolled back (readonly mode).")
            else:
                session.commit()
                logger.debug("Session committed successfully.")

        except Exception as e:  # pylint: disable=W0703
            session.rollback()
            logger.error("Session rollback due to error: %s", repr(e))
            raise

        finally:
            session.close()

    @log_performance()
    def create_schema(self):
        """Create all tables in the database based on ORM models."""
        if self._engine is None:
            raise RuntimeError("Database engine not initialized")

        try:
            Base.metadata.create_all(self._engine)
            logger.info("Database schema created successfully.")
        except SQLAlchemyError as e:
            logger.error("Failed to create database schema: %s", e)
            raise

    @log_performance()
    def drop_schema(self, force=False, raw=False, schema="public"):
        """Drop all tables in the database based on ORM models."""
        if self._engine is None:
            raise RuntimeError("Database engine not initialized")

        try:
            if raw:
                with self.get_session() as session:
                    session.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE;"))
                    session.execute(text(f"CREATE SCHEMA {schema};"))
                    logger.info("Schema %s dropped and recreated.", schema)
                return

            if force:

                @compiles(DropTable, "postgresql")
                def _compile_drop_table(element, compiler, **_kw):
                    return compiler.visit_drop_table(element) + " CASCADE"

            Base.metadata.drop_all(bind=self._engine, checkfirst=True)
            logger.info("Database schema dropped successfully.")

        except SQLAlchemyError as e:
            logger.error("Failed to drop database schema: %s", e)
            raise

    @log_calls()
    def get_pool_status(self):
        """Get current status of the connection pool."""
        if self._engine is None or not hasattr(self._engine, "pool"):
            return {"status": "not_initialized"}

        pool = self._engine.pool
        return {
            "status": "active",
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "pool_class": pool.__class__.__name__,
            "engine_url": str(self._engine.url).replace(
                f":{self.db_config['password']}", ":***"
            ),
        }

    @log_database_operations()
    def execute(self, query, params=None, fetch=False, dict_cursor=True):
        """Execute a query using either pooled or individual connection."""
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                logger.debug("Executed query: %s", query)
                if fetch:
                    if dict_cursor:
                        return result.mappings().all()
                    return result.fetchall()
                return None
        except SQLAlchemyError as e:
            logger.error("Failed to execute query: %s", e)
            raise

    @log_database_operations()
    def execute_many(self, query, param_list):
        """Execute a query with multiple sets of parameters."""
        try:
            with self.get_session() as session:
                session.execute(text(query), param_list)
                logger.debug("Executed query with multiple parameter sets.")
                return
        except SQLAlchemyError as e:
            logger.error("Failed to execute query: %s", e)
            raise

    def execute_file(self, file_path, params=None, fetch=False, dict_cursor=True):
        """Execute a SQL file."""
        try:
            with Path(file_path).open(encoding="utf-8") as file:
                sql = file.read()

            result = self.execute(sql, params, fetch, dict_cursor)
            logger.info("Executed SQL file: %s", file_path)
            return result
        except SQLAlchemyError as e:
            logger.error("Failed to execute SQL file: %s", e)
            raise

    def close(self):
        """Dispose the engine and close all connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database engine disposed and all connections closed.")

    # Migration utilities
    @log_calls()
    def run_migrations(self, revision="head"):
        """Run Alembic migrations programmatically.

        Args:
            revision (str): Target revision to upgrade to. Defaults to "head" (latest).
        """
        try:
            config = Config("alembic.ini")
            command.upgrade(config, revision)
            logger.info("Successfully upgraded database to revision: %s", revision)
        except ImportError:
            logger.error("Alembic not installed. Install with: pip install alembic")
            raise
        except Exception as e:
            logger.error("Failed to run migrations: %s", e)
            raise

    @log_calls()
    def create_migration(self, message, autogenerate=True):
        """Create a new Alembic migration.

        Args:
            message (str): Description of the migration.
            autogenerate (bool): Whether to auto-generate migration from model changes.

        Returns:
            str: The revision ID of the created migration.
        """
        try:
            config = Config("alembic.ini")
            revision = command.revision(
                config, message=message, autogenerate=autogenerate
            )
            logger.info("Created migration: %s", message)
            return revision.revision if revision else None
        except ImportError:
            logger.error("Alembic not installed. Install with: pip install alembic")
            raise
        except Exception as e:
            logger.error("Failed to create migration: %s", e)
            raise

    def get_migration_history(self):
        """Get the migration history.

        Returns:
            list: List of migration revisions and descriptions.
        """
        try:
            config = Config("alembic.ini")

            # Capture the output
            output = StringIO()
            config.set_main_option("script_location", "alembic")

            # Get history (this will print to stdout, so we capture it)
            old_stdout = sys.stdout
            sys.stdout = output
            try:
                command.history(config)
                history = output.getvalue()
            finally:
                sys.stdout = old_stdout

            return history.strip().split("\n") if history.strip() else []
        except ImportError:
            logger.error("Alembic not installed. Install with: pip install alembic")
            raise
        except Exception as e:
            logger.error("Failed to get migration history: %s", e)
            raise

    def get_current_revision(self):
        """Get the current database revision.

        Returns:
            str: Current revision ID or None if no migrations have been applied.
        """
        try:
            with self.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision()

        except ImportError:
            logger.error("Alembic not installed. Install with: pip install alembic")
            raise
        except Exception as e:
            logger.error("Failed to get current revision: %s", e)
            raise

    @log_calls()
    def rollback_migration(self, steps=1):
        """Rollback migrations by a number of steps.

        Args:
            steps (int): Number of steps to rollback. Defaults to 1.
        """
        try:
            config = Config("alembic.ini")

            # First, check current revision
            current = self.get_current_revision()
            if not current:
                logger.warning("No migrations to rollback - database not initialized")
                return

            # Get migration history to check if we have enough migrations to rollback
            try:
                # Use alembic script to get available revisions
                script = ScriptDirectory.from_config(config)
                revisions = list(script.walk_revisions())

                if len(revisions) < steps:
                    logger.warning(
                        "Only %d migration(s) available, requested %d",
                    )
                    if len(revisions) == 0:
                        logger.info("No migrations to rollback")
                        return
                    steps = len(revisions)
                    logger.info("Rolling back %d migration(s) instead", steps)
            except Exception as e:  # pylint: disable=W0703
                logger.warning("Could not check migration count: %s", e)
                # Continue with original logic

            target = f"-{steps}" if steps > 0 else str(steps)
            command.downgrade(config, target)
            logger.info("Successfully rolled back %d migration(s)", steps)
        except ImportError:
            logger.error("Alembic not installed. Install with: pip install alembic")
            raise
        except Exception as e:
            logger.error("Failed to rollback migrations: %s", e)
            raise

    def rollback_to_revision(self, revision):
        """Rollback to a specific revision.

        Args:
            revision (str): Target revision to rollback to.
        """
        try:
            config = Config("alembic.ini")
            command.downgrade(config, revision)
            logger.info("Successfully rolled back to revision: %s", revision)
        except ImportError:
            logger.error("Alembic not installed. Install with: pip install alembic")
            raise
        except Exception as e:
            logger.error("Failed to rollback to revision %s: %s", revision, e)
            raise

    def __del__(self):
        """Ensure engine is disposed on deletion."""
        self.close()
