"""Alembic migration environment setup."""

# pylint: disable=no-member
from alembic import context
from tracker import Base, DBManager

# Instantiate your DBManager (this will load env vars via dotenv)
db_manager = DBManager()

# Tell Alembic to use your Base metadata
target_metadata = Base.metadata


def run_migrations_offline():
    """Run migrations without DB connection (generate SQL)."""
    url = str(db_manager.engine.url)  # reuse DBManager's engine url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations with a DB connection from DBManager."""
    with db_manager.engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
