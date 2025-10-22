"""Repository pattern implementation for database operations using SQLAlchemy ORM."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Generic, TypeVar

import pandas as pd
from sqlalchemy import Column, and_, inspect, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as lite_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from tracker.config.decorators import log_database_operations, log_performance
from tracker.config.logger import get_logger
from tracker.data.managers.db_manager import DBManager
from tracker.data.orm.base import Base
from tracker.data.orm.transaction import TransactionMethod

Model = TypeVar("Model", bound=Base)

logger = get_logger(__name__)


class Repository(Generic[Model]):
    """Generic repository for database operations on a given SQLAlchemy model."""

    def __init__(self, model: type[Model], db_manager: DBManager | None = None):
        """Initialize repository with model and database manager."""
        self.model = model
        self.db_manager = db_manager or DBManager()

    @property
    def columns(self) -> dict[str, Column]:
        """List of column names for the model."""
        mapper = inspect(self.model)
        return {col.key: col for col in mapper.columns}

    @contextmanager
    def get_session(self, session: Session) -> Generator[Session, None, None]:
        """Provide a database session for the duration of the context."""
        if session is not None:
            yield session
        else:
            with self.db_manager.get_session() as new_session:
                yield new_session

    @log_database_operations(operation_type="CREATE")
    def create(self, session: Session | None = None, **kwargs) -> Model:
        """Create new instance in the database."""
        instance = self.model(**kwargs)
        with self.get_session(session) as session:
            session.add(instance)
            logger.debug("Created new instance: %s", instance)

        return instance

    @log_database_operations(operation_type="CREATE")
    @log_performance()
    def create_many(
        self,
        data_list: list[dict[str, Any]],
        strategy: str = "add",
        batch_size: int = 1000,
        return_instances: bool = False,
        session: Session | None = None,
        method: TransactionMethod | None = None,
    ) -> list[Model] | int:
        """Create multiple instances in the database."""
        if not data_list:
            logger.warning("No data provided for bulk create.")
            return []
        if method:
            logger.warning(
                "Method parameter is a placeholder for transaction repository."
            )

        if strategy == "upsert" and not self.model.conflict_fields():
            strategy = "add"
            logger.warning(
                "Model %s does not define conflict_fields; switching strategy to 'add'.",
                self.model.__name__,
            )

        with self.get_session(session) as session:
            try:
                if strategy == "add":
                    instances = [self.model(**data) for data in data_list]
                    for i in range(0, len(instances), batch_size):
                        batched_instances = instances[i : i + batch_size]
                        session.add_all(batched_instances)
                        logger.debug(
                            "Created %d instances: %s",
                            len(batched_instances),
                            batched_instances,
                        )

                    return instances if return_instances else len(instances)

                if strategy == "bulk":
                    for i in range(0, len(data_list), batch_size):
                        batched_data = data_list[i : i + batch_size]
                        session.bulk_insert_mappings(self.model, batched_data)
                        logger.debug(
                            "Bulk created %d instances: %s",
                            len(batched_data),
                            batched_data,
                        )

                    if return_instances:
                        logger.warning(
                            "Return instances is not supported with bulk strategy."
                        )

                    return len(data_list)

                if strategy == "upsert":
                    # Dialect check (SQLite vs PostgreSQL)
                    dialect = session.bind.dialect.name
                    if dialect == "postgresql":
                        insert_stmt = pg_insert(self.model)
                    elif dialect == "sqlite":
                        insert_stmt = lite_insert(self.model)
                    else:
                        raise NotImplementedError(
                            f"Upsert not supported for dialect '{dialect}'."
                        )

                    # Default update fields = all except conflict_fields
                    conflict_fields = self.model.conflict_fields()
                    update_fields = [
                        c.name
                        for c in self.model.__table__.columns
                        if c.name
                        not in (*conflict_fields, "id", "created_at", "updated_at")
                    ]

                    for i in range(0, len(data_list), batch_size):
                        batched_data = data_list[i : i + batch_size]
                        stmt = insert_stmt.values(batched_data).on_conflict_do_update(
                            index_elements=conflict_fields,
                            set_={
                                field: getattr(insert_stmt.excluded, field)
                                for field in update_fields
                            },
                        )
                        session.execute(stmt)

                    return len(data_list)

                raise ValueError(f"Unknown strategy: {strategy}")

            except SQLAlchemyError as e:
                logger.exception("Error during create_many: %s", e)
                session.rollback()
                raise

    def _build_and_conditions(self, all_filters: dict[str, Any]) -> list[Any]:
        and_conditions = []
        for attr, value in all_filters.items():
            column = getattr(self.model, attr)

            if isinstance(value, dict):
                # Handle special operators
                if "in" in value:
                    and_conditions.append(column.in_(value["in"]))
                    logger.debug("Filtering %s IN %s", column, value["in"])
                elif "gt" in value:
                    and_conditions.append(column > value["gt"])
                    logger.debug("Filtering %s > %s", column, value["gt"])
                elif "lt" in value:
                    and_conditions.append(column < value["lt"])
                    logger.debug("Filtering %s < %s", column, value["lt"])
                elif "gte" in value:
                    and_conditions.append(column >= value["gte"])
                    logger.debug("Filtering %s >= %s", column, value["gte"])
                elif "lte" in value:
                    and_conditions.append(column <= value["lte"])
                    logger.debug("Filtering %s <= %s", column, value["lte"])
                elif "like" in value:
                    and_conditions.append(column.like(value["like"]))
                    logger.debug("Filtering %s LIKE %s", column, value["like"])
                elif "ilike" in value:
                    and_conditions.append(column.ilike(value["ilike"]))
                    logger.debug("Filtering %s ILIKE %s", column, value["ilike"])
            else:
                # Simple equality
                and_conditions.append(column == value)
                logger.debug("Filtering %s = %s", column, value)

        return and_conditions

    def _build_or_conditions(self, or_filters: list[dict[str, Any]]) -> list[Any]:
        or_conditions = []
        for filter_dict in or_filters:
            filter_conditions = []
            for attr, value in filter_dict.items():
                column = getattr(self.model, attr)

                if isinstance(value, dict) and "in" in value:
                    filter_conditions.append(column.in_(value["in"]))
                    logger.debug("Filtering %s IN %s", column, value["in"])
                else:
                    filter_conditions.append(column == value)
                    logger.debug("Filtering %s = %s", column, value)

            if filter_conditions:
                or_conditions.append(and_(*filter_conditions))
                logger.debug("Constructed OR condition: %s", filter_conditions)

        return or_conditions

    @log_database_operations(operation_type="READ")
    def get(
        self,
        method: str = "all",
        load: bool = False,
        filters: dict[str, Any] | None = None,
        or_filters: list[dict[str, Any]] | None = None,
        session: Session | None = None,
        **kwargs,
    ) -> Model | list[Model] | None:
        """Retrieve instances with flexible filtering options.

        Args:
            method: "one", "one_or_none", "all", "first"
            load: Whether to apply eager loading options
            filters: Simple filters {attr: value} or {attr: {"in": [values]}}
            or_filters: List of filter dicts for OR conditions
            session: Optional SQLAlchemy session
            **kwargs: Additional simple filters

        Examples:
            # Simple filtering
            repo.get(name="Portfolio1", status="active")

            # IN filtering
            repo.get(filters={"status": {"in": ["active", "pending"]}})

            # OR filtering
            repo.get(or_filters=[{"name": "Portfolio1"}, {"status": "active"}])

            # Combined
            repo.get(
                filters={"type": "equity", "status": {"in": ["active", "pending"]}},
                or_filters=[{"name": "Portfolio1"}, {"name": "Portfolio2"}]
            )
        """
        # Combine explicit filters with kwargs
        all_filters = {**(filters or {}), **kwargs}

        with self.get_session(session) as session:
            # Handle simple ID lookup for performance
            if (
                len(all_filters) == 1
                and "id" in all_filters
                and not or_filters
                and isinstance(all_filters["id"], int)
            ):
                instance = session.get(self.model, all_filters["id"])
                logger.debug("Retrieved instance by ID: %s", instance)
                return instance

            # Build query with modern SQLAlchemy 2.0 style
            stmt = select(self.model)

            # Apply AND filters
            if all_filters:
                and_conditions = self._build_and_conditions(all_filters)
                if and_conditions:
                    stmt = stmt.where(and_(*and_conditions))
                    logger.debug("Applied AND filters: %s", and_conditions)

            # Apply OR filters
            if or_filters:
                or_conditions = self._build_or_conditions(or_filters)

                if or_conditions:
                    stmt = stmt.where(or_(*or_conditions))
                    logger.debug("Applied OR filters: %s", or_conditions)

            # Apply eager loading if specified
            if load:
                load_options = self.model.load_options()
                stmt = stmt.options(*load_options)
                logger.debug("Applied eager loading options: %s", load_options)

            # Execute query based on method
            result = session.execute(stmt)
            logger.debug("Executed query: %s", stmt)

            if method == "one":
                instance = result.scalar_one()
            elif method == "one_or_none":
                instance = result.scalar_one_or_none()
            elif method == "first":
                instance = result.scalar()
            else:  # "all"
                instance = result.scalars().all()

            logger.debug("Retrieved instance(s): %s", instance)
            return instance

    @log_database_operations(operation_type="DELETE")
    def delete(self, instances: Model | list[Model]) -> None:
        """Delete instance(s) from the database."""
        if not instances:
            logger.warning("No instances provided for deletion.")
            return

        if not isinstance(instances, list):
            instances = [instances]

        with self.get_session() as session:
            for instance in instances:
                session.delete(instance)
                logger.debug("Deleted instance: %s", instance)

    def to_df(self, instances: list[Model]) -> pd.DataFrame:
        """Convert a list of model instances to a pandas DataFrame."""
        if not instances:
            return pd.DataFrame()

        df = pd.DataFrame([instance.to_dict() for instance in instances])
        logger.debug("Converted %d instances to DataFrame", len(instances))
        return df
