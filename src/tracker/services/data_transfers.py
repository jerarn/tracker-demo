"""Services for data import and export."""

from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from sqlalchemy import Column

from tracker.config.logger import get_logger
from tracker.data.managers.csv_manager import CSVManager
from tracker.data.managers.db_manager import DBManager
from tracker.data.managers.fk_resolver import FKResolver
from tracker.data.orm.account import Account
from tracker.data.orm.base import Base
from tracker.data.orm.currency import Currency
from tracker.data.orm.instrument import Instrument
from tracker.data.orm.lookups import AssetClass, DataSource, DataType
from tracker.data.orm.portfolio import Portfolio
from tracker.data.orm.transaction import Transaction, TransactionMethod

from .service import Service

logger = get_logger(__name__)


class ImportService(Service):
    """Service for data ingestion."""

    fk_map: ClassVar[dict[str, tuple[type[Base], str]]] = {
        "currency": (Currency, "code"),
        "instrument": (Instrument, "ticker"),
        "base_ticker": (Instrument, "ticker"),
        "quote_ticker": (Instrument, "ticker"),
        "fee_ticker": (Instrument, "ticker"),
        "account": (Account, "name"),
        "data_source": (DataSource, "name"),
        "data_type": (DataType, "name"),
        "asset_class": (AssetClass, "name"),
    }

    def __init__(self, db_manager: DBManager | None = None):
        """Initialize the ingestion service."""
        super().__init__(db_manager=db_manager)
        self.fk_cache: dict[str, dict[str, int]] = {}

    @staticmethod
    def _cast_value(value: str | None, column: Column) -> Any:
        if value in (None, "", "NULL", "NaN"):
            return None

        try:
            col_type = getattr(column.type, "python_type", None)
            if col_type is None:
                return value
            if col_type is bool:
                return str(value).strip().lower() in ("true", "1", "yes", "y")
            if col_type in (datetime, date):
                return pd.to_datetime(value, utc=True).to_pydatetime()

            return col_type(value)
        except Exception as e:
            logger.warning(f"Failed to cast value '{value}' to {col_type}: {e}")
            return value

    @staticmethod
    def _resolve_fk(row: dict[str, Any], resolver: FKResolver) -> None:
        """Resolve foreign key references in a CSV row using FKResolver."""
        for account_col, portfolio_col in [
            ("account", "portfolio"),
            ("account_from", "portfolio_from"),
            ("account_to", "portfolio_to"),
        ]:
            if row.get(account_col) and row.get(portfolio_col):
                acc, port = row.pop(account_col, None), row.pop(portfolio_col, None)
                row[portfolio_col + "_id"] = (
                    resolver.resolve(Portfolio, ("account_name", "name"), (acc, port))
                    if acc and port
                    else None
                )

        for csv_col, (fk_model, fk_field) in ImportService.fk_map.items():
            if row.get(csv_col) is not None:
                fk_id = resolver.resolve(fk_model, fk_field, row.pop(csv_col))
                if fk_id is None:
                    logger.warning(
                        f"Could not resolve foreign key for {csv_col}='{row.get(csv_col)}'"
                    )
                row[csv_col + "_id"] = fk_id

    def load_from_csv(
        self, model: type[Base], file_path: str, method: TransactionMethod | None = None
    ) -> None:
        """Load data from a CSV file into the database."""
        repo = self.get_repository(model)
        data = CSVManager.read_csv(file_path)

        if not data:
            logger.warning(f"No data found in CSV file: {file_path}")
            return

        with self.db_manager.get_session() as session:
            resolver = FKResolver(session)
            for row in data:
                self._resolve_fk(row, resolver)
                for col_name, column in repo.columns.items():
                    if col_name in row:
                        row[col_name] = self._cast_value(row[col_name], column)

        strategy = "add" if model is Transaction else "upsert"
        batch_size = 1 if model is Transaction else 1000

        repo.create_many(
            data_list=data,
            strategy=strategy,
            batch_size=batch_size,
            return_instances=False,
            session=None,
            method=method,
        )

        return


class ExportService(Service):
    """Service for data export."""

    def export_to_csv_sql(
        self, sql_filepath: str, csv_filepath: str, params: dict[str, Any] | None = None
    ) -> None:
        """Export data from the database to a CSV file using a SQL query."""
        if not Path(sql_filepath).exists():
            logger.error(f"SQL file not found: {sql_filepath}")
            return

        data = self.db_manager.execute_file(sql_filepath, fetch=True, params=params)
        if not data:
            logger.warning(f"No data returned from SQL file: {sql_filepath}")
            return

        CSVManager.write_csv(data, csv_filepath)
        logger.info(f"Exported {len(data)} rows to {csv_filepath}")
