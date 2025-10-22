"""CSV Manager for reading/writing CSV data."""

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tracker.config.decorators import log_calls, log_performance
from tracker.config.logger import get_logger

logger = get_logger(__name__)


class CSVManager:
    """Manager for reading/writing CSV data."""

    @log_calls()
    @log_performance()
    @staticmethod
    def read_csv(
        file_path: str,
        use_pandas: bool = True,
        low_memory: bool = False,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Read a CSV file and return a list of dictionaries.

        Args:
            file_path: Path to the CSV file.
            use_pandas: Use pandas for fast loading and type inference.
            low_memory: Whether to use low_memory mode in pandas (only if use_pandas is True).
            chunk_size: Number of rows to read at a time (only if use_pandas is True).
            **kwargs: Additional arguments to pass to pandas.read_csv.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("CSV file not found: %s", file_path)
            return []

        if use_pandas:
            df = pd.read_csv(
                path, low_memory=low_memory, chunksize=chunk_size, **kwargs
            ).replace({np.nan: None})
            logger.info("Read %d rows from %s", len(df), file_path)
            return df.to_dict(orient="records")

        # fallback to native CSV
        with path.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = [dict(r) for r in reader]
        logger.info("Read %d rows from %s", len(rows), file_path)
        return rows

    @log_calls()
    @log_performance()
    @staticmethod
    def write_csv(items: list[dict[str, Any]], file_path: str) -> None:
        """Write a list of dictionaries to a CSV file."""
        if not items:
            logger.warning("No items to write to CSV.")
            return

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = items[0].keys()

        with path.open(mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(items)

        logger.info("Wrote %d items to %s", len(items), file_path)
