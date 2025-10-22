"""Abstract base class and Lot dataclass for accounting strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Lot:
    """Represents a lot/parcel of holdings for FIFO/LIFO accounting."""

    qty: float
    price: float
    date: pd.Timestamp | None = None


class AccountingStrategy(ABC):
    """Abstract base class for different accounting methods."""

    @abstractmethod
    def process_transaction(
        self, state: dict, tr_type: str, qty: float, price: float
    ) -> tuple[dict, float]:
        """Process a single transaction and return updated state and realized P&L."""

    @abstractmethod
    def get_current_metrics(self, state: dict) -> dict:
        """Get current qty, cost_basis, and avg_cost from state."""
