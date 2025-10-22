"""Service layer for higher-level operations."""

from tracker.config.logger import get_logger
from tracker.data.managers.db_manager import DBManager
from tracker.data.orm.base import Base
from tracker.data.orm.cash_flow import CashFlow
from tracker.data.orm.market_data import MarketData
from tracker.data.orm.transaction import Transaction
from tracker.data.repos.cf_repo import CashFlowRepository
from tracker.data.repos.mkt_repo import MarketDataRepository
from tracker.data.repos.repository import Repository
from tracker.data.repos.tx_repo import TransactionRepository

logger = get_logger(__name__)

_REPO_MAP = {
    Transaction: TransactionRepository,
    CashFlow: CashFlowRepository,
    MarketData: MarketDataRepository,
}


class Service:
    """Base class for services."""

    def __init__(self, db_manager: DBManager | None = None):
        """Initialize the service."""
        self.db_manager = db_manager or DBManager()
        self._repositories: dict[type[Base], Repository] = {}

    def get_repository(self, model: type[Base]) -> Repository:
        """Get the repository for a given model."""
        if model not in self._repositories:
            repo_cls = _REPO_MAP.get(model)
            self._repositories[model] = (
                Repository(model, self.db_manager)
                if repo_cls is None
                else repo_cls(self.db_manager)
            )
        return self._repositories[model]
