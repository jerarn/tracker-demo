"""Portfolio Tracker - A comprehensive portfolio analysis toolkit."""

__version__ = "0.1.0"
__description__ = "A comprehensive portfolio tracking and analysis toolkit"

# Analytics - core analysis tools
from .analytics.accounting.avg_cost import AverageCost
from .analytics.accounting.cash import Cash
from .analytics.accounting.fifo import FIFO
from .analytics.accounting.lifo import LIFO
from .analytics.analyzer import PortfolioAnalyzer
from .analytics.core.holdings import Holdings
from .analytics.core.performance import Performance
from .analytics.core.price_manager import PriceManager
from .analytics.core.risk_metrics import RiskMetrics

# Configuration - logging and other settings
from .config.decorators import (
    LoggerMixin,
    audit_log,
    log_api_calls,
    log_calls,
    log_database_operations,
    log_dataframe_operations,
    log_performance,
)
from .config.logger import (
    LogFileConfig,
    cleanup_logs,
    get_log_stats,
    get_logger,
    setup_logging,
)

# Data access managers - for fetching external data
from .data.access.binance_api import BinanceAPI
from .data.access.morningstar_api import MorningstarAPI
from .data.access.yf_api import YFinanceAPI

# Data management - for handling data storage and retrieval
from .data.managers.csv_manager import CSVManager
from .data.managers.db_manager import DBManager

# Data models - core data structures
from .data.orm.account import Account
from .data.orm.base import Base
from .data.orm.cash_flow import CashFlow, CashFlowType
from .data.orm.currency import Currency
from .data.orm.instrument import Instrument
from .data.orm.lookups import AssetClass, DataSource, DataType
from .data.orm.market_data import MarketData
from .data.orm.portfolio import Portfolio
from .data.orm.transaction import (
    Transaction,
    TransactionMethod,
    TransactionType,
)

# Data repositories - for database interactions
from .data.repos.cf_repo import CashFlowRepository
from .data.repos.mkt_repo import MarketDataRepository
from .data.repos.repository import (
    Repository,
)
from .data.repos.tx_repo import TransactionRepository

# Services - higher-level operations
from .services.analytics import AnalyticsService
from .services.data_transfers import ExportService, ImportService
from .services.service import Service

__all__ = [
    "FIFO",
    "LIFO",
    "Account",
    "AnalyticsService",
    "AssetClass",
    "AverageCost",
    "Base",
    "BinanceAPI",
    "CSVManager",
    "Cash",
    "CashFlow",
    "CashFlowRepository",
    "CashFlowType",
    "Currency",
    "DBManager",
    "DataSource",
    "DataType",
    "ExportService",
    "Holdings",
    "ImportService",
    "Instrument",
    "LogFileConfig",
    "LoggerMixin",
    "MarketData",
    "MarketDataRepository",
    "MorningstarAPI",
    "Performance",
    "Portfolio",
    "PortfolioAnalyzer",
    "PriceManager",
    "Repository",
    "RiskMetrics",
    "Service",
    "Transaction",
    "TransactionMethod",
    "TransactionRepository",
    "TransactionType",
    "YFinanceAPI",
    "audit_log",
    "cleanup_logs",
    "get_log_stats",
    "get_logger",
    "log_api_calls",
    "log_calls",
    "log_database_operations",
    "log_dataframe_operations",
    "log_performance",
    "setup_logging",
]
