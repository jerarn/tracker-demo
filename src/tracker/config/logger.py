"""Simple, clean logging configuration for the tracker project.

Based on Python logging best practices.
"""

from collections import defaultdict
import contextlib
from datetime import datetime, timezone
import logging
import logging.handlers
import os
from pathlib import Path
import time

from dotenv import load_dotenv


class LogFileConfig:
    """Configuration class for log file management."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        load_dotenv()

        # File rotation settings
        self.max_file_size = self._parse_size(os.getenv("LOG_MAX_FILE_SIZE", "10MB"))
        self.backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
        self.rotation_type = os.getenv(
            "LOG_ROTATION_TYPE", "size"
        ).lower()  # size, time, both

        # Time-based rotation settings
        self.rotation_interval = os.getenv(
            "LOG_ROTATION_INTERVAL", "daily"
        ).lower()  # daily, weekly, monthly
        self.rotation_when = os.getenv("LOG_ROTATION_WHEN", "midnight").lower()

        # File organization
        self.log_dir = Path(os.getenv("LOG_DIR", "logs"))
        self.base_filename = os.getenv("LOG_BASE_FILENAME", "tracker")
        self.file_pattern = os.getenv(
            "LOG_FILE_PATTERN", "{base}.log"
        )  # {base}.log, {base}-{date}.log

        # Compression and cleanup
        self.compress_rotated = (
            os.getenv("LOG_COMPRESS_ROTATED", "false").lower() == "true"
        )
        self.auto_cleanup_days = int(os.getenv("LOG_AUTO_CLEANUP_DAYS", "30"))

        # Archive settings
        self.archive_old_logs = (
            os.getenv("LOG_ARCHIVE_OLD_LOGS", "false").lower() == "true"
        )
        self.archive_dir = Path(os.getenv("LOG_ARCHIVE_DIR", "logs/archive"))

    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '10MB', '1GB' to bytes."""
        size_str = size_str.upper().strip()

        if size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        if size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        if size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        # Assume bytes
        return int(size_str)

    def get_log_file_path(self) -> Path:
        """Get the main log file path."""
        if "{date}" in self.file_pattern:
            date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            filename = self.file_pattern.format(base=self.base_filename, date=date_str)
        else:
            filename = self.file_pattern.format(base=self.base_filename)

        return self.log_dir / filename

    def create_log_directory(self):
        """Create log directory structure."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.archive_old_logs:
            self.archive_dir.mkdir(parents=True, exist_ok=True)


class PerformanceFilter(logging.Filter):
    """Simple filter to reduce high-frequency log noise in production.

    Limits the rate of similar log messages to prevent spam.
    """

    def __init__(self, max_rate: int = 10, window_seconds: int = 60):
        """Initialize the performance filter.

        Args:
            max_rate: Maximum number of similar messages per window
            window_seconds: Time window in seconds for rate limiting
        """
        super().__init__()
        self.max_rate = max_rate
        self.window_seconds = window_seconds
        self.message_counts = defaultdict(list)

    def filter(self, record):
        """Filter log records based on frequency.

        Returns:
            True if record should be logged, False otherwise
        """
        # Only apply filtering in production
        environment = os.getenv("ENVIRONMENT", "development").lower()
        if environment != "production":
            return True

        # Create a key for this type of message
        # Use function name + level to group similar messages
        message_key = f"{record.funcName}:{record.levelname}"

        current_time = time.time()

        # Clean old entries outside the window
        cutoff_time = current_time - self.window_seconds
        self.message_counts[message_key] = [
            timestamp
            for timestamp in self.message_counts[message_key]
            if timestamp > cutoff_time
        ]

        # Check if we're under the rate limit
        if len(self.message_counts[message_key]) < self.max_rate:
            self.message_counts[message_key].append(current_time)
            return True
        # Rate limit exceeded - drop this message
        return False


class LogFileManager:
    """Manages log file rotation, archiving, and cleanup."""

    def __init__(self, config: LogFileConfig):
        """Initialize with a LogFileConfig instance."""
        self.config = config

    def create_file_handler(self) -> logging.Handler:
        """Create appropriate file handler based on configuration."""
        self.config.create_log_directory()
        log_file_path = self.config.get_log_file_path()

        if self.config.rotation_type == "time":
            # Time-based rotation
            handler = self._create_time_based_handler(log_file_path)
        elif self.config.rotation_type == "both":
            # Both size and time-based rotation (use TimedRotatingFileHandler with size check)
            handler = self._create_time_based_handler(log_file_path)
            # Note: For true both-type rotation, you'd need a custom handler
        else:
            # Size-based rotation (default)
            handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
            )

        return handler

    def _create_time_based_handler(self, log_file_path: Path) -> logging.Handler:
        """Create time-based rotating file handler."""
        when_mapping = {
            "daily": "midnight",
            "weekly": "W0",  # Monday
            "monthly": "midnight",  # Will rotate monthly via interval
        }

        when = when_mapping.get(self.config.rotation_interval, "midnight")
        interval = 1

        if self.config.rotation_interval == "monthly":
            when = "midnight"
            interval = 30  # Approximate monthly rotation

        return logging.handlers.TimedRotatingFileHandler(
            log_file_path,
            when=when,
            interval=interval,
            backupCount=self.config.backup_count,
        )

    def cleanup_old_logs(self):
        """Clean up old log files based on configuration."""
        if self.config.auto_cleanup_days <= 0:
            return

        cutoff_time = time.time() - (self.config.auto_cleanup_days * 24 * 3600)

        # Clean up main log directory
        self._cleanup_directory(self.config.log_dir, cutoff_time)

        # Clean up archive directory if it exists
        if self.config.archive_dir.exists():
            self._cleanup_directory(self.config.archive_dir, cutoff_time)

    def _cleanup_directory(self, directory: Path, cutoff_time: float):
        """Clean up files older than cutoff_time in directory."""
        for log_file in directory.glob("*.log*"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    if (
                        self.config.archive_old_logs
                        and directory == self.config.log_dir
                    ):
                        # Move to archive instead of deleting
                        self._archive_file(log_file)
                    else:
                        log_file.unlink()
            except OSError:
                # Skip files we can't process
                continue

    def _archive_file(self, log_file: Path):
        """Archive a log file to the archive directory."""
        try:
            self.config.archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = self.config.archive_dir / log_file.name
            log_file.rename(archive_path)
        except OSError:
            # If archiving fails, just delete the file
            with contextlib.suppress(OSError):
                log_file.unlink()


def setup_logging(config: LogFileConfig = None):
    """Set up logging configuration for the entire application.

    Call this once at application startup.

    Args:
        config: Optional LogFileConfig for custom file management settings
    """
    load_dotenv()

    # Use provided config or create default
    if config is None:
        config = LogFileConfig()

    # Determine environment and log level
    environment = os.getenv("ENVIRONMENT", "development").lower()
    log_level = os.getenv("LOG_LEVEL", "").upper()
    third_party_level = os.getenv("LOG_THIRD_PARTY_LEVEL", "WARNING").upper()

    if log_level and hasattr(logging, log_level):
        level = getattr(logging, log_level)
    elif environment == "production":
        level = logging.INFO
    else:
        level = logging.DEBUG

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler with simple format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)

    # File handler with configurable rotation
    file_manager = LogFileManager(config)
    file_handler = file_manager.create_file_handler()
    file_handler.setLevel(logging.DEBUG)  # Always capture all levels in file
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    # Add performance filter to reduce noise in production
    if environment == "production":
        # More restrictive rate limiting in production
        performance_filter = PerformanceFilter(max_rate=5, window_seconds=60)
        console_handler.addFilter(performance_filter)
        # Apply lighter filtering to file handler to keep more detailed logs
        file_performance_filter = PerformanceFilter(max_rate=20, window_seconds=60)
        file_handler.addFilter(file_performance_filter)
    elif environment == "development":
        # Lighter rate limiting in development (mostly for very spammy operations)
        dev_filter = PerformanceFilter(max_rate=50, window_seconds=30)
        console_handler.addFilter(dev_filter)

    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Quiet down noisy third-party libraries
    logging.getLogger("urllib3").setLevel(third_party_level)
    logging.getLogger("requests").setLevel(third_party_level)
    logging.getLogger("sqlalchemy.engine").setLevel(third_party_level)
    logging.getLogger("alembic").setLevel(third_party_level)

    # Perform log cleanup if configured
    file_manager.cleanup_old_logs()

    # Log the startup with configuration info
    logger = logging.getLogger("tracker.config")
    logger.info(
        f"Logging initialized - Environment: {environment}, Level: {logging.getLevelName(level)}"
    )
    logger.info(
        f"Log rotation: {config.rotation_type}, Max size: {config.max_file_size / 1024 / 1024:.1f}MB, "
        f"Backups: {config.backup_count}, Cleanup: {config.auto_cleanup_days} days"
    )


def get_logger(name: str | None = None):
    """Get a logger for a module.

    Args:
        name: Logger name, typically __name__ from the calling module

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    if name is None:
        name = "tracker"
    elif callable(name):
        # Handle case where a function object is passed instead of string
        name = getattr(name, "__module__", "tracker")
    elif not isinstance(name, str):
        # Convert to string if it's not a string
        name = str(name)

    if not name.startswith("tracker"):
        # Ensure all loggers are under the 'tracker' hierarchy
        name = f"tracker.{name}" if name else "tracker"

    return logging.getLogger(name)


def cleanup_logs(config: LogFileConfig = None):
    """Manually trigger log cleanup.

    Args:
        config: Optional LogFileConfig, defaults to environment settings
    """
    if config is None:
        config = LogFileConfig()

    manager = LogFileManager(config)
    manager.cleanup_old_logs()

    logger = get_logger("tracker.config")
    logger.info("Manual log cleanup completed")


def get_log_stats(config: LogFileConfig = None) -> dict:
    """Get statistics about current log files.

    Args:
        config: Optional LogFileConfig

    Returns:
        Dictionary with log file statistics
    """
    if config is None:
        config = LogFileConfig()

    stats = {
        "log_dir": str(config.log_dir),
        "total_files": 0,
        "total_size_mb": 0,
        "files": [],
    }

    if config.log_dir.exists():
        for log_file in config.log_dir.glob("*.log*"):
            try:
                size = log_file.stat().st_size
                stats["files"].append(
                    {
                        "name": log_file.name,
                        "size_mb": round(size / 1024 / 1024, 2),
                        "modified": datetime.fromtimestamp(
                            log_file.stat().st_mtime, tz=timezone.utc
                        ).isoformat(),
                    }
                )
                stats["total_size_mb"] += size
                stats["total_files"] += 1
            except OSError:
                continue

    stats["total_size_mb"] = round(stats["total_size_mb"] / 1024 / 1024, 2)
    return stats
