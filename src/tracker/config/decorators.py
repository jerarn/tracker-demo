"""Logging decorators for automatic function and operation tracking.

Provides simple decorators to add logging to functions without manual logging code.
"""

from collections.abc import Callable
import functools
import time
from typing import ParamSpec, TypeVar

from .logger import get_logger

P = ParamSpec("P")
T = TypeVar("T")


def log_calls(
    logger_name: str | None = None,
    log_args: bool = True,
    log_result: bool = True,
    log_timing: bool = True,
    level: str = "DEBUG",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to automatically log function calls with arguments, results, and timing.

    Args:
        logger_name: Custom logger name, defaults to function's module
        log_args: Whether to log function arguments
        log_result: Whether to log function return value
        log_timing: Whether to log execution time
        level: Log level (DEBUG, INFO, WARNING, ERROR)

    Example:
        @log_calls()
        def calculate_portfolio_value(portfolio_id):
            return 1000.0
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        # Get logger for the function's module
        logger = get_logger(logger_name or func.__module__)
        log_level = getattr(logger, level.lower())

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            func_name = func.__name__

            # Log function entry
            args_str = ""
            if log_args and (args or kwargs):
                args_parts = []
                if args:
                    args_parts.extend(
                        [str(arg)[:100] for arg in args]
                    )  # Limit arg length
                if kwargs:
                    args_parts.extend(
                        [f"{k}={str(v)[:100]}" for k, v in kwargs.items()]
                    )
                args_str = f" with args: ({', '.join(args_parts)})"

            log_level(f"Calling {func_name}{args_str}")

            # Execute function with timing
            start_time = time.time() if log_timing else None
            try:
                result = func(*args, **kwargs)

                # Log success
                timing_str = ""
                if log_timing:
                    elapsed = time.time() - start_time
                    timing_str = f" (took {elapsed:.3f}s)"

                result_str = ""
                if log_result:
                    result_preview = str(result)[:200] if result is not None else "None"
                    result_str = f" -> {result_preview}"

                log_level(f"Completed {func_name}{timing_str}{result_str}")
                return result

            except Exception as e:
                # Log exception
                timing_str = ""
                if log_timing:
                    elapsed = time.time() - start_time
                    timing_str = f" (failed after {elapsed:.3f}s)"

                logger.error(f"Failed {func_name}{timing_str}: {type(e).__name__}: {e}")
                raise

        return wrapper

    return decorator


def log_database_operations(
    operation_type: str = "DATABASE",
    log_queries: bool = False,
    log_results: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for logging database operations with automatic timing and error handling.

    Args:
        operation_type: Type of DB operation (CREATE, READ, UPDATE, DELETE)
        log_queries: Whether to log SQL queries (be careful with sensitive data)
        log_results: Whether to log operation results

    Example:
        @log_database_operations("CREATE")
        def create_portfolio(self, portfolio_data):
            return self.session.add(portfolio_data)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            func_name = func.__name__

            # Log operation start
            logger.debug(f"Starting {operation_type} operation: {func_name}")

            start_time = time.time()
            try:
                # Log query if requested
                if log_queries and "query" in kwargs:
                    logger.debug(f"Query: {kwargs['query']}")

                result = func(*args, **kwargs)

                # Log success
                elapsed = time.time() - start_time
                logger.debug(
                    f"Completed {operation_type} operation: {func_name} ({elapsed:.3f}s)"
                )

                # Log results if requested
                if log_results and result is not None:
                    if hasattr(result, "__len__"):
                        logger.debug(f"Operation returned {len(result)} records")
                    else:
                        logger.debug(f"Operation result: {str(result)[:100]}")

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Failed {operation_type} operation: {func_name} ({elapsed:.3f}s): {e}"
                )
                raise

        return wrapper

    return decorator


def log_api_calls(
    service_name: str = "API",
    log_requests: bool = True,
    log_responses: bool = False,
    max_response_length: int = 500,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for logging external API calls with retry patterns and rate limiting awareness.

    Args:
        service_name: Name of the external service
        log_requests: Whether to log request details
        log_responses: Whether to log response data
        max_response_length: Maximum length of response data to log

    Example:
        @log_api_calls("YahooFinance")
        def fetch_stock_price(self, symbol):
            return requests.get(f"https://api.yahoo.com/v1/quote/{symbol}")
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            func_name = func.__name__

            # Extract common request info
            request_info = ""
            if log_requests:
                if "url" in kwargs:
                    request_info = f" to {kwargs['url']}"
                elif (
                    len(args) > 1
                    and isinstance(args[1], str)
                    and args[1].startswith("http")
                ):
                    request_info = f" to {args[1]}"

            logger.info(f"Calling {service_name} API: {func_name}{request_info}")

            start_time = time.time()
            try:
                result = func(*args, **kwargs)

                elapsed = time.time() - start_time
                logger.info(
                    f"Completed {service_name} API call: {func_name} ({elapsed:.3f}s)"
                )

                # Log response if requested
                if log_responses and result is not None:
                    response_preview = str(result)[:max_response_length]
                    logger.debug(f"API response: {response_preview}")

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Failed {service_name} API call: {func_name} ({elapsed:.3f}s): {e}"
                )
                raise

        return wrapper

    return decorator


def log_dataframe_operations(
    log_memory: bool = True, log_dtypes: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for logging pandas DataFrame operations with memory usage and shape info.

    Args:
        log_memory: Whether to log memory usage
        log_dtypes: Whether to log data types

    Example:
        @log_dataframe_operations()
        def process_portfolio_data(self, df):
            return df.groupby('symbol').sum()
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            func_name = func.__name__

            # Find DataFrame arguments
            df_info = []
            for i, arg in enumerate(args):
                if hasattr(arg, "shape"):  # Duck typing for DataFrame-like objects
                    info = f"arg{i}: {arg.shape}"
                    if log_memory and hasattr(arg, "memory_usage"):
                        memory_mb = arg.memory_usage(deep=True).sum() / 1024 / 1024
                        info += f" ({memory_mb:.2f}MB)"
                    df_info.append(info)

            for name, value in kwargs.items():
                if hasattr(value, "shape"):
                    info = f"{name}: {value.shape}"
                    if log_memory and hasattr(value, "memory_usage"):
                        memory_mb = value.memory_usage(deep=True).sum() / 1024 / 1024
                        info += f" ({memory_mb:.2f}MB)"
                    df_info.append(info)

            input_info = f" [Input: {', '.join(df_info)}]" if df_info else ""
            logger.info(f"Processing DataFrame operation: {func_name}{input_info}")

            start_time = time.time()
            try:
                result = func(*args, **kwargs)

                elapsed = time.time() - start_time

                # Log result info
                result_info = ""
                if hasattr(result, "shape"):
                    result_info = f" -> {result.shape}"
                    if log_memory and hasattr(result, "memory_usage"):
                        memory_mb = result.memory_usage(deep=True).sum() / 1024 / 1024
                        result_info += f" ({memory_mb:.2f}MB)"

                    if log_dtypes:
                        logger.debug(f"Result dtypes: {dict(result.dtypes)}")

                logger.info(
                    f"Completed DataFrame operation: {func_name} ({elapsed:.3f}s){result_info}"
                )
                return result

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Failed DataFrame operation: {func_name} ({elapsed:.3f}s): {e}"
                )
                raise

        return wrapper

    return decorator


def log_performance(
    warn_threshold: float = 1.0,
    error_threshold: float = 5.0,
    memory_tracking: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for performance monitoring with configurable thresholds.

    Args:
        warn_threshold: Seconds after which to log a warning
        error_threshold: Seconds after which to log an error
        memory_tracking: Whether to track memory usage

    Example:
        @log_performance(warn_threshold=0.5, error_threshold=2.0)
        def expensive_calculation(self, data):
            return heavy_computation(data)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            func_name = func.__name__

            # Memory tracking setup
            import psutil

            process = psutil.Process() if memory_tracking else None
            start_memory = process.memory_info().rss / 1024 / 1024 if process else None

            start_time = time.time()
            try:
                result = func(*args, **kwargs)

                elapsed = time.time() - start_time

                # Determine log level based on timing
                memory_str = ""
                if memory_tracking and process:
                    end_memory = process.memory_info().rss / 1024 / 1024
                    memory_diff = end_memory - start_memory
                    memory_str = f" (memory: {memory_diff:+.1f}MB)"

                message = (
                    f"Performance: {func_name} completed in {elapsed:.3f}s{memory_str}"
                )

                if elapsed >= error_threshold:
                    logger.error(f"SLOW PERFORMANCE: {message}")
                elif elapsed >= warn_threshold:
                    logger.warning(f"PERFORMANCE WARNING: {message}")
                else:
                    logger.debug(message)

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Performance: {func_name} failed after {elapsed:.3f}s: {e}"
                )
                raise

        return wrapper

    return decorator


class LoggerMixin:
    """Mixin class that provides logging capabilities to any class.

    Automatically creates a logger based on the class name.

    Example:
        class PortfolioManager(LoggerMixin):
            def create_portfolio(self, data):
                self.logger.debug("Creating new portfolio")
                return self._do_create(data)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the mixin and set up the logger."""
        super().__init__(*args, **kwargs)
        self.logger = get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    def log_method_call(self, method_name: str, *args, **kwargs):
        """Manually log a method call with arguments."""
        args_str = ", ".join([str(arg)[:50] for arg in args])
        kwargs_str = ", ".join([f"{k}={str(v)[:50]}" for k, v in kwargs.items()])
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        self.logger.debug(f"Calling {method_name}({all_args})")

    def log_operation_start(self, operation: str, details: str = ""):
        """Log the start of an operation."""
        details_str = f": {details}" if details else ""
        self.logger.debug(f"Starting {operation}{details_str}")

    def log_operation_success(
        self, operation: str, duration: float | None = None, details: str = ""
    ):
        """Log successful completion of an operation."""
        timing_str = f" ({duration:.3f}s)" if duration else ""
        details_str = f": {details}" if details else ""
        self.logger.debug(f"Completed {operation}{timing_str}{details_str}")

    def log_operation_error(
        self, operation: str, error: Exception, duration: float | None = None
    ):
        """Log operation failure."""
        timing_str = f" ({duration:.3f}s)" if duration else ""
        self.logger.error(
            f"Failed {operation}{timing_str}: {type(error).__name__}: {error}"
        )


def audit_log(
    action: str,
    audit_logger_name: str = "tracker.audit",
    include_user: bool = True,
    include_timestamp: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for audit logging of sensitive operations.

    Args:
        action: Description of the action being audited
        audit_logger_name: Name of the audit logger
        include_user: Whether to include user information
        include_timestamp: Whether to include timestamp

    Example:
        @audit_log("DELETE_PORTFOLIO")
        def delete_portfolio(self, portfolio_id):
            return self.db.delete(portfolio_id)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        audit_logger = get_logger(audit_logger_name)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            import datetime

            # Build audit message
            audit_parts = [f"ACTION={action}"]

            if include_timestamp:
                audit_parts.append(
                    f"TIMESTAMP={datetime.datetime.now(tz=datetime.timezone.utc).isoformat()}"
                )

            if include_user:
                # Try to extract user info from common locations
                user = "UNKNOWN"
                if hasattr(args[0], "current_user"):
                    user = getattr(args[0].current_user, "username", "UNKNOWN")
                elif "user_id" in kwargs:
                    user = kwargs["user_id"]
                audit_parts.append(f"USER={user}")

            # Add function arguments (be careful with sensitive data)
            if args[1:]:  # Skip 'self' argument
                audit_parts.append(f"ARGS={args[1:]}")
            if kwargs:
                audit_parts.append(f"KWARGS={kwargs}")

            audit_message = " | ".join(audit_parts)

            try:
                result = func(*args, **kwargs)
                audit_logger.info(f"AUDIT SUCCESS: {audit_message}")
                return result
            except Exception as e:
                audit_logger.error(f"AUDIT FAILURE: {audit_message} | ERROR={e}")
                raise

        return wrapper

    return decorator
