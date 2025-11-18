"""Structured logging configuration for GRYPHGEN agents."""

import logging
import sys
from typing import Optional
from pathlib import Path

import structlog
from pythonjsonlogger import jsonlogger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    structured: bool = True,
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        structured: Whether to use structured logging (JSON format)
    """
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure stdlib logging
    if structured:
        # JSON formatter for structured logging
        formatter = jsonlogger.JsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s",
            rename_fields={"levelname": "level", "asctime": "timestamp"},
        )
    else:
        # Standard formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)

    # Configure structlog if using structured logging
    if structured:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class ContextLogger:
    """Logger with contextual information."""

    def __init__(self, name: str, **context):
        """Initialize context logger.

        Args:
            name: Logger name
            **context: Additional context to include in all log messages
        """
        self.logger = structlog.get_logger(name)
        self.context = context

    def _log(self, level: str, message: str, **kwargs):
        """Log with context.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional context for this log entry
        """
        log_func = getattr(self.logger, level)
        log_func(message, **{**self.context, **kwargs})

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log("error", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log("critical", message, **kwargs)

    def bind(self, **new_context):
        """Add new context and return new logger.

        Args:
            **new_context: New context to add

        Returns:
            New ContextLogger with combined context
        """
        return ContextLogger(
            self.logger.name,
            **{**self.context, **new_context},
        )
