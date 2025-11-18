"""Logging configuration for agentic services."""

import sys
from typing import Any, Dict

from loguru import logger

from .config import get_settings


def setup_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Configure format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Add stdout handler
    logger.add(
        sys.stdout,
        format=log_format,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Add file handler for errors
    logger.add(
        "logs/error.log",
        format=log_format,
        level="ERROR",
        rotation="10 MB",
        retention="10 days",
        compression="zip",
    )

    # Add JSON file handler for structured logging
    logger.add(
        "logs/app.log",
        format="{time} {level} {message}",
        level=settings.log_level,
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        serialize=True,
    )

    logger.info(f"Logging initialized with level: {settings.log_level}")


def get_logger(name: str) -> Any:
    """Get a logger instance with the given name."""
    return logger.bind(name=name)
