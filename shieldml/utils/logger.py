"""Standardized structured logger for ShieldML."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "shieldml", level: Optional[str] = None) -> logging.Logger:
    """Returns a configured logger with clean formatted output.

    Args:
        name: Name of the logger module.
        level: Optional log level override (e.g. 'DEBUG', 'INFO').

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level or "INFO", logging.INFO))
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level or "INFO", logging.INFO))
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
