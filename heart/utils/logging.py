"""Logging configuration for H.E.A.R.T.

Provides a centralized logging setup with configurable levels and formatting.
"""

import logging
import sys


_CONFIGURED = False


def configure_logging(level: int = logging.INFO, log_to_file: str | None = None) -> None:
    """Configure the H.E.A.R.T. logging system.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_to_file: Optional file path to also write logs to.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root_logger = logging.getLogger("heart")
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_to_file:
        file_handler = logging.FileHandler(log_to_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific H.E.A.R.T. module.

    Creates a child logger under the 'heart' namespace.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(f"heart.{name}")
