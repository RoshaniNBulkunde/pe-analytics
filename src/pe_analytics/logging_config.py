"""
logging_config.py
-----------------
Central logging configuration for the pe-analytics package.
Import setup_logger() in every other module.
"""

import logging  ##This module defines functions and classes which implement a flexible event logging system for applications and libraries.
import os
from datetime import datetime


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a logger with console and file handlers.

    Args:
        name: Name of the logger, typically __name__ from calling module.
        level: Logging level. Defaults to INFO.

    Returns:
        Configured logger instance.
    """
    # Create logs directory if it does not exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Log filename includes date so each day gets its own file
    log_filename = os.path.join(
        log_dir,
        f"pe_analytics_{datetime.now().strftime('%Y%m%d')}.log"
    )

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Format: timestamp | level | module | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — prints to Spyder console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File handler — writes to logs/ folder
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
