"""Centralised logging configuration for hallucination-reduced-vqa.

Call ``setup_logging()`` once at the entry point of each script or notebook.
All sub-modules then get loggers via ``logging.getLogger(__name__)``.

Example:
    >>> from src.utils.logging import setup_logging
    >>> setup_logging("logs/data_preparation.log")
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """Configure logging for the entire project.

    Sets up a stream handler (stdout) and optionally a file handler.
    Safe to call multiple times — existing handlers are cleared first.

    Args:
        log_file: Optional path to log file.  Parent directories are created
            automatically.  Pass ``None`` to log to console only.
        level: Logging level (default ``logging.INFO``).
        fmt: Log message format string.

    Example:
        >>> setup_logging("logs/training.log", level=logging.DEBUG)
    """
    root_logger = logging.getLogger()
    # Clear any pre-existing handlers to avoid duplicate messages on re-calls
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    formatter = logging.Formatter(fmt)

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # --- File handler (optional) ---
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logging.getLogger(__name__).info(f"Logging to file: {log_path}")

    # Quiet down noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def get_log_level_from_env() -> int:
    """Read LOG_LEVEL environment variable and return corresponding int.

    Returns:
        Logging level integer.  Defaults to ``logging.INFO`` if ``LOG_LEVEL``
        is not set or is unrecognised.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    return level
