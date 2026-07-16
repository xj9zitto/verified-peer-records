"""
logger_setup.py
Shared loguru configuration for all project modules.

Usage (call once at the top of each entry script):
    from src.logger_setup import setup_logger
    setup_logger("scenario_gen")

Log files are written to logs/<name>.log.
Console output shows INFO and above.
File output includes DEBUG and rotates at 10 MB.
"""

from __future__ import annotations
from loguru import logger
from pathlib import Path
import sys


def setup_logger(
    log_name:      str = "run",
    log_dir:       str = "logs",
    console_level: str = "INFO",
    file_level:    str = "DEBUG",
) -> None:
    """
    Configure loguru for console + rotating file output.

    Parameters
    ----------
    log_name      : base filename (e.g. "scenario_gen" → logs/scenario_gen.log)
    log_dir       : directory for log files (created automatically)
    console_level : minimum level printed to stderr  (default INFO)
    file_level    : minimum level written to file    (default DEBUG)
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Remove the default handler so we fully control format
    logger.remove()

    # ── Console ────────────────────────────────────────────────
    logger.add(
        sys.stderr,
        level=console_level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # ── File ───────────────────────────────────────────────────
    log_path = Path(log_dir) / f"{log_name}.log"
    logger.add(
        str(log_path),
        level=file_level,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,        # thread-safe async writes
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level:<8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
    )

    logger.debug(f"Logger initialised — writing to {log_path.resolve()}")