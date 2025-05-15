"""
Centralized logging configuration for the standard data format package.
"""

import logging
from pathlib import Path


def setup_logger(
    chunk_id: int = None, verbose: bool = False, log_dir: Path = Path("logs")
) -> logging.Logger:
    """
    Configure and return a logger with file and console handlers.

    Args:
        chunk_id (int, optional): ID of the current processing chunk
        verbose (bool): Whether to show detailed output in console
        log_dir (Path): Directory for log files

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir.mkdir(exist_ok=True)

    # Configure root logger to ERROR level to suppress other loggers
    logging.getLogger().setLevel(logging.ERROR)

    # Create our specific logger
    logger = logging.getLogger("standard_data_format")

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | Chunk %(chunk)d | %(message)s",
        defaults={"chunk": chunk_id if chunk_id is not None else -1},
    )

    # File handler - always log everything to file
    file_handler = logging.FileHandler(
        (
            log_dir / f"processing_chunk_{chunk_id}.log"
            if chunk_id is not None
            else log_dir / "processing.log"
        ),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Console handler - configure based on verbose flag
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.addHandler(console_handler)

    # Set overall logger level
    logger.setLevel(logging.INFO)

    # Explicitly disable other loggers
    for log_name in ["marker", "PIL", "httpx", "tqdm", "urllib3", "matplotlib"]:
        logging.getLogger(log_name).setLevel(logging.ERROR)

    return logger
