import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_file: str, max_bytes: int = 5_000_000, backup_count: int = 3
) -> None:
    """Configure logging to write both to a rotating file and console.

    Args:
        log_file: Full path to the log file (directory will be created)
        max_bytes: Max size per log file before rotation
        backup_count: Number of rotated files to keep
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates when re-running in Hydra cwd
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=str(log_path), maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
