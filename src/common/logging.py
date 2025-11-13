import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch


def setup_logging(log_name: str) -> logging.Logger:
    """Configure logging

    Args:
        log_name: Name of the file the logs will come from
    """
    # Determine log directory and file name
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / (Path(log_name).stem + ".log")

    logger = logging.getLogger(Path(log_name).stem)
    logger.setLevel(logging.INFO)
    logger.propagate = (
        False  # Prevent double logging if root logger is configured elsewhere
    )

    # Only add handlers if not already present (avoid duplicate logs)
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

        # Rotating file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger


def log_gpu_memory(logger: logging.Logger) -> None:
    """Log GPU memory usage if available."""
    if torch.cuda.is_available():
        max_mem_allocated = torch.cuda.max_memory_allocated() / (1024**2)
        logger.info(f"  Max GPU Memory Allocated: {max_mem_allocated:.2f} MB")
        torch.cuda.reset_peak_memory_stats()
