import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch

from src.common.utils import DEVICE


class BriefConsoleFilter(logging.Filter):
    """Drop long INFO/DEBUG messages on the console while keeping WARN+ untouched."""

    def __init__(self, max_len: int = 720) -> None:
        super().__init__()
        self.max_len = max_len

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        return len(msg) <= self.max_len


def short_path_for_log(path: Path | str) -> str:
    """Best-effort short path for logs (relative to cwd when under it)."""
    p = Path(path).expanduser().resolve()
    try:
        return str(p.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


def artifact_rel_to_experiment(
    file_path: Path | str, experiment_root: Path | str
) -> str:
    """Path relative to experiment root for compact one-line logs (e.g. ``visualizations/x.png``)."""
    p = Path(file_path).expanduser().resolve()
    root = Path(experiment_root).expanduser().resolve()
    try:
        return str(p.relative_to(root))
    except ValueError:
        return short_path_for_log(p)


def setup_logging(log_name: str) -> logging.Logger:
    """Configure a module-scoped logger (console + file under ./logs/).

    Prefer :func:`configure_run_logging` for Hydra training runs — pass Hydra ``runtime.output_dir``, not cwd.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / (Path(log_name).stem + ".log")

    named = logging.getLogger(Path(log_name).stem)
    named.setLevel(logging.INFO)
    named.propagate = False

    if not named.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
        named.addHandler(console_handler)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_fmt)
        named.addHandler(file_handler)
    return named


def _quiet_third_party_loggers() -> None:
    """Reduce hub/HTTP/font noise so run logs stay readable."""
    for name in (
        "timm",
        "urllib3",
        "PIL",
        "huggingface_hub",
        "requests",
        "httpcore",
        "matplotlib",
        "matplotlib.font_manager",
        "matplotlib.pyplot",
        "git",
        "git.cmd",
        "git.util",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    
    for name in ("mlflow", "mlflow.pytorch", "mlflow.utils.requirements_utils",
             "mlflow.utils.uv_utils", "mlflow.utils.environment"):
        logging.getLogger(name).setLevel(logging.ERROR)


def configure_run_logging(
    *,
    run_directory: Path | str | None = None,
    logs_subdir: str = "logs",
    full_log_name: str = "main.log",
    console_max_msg_len: int = 720,
    use_rich_console: bool = False,
    quiet_third_party_loggers: bool = False,
    rich_show_path: bool = False,
) -> Path:
    """Configure root logging with a rotating file plus console.

    Pass the same **experiment root directory** as ``*_Trainer._get_root_dir_path()`` —
    e.g. :func:`classification_experiment_root` / :func:`segmentation_experiment_root`
    (``.resolve()``). Do **not** use raw ``os.getcwd()`` — that often points at the repo
    root while Hydra writes under ``output/...``.

    **Hydra-only** entrypoints (e.g. eval whose ``hydra.run.dir`` differs from training)
    may pass ``Path(HydraConfig.get().runtime.output_dir).resolve()``.

    Writes to ``<run_directory>/<logs_subdir>/<full_log_name>``.

    Without ``run_directory``, falls back to the current working directory (non-Hydra scripts).

    Returns:
        Absolute path of the rotating full log file.
    """

    base = (
        Path(run_directory).expanduser().resolve()
        if run_directory is not None
        else Path.cwd().resolve()
    )
    log_root = base / logs_subdir
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / full_log_name

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(
        str(log_path),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if quiet_third_party_loggers:
        _quiet_third_party_loggers()

    if use_rich_console:
        from rich.console import Console
        from rich.logging import RichHandler

        console_handler = RichHandler(
            console=Console(stderr=True, soft_wrap=True),
            show_time=True,
            show_level=True,
            show_path=rich_show_path,
            markup=False,
            rich_tracebacks=True,
            tracebacks_word_wrap=False,
            log_time_format="[%Y-%m-%d %X]",
        )
        console_handler.setLevel(logging.INFO)
        console_handler.addFilter(BriefConsoleFilter(max_len=console_max_msg_len))
        root.addHandler(console_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.addFilter(BriefConsoleFilter(max_len=console_max_msg_len))
        console_handler.setFormatter(fmt)
        root.addHandler(console_handler)
    return log_path.resolve()


def log_gpu_memory(logger: logging.Logger) -> None:
    """Log GPU memory usage if available."""
    if torch.cuda.is_available():
        max_mem_allocated = torch.cuda.max_memory_allocated() / (1024**2)
        logger.info(f"  Max GPU Memory Allocated: {max_mem_allocated:.2f} MB")
        torch.cuda.reset_peak_memory_stats()


def gpu_epoch_log_suffix() -> str:
    """Compact suffix for epoch summary lines (peak alloc since last reset)."""
    if torch.cuda.is_available():
        mb = torch.cuda.max_memory_allocated() / (1024**2)
        torch.cuda.reset_peak_memory_stats()
        return f"  |  GPU max alloc {mb:.0f} MB"
    return ""

def run_train_info_log(logger_name: str, run_log_path: Path, cfg: str) -> None:
    logger = logging.getLogger(logger_name)
    logger.info("--------------------------------")

    logger.info("Run information:")
    logger.info("Full run log (all levels): %s", run_log_path)
    logger.info("Resolved configuration:\n%s", cfg)
    logger.info("Device: %s", DEVICE)
