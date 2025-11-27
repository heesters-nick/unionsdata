import logging
import sys
from pathlib import Path

from concurrent_log_handler import ConcurrentRotatingFileHandler
from rich.logging import RichHandler
from tqdm import tqdm


def setup_logger(
    log_dir: Path, name: str, logging_level: int = logging.INFO, *, force: bool = True
) -> None:
    """
    Set up a custom logger for a given script

    Args:
        log_dir: directory where logs should be saved
        name: logger name
        logging_level: logging level (e.g. logging.INFO, logging.DEBUG)
    """
    log_filename = log_dir / f'{name}.log'

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - ID %(process)d - %(levelname)s - %(message)s')
    # console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Filter redundant logging messages to decrease clutter
    log_filter = LoggingFilter()

    # Set up file handler
    file_handler = ConcurrentRotatingFileHandler(
        log_filename,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=1000,  # keep all log files
        encoding='utf-8',
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(log_filter)

    # Set up console handler
    # console_handler = TqdmLoggingHandler()
    console_handler = RichHandler(
        rich_tracebacks=True, show_time=False, show_level=True, show_path=False, markup=True
    )
    # console_handler.setFormatter(console_formatter)
    console_handler.addFilter(log_filter)

    # Configure root logger
    # logging.basicConfig(
    #     level=logging_level,
    #     handlers=[file_handler, console_handler],
    #     force=force,  # Overwrite any existing logging configuration
    # )
    logging.basicConfig(
        level=logging_level,
        format='%(message)s',
        datefmt='[%X]',
        handlers=[file_handler, console_handler],
        force=force,
    )


class TqdmLoggingHandler(logging.Handler):
    """
    A custom logging handler that uses tqdm.write() to print logs.
    This allows log messages to be printed 'above' the active progress bar
    instead of overwriting it or causing visual glitches.
    """

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Use tqdm.write to print safely.
            # We write to stderr to match standard logging behavior.
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


class LoggingFilter(logging.Filter):
    """Filter to suppress redundant VOSpace config messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.msg.startswith('Using config file')
            and 'default-vos-config' in record.msg
            and record.levelno == logging.INFO
        )
