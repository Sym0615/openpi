import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


class StreamToLogger(TextIO):
    """Redirect writes to a logger while preserving line boundaries."""

    def __init__(
        self,
        logger: logging.Logger,
        level: int,
        fallback_stream: TextIO | None = None,
    ):
        self.logger = logger
        self.level = level
        self._buffer = ""
        self._fallback_stream = fallback_stream

    def write(self, message: str) -> int:
        if not isinstance(message, str):
            message = str(message)
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)
        return len(message)

    def flush(self) -> None:
        line = self._buffer.rstrip()
        if line:
            self.logger.log(self.level, line)
        self._buffer = ""

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        if self._fallback_stream is None:
            raise OSError("No valid file descriptor available for redirected stream")
        return self._fallback_stream.fileno()


def get_run_log_dir(current_dir: str, run_id: str | None) -> str:
    safe_run_id = run_id if isinstance(run_id, str) and run_id.strip() else "default"
    log_dir = Path(current_dir) / "logs" / safe_run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def make_log_file_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def build_file_logger(
    logger_name: str,
    log_path: str,
    level: int = logging.INFO,
    propagate: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = propagate

    target_path = os.path.abspath(log_path)
    has_handler = False
    for handler in logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == target_path
        ):
            has_handler = True
            break

    if not has_handler:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        handler = logging.FileHandler(target_path)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger


def redirect_std_streams(
    stdout_logger: logging.Logger | None = None,
    stderr_logger: logging.Logger | None = None,
) -> None:
    if stdout_logger is not None:
        sys.stdout = StreamToLogger(stdout_logger, logging.INFO, sys.__stdout__)
    if stderr_logger is not None:
        sys.stderr = StreamToLogger(stderr_logger, logging.ERROR, sys.__stderr__)
