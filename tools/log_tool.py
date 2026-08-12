from datetime import datetime
import logging
import os
import sys

from path_tool import get_abs_path

# ── ANSI terminal colors ──────────────────────────────────────────────────
_COLORS = {
    "DEBUG":    "\033[90m",   # gray
    "INFO":     "\033[0m",    # default (white)
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}


class _ColorFormatter(logging.Formatter):
    """Wrap the levelname in an ANSI color code for terminal readability."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        reset = _COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


# ── File log (no colors, full detail) ─────────────────────────────────────
log_path = get_abs_path("logs")
if not os.path.exists(log_path):
    os.makedirs(log_path)

file_log_template = logging.Formatter(
    "%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d -  %(message)s"
)

# ── Console log (colored level, compact) ─────────────────────────────────
console_log_template = _ColorFormatter(
    "%(asctime)s - %(name)s - %(levelname)s  -  %(message)s"
)


def get_logger(name: str = "agent",
               console_level: int = logging.INFO,
               file_level: int = logging.DEBUG,
               log_file=None):
    """Return a configured logger with colored console + file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    # Console: stdout (not stderr → no forced red), colored level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_log_template)
    logger.addHandler(console_handler)

    # File: persistent disk log
    if not log_file:
        log_file = os.path.join(log_path, f"{name}-{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_log_template)
    logger.addHandler(file_handler)

    return logger
