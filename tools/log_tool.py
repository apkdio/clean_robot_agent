from datetime import datetime
import logging
import os
import sys

from path_tool import get_abs_path

# ── ANSI 终端颜色 ─────────────────────────────────────────────────────────
_COLORS = {
    "DEBUG":    "\033[90m",   # 灰色
    "INFO":     "\033[0m",    # 默认（白色）
    "WARNING":  "\033[33m",   # 黄色
    "ERROR":    "\033[31m",   # 红色
    "CRITICAL": "\033[35m",   # 洋红色
    "RESET":    "\033[0m",
}


class _ColorFormatter(logging.Formatter):
    """将 levelname 包裹在 ANSI 颜色码中，便于终端阅读。"""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        reset = _COLORS["RESET"]
        # 保存并恢复，避免共享的 record 被其他 handler 修改
        original = record.levelname
        record.levelname = f"{color}{original}{reset}"
        result = super().format(record)
        record.levelname = original
        return result


# ── 文件日志（无颜色，完整详情）───────────────────────────────────────────
log_path = get_abs_path("logs")
if not os.path.exists(log_path):
    os.makedirs(log_path)

file_log_template = logging.Formatter(
    "%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d -  %(message)s"
)

# ── 控制台日志（彩色级别，紧凑）───────────────────────────────────────────
console_log_template = _ColorFormatter(
    "%(asctime)s - %(name)s - %(levelname)s  -  %(message)s"
)


def get_logger(name: str = "agent",
               console_level: int = logging.INFO,
               file_level: int = logging.DEBUG,
               log_file=None):
    """返回一个已配置的 logger，包含彩色控制台输出 + 文件输出。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    # 控制台：输出到 stdout（而非 stderr → 不会强制红色），级别着色
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_log_template)
    logger.addHandler(console_handler)

    # 文件：持久化磁盘日志
    if not log_file:
        log_file = os.path.join(log_path, f"{name}-{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_log_template)
    logger.addHandler(file_handler)

    return logger
