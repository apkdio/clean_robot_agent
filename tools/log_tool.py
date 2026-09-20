from datetime import datetime
import logging
import logging.handlers
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

def _resolve_file_level() -> int:
    """文件日志级别，优先级：环境变量 LOG_LEVEL > agent.yaml behavior.verbose_log > INFO。

    默认 INFO 是为了避免检索结果的 DEBUG 详情（top-N chunk 内容）撑大日志文件；
    需要排查检索问题时，把 `agent.yaml` 的 `behavior.verbose_log` 设为 true 即可，
    也可临时用环境变量 `LOG_LEVEL=DEBUG`（优先级更高、无需改配置）。

    只影响**文件**日志；控制台始终按 INFO 输出，避免调试细节刷屏。
    """
    env_level = os.environ.get("LOG_LEVEL", "").strip().upper()
    if env_level:
        return getattr(logging, env_level, logging.INFO)
    try:
        from config_tool import load_config
        verbose = bool(load_config("agent").get("behavior", {}).get("verbose_log", False))
    except Exception:
        verbose = False
    return logging.DEBUG if verbose else logging.INFO


_DEFAULT_FILE_LEVEL = _resolve_file_level()

# 单个日志文件上限（字节）与滚动备份数：防止同一天内文件无限增长
_LOG_MAX_BYTES = 5 * 1024 * 1024   # 5MB
_LOG_BACKUP_COUNT = 2              # 保留 2 份滚动备份

file_log_template = logging.Formatter(
    "%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d -  %(message)s"
)

# ── 控制台日志（彩色级别，紧凑）───────────────────────────────────────────
console_log_template = _ColorFormatter(
    "%(asctime)s - %(name)s - %(levelname)s  -  %(message)s"
)


def get_logger(name: str = "agent",
               console_level: int = logging.INFO,
               file_level: int | None = None,
               log_file=None):
    """返回一个已配置的 logger，包含彩色控制台输出 + 文件输出。

    file_level 默认按「环境变量 LOG_LEVEL → agent.yaml behavior.verbose_log → INFO」
    依次解析；传显式值时覆盖。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    # 控制台：输出到 stdout（而非 stderr → 不会强制红色），级别着色
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_log_template)
    logger.addHandler(console_handler)

    # 文件：按大小滚动（超过上限切到 .1/.2），避免单文件无限增长
    if file_level is None:
        file_level = _DEFAULT_FILE_LEVEL
    if not log_file:
        # 按 /模块/日期/ 分目录：logs/<name>/<YYYY-MM-DD>/<name>.log
        date_dir = os.path.join(log_path, name, datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(date_dir, exist_ok=True)
        log_file = os.path.join(date_dir, f"{name}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, mode="a", encoding="utf-8",
        maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT,
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_log_template)
    logger.addHandler(file_handler)

    return logger
