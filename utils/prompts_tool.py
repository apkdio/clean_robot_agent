"""Prompt 文件加载与渲染：缓存内容，渲染前校验占位符。

prompts/*.md 存完整可发送模板，动态内容用 $var 占位；调用方只传上下文，
由 render 一次性填充，不再自行拼接字符串。
生产模式缺占位参数时返回 None 并告警，由调用方走确定性兜底；
strict 模式用于测试，缺参直接抛 ValueError。
"""
from __future__ import annotations

import string
import threading

from utils.config_tool import get_config
from utils.log_tool import get_logger
from utils.path_tool import get_abs_path

logger = get_logger(name="prompts_tool")

_LOCK = threading.RLock()
_CACHE: dict[str, tuple[str, str]] = {}
_STRICT = False


def set_strict(enabled: bool = True) -> None:
    """strict 模式：占位参数不匹配时直接抛 ValueError，供测试使用。"""
    global _STRICT
    _STRICT = bool(enabled)


def is_strict() -> bool:
    return _STRICT


def reset() -> None:
    """清空文件内容缓存；配置缓存由 config_tool.reload_configs() 负责。"""
    with _LOCK:
        _CACHE.clear()


def _prompts_config() -> dict:
    return get_config("prompts") or {}


def prompt_paths() -> list[str]:
    """prompts.yaml 中 prompt_files 声明的所有文件（相对项目根目录）。"""
    files = _prompts_config().get("prompt_files") or {}
    return sorted({str(p) for p in files.values() if p})


def _path_of(key: str) -> str:
    rel = (_prompts_config().get("prompt_files") or {}).get(key)
    if not rel:
        raise KeyError("prompts.yaml 未配置 prompt：%s" % key)
    return get_abs_path(rel)


def load(key: str) -> str:
    """读取并缓存 prompt 文件；文件路径来自 prompts.yaml。"""
    path = _path_of(key)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and cached[0] == path:
            return cached[1]
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        logger.error("[Prompts] 读取失败 %s（%s）：%s", key, path, e)
        raise
    with _LOCK:
        _CACHE[key] = (path, text)
    return text


def _placeholders(text: str) -> set[str]:
    """提取 string.Template 的 $var / ${var} 占位名，忽略 $$ 转义。"""
    found = set()
    for match in string.Template.pattern.finditer(text):
        name = match.group("named") or match.group("braced")
        if name:
            found.add(name)
    return found


def _render_text(prompt_name: str, text: str, ctx: dict,
                 warn_extra: bool = True) -> str | None:
    declared = _placeholders(text)
    provided = set(ctx)
    missing = sorted(declared - provided)
    extra = sorted(provided - declared)

    # SOP 的 ctx 天然带全部槽位，内联模板只取其中一部分；文件模板才按精确 key 告警。
    if extra and warn_extra:
        logger.warning("[Prompts] %s 提供了未声明的占位参数：%s",
                       prompt_name, ",".join(extra))
    if missing:
        message = "[Prompts] %s 缺少占位参数：%s" % (prompt_name, ",".join(missing))
        if _STRICT:
            raise ValueError(message)
        logger.warning(message)
        return None
    try:
        return string.Template(text).substitute(ctx)
    except (KeyError, ValueError) as e:
        message = "[Prompts] %s 渲染失败：%s" % (prompt_name, e)
        if _STRICT:
            raise ValueError(message) from e
        logger.warning(message)
        return None


def render(prompt_name: str, *, template: str | None = None, **ctx) -> str | None:
    """渲染完整 prompt 文件，或调用方传入的内联 SOP 模板，统一使用 $var。"""
    if template is None:
        return _render_text(prompt_name, load(prompt_name), ctx)
    return _render_text(prompt_name, template, ctx, warn_extra=False)


def load_main_prompts() -> str:
    """主系统提示词；走统一缓存。"""
    return load("main")
