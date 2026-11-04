import os
import threading

import yaml
from tools.path_tool import get_abs_path

# 配置名 → 相对项目根目录的路径
_CONFIG_PATHS = {
    "rag": "config/rag.yaml",
    "chroma": "config/chroma.yaml",
    "prompts": "config/prompts.yaml",
    "agent": "config/agent.yaml",
    "redis": "config/redis.yaml",
    "context": "config/context.yaml",
}

_CACHE = {}
_LOCK = threading.RLock()
_OPTIONAL = ("redis", "context")


def _config_path(name: str) -> str:
    return get_abs_path(_CONFIG_PATHS.get(name, name))


def get_config(name: str, *, encoding: str = "utf-8", required: bool | None = None,
               default=None):
    """读取并缓存 YAML 配置；同一路径只读一次，热更新时用 reload_configs()。

    redis / context 默认可选（缺失返回 default）；其它配置默认必需。
    缓存对象按只读使用，调用方不要原地修改。
    """
    abs_path = _config_path(name)
    key = (abs_path, encoding)
    if not os.path.isfile(abs_path):
        if required is None:
            required = name not in _OPTIONAL
        if required:
            stem = os.path.splitext(os.path.basename(abs_path))[0]
            hint = (f"配置缺失：{abs_path}。首次使用请复制模板："
                    f"copy config/{stem}_template.yaml {name}")
            raise FileNotFoundError(hint)
        return default

    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

    with open(abs_path, "r", encoding=encoding) as f:
        value = yaml.load(f, Loader=yaml.FullLoader)

    with _LOCK:
        _CACHE[key] = value
    return value


def reload_configs() -> None:
    """清空配置缓存；下一次读取重新落盘。"""
    with _LOCK:
        _CACHE.clear()


def load_config(name: str, encoding: str = "utf-8"):
    """兼容旧调用：读取 YAML 配置（结果由 get_config 统一缓存）。"""
    return get_config(name, encoding=encoding, required=name not in _OPTIONAL)


def get_data_dir() -> str:
    """知识库源目录的绝对路径（rag.yaml 的 data_dir，默认 data/knowledge）。

    热更新、批量入库、数据集构建统一走这里，避免各自硬编码目录。
    """
    return get_abs_path(get_config("rag").get("data_dir") or "data/knowledge")
