import os

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


def load_config(name: str, encoding: str = "utf-8"):
    """读取 YAML 配置。

    name 可以是 rag / chroma / prompts / agent 之一，或直接传配置文件路径。
    redis / context 缺失属正常（可选依赖），直接返回 None；
    其余配置缺失时给出可操作的提示（复制哪个模板），而不是裸 FileNotFoundError。
    """
    path = _CONFIG_PATHS.get(name, name)
    abs_path = get_abs_path(path)
    if not os.path.isfile(abs_path):
        if name in ("redis", "context"):
            return None
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        hint = (f"配置缺失：{abs_path}。首次使用请复制模板："
                f"copy config/{stem}_template.yaml {path}")
        raise FileNotFoundError(hint)
    with open(abs_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_data_dir() -> str:
    """知识库源目录的绝对路径（rag.yaml 的 data_dir，默认 data/knowledge）。

    热更新、批量入库、数据集构建统一走这里，避免各自硬编码目录。
    """
    return get_abs_path(load_config("rag").get("data_dir") or "data/knowledge")
