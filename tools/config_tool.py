import yaml
from path_tool import get_abs_path

# 配置名 → 相对项目根目录的路径
_CONFIG_PATHS = {
    "rag": "config/rag.yaml",
    "chroma": "config/chroma.yaml",
    "prompts": "config/prompts.yaml",
    "agent": "config/agent.yaml",
    "redis": "config/redis.yaml",
}


def load_config(name: str, encoding: str = "utf-8"):
    """读取 YAML 配置。

    name 可以是 rag / chroma / prompts / agent 之一，或直接传配置文件路径。
    """
    path = _CONFIG_PATHS.get(name, name)
    with open(get_abs_path(path), "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
