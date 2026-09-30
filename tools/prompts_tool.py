from tools.log_tool import get_logger
from tools.path_tool import get_abs_path
from tools.config_tool import load_config

prompts_config = load_config("prompts")
logger = get_logger(name="prompts_tool")


def _load_prompt_by_key(key: str) -> str:
    """通用加载器：在 prompts.yaml 中查找路径键并读取文件。"""
    try:
        prompt_path = get_abs_path(prompts_config[key])
    except KeyError as e:
        logger.error(f"[prompts] key '{key}' not found in prompts config.")
        raise e
    try:
        with open(prompt_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[prompts] Failed to read {prompt_path}: {e}")
        raise e


def load_main_prompts() -> str:
    """加载主系统提示词。"""
    return _load_prompt_by_key("main_prompts_path")
