from log_tool import get_logger
from path_tool import get_abs_path
from config_tool import load_prompts_config

prompts_config = load_prompts_config()
logger = get_logger(name="prompts_tool")


def _load_prompt_by_key(key: str) -> str:
    """Generic loader: look up a path key in prompts.yaml and read the file."""
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
    """Load the main system prompt."""
    return _load_prompt_by_key("main_prompts_path")


def load_rag_summarize_prompts() -> str:
    """Load the RAG summarization prompt template."""
    return _load_prompt_by_key("rag_summarize_prompts_path")


def load_report_prompts() -> str:
    """Load the report generation prompt template."""
    return _load_prompt_by_key("report_prompts_path")
