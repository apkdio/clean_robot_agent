"""工具注册：收敛「暴露给模型的工具」白名单（P1-5 编排层用）。

暴露面只 3 个：`find_models`（结构化直出）/ `search_kb`（检索）/ `find_nearest_service_point`（网点）。
其余工具（`extract_models` / `calc_date_range` / `extract_budget` / `classify_symptom`）退成
**内部函数或校验器**：工具越多，选错的概率越高——实测把 `find_models` + `extract_models` +
`calc_date_range` 一起给 7b，问型号时它会选 `calc_date_range`。
"""

from __future__ import annotations

from typing import Dict, List

# 白名单：工具名 → (模块, schema 属性名)。schema 归属原模块，这里只做登记。
_TOOLS: Dict[str, tuple] = {
    "find_models": ("function_tools.model_tool", "FIND_MODELS_TOOL_SCHEMA"),
    "search_kb": ("function_tools.kb_tool", "SEARCH_KB_TOOL_SCHEMA"),
    "find_nearest_service_point": ("function_tools.service_point_tool", "SERVICE_POINT_TOOL_SCHEMA"),
}


def tool_names() -> List[str]:
    """白名单里的工具名（顺序固定，便于日志与离线对照复现）。"""
    return list(_TOOLS)


def build_tool_schemas(names: List[str] | None = None) -> List[Dict]:
    """按白名单返回工具 schema 列表；names 指定子集（离线对照用）。

    未知工具名直接报错，不做静默忽略：写错名字应当立刻暴露，而不是模型少了一个工具。
    """
    import importlib

    picked = list(names) if names else tool_names()
    unknown = [n for n in picked if n not in _TOOLS]
    if unknown:
        raise ValueError("未知工具 %s；白名单：%s" % (unknown, tool_names()))
    return [getattr(importlib.import_module(_TOOLS[n][0]), _TOOLS[n][1]) for n in picked]


def orchestration_mode() -> str:
    """`agent.yaml tools.orchestration`：off（默认，走现有 if 链）/ shadow（选路只记日志）/ on（选路生效）。

    取值请写成**带引号的字符串**：YAML 会把裸 `off` / `on` 解析成布尔（实测裸 `off` 读出来是 False），
    这里做一次归一化，避免开关“看着是 off、读出来是 false”。
    """
    from tools.config_tool import load_config

    try:
        raw = ((load_config("agent") or {}).get("tools") or {}).get("orchestration", "off")
    except Exception:  # noqa: BLE001
        return "off"
    if raw is True:
        return "on"
    if raw is False:
        return "off"
    return str(raw).lower()
