"""LLM function calling 的型号搜索工具。

让模型自主决策要查询哪个型号（处理指代「这两个/它」、对比「区别」），
再按型号名精准检索型号详情，作为"型号详情咨询 / 上下文对比"的兜底。

与 budget_tool/symptom_tool 不同：这里用 7b——型号名是 string 参数，
且需要理解上下文指代（"这两个"指谁），3b 的 function calling 对 string
参数会原样返回 query、不做提取，不稳定。

对外提供：
  - MODEL_TOOL_SCHEMA     : 供 LLM 使用的 function-calling schema
  - MODEL_TOOL_MODEL      : 提取型号用的小模型名
  - get_all_model_names   : 全量型号名列表（缓存，供型号上下文判断）
  - search_models_by_names: 型号名列表（逗号分隔）→ 检索型号详情
"""

from __future__ import annotations

from typing import Dict, List

# 型号提取用 7b（string 参数 + 上下文指代理解，3b 不稳定）
MODEL_TOOL_MODEL = "qwen2.5:7b"

# 品牌名（型号名前缀，型号上下文判断时去掉，得到「云顶 X2」这类用户口中的型号）
_BRAND_NAME = "不染一尘"


MODEL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_models",
        "description": (
            "从用户的话和对话上下文里，提取用户想查询的扫地机器人具体型号名"
            "（如「云顶 X2」「净白 S1」）。用户可能用指代词（「这两个」「它」）指代上文提到的具体型号，"
            "或直接说型号名。多个型号用逗号分隔。不要提取系列名（如「云顶 X 系列」）。"
            "没有明确型号时返回空字符串。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_names": {
                    "type": "string",
                    "description": "要查询的型号名，多个用逗号分隔（如「云顶 X2」或「云顶 X2,净白 S1」）；无明确型号返回空字符串",
                },
            },
            "required": ["model_names"],
        },
    },
}


def _enumerate_model_infos() -> List[Dict]:
    """枚举全量型号信息（去重），复用 metadata_extractor 的条目校验。"""
    from tools.metadata_extractor import enumerate_models
    return enumerate_models({"file_name": {"$ne": "__never__"}}, require_field="price")


_model_names_cache: List[str] | None = None


def get_all_model_names() -> List[str]:
    """全量型号名列表（模块级缓存），供型号上下文判断用。

    用 require_field="price" 过滤，只保留真正的型号条目（排除品牌介绍 FAQ
    里加粗但无价格的标题，如「不染一尘和云境智能是什么关系？」）。
    """
    global _model_names_cache
    if _model_names_cache is None:
        _model_names_cache = [m["name"] for m in _enumerate_model_infos() if m.get("name")]
    return _model_names_cache


def search_models_by_names(model_names: str) -> List[Dict]:
    """根据型号名（逗号分隔）检索型号详情。

    对全量型号做子串匹配：查询名（去掉「系列」后缀）含于型号名即命中，
    兼容 LLM 偶发提取成系列名（「云顶 X 系列」→ 命中云顶 X1/X2 等）。
    """
    names = [n.strip() for n in (model_names or "").split(",") if n.strip()]
    if not names:
        return []

    models = _enumerate_model_infos()
    keys = [n.replace("系列", "").strip() for n in names]
    matched = []
    for m in models:
        mn = m.get("name", "")
        if any(k and k in mn for k in keys):
            matched.append(m)
    return matched


def model_name_in_query(query: str) -> bool:
    """判断 query 里是否直接出现具体型号名（去掉品牌前缀后子串匹配）。

    如型号「不染一尘云顶 X2」→ 去掉品牌前缀得「云顶 X2」，query 含「云顶 X2」
    即命中。用于「云顶 X2 怎么样」这类直接报型号名的详情咨询。
    """
    for name in get_all_model_names():
        short = name.replace(_BRAND_NAME, "").strip()
        if short and short in query:
            return True
    return False
