"""型号搜索工具：让模型自主决策要查哪个型号，再按型号名精准检索型号详情。

用于「型号详情咨询 / 上下文对比」兜底（处理指代「这两个/它」、对比「区别」）。
与 budget_tool/symptom_tool 不同，这里用主生成模型（默认 7b）——型号名是 string
参数且需理解上下文指代，3b 对 string 参数会原样返回 query、不做提取。
"""

from __future__ import annotations

from typing import Dict, List

from tools.llm_tool import get_chat_model_name

# 型号提取跟随主生成模型（string 参数 + 上下文指代理解，3b 不稳定）
MODEL_TOOL_MODEL = get_chat_model_name()

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
    """根据型号名（逗号分隔，LLM 提取结果）检索型号详情。

    两段式匹配（见 BC-20260921-06）：① 去品牌前缀后**以查询名结尾**算精确命中，
    任一 key 精确命中时只返回精确结果（「净界 P2」不再带出 P2 Lite / P2 Pro）；
    ② 精确全未命中才退回子串匹配，兼容 LLM 偶发把系列名当型号（「云顶 X」）。
    """
    names = [n.strip() for n in (model_names or "").split(",") if n.strip()]
    if not names:
        return []

    models = _enumerate_model_infos()
    # 两边都去品牌前缀：型号名来自知识库（带「不染一尘」），而 LLM 提取的 key 可能带也可能不带
    def bare(name: str) -> str:
        return name[len(_BRAND_NAME):] if name.startswith(_BRAND_NAME) else name

    keys = [bare(n.replace("系列", "").strip()) for n in names]
    exact = [m for m in models if any(k and bare(m.get("name", "")).endswith(k) for k in keys)]
    if exact:
        return exact

    return [m for m in models if any(k and k in bare(m.get("name", "")) for k in keys)]


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
