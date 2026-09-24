"""编排层骨架（P1-5）：只出"决策"，不执行、不改动现有链路。

一轮请求里，现有 if 链照常把答案给用户；本模块只在旁边做一次**并行的决策尝试**——
让模型在工具白名单里选工具、填参数，然后把决策对象交给调用方（影子阶段用于落日志、离线阶段用于对照）。
刻意不做的事：不 import agent、不执行任何工具、不产生任何面向用户的输出。

先有它的理由：影子阶段要"在真实流量上并行做一遍决定"，离线沙盘也要复用同一套决策逻辑，
两处都需要一个**可独立调用、无副作用**的函数。

决策类目（`TASK_CATEGORIES`）：链路侧是离散分支，FC 侧是工具名，两边映射到同一张表上才谈得上
"选路准确率"。SOP 与硬拦截没有对应工具——那几类本来就不该由 FC 决定（状态机留代码）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from log_tool import get_logger

logger = get_logger(name="orchestrator")

# ── 类目表（对照统计的口径，指标定义归 test-agent）─────────────────────
# 为什么分两张表：“要不要检索 / 要不要拒答”**不由 FC 决定**——那是砸词表硬规则（危险/注入）
# 与意图分类（含低置信承接）的活（A 类主路）。FC 只在**已经确定要走检索或结构化筛选**的轮次里参与，
# 贡献的是“用哪个工具 + 填什么参数”，其中 `search_kb` 的实际价值是 `query_summarization`（检索式）
# 与 `domain`（域）——这两件事意图分类从来没做过，今天靠余弦选轮拼接 + 词表域路由。
# 所以 `fc_eligible=False` 的类目压根不调 `decide()`，也就谈不上“拿 FC 再查一遍意图”。
CHAIN_CATEGORIES: Dict[str, Dict] = {
    "kb_search": {"label": "知识库检索问答", "fc_eligible": True,
                  "fc_value": "检索式（query_summarization）+ 域（domain）"},
    "model_query": {"label": "型号筛选/枚举/详情", "fc_eligible": True,
                    "fc_value": "参数筛选 / 排序 / 枚举直出"},
    "service_point": {"label": "售后网点查询", "fc_eligible": True,
                      "fc_value": "城市名提取"},
    "sop_flow": {"label": "故障/流程类（状态机）", "fc_eligible": False,
                 "fc_value": ""},
    "hard_block": {"label": "硬拦截（危险现象 / 注入）", "fc_eligible": False,
                   "fc_value": ""},
    "chitchat": {"label": "情绪 / 闲聊", "fc_eligible": False, "fc_value": ""},
    "refuse": {"label": "低置信拒答 / 无上文可承接", "fc_eligible": False, "fc_value": ""},
}

# FC 侧：工具 → 类目。未列出的工具（或模型弃权）归 ABSTAIN——表示本轮不采用工具、沿用原链路。
TOOL_CATEGORIES: Dict[str, str] = {
    "search_kb": "kb_search",
    "find_models": "model_query",
    "find_nearest_service_point": "service_point",
}
ABSTAIN = "no_tool"


def category_of_tool(tool: Optional[str]) -> str:
    """工具名 → 类目；None / 白名单外 → `ABSTAIN`（弃权，本轮不采用工具）。"""
    return TOOL_CATEGORIES.get(tool or "", ABSTAIN)


def chain_fc_eligible(category: str) -> bool:
    """该类链路的轮次是否该调 `decide()`。状态机 / 硬拦截 / 情绪 / 拒答一律不调。"""
    return bool((CHAIN_CATEGORIES.get(category) or {}).get("fc_eligible", False))


def fc_value_of(category: str) -> str:
    """该链路类目下，FC 能贡献什么（写进日志便于回看“它到底帮上了什么”）。"""
    return str((CHAIN_CATEGORIES.get(category) or {}).get("fc_value", ""))


@dataclass
class Decision:
    """一轮的编排决策：只说"会怎么做"，不含执行结果。"""

    tool: Optional[str] = None
    args: Dict = field(default_factory=dict)
    extra_calls: List[Dict] = field(default_factory=list)  # 同一轮命中的其他工具（多意图）
    category: str = ABSTAIN
    notes: List[str] = field(default_factory=list)  # 校验提醒（参数被纠正 / 域冲突等）
    ms: float = 0.0
    error: str = ""

    @property
    def decided(self) -> bool:
        """是否真的决定要调工具（False = 模型选择不动，或本次决策失败）。"""
        return bool(self.tool)

    def to_log(self) -> Dict:
        return {
            "tool": self.tool,
            "category": self.category,
            "args": self.args,
            "extra": [c.get("name") for c in self.extra_calls],
            "notes": self.notes,
            "ms": round(self.ms, 1),
            "error": self.error,
        }


_SYSTEM = (
    "你是扫地机器人客服系统的调度器。今天日期：{today}。\n"
    "你的任务只有一个：判断用户这一句该用哪个工具，并把参数填好。\n"
    "规则：\n"
    "1. 只在给定工具里选；拿不准该不该用工具时**不要调用任何工具**。\n"
    "2. `query` 参数填用户的**原话**，不要加任何前缀、标签或改写。\n"
    "3. 金额、时间这类把原话搬过去（如「5000元以上」「最近半年」），不要自己换算。\n"
    "4. 安全告警（冒烟、焦味、漏电）、情绪抱怨、闲聊、与扫地机器人无关的请求，都不要调工具。\n"
    "5. 用户同时提了多件事时可以调多个工具，按依赖顺序给出。\n"
    "6. 不要输出任何解释性文字。{extra}"
)


def decide(query: str, effective_query: str = "", history_block: str = "",
           domain_hint: Optional[str] = None, model: str = "") -> Decision:
    """让模型做一次决策（**不执行**任何工具）。

    `effective_query` 是 agent 侧改写后的自足问句（有就给模型参考）；
    `domain_hint` 是词表域路由给出的域键（无则 None）——影子/离线阶段用来和模型选的 `domain` 比对，
    不一致记进 `notes`，不在这里裁决（裁决口径见 §11 词表分工）。

    消息形态刻意做成**自然对话**：历史放 system 块，当前这一句就是 human 消息本身
    （不做「用户当前这一句：…」这种标签——实测模型会把标签一起拄进 `query` 参数）。
    """
    from function_tools.model_tool import validate_find_models_args
    from function_tools.registry import build_tool_schemas, tool_names
    from llm_tool import chat_with_tools, get_chat_model_name

    extra = ""
    if history_block:
        extra += "\n\n最近对话（仅供理解指代，**不要把它的内容或标签填进参数**）：\n%s" % history_block
    if effective_query and effective_query != query:
        extra += "\n\n（结合上文补充后的问句：%s；仅供参考，`query` 参数仍填用户原话）" % effective_query

    from langchain_core.messages import HumanMessage, SystemMessage

    tools = build_tool_schemas()
    allowed = set(tool_names())
    started = time.perf_counter()
    try:
        resp = chat_with_tools(
            [SystemMessage(content=_SYSTEM.format(today=date.today().isoformat(), extra=extra)),
             HumanMessage(content=query)],
            tools, model=model or get_chat_model_name(), temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[Orchestrator] decide failed: %s", e)
        return Decision(error=str(e)[:120], ms=(time.perf_counter() - started) * 1000)
    elapsed_ms = (time.perf_counter() - started) * 1000

    calls = getattr(resp, "tool_calls", None) or []
    if not calls:
        return Decision(category=ABSTAIN, ms=elapsed_ms)

    decision = Decision(ms=elapsed_ms)
    for i, call in enumerate(calls):
        name = call.get("name") or (call.get("function") or {}).get("name") or ""
        args = call.get("args") or {}
        if name not in allowed:
            decision.notes.append("工具名不在白名单：%s" % name)
            continue
        if i == 0:
            decision.tool, decision.args = name, args
        else:
            decision.extra_calls.append({"name": name, "args": args})
    decision.category = category_of_tool(decision.tool)

    # 参数校验：不做纠正之外的裁决，把提醒一并带出去（影子日志要能看到"模型填得对不对"）
    if decision.tool == "find_models":
        fixed, notes = validate_find_models_args(query, decision.args)
        decision.args, decision.notes = fixed, decision.notes + notes
    elif decision.tool == "search_kb":
        from function_tools.kb_tool import retrieval_query_of
        chosen = str(decision.args.get("domain") or "").strip()
        if not retrieval_query_of(decision.args):
            decision.notes.append("search_kb 没给检索式")
        if chosen and domain_hint and chosen != domain_hint:
            decision.notes.append("域与词表信号不一致：模型=%s 词表=%s" % (chosen, domain_hint))
    return decision


def shadow_line(decision: Decision, origin: str, chain_category: str,
                chain_detail: str = "") -> str:
    """影子/离线对照的一行日志：链路怎么判、FC 会怎么判、是否一致。

    格式固定，便于事后 grep 统计（`[Shadow]` 前缀）。
    """
    fc = decision.tool or "-"
    if decision.args:
        fc += " " + str(decision.args)
    agree = decision.category == chain_category
    return "[Shadow] origin=%s | chain=%s(%s) | fc=%s | agree=%s | fc_ms=%.0f%s%s" % (
        origin[:40], chain_category, chain_detail or "-", fc, agree, decision.ms,
        "".join(" | note=%s" % n for n in decision.notes),
        " | error=%s" % decision.error if decision.error else "",
    )
