"""故障现象分类工具（repair SOP 关键词规则未命中时的 LLM 兜底）。

`_SYMPTOM_MAP` 的关键词匹配会漏掉"不动弹""拖地留水印"这类口语描述，
这里用 3b 把故障描述归类到标准类型，再映射回标准检索 query。

与 budget_tool 同理：qwen2.5:3b 只能稳定处理「单一 integer 参数」，故用编号
symptom_id（0-10）而非多字段，编号含义写进 description。
"""

from __future__ import annotations

from typing import Dict, Optional


from config.word_dict_config import SYMPTOM_QUERY_MAP  # 故障类型编号 → 标准检索 query


SYMPTOM_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "classify_symptom",
        "description": (
            "把用户描述的扫地机器人故障归类到标准故障类型，返回编号。"
            "编号含义：1=不移动/卡住/不走，2=水箱漏水，3=异响/噪音大，"
            "4=充不进电，5=找不到充电座/回充失败，6=APP连不上/联网失败，"
            "7=吸力下降/吸力变弱，8=建图不完整/地图错误，"
            "9=清扫不干净/拖不干净，10=拖布有异味/发臭。"
            "没有明确故障填 0。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symptom_id": {
                    "type": "integer",
                    "description": "故障类型编号（0-10），无明确故障填 0",
                },
            },
            "required": ["symptom_id"],
        },
    },
}


def symptom_id_to_query(args: Dict) -> Optional[str]:
    """把 LLM 返回的 {symptom_id} 映射成标准检索 query。

    编号 0 / 空 / 非数字 / 越界 → None（视为未识别，SOP 走 retry 重新问）。
    """
    if not args:
        return None

    sid = args.get("symptom_id")
    if sid is None:
        return None

    try:
        sid = int(sid)
    except (TypeError, ValueError):
        return None

    return SYMPTOM_QUERY_MAP.get(sid)
