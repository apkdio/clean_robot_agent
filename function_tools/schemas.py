"""主链单次抽取 schema。

主链把 function calling 当作“带类型的参数提取器”：一次只问一个字段，
拿到 args 后由代码校验和转换。这里不登记 agent 可选的完整工具，也不要求
模型在多个工具之间选路。
"""

from __future__ import annotations

EXTRACT_MODELS_SCHEMA = {
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


EXTRACT_DATE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calc_date_range",
        "description": (
            "把用户话里的日期表达换算成具体日期范围。"
            "支持绝对日期（「2025年三月」「2025年3月」「2025年」）和相对日期"
            "（「最近半年」「近三个月」「今年」「去年」）。"
            "只应在用户询问某时间段内发布/上市的产品时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "用户话里的日期表达，如：2025年三月、2025年3月、最近半年、今年、去年",
                }
            },
            "required": ["expression"],
        },
    },
}


EXTRACT_BUDGET_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_budget",
        "description": (
            "提取用户话里的购机预算上限（单位：元）。"
            "单一上限（「一千以内」「3000元以下」）填该数值；"
            "模糊表达（「一千来块」「两千左右」「1500上下」「两千出头」「八百多」）填该数值；"
            "区间（「一千到两千」）填区间上限。"
            "完全没有预算时返回空字符串。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "budget_max": {
                    "type": ["integer", "string"],
                    "description": "预算上限（元）；无预算返回空字符串",
                },
            },
            "required": ["budget_max"],
        },
    },
}


EXTRACT_SYMPTOM_SCHEMA = {
    "type": "function",
    "function": {
        "name": "classify_symptom",
        "description": (
            "把用户描述的扫地机器人故障归类到标准故障类型，返回编号。"
            "编号含义：1=不移动/卡住/不走，2=水箱漏水，3=异响/噪音大，"
            "4=充不进电，5=找不到充电座/回充失败，6=APP连不上/联网失败，"
            "7=吸力下降/吸力变弱，8=建图不完整/地图错误，"
            "9=清扫不干净/拖不干净，10=拖布有异味/发臭，11=拖地后地面有明显水痕/水渍。"
            "没有明确故障填 0。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symptom_id": {
                    "type": "integer",
                    "description": "故障类型编号（0-11），无明确故障填 0",
                },
            },
            "required": ["symptom_id"],
        },
    },
}


EXTRACT_LOCATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_location",
        "description": (
            "从用户话里提取城市或区名（如「上海」「北京」「浦东」），"
            "用于主链网点查询的槽位识别。没有明确地点时返回空字符串。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市或区名（如「上海」「浦东」）；无地点返回空字符串",
                },
            },
            "required": ["location"],
        },
    },
}
