"""LLM 工具调用（function calling）工具包。

本包中的每个工具都通过 function calling（bind_tools）暴露给 LLM。
一个工具应提供：
  - `*_TOOL_SCHEMA`：OpenAI 格式的 function schema，供 LLM 决策是否调用
  - 执行函数：真正执行工具逻辑，由 agent 拿到 LLM 的 tool_call 后调用

当前工具：
  - date_tool: calc_date_range —— 把日期表达（"最近半年"/"2025年三月"）换算成日期范围
"""

from function_tools.date_tool import (  # noqa: F401
    DATE_TOOL_SCHEMA,
    build_date_filter,
    calc_date_range,
    parse_absolute_date,
    parse_date,
    parse_relative_date,
)
