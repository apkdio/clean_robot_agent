"""LLM function calling 的日期计算工具。

把日期表达换算成具体的 ISO 日期范围。同时支持：
  - 绝对日期（"2025年三月"、"2025年3月"、"2025年"、"三月"）
  - 相对日期（"最近半年"、"近三个月"、"今年"、"去年"）

对外提供：
  - DATE_TOOL_SCHEMA : 供 LLM 使用的 OpenAI function-calling schema
  - calc_date_range  : 工具执行器（LLM 发出 tool_call 后调用）
  - parse_date       : 确定性规则解析器（先绝对后相对）

日期字符串用 ISO 格式 "YYYY-MM-DD"；Chroma metadata 过滤会把它转成
int（YYYYMMDD），以便 $gte/$lte 做数值比较。
"""

from __future__ import annotations

import re
from datetime import date, timedelta

_DATE_TODAY = None  # 测试时可注入


def _today() -> date:
    return _DATE_TODAY or date.today()


def _cn_to_int(s: str) -> int | None:
    """把简单中文数字（'三'、'十二'）转成数字。"""
    num_map = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + num_map.get(s[1:], 0) if len(s) > 1 else 10
    if "十" in s:
        a, b = s.split("十", 1)
        return num_map.get(a, 0) * 10 + (num_map.get(b, 0) if b else 0)
    return num_map.get(s)


_MONTH_CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
             "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}


def _cn_month_to_int(s: str) -> int | None:
    """把月份 token（'3'、'三月'、'十一'）转成 1..12 的 int。"""
    s = s.strip()
    if s.isdigit():
        n = int(s)
        return n if 1 <= n <= 12 else None
    return _MONTH_CN.get(s)


def _days_in_month(y: int, m: int) -> int:
    if m == 12:
        nxt = date(y + 1, 1, 1)
    else:
        nxt = date(y, m + 1, 1)
    return (nxt - date(y, m, 1)).days


def _subtract_months(d: date, months: int) -> date:
    y = d.year
    m = d.month - months
    while m <= 0:
        m += 12
        y -= 1
    day = min(d.day, _days_in_month(y, m))
    return date(y, m, day)


def parse_absolute_date(expression: str) -> tuple[str, str] | None:
    """把绝对日期解析成 (start_date, end_date) ISO 字符串。

    支持 "2025年三月"、"2025年3月"、"2025年"、"2025-03"、"三月"。
    未识别到绝对日期时返回 None。
    """
    expr = expression.strip()

    # YYYY年M月 → 该月
    m = re.search(r"(\d{4})\s*年\s*([0-9一二三四五六七八九十]+)\s*月", expr)
    if m:
        y = int(m.group(1))
        mo = _cn_month_to_int(m.group(2))
        if mo:
            return (f"{y}-{mo:02d}-01", f"{y}-{mo:02d}-{_days_in_month(y, mo):02d}")

    # YYYY年 → 该年
    m = re.search(r"(\d{4})\s*年", expr)
    if m:
        y = int(m.group(1))
        return (f"{y}-01-01", f"{y}-12-31")

    # YYYY-MM / YYYY/M（但不是 YYYY-MM-DD）
    m = re.search(r"(\d{4})\s*[-/]\s*(\d{1,2})(?!\s*[-/]\d)", expr)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2))
        if 1 <= mo <= 12:
            return (f"{y}-{mo:02d}-01", f"{y}-{mo:02d}-{_days_in_month(y, mo):02d}")

    # M月（无年份）→ 当年的该月
    m = re.search(r"(?<!\d)([一二三四五六七八九十0-9]+)\s*月", expr)
    if m:
        mo = _cn_month_to_int(m.group(1))
        if mo:
            y = _today().year
            return (f"{y}-{mo:02d}-01", f"{y}-{mo:02d}-{_days_in_month(y, mo):02d}")

    return None


def parse_relative_date(expression: str) -> tuple[str, str] | None:
    """把相对日期表达解析成 (start_date, end_date) ISO 字符串。

    表达不是可识别的相对日期时返回 None。
    """
    expr = expression.strip()
    today = _today()
    t = today.isoformat()

    # 今年 / 去年 / 前年
    if "今年" in expr:
        return (f"{today.year}-01-01", t)
    if "去年" in expr:
        return (f"{today.year - 1}-01-01", f"{today.year - 1}-12-31")
    if "前年" in expr:
        return (f"{today.year - 2}-01-01", f"{today.year - 2}-12-31")

    # 最近/近 X 年
    m = re.search(r"(?:最近|近)([0-9一二两三四五六七八九十]+)\s*年", expr)
    if m:
        n = _cn_to_int(m.group(1))
        if n:
            return (_subtract_months(today, n * 12).isoformat(), t)

    # X 年内（"一年内"、"两年内"）
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*年\s*(?:内|以内|之内|以来)", expr)
    if m:
        n = _cn_to_int(m.group(1))
        if n:
            return (_subtract_months(today, n * 12).isoformat(), t)

    # 半年
    if "半年" in expr:
        return (_subtract_months(today, 6).isoformat(), t)

    # X 个月内（"6个月内"、"三个月以内"）
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*个?月\s*(?:内|以内|之内|以来)", expr)
    if m:
        n = _cn_to_int(m.group(1))
        if n:
            return (_subtract_months(today, n).isoformat(), t)

    # 最近/近 X 个月（含"个"月）
    m = re.search(r"(?:最近|近|这)([0-9一二两三四五六七八九十]+)\s*个?月", expr)
    if m:
        n = _cn_to_int(m.group(1))
        if n:
            return (_subtract_months(today, n).isoformat(), t)

    # X 周内（"两周内"）
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*个?周\s*(?:内|以内|之内|以来)", expr)
    if m:
        n = _cn_to_int(m.group(1))
        if n:
            return ((today - timedelta(weeks=n)).isoformat(), t)

    # 最近/近 X 周
    m = re.search(r"(?:最近|近|这)([0-9一二两三四五六七八九十]+)\s*个?周", expr)
    if m:
        n = _cn_to_int(m.group(1))
        if n:
            return ((today - timedelta(weeks=n)).isoformat(), t)

    # X 天内（"30天内"）
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*天\s*(?:内|以内|之内|以来)", expr)
    if m:
        n = _cn_to_int(m.group(1))
        if n:
            return ((today - timedelta(days=n)).isoformat(), t)

    # 最近/近 X 天
    m = re.search(r"(?:最近|近|这)([0-9一二两三四五六七八九十]+)\s*天", expr)
    if m:
        n = _cn_to_int(m.group(1))
        if n:
            return ((today - timedelta(days=n)).isoformat(), t)

    return None


def parse_date(expression: str) -> tuple[str, str] | None:
    """把绝对或相对日期表达解析成日期范围。"""
    return parse_absolute_date(expression) or parse_relative_date(expression)


def calc_date_range(expression: str) -> dict:
    """工具执行器：把绝对/相对日期表达换算成日期范围。

    成功返回 {"start_date", "end_date", "expression"}，
    无法解析时返回 {"error": ...}。
    """
    r = parse_date(expression)
    if r is None:
        return {"error": f"无法识别的日期表达：{expression}"}
    return {"start_date": r[0], "end_date": r[1], "expression": expression}


DATE_TOOL_SCHEMA = {
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


def _iso_to_int(iso: str) -> int:
    """把 ISO 日期字符串 'YYYY-MM-DD' 转成 int YYYYMMDD。

    Chroma 的 $gte/$lte 只接受 int/float 操作数，所以 publish_date
    存成整数（如 20260310）以便数值排序。
    """
    return int(iso.replace("-", ""))


def build_date_filter(start_date: str, end_date: str | None = None) -> dict:
    """从 ISO 日期范围构建 publish_date 的 Chroma `where` 过滤条件。

    Chroma 要求每个表达式只能有一个操作符，且操作数为 int/float，
    因此闭区间用 $and 组合两个单操作符 int 条件表示。
    """
    if end_date:
        return {
            "$and": [
                {"publish_date": {"$gte": _iso_to_int(start_date)}},
                {"publish_date": {"$lte": _iso_to_int(end_date)}},
            ]
        }
    return {"publish_date": {"$gte": _iso_to_int(start_date)}}
