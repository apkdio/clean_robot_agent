"""序号解析：单序号、多选序号与固定跳过命令共用一套规则。

只认序号本身，不做标签文字/操作别名匹配；别名由调用方在序号解析失败后自己处理。
"""
from __future__ import annotations

import re
import unicodedata

from config.word_dict_config import TAG_SKIP_WORDS

_SELECTION_SEP_RE = re.compile(r"[\s,，、；;/／]+")
_CN_DIGIT = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_INDEX_RE = re.compile(r"^(?:第|选)?\s*([0-9]+|[一二两三四五六七八九十]+)\s*(?:个|项|号)?$")
_SELECTOR_RE = re.compile(r"(?:选|第)\s*([0-9]+|[一二两三四五六七八九十]+)\s*(?:个|项|号)?")


def _normalize(text: str) -> str:
    # NFKC 处理全角数字/标点；中文逗号/顿号统一成半角逗号
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("，", ",").replace("、", ",").replace("；", ",").replace(";", ",")
    return text.replace("／", "/").strip()


def _to_index(token: str):
    if token.isdigit():
        return int(token)
    return _CN_DIGIT.get(token)


def parse_index(text: str, n: int, allow_skip: bool = False):
    """解析单个序号，返回 ("ok", idx) / ("skip", None) / ("invalid", None)。

    整串必须是序号（可带「选/第」「个/项/号」）；越界直接 invalid，不能把 "10"
    截成 1。句中带明确「选/第」的句子（如「2个都不对，选1」）取最后一个选择标记。
    """
    raw = _normalize(text)
    if not raw:
        return "invalid", None
    compact = re.sub(r"\s+", "", raw)
    if allow_skip and compact in TAG_SKIP_WORDS:
        return "skip", None
    if n <= 0:
        return "invalid", None

    m = _INDEX_RE.fullmatch(compact)
    if m:
        idx = _to_index(m.group(1))
        return ("ok", idx) if idx is not None and 1 <= idx <= n else ("invalid", None)

    # 整串不是纯序号时，只接受句末明确的选择标记；裸数字仍按越界失败处理。
    matches = _SELECTOR_RE.findall(compact)
    if not matches:
        return "invalid", None
    idx = _to_index(matches[-1])
    return ("ok", idx) if idx is not None and 1 <= idx <= n else ("invalid", None)


def parse_multi_select(text: str, max_n: int):
    """返回 ("ok", [1,3]) / ("skip", []) / ("invalid", [])。

    只接受序号（阿拉伯数字或单个中文数字）；不识别标签文字、不做模糊匹配。
    """
    raw = _normalize(text)
    if not raw:
        return "invalid", []
    compact = re.sub(r"[\s,]+", "", raw)
    if compact in TAG_SKIP_WORDS:
        return "skip", []
    parts = [p for p in _SELECTION_SEP_RE.split(raw) if p]
    if not parts:
        return "invalid", []
    out = []
    for part in parts:
        status, idx = parse_index(part, max_n)
        if status != "ok":
            return "invalid", []
        if idx not in out:
            out.append(idx)
    return "ok", out


def is_skip(text: str) -> bool:
    return re.sub(r"[\s,]+", "", _normalize(text)) in TAG_SKIP_WORDS
