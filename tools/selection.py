"""多选序号解析：只认序号与固定跳过命令，不做标签别名匹配。"""
from __future__ import annotations

import re
import unicodedata

from config.word_dict_config import TAG_SKIP_WORDS

_SELECTION_SEP_RE = re.compile(r"[\s,，、；;/／]+")
_CN_DIGIT = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _normalize(text: str) -> str:
    # NFKC 处理全角数字/标点；中文逗号/顿号统一成半角逗号
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("，", ",").replace("、", ",").replace("；", ",").replace(";", ",")
    return text.replace("／", "/").strip()


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
        token = re.sub(r"^(?:第|选)|(?:个|项|号)$", "", part.strip())
        if token.isdigit():
            n = int(token)
        elif token in _CN_DIGIT:
            n = _CN_DIGIT[token]
        else:
            return "invalid", []
        if n < 1 or n > max_n:
            return "invalid", []
        if n not in out:
            out.append(n)
    return "ok", out


def is_skip(text: str) -> bool:
    return re.sub(r"[\s,]+", "", _normalize(text)) in TAG_SKIP_WORDS
