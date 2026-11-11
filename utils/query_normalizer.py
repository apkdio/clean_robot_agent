"""query 入口归一化：纯规则清洗标点、空格与全半角，另做有向的同义词映射。

调用点在注入、危险、情绪三条硬规则之后，结果用于 SOP 触发匹配、域路由与检索 query；
安全判断、生成与会话记录仍用原文。函数幂等，没变化时原样返回。
"""

from __future__ import annotations

import re

from config.word_dict_config import QUERY_HALF_PUNCT, QUERY_PUNCT, QUERY_SYNONYMS

# 全角字母 / 数字（ＡＰＰ / １２３）→ 半角；全角标点保持知识库里的中文形态不动
_FULLWIDTH_ALNUM = re.compile(r"[\uff10-\uff19\uff21-\uff3a\uff41-\uff5a]")
_WS = re.compile(r"[\s\u3000]+")
_SPACE_NEAR_PUNCT = re.compile(r"\s*([" + QUERY_PUNCT + r"])\s*")
_DUP_PUNCT = re.compile(r"([" + QUERY_PUNCT + r"])\1+")
_CJK = r"\u4e00-\u9fff"
_CJK_SPACE = re.compile(r"(?<=[%s])\s+(?=[%s])" % (_CJK, _CJK))


def _widen_punct(text: str) -> str:
    """半角标点 → 全角（知识库的写法）；数字之间的小数点与千分位保持原样。"""
    out = []
    for i, ch in enumerate(text):
        if ch in ".," and 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
            out.append(ch)
        else:
            out.append(QUERY_HALF_PUNCT.get(ch, ch))
    return "".join(out)


def _tighten(text: str) -> str:
    """全角字母数字转半角、空白压成单空格、去掉标点两侧与中文字符之间的空格。"""
    text = _FULLWIDTH_ALNUM.sub(lambda m: chr(ord(m.group()) - 0xFEE0), text)
    text = _WS.sub(" ", text)
    text = _SPACE_NEAR_PUNCT.sub(r"\1", text)
    text = _CJK_SPACE.sub("", text)
    return text.strip()


def normalize(text: str) -> str:
    """归一化 query。只做形态清洗与同义词替换，不删任何实词，语义标记也不碰。"""
    if not text:
        return text
    out = _tighten(_widen_punct(text))
    for src, dst in QUERY_SYNONYMS.items():
        out = out.replace(src, dst)
    out = _DUP_PUNCT.sub(r"\1", out)
    return out or text
