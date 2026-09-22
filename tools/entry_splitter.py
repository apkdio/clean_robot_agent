"""编号条目切分器：把结构化为编号列表的文档按条目切分为 chunk。

支持两种形式：「1. **加粗标题**」后跟「- 内容」行，以及「1. 纯文本」单行条目。
章节标题（## / ###）作为前缀保留在其后第一条目上，使 chunk 携带分类上下文。
不含编号条目时返回空列表，由调用方回退到通用切分器。
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

# 条目以 "1. "、"2. "、"99. " … 开头（可选后跟 **加粗**）
_ENTRY_RE = re.compile(r"^\d+\.\s+")
# 章节标题："# 标题"、"## 标题"、"### 标题"
_SECTION_RE = re.compile(r"^#{1,6}\s+")


def _detect_entries(text: str) -> List[tuple]:
    """返回 [(章节前缀, 条目文本), ...]；章节前缀是最近的上级标题，无则空串。"""
    entries: List[tuple] = []
    current_section = ""
    current_lines: List[str] = []
    started = False

    for line in text.split("\n"):
        # 章节标题 → 记住作为前缀，不开始新条目
        if _SECTION_RE.match(line.strip()):
            # 切换章节前先刷出挂起的条目
            if started and current_lines:
                entries.append((current_section, "\n".join(current_lines)))
                current_lines = []
                started = False
            current_section = line.strip()
            continue

        # 新的编号条目
        if _ENTRY_RE.match(line.strip()):
            if started and current_lines:
                entries.append((current_section, "\n".join(current_lines)))
            current_lines = [line.strip()]
            started = True
            continue

        # 当前条目的续行（"- 内容"、折行等）
        if started:
            if line.strip():  # 跳过条目内部的空白分隔行
                current_lines.append(line.rstrip())
            continue

        # 否则：文件标题 / 空行 / 首条之前的任何内容 → 跳过

    # 刷出最后一条
    if started and current_lines:
        entries.append((current_section, "\n".join(current_lines)))

    return entries


def split_numbered_entries(text: str, metadata: dict | None = None) -> List[Document]:
    """把编号列表文档切分为每个条目一个 Document；不含编号条目时返回空列表。"""
    entries = _detect_entries(text)
    if not entries:
        return []

    docs: List[Document] = []
    base = dict(metadata or {})
    for section, entry_text in entries:
        # 将章节作为前缀附加，使 chunk 保留其分类上下文
        content = f"{section}\n{entry_text}" if section else entry_text
        docs.append(Document(page_content=content, metadata=dict(base)))
    return docs
