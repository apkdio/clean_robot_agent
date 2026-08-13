"""Numbered-entry splitter.

Splits documents structured as numbered lists into one chunk per entry.

Handles both variants found in the knowledge base:
  - "1. **加粗标题**" followed by "- 内容" lines  (FAQ / 型号)
  - "1. 纯文本内容" single-line entries            (故障排除 / 维护保养 / 选购指南)

Section headers (## / ###) are kept as a prefix on the first entry that follows,
so each chunk carries its category context (e.g. "入门级" + 型号内容).

A file that does not contain numbered entries (e.g. a plain paragraph or CSV)
falls back to the caller's generic splitter.
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

# An entry starts with "1. ", "2. ", "99. " … (optionally followed by **bold**)
_ENTRY_RE = re.compile(r"^\d+\.\s+")
# Section headers: "# 标题", "## 标题", "### 标题"
_SECTION_RE = re.compile(r"^#{1,6}\s+")


def _detect_entries(text: str) -> List[tuple]:
    """Scan text and return [(section_prefix, entry_text), ...].

    section_prefix is the nearest preceding ##/### header, or "" if none.
    """
    entries: List[tuple] = []
    current_section = ""
    current_lines: List[str] = []
    started = False

    for line in text.split("\n"):
        # Section header → remember as prefix, do not start an entry
        if _SECTION_RE.match(line.strip()):
            # Flush a pending entry before switching section
            if started and current_lines:
                entries.append((current_section, "\n".join(current_lines)))
                current_lines = []
                started = False
            current_section = line.strip()
            continue

        # New numbered entry
        if _ENTRY_RE.match(line.strip()):
            if started and current_lines:
                entries.append((current_section, "\n".join(current_lines)))
            current_lines = [line.strip()]
            started = True
            continue

        # Continuation of current entry ("- 内容", wrapped lines, etc.)
        if started:
            if line.strip():  # skip blank separator lines inside an entry
                current_lines.append(line.rstrip())
            continue

        # Otherwise: file title / blank line / anything before first entry → skip

    # Flush last entry
    if started and current_lines:
        entries.append((current_section, "\n".join(current_lines)))

    return entries


def split_numbered_entries(text: str, metadata: dict | None = None) -> List[Document]:
    """Split a numbered-list document into one Document per entry.

    Args:
        text: Raw document text.
        metadata: Base metadata dict copied onto every produced Document.

    Returns:
        List of Documents. Empty list if the text has no numbered entries.
    """
    entries = _detect_entries(text)
    if not entries:
        return []

    docs: List[Document] = []
    base = dict(metadata or {})
    for section, entry_text in entries:
        # Attach section as a prefix so the chunk keeps its category context
        content = f"{section}\n{entry_text}" if section else entry_text
        docs.append(Document(page_content=content, metadata=dict(base)))
    return docs
