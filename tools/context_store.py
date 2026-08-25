"""对话上下文存储：按 session_id 分隔的 jsonl 文件 + 内存缓存。

每轮追加一条消息（role + content + 可选元数据），文件保留全量（持久化、
重启可恢复），内存缓存与读取时保留最近 6 轮（12 条消息）供自由指代与
LLM 生成拼接。

文件布局：
  data/context/<session_id>.jsonl   每行一条 JSON 消息

消息格式：
  {"role": "user"|"assistant", "content": "...", "intent": "...", "models": [...], "ts": "..."}
"""

import json
import os
import re
import uuid
from datetime import datetime

from path_tool import get_abs_path

_CONTEXT_DIR = get_abs_path("data/context")
_MAX_TURNS = 6                      # 上下文保留最近 6 轮
_MAX_MESSAGES = _MAX_TURNS * 2      # 一轮 = 用户 + 客服，共 12 条消息

# 内存缓存：session_id → 最近的消息列表（供快速访问，避免频繁读文件）
_cache = {}


def _file_path(session_id: str) -> str:
    return os.path.join(_CONTEXT_DIR, f"{session_id}.jsonl")


def append_message(session_id: str, role: str, content: str, **meta):
    """追加一条消息：写内存缓存 + 追加 jsonl 文件。

    meta 可携带 intent / models（上一轮推荐的结构化型号，供自由指代）等。
    """
    os.makedirs(_CONTEXT_DIR, exist_ok=True)
    msg = {"role": role, "content": content, "ts": datetime.now().isoformat(timespec="seconds")}
    msg.update(meta)

    cache = _cache.setdefault(session_id, [])
    cache.append(msg)
    if len(cache) > _MAX_MESSAGES:
        cache.pop(0)

    with open(_file_path(session_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def get_recent(session_id: str, n: int = None) -> list:
    """取最近 n 条消息（默认 6 轮）。优先内存缓存，未缓存则读文件。"""
    n = n or _MAX_MESSAGES
    cache = _cache.get(session_id)
    if cache is not None:
        return cache[-n:]

    msgs = []
    fp = _file_path(session_id)
    if os.path.exists(fp):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msgs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    msgs = msgs[-n:]
    _cache[session_id] = msgs
    return msgs


def get_last_models(session_id: str) -> list:
    """取最近一次推荐的结构化型号列表。

    从最近的消息倒序找第一条带 models 的 assistant 消息，返回其 models。
    """
    msgs = get_recent(session_id)
    for m in reversed(msgs):
        if m.get("role") == "assistant" and m.get("models"):
            return m["models"]
    return []


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def ensure_session_id(session_id):
    """双端验证 session_id：非合法 UUID 则重新生成（防御前端异常）。

    前端用 crypto.randomUUID() 生成 UUID；后端校验格式兑底，
    保证 sessionID 始终是合法 UUID，避免非法/重复值写入文件名。
    """
    if isinstance(session_id, str) and _UUID_RE.match(session_id):
        return session_id
    return str(uuid.uuid4())


def list_sessions() -> list:
    """列出所有会话：session_id + 最近一条用户消息摘要 + 消息数 + 时间。"""
    os.makedirs(_CONTEXT_DIR, exist_ok=True)
    sessions = []
    for fn in os.listdir(_CONTEXT_DIR):
        if not fn.endswith(".jsonl"):
            continue
        sid = fn[:-6]
        fp = os.path.join(_CONTEXT_DIR, fn)
        msgs = []
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msgs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not msgs:
            continue
        summary = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                summary = m.get("content", "")[:30]
                break
        sessions.append({
            "session_id": sid,
            "summary": summary or "（空会话）",
            "messages": len(msgs),
            "ts": msgs[-1].get("ts", ""),
        })
    sessions.sort(key=lambda s: s.get("ts", ""), reverse=True)
    return sessions
