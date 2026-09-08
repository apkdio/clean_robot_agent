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
_CONTEXT_META_DATA_DIR = get_abs_path("data/context_meta")
_MAX_TURNS = 6                      # 上下文保留最近 6 轮
_MAX_MESSAGES = _MAX_TURNS * 2      # 一轮 = 用户 + 客服，共 12 条消息

# 内存缓存：session_id → 最近的消息列表（供快速访问，避免频繁读文件）
_cache = {}


def _file_path(session_id: str) -> str:
    return os.path.join(_CONTEXT_DIR, f"{session_id}.jsonl")


def _meta_path(session_id: str) -> str:
    return os.path.join(_CONTEXT_META_DATA_DIR, f"{session_id}.meta.json")


def _get_meta_field(session_id: str, field: str):
    """读取会话 meta 文件里的某个字段。"""
    fp = _meta_path(session_id)
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, encoding="utf-8") as f:
            return json.load(f).get(field)
    except Exception:
        return None


def get_session_title(session_id: str) -> str | None:
    """获取会话标题（若已生成）。"""
    return _get_meta_field(session_id, "title")


def _update_meta(session_id: str, **fields) -> dict:
    """读-改-写会话 meta 文件（保留已有字段，如 title / last_models）。"""
    os.makedirs(_CONTEXT_DIR, exist_ok=True)
    os.makedirs(_CONTEXT_META_DATA_DIR, exist_ok=True)
    file_path = _meta_path(session_id)
    data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.update(fields)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass
    return data


def set_session_title(session_id: str, title: str):
    """保存会话标题至 .meta.json（读-改-写，不覆盖其它字段）。"""
    _update_meta(session_id, title=title)


def set_last_models(session_id: str, models):
    """保存上一轮推荐结果到 meta（供追问使用，跨重启有效）。"""
    _update_meta(session_id, last_models=models)


def generate_session_title(query: str, answer: str) -> str:
    """调用 LLM (qwen2.5:3b) 为会话生成简短标题（6~10字）。

    失败或异常时自动回退到首条 query 截断 20 字。
    """
    fallback_title = (query.strip()[:20] or "新会话")
    try:
        from tools.llm_tool import get_chat_model
        from langchain_core.messages import HumanMessage

        llm = get_chat_model(model="qwen2.5:3b", temperature=0.3)
        prompt = (
            "请根据以下第一轮用户与客服的对话，生成一个简短的中文标题（6~10个字）。\n"
            "要求：直接返回标题文字本身，不要包含引号、书名号、序号或标点符号，不要任何解释说明。\n\n"
            f"用户：{query}\n"
            f"客服：{answer[:120]}\n\n"
            "标题："
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw_title = getattr(resp, "content", "") or ""
        cleaned = re.sub(r'["\'《》“”‘’\n\r\t。，！？,.!?]', '', raw_title).strip()
        if cleaned:
            cleaned = re.sub(r'^(?:会话|对话)?标题[：:]\s*', '', cleaned).strip()
            if cleaned:
                return cleaned[:20]
    except Exception:
        pass
    return fallback_title


def delete_session(session_id: str) -> bool:
    """删除指定会话：清理 jsonl 数据文件、meta 文件与内存缓存。"""
    _cache.pop(session_id, None)
    fp = _file_path(session_id)
    meta_fp = _meta_path(session_id)
    deleted = False
    if os.path.exists(fp):
        try:
            os.remove(fp)
            deleted = True
        except OSError:
            pass
    if os.path.exists(meta_fp):
        try:
            os.remove(meta_fp)
        except OSError:
            pass
    return deleted


def append_message(session_id: str, role: str, content: str, **meta):
    """追加一条消息：写内存缓存 + 追加 jsonl 文件。

    meta 可携带 intent / models（上一轮推荐的结构化型号，供自由指代）等。
    """
    os.makedirs(_CONTEXT_DIR, exist_ok=True)
    os.makedirs(_CONTEXT_META_DATA_DIR, exist_ok=True)
    msg = {"role": role, "content": content, "ts": datetime.now().isoformat(timespec="seconds")}
    msg.update(meta)

    cache = _cache.setdefault(session_id, [])
    cache.append(msg)
    if len(cache) > _MAX_MESSAGES:
        cache.pop(0)

    with open(_file_path(session_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def rollback_last_user_message(session_id: str):
    """停止回答时回滚最后一条用户消息（这一轮不进入上下文）。

    jsonl 追加写，回滚 = 删除最后一条 user 及其后的消息（此时 assistant 尚未写入）。
    若删除后会话为空（第一句就取消），删除整个会话（jsonl + meta + 缓存），不落地磁盘。
    """
    # 1. 内存缓存回滚：pop 最后一条 user 及其后的消息
    cache = _cache.get(session_id)
    if cache is not None:
        while cache and cache[-1].get("role") != "user":
            cache.pop()
        if cache:
            cache.pop()

    # 2. jsonl 回滚：删除最后一条 user 及其后的行
    fp = _file_path(session_id)
    if not os.path.exists(fp):
        return
    try:
        with open(fp, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return

    while lines:
        try:
            last = json.loads(lines[-1])
        except json.JSONDecodeError:
            lines.pop()
            continue
        if last.get("role") != "user":
            lines.pop()
            continue
        lines.pop()  # 删除最后一条 user
        break

    if not lines:
        # 第一句就取消 → 会话不落地，删除 jsonl + meta + 缓存
        delete_session(session_id)
        return

    try:
        with open(fp, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception:
        pass


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


def get_last_models(session_id: str):
    """读取上一轮推荐结果（从 meta 的 last_models 字段，跨重启有效）。"""
    return _get_meta_field(session_id, "last_models")


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
    """列出所有会话：session_id + 标题(title) + 消息数 + 更新时间(updated_at/ts)。"""
    os.makedirs(_CONTEXT_DIR, exist_ok=True)
    os.makedirs(_CONTEXT_META_DATA_DIR, exist_ok=True)
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

        # 优先从 meta 读取标题，若无则取首条用户消息截断兜底
        title = get_session_title(sid)
        if not title:
            for m in msgs:
                if m.get("role") == "user":
                    title = m.get("content", "")[:20]
                    break
        title = title or "（空会话）"

        ts = msgs[-1].get("ts", "")
        sessions.append({
            "session_id": sid,
            "title": title,
            "summary": title,
            "messages": len(msgs),
            "ts": ts,
            "updated_at": ts,
        })
    sessions.sort(key=lambda s: s.get("updated_at") or s.get("ts", ""), reverse=True)
    return sessions
