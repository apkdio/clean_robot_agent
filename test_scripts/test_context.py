"""上下文存储测试：tools/context_store.py 的会话持久化与读取逻辑。

覆盖：
  - append_message / get_recent（内存缓存 + 文件回读）
  - 最近 12 条（6 轮）截断
  - set_last_models / get_last_models（跨重启的推荐结果 meta）
  - set_session_title / get_session_title（会话标题 meta）
  - delete_session（清理 jsonl + meta + 缓存）
  - ensure_session_id（UUID 双端校验）
  - list_sessions（会话列表）
  - generate_session_title（LLM 失败时的 fallback，monkeypatch 强制异常）

运行：
  .venv\\Scripts\\python.exe test_context.py
"""
import sys
import uuid
from _runner import *
import tools.context_store as ctx


def _new_sid():
    return "test_" + str(uuid.uuid4())


# ──────────────────────────────────────────────────────────────
# 1. 消息追加与读取
# ──────────────────────────────────────────────────────────────

def test_append_and_recent():
    sid = _new_sid()
    ctx.append_message(sid, "user", "你好")
    ctx.append_message(sid, "assistant", "你好呀")
    recent = ctx.get_recent(sid)
    _assert_eq(len(recent), 2, "追加 2 条→读到 2 条")
    _assert_eq(recent[0]["role"], "user", "第一条 role=user")
    _assert_eq(recent[1]["content"], "你好呀", "第二条 content 正确")
    _assert_in("ts", recent[0], "消息带时间戳")
    ctx.delete_session(sid)


def test_recent_truncation():
    sid = _new_sid()
    for i in range(13):
        ctx.append_message(sid, "user", f"msg{i}")
    recent = ctx.get_recent(sid)
    _assert_eq(len(recent), 12, "缓存截断为最近 12 条")
    _assert_eq(recent[-1]["content"], "msg12", "最新一条保留")
    _assert_eq(recent[0]["content"], "msg1", "最早一条被截断")
    ctx.delete_session(sid)


def test_recent_from_file():
    sid = _new_sid()
    ctx.append_message(sid, "user", "a")
    ctx.append_message(sid, "user", "b")
    ctx._cache.pop(sid, None)  # 清空缓存，强制从文件回读
    recent = ctx.get_recent(sid)
    _assert_eq(len(recent), 2, "文件回读 2 条")
    _assert_eq(recent[1]["content"], "b", "文件内容正确")
    ctx.delete_session(sid)


# ──────────────────────────────────────────────────────────────
# 2. meta：推荐结果 / 会话标题
# ──────────────────────────────────────────────────────────────

def test_last_models_roundtrip():
    sid = _new_sid()
    models = [{"name": "不染一尘净白 S1", "price": 899}]
    ctx.set_last_models(sid, models)
    got = ctx.get_last_models(sid)
    _assert_eq(got, models, "last_models 写读一致")
    _assert(ctx.get_last_models("nonexistent_" + str(uuid.uuid4())) is None, "无 meta→None")
    ctx.delete_session(sid)


def test_session_title_roundtrip():
    sid = _new_sid()
    ctx.set_session_title(sid, "扫地机选购")
    _assert_eq(ctx.get_session_title(sid), "扫地机选购", "标题写读一致")
    ctx.set_session_title(sid, "新标题")
    _assert_eq(ctx.get_session_title(sid), "新标题", "标题覆盖更新")
    ctx.delete_session(sid)


# ──────────────────────────────────────────────────────────────
# 3. 删除 / UUID 校验 / 会话列表
# ──────────────────────────────────────────────────────────────

def test_delete_session():
    sid = _new_sid()
    ctx.append_message(sid, "user", "x")
    ctx.set_session_title(sid, "t")
    _assert(ctx.delete_session(sid), "删除已存在会话返回 True")
    _assert_eq(ctx.get_recent(sid), [], "删除后读到空")
    _assert(ctx.get_session_title(sid) is None, "删除后标题为空")
    _assert(not ctx.delete_session(sid), "重复删除返回 False")


def test_ensure_session_id():
    valid = str(uuid.uuid4())
    _assert_eq(ctx.ensure_session_id(valid), valid, "合法 UUID 原样返回")
    for bad in ("abc", "123", "", "  "):
        out = ctx.ensure_session_id(bad)
        _assert(out != bad and len(out) == 36, f"非法值 {bad!r} 重新生成 UUID")
    out = ctx.ensure_session_id(None)
    _assert(len(out) == 36, "None→生成 UUID")


def test_list_sessions():
    sid = _new_sid()
    ctx.append_message(sid, "user", "帮我推荐一款扫地机器人")
    ctx.set_session_title(sid, "选购咨询")
    ids = {s["session_id"] for s in ctx.list_sessions()}
    _assert(sid in ids, "新会话出现在列表中")
    entry = next(s for s in ctx.list_sessions() if s["session_id"] == sid)
    _assert_eq(entry["title"], "选购咨询", "列表读标题")
    _assert_eq(entry["messages"], 1, "列表消息数")
    ctx.delete_session(sid)


# ──────────────────────────────────────────────────────────────
# 4. 会话标题生成（LLM 失败 fallback）
# ──────────────────────────────────────────────────────────────

def test_generate_session_title_fallback():
    import tools.llm_tool as lt
    orig = lt.get_chat_model

    def _boom(*a, **k):
        raise RuntimeError("no llm")

    lt.get_chat_model = _boom
    try:
        q = "你好，帮我推荐一款扫地机器人"
        title = ctx.generate_session_title(q, "好的")
        _assert_eq(title, q[:20], "LLM 失败回退到 query 截断")
        _assert_eq(ctx.generate_session_title("", ""), "新会话", "空 query 回退为「新会话」")
    finally:
        lt.get_chat_model = orig


TESTS = [
    test_append_and_recent,
    test_recent_truncation,
    test_recent_from_file,
    test_last_models_roundtrip,
    test_session_title_roundtrip,
    test_delete_session,
    test_ensure_session_id,
    test_list_sessions,
    test_generate_session_title_fallback,
]


def run():
    reset()
    return run_tests("上下文存储测试（context_store）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
