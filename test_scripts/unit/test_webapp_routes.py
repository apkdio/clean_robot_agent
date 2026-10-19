"""webapp 路由层契约测试。

只测**不需要模型、也不写真实数据**的那一层：入参校验、状态码与错误语义、并发锁拒绝。
刻意不测的（会污染真实数据或需要 Ollama，交给 e2e 行为轨与人工）：
  - `/api/chat/stream` 的正常流式生成（需 Ollama；这里只测 400/409 两条拒绝路径）
  - `/api/ingest` 的成功上传（会写进真实 `data/knowledge/` 并触发入库）
  - `/api/reset` 的成功重置（会清空真实向量库；这里只测锁被占时的 409）

隔离：**必须先 import `test_scripts._runner`**——它把 CONTEXT_DIR / TRACE_DIR 指到测试目录，
之后 import webapp 才会用测试目录；顺序反了会把测试数据写进真实会话目录（_runner 里有守卫）。

单独运行：.venv\\Scripts\\python.exe test_scripts/unit/test_webapp_routes.py
"""
import io
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *          # noqa: E402  必须早于 webapp（设测试隔离目录）

from webapp.app import app                  # noqa: E402

_SID = "11111111-2222-3333-4444-555555555555"      # 合法 UUID：路由接受


def _client():
    app.config["TESTING"] = True
    return app.test_client()


def _hold(key):
    """占住一把锁，返回释放回调——用来验证「锁被占时路由怎么响应」。"""
    from tools.redis_store import acquire_lock, release_lock

    got = acquire_lock(key)
    return got, (lambda: release_lock(key))


def test_index_and_config():
    c = _client()
    r = c.get("/")
    _assert_eq(r.status_code, 200, "GET / 渲染聊天页")
    _assert_in("<html", r.get_data(as_text=True).lower(), "返回的是 HTML 页面")
    r = c.get("/api/config")
    _assert_eq(r.status_code, 200, "GET /api/config 200")
    body = r.get_json() or {}
    _assert_eq(body.get("status"), "ok", "config status=ok")
    _assert_true(body.get("mode"), "带 mode（供前端标识当前模式）")


def test_sessions_list_and_invalid_sid():
    c = _client()
    r = c.get("/api/sessions")
    _assert_eq(r.status_code, 200, "GET /api/sessions 200")
    _assert_true(isinstance((r.get_json() or {}).get("sessions"), list), "sessions 是列表")
    _assert_eq(c.get("/api/sessions/not-a-uuid/messages").status_code, 400,
               "非法 sid 取消息 → 400")
    _assert_eq(c.delete("/api/sessions/not-a-uuid").status_code, 400,
               "非法 sid 删会话 → 400")


def test_session_messages_and_delete():
    c = _client()
    r = c.get(f"/api/sessions/{_SID}/messages")
    _assert_eq(r.status_code, 200, "合法 sid 取消息 → 200（无消息即空列表）")
    _assert_true(isinstance((r.get_json() or {}).get("messages"), list), "messages 是列表")
    r = c.delete(f"/api/sessions/{_SID}")
    _assert_eq(r.status_code, 200, "删会话 → 200")
    _assert_in("deleted", r.get_json() or {}, "返回 deleted 字段")


def test_feedback_validation():
    c = _client()
    bad = [
        ({"session_id": _SID, "query": "x", "rating": "maybe"}, "rating 非枚举"),
        ({"session_id": _SID, "query": "", "rating": "useful"}, "缺 query"),
        ({"session_id": _SID, "query": "x", "rating": "useless"}, "无用必须给枚举原因"),
        ({"session_id": _SID, "query": "x", "rating": "useless", "reason": "瞎写的"},
         "无用原因越界"),
    ]
    for payload, why in bad:
        _assert_eq(c.post("/api/feedback", json=payload).status_code, 400, f"feedback {why} → 400")
    r = c.post("/api/feedback", json={"session_id": _SID, "query": "这句从没发生过", "rating": "useful"})
    _assert_eq(r.status_code, 404, "对不上轮次 → 404（如实告诉前端，不静默吞）")


def test_stop_and_stream_rejects():
    c = _client()
    _assert_eq(c.post("/api/chat/stop", json={"session_id": _SID}).status_code, 200,
               "停止请求 → 200（只置标志，不依赖流在跑）")
    _assert_eq(c.post("/api/chat/stream", json={}).status_code, 400, "缺 query → 400")
    _assert_eq(c.post("/api/chat/stream", json={"query": "   "}).status_code, 400, "空 query → 400")

    got, release = _hold(f"lock:session:{_SID}")
    _assert_true(got, "测试占住该会话的锁")
    try:
        r = c.post("/api/chat/stream", json={"query": "测试", "session_id": _SID})
        _assert_eq(r.status_code, 409, "同一会话并发 → 409 busy（前端据此结束 loading）")
        _assert_eq((r.get_json() or {}).get("status"), "busy",
                   "409 带 status=busy，前端可判并结束 loading")
    finally:
        release()


def test_ingest_rejects_and_busy():
    c = _client()
    _assert_eq(c.post("/api/ingest", data={}).status_code, 400, "没有 file 字段 → 400")
    _assert_eq(c.post("/api/ingest", data={"file": (io.BytesIO(b"x"), "")}).status_code, 400,
               "空文件名 → 400")
    _assert_eq(c.post("/api/ingest", data={"file": (io.BytesIO(b"x"), "x.exe")}).status_code, 400,
               "扩展名不在白名单 → 400（拿锁之前就挡掉）")

    got, release = _hold("lock:ingest")
    _assert_true(got, "测试占住入库锁")
    try:
        r = c.post("/api/ingest", data={"file": (io.BytesIO(b"# t\n"), "路由测试.md")})
        _assert_eq(r.status_code, 409, "入库锁被占 → 409，且不落盘（先拿锁再写文件）")
        _assert_eq(c.post("/api/reset").status_code, 409, "重置同样先拿锁 → 409 busy")
    finally:
        release()


TESTS = [
    test_index_and_config,
    test_sessions_list_and_invalid_sid,
    test_session_messages_and_delete,
    test_feedback_validation,
    test_stop_and_stream_rejects,
    test_ingest_rejects_and_busy,
]


def run():
    reset()
    return run_tests("webapp 路由层契约测试", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
