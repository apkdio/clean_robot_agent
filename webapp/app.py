"""Flask Web 应用：知识库上传与向量化管理。

运行方式：
  python -m webapp.app          # 从项目根目录运行
  python webapp/app.py          # 或直接运行
然后打开 http://localhost:5050
"""

import os
import sys

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# 确保项目根目录位于 sys.path 中（service.* / utils.* 等包导入依赖它）。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from service.agent import ask_stream
from service.chat_service import (
    acquire_turn_lock, clear_stop, release_turn_lock, set_stop, stream_turn,
)
from service.feedback_service import reasons as feedback_reasons, submit as submit_feedback
from service.knowledge_service import (
    acquire_ingest_lock,
    collections_info,
    ingest_uploaded_file,
    release_ingest_lock,
    reset_knowledge_base,
    start_background_ingest,
    supported_exts,
)
from service.session_service import (
    all_sessions,
    is_valid_session_id,
    normalize_session_id,
    recent_messages,
    remove_session as remove_session_record,
    rollback_last_user,
)
from service.status_service import agent_display_config
from utils.dto.feedback import FeedbackRequest
from utils.dto.request import ChatRequest
from utils.log_tool import get_logger

logger = get_logger(name="webapp")

# 500 一律回显这句：内部路径/异常细节只进日志，不回给前端
_INTERNAL_ERROR_MSG = "内部错误，请查看服务端日志"

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)
# 上传体积上限（超出由 Flask 返回 413），防任意大文件打满磁盘
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

_upload_dir = os.path.join(str(_PROJECT_ROOT), "temp", "uploads")
os.makedirs(_upload_dir, exist_ok=True)


def _supported_exts() -> tuple:
    return supported_exts()


def _safe_upload_name(raw: str) -> str:
    """净化上传文件名：取 basename 防路径穿越、清文件系统非法字符；非法或空名返回空串。

    中文名要保留：入库后的 file_name 元数据与域路由都按知识文件名匹配。
    """
    name = os.path.basename((raw or "").replace("\\", "/"))
    name = "".join(ch for ch in name if ch not in '<>:"|?*' and ord(ch) >= 32)
    return name.strip().strip(".")[:120]

@app.route("/")
def index():
    return render_template("index.html", feedback_reasons=feedback_reasons())


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """记录用户对某轮回复的「有用/无用」标注，落到 trace 旁路文件 <sid>.feedback.jsonl。"""
    data = request.get_json(silent=True) or {}
    try:
        feedback_request = FeedbackRequest(
            query=data.get("query") or "",
            session_id=normalize_session_id(data.get("session_id", "")),
            rating=data.get("rating") or "",
            reason=data.get("reason") or "",
            note=data.get("note") or "",
            date=data.get("date") or None,
            turn_index=data.get("turn_index"),
        )
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    result = submit_feedback(feedback_request)
    return jsonify(result.payload()), {"ok": 200, "no_turn": 404}.get(result.status, 400)


@app.route("/api/stats", methods=["GET"])
def stats():
    try:
        return jsonify(collections_info().payload())
    except Exception as e:
        logger.error(f"[Stats] {e}")
        return jsonify({"status": "error", "message": _INTERNAL_ERROR_MSG}), 500


@app.route("/api/ingest", methods=["POST"])
def ingest():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part in request."}), 400

    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"status": "error", "message": "Empty filename."}), 400

    # 文件名净化后再校验扩展名；realpath 双保险确保落在 upload 目录内
    base = _safe_upload_name(upload.filename)
    ext = os.path.splitext(base)[1].lower()
    if not base or not ext.endswith(_supported_exts()):
        msg = f"Unsupported or unsafe file name. Supported: {', '.join(_supported_exts())}"
        logger.warning(f"[Ingest] {msg}: {upload.filename!r}")
        return jsonify({"status": "error", "message": msg}), 400

    # 先拿锁再落盘：两个并发同名上传会互相覆盖/删掉对方正在摄入的源文件；
    # 锁拿不到时文件还没写盘，直接 409 不留垃圾
    if not acquire_ingest_lock():
        return jsonify({"status": "busy", "message": "正在摄入知识库，请稍候"}), 409

    try:
        save_path = os.path.realpath(os.path.join(_upload_dir, base))
        if os.path.dirname(save_path) != os.path.realpath(_upload_dir):
            logger.warning(f"[Ingest] path escapes upload dir: {upload.filename!r}")
            return jsonify({"status": "error", "message": "Invalid file name."}), 400
        upload.save(save_path)
        logger.info(f"[Ingest] Saved upload: {save_path}")
        try:
            result = ingest_uploaded_file(save_path)
        finally:
            # 摄入完成后清理临时文件（只删自己刚命名的这个）
            try:
                os.remove(save_path)
            except OSError:
                pass
    finally:
        release_ingest_lock()

    code = 200 if result.status == "ok" else 500
    return jsonify(result.payload()), code


@app.route("/api/reset", methods=["POST"])
def reset():
    try:
        result = reset_knowledge_base()
        if result.status == "busy":
            return jsonify({"status": "busy", "message": "正在摄入知识库，请稍候"}), 409
        return jsonify(result.payload())
    except Exception as e:
        logger.error(f"[Reset] {e}")
        return jsonify({"status": "error", "message": _INTERNAL_ERROR_MSG}), 500


@app.route("/api/chat/stop", methods=["POST"])
def chat_stop():
    """设置停止标志，中断当前会话的流式回答。"""
    data = request.get_json(silent=True) or {}
    session_id = normalize_session_id(data.get("session_id", ""))
    set_stop(session_id)
    logger.info(f"[Stream] stop requested, session={session_id}")
    return jsonify({"status": "ok"})


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """通过 Server-Sent Events 流式返回回答片段。"""
    data = request.get_json(silent=True)
    if not data or "query" not in data:
        return jsonify({"status": "error", "message": "Missing 'query' in JSON body."}), 400
    query = data["query"].strip()
    if not query:
        return jsonify({"status": "error", "message": "Empty query."}), 400
    session_id = normalize_session_id(data.get("session_id", ""))
    lng = data.get("lng")
    lat = data.get("lat")
    is_retry = bool(data.get("retry"))     # 前端点「重新回答」时显式带上；不靠"和上一句相同"猜
    try:
        chat_request = ChatRequest(query=query, session_id=session_id,
                                   lng=lng, lat=lat, retry=is_retry)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    # 会话级锁：同一会话回答未完成时拒绝新请求（Redis SETNX + 本地降级）
    if not acquire_turn_lock(session_id):
        return jsonify({"status": "busy", "message": "正在回答中，请稍候"}), 409

    # 清掉上一轮残留的停止标志：晚到/重复的 stop 请求会在流结束后留在 Redis，
    # 不清的话下一轮在首个 chunk 前就被误停、user 消息被回滚
    clear_stop(session_id)
    logger.info(f"[Stream] query: {query[:60]}... session={session_id}")

    def generate():
        completed = False
        try:
            import json

            for event in stream_turn(chat_request, ask_stream_func=ask_stream):
                if event.kind == "delta":
                    yield f"data: {json.dumps(event.text, ensure_ascii=False)}\n\n"
                elif event.kind == "error":
                    yield f"event: error\ndata: {json.dumps(event.text, ensure_ascii=False)}\n\n"
                elif event.kind == "stopped":
                    yield "data: [STOPPED]\n\n"
                elif event.kind == "meta":
                    yield f"event: meta\ndata: {json.dumps(event.meta, ensure_ascii=False)}\n\n"
                elif event.kind == "done":
                    yield "data: [DONE]\n\n"
            completed = True
        finally:
            # 客户端断连（GeneratorExit，BaseException 捕不住上面的 except）与
            # 停止同一语义：回滚这轮的 user 消息，不留「问了没答」的上下文残局
            if not completed:
                try:
                    rollback_last_user(session_id)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[Stream] rollback on disconnect failed: {e}")
            release_turn_lock(session_id)
            clear_stop(session_id)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/sessions", methods=["GET"])
def sessions():
    """列出所有历史会话（供前端对话列表）。"""
    try:
        return jsonify({"status": "ok", "sessions": all_sessions()})
    except Exception as e:
        logger.error(f"[Sessions] {e}")
        return jsonify({"status": "error", "message": _INTERNAL_ERROR_MSG}), 500


@app.route("/api/sessions/<sid>/messages", methods=["GET"])
def session_messages(sid):
    """返回指定会话的完整消息（供前端加载历史对话）。"""
    if not is_valid_session_id(sid):
        return jsonify({"status": "error", "message": "Invalid session id."}), 400
    try:
        msgs = recent_messages(sid, n=1000)  # 文件保留全量，这里取足够多
        return jsonify({"status": "ok", "messages": msgs})
    except Exception as e:
        logger.error(f"[SessionMessages] {e}")
        return jsonify({"status": "error", "message": _INTERNAL_ERROR_MSG}), 500


@app.route("/api/sessions/<sid>", methods=["DELETE"])
def remove_session(sid):
    """删除指定会话。"""
    if not is_valid_session_id(sid):
        return jsonify({"status": "error", "message": "Invalid session id."}), 400
    try:
        ok = remove_session_record(sid)
        return jsonify({"status": "ok", "deleted": ok})
    except Exception as e:
        logger.error(f"[DeleteSession] {e}")
        return jsonify({"status": "error", "message": _INTERNAL_ERROR_MSG}), 500


@app.route("/api/config", methods=["GET"])
def agent_config():
    """返回当前的 agent 与检索配置，用于前端标识。"""
    try:
        return jsonify(agent_display_config())
    except Exception as e:
        logger.error(f"[Config] {e}")
        return jsonify({"status": "error", "message": _INTERNAL_ERROR_MSG}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    logger.info(f"[WebApp] Starting on http://localhost:{port}")

    # 启动热更新守护线程（每 30 分钟扫描一次 data/knowledge）
    start_background_ingest()

    app.run(host="0.0.0.0", port=port, debug=False)
