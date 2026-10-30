"""Flask Web 应用：知识库上传与向量化管理。

运行方式：
  python -m webapp.app          # 从项目根目录运行
  python webapp/app.py          # 或直接运行
然后打开 http://localhost:5050
"""

import os
import sys

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# 确保项目根目录位于 sys.path 中（tools.* / sops.* 等包导入依赖它）。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.agent import ask_stream
from tools.chat_service import (
    acquire_turn_lock, clear_stop, release_turn_lock, set_stop, stream_turn,
)
from tools.dto.request import ChatRequest
from tools.config_tool import load_config
from tools.context_store import ensure_session_id, list_sessions
from tools.log_tool import get_logger
from tools.vector_store import (
    ingest_file,
    list_collections_info,
    reset_collection,
)

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

_rag_cfg = load_config("rag")
_upload_dir = os.path.join(str(_PROJECT_ROOT), "temp", "uploads")
os.makedirs(_upload_dir, exist_ok=True)

_supported_exts = tuple(_rag_cfg.get("supported_exts", [".txt", ".pdf"]))


def _safe_upload_name(raw: str) -> str:
    """净化上传文件名：取 basename 防路径穿越、清文件系统非法字符；非法或空名返回空串。

    中文名要保留：入库后的 file_name 元数据与域路由都按知识文件名匹配。
    """
    name = os.path.basename((raw or "").replace("\\", "/"))
    name = "".join(ch for ch in name if ch not in '<>:"|?*' and ord(ch) >= 32)
    return name.strip().strip(".")[:120]

@app.route("/")
def index():
    from config.word_dict_config import FEEDBACK_REASONS
    return render_template("index.html", feedback_reasons=FEEDBACK_REASONS)


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """记录用户对某轮回复的「有用/无用」标注，落到 trace 旁路文件 <sid>.feedback.jsonl。"""
    data = request.get_json(silent=True) or {}
    session_id = ensure_session_id(data.get("session_id", ""))
    query = (data.get("query") or "").strip()
    rating = data.get("rating")
    if rating not in ("useful", "useless"):
        return jsonify({"status": "error", "message": "rating 必须是 useful / useless"}), 400
    if not query:
        return jsonify({"status": "error", "message": "缺少 query（用于定位轮次）"}), 400

    from config.word_dict_config import FEEDBACK_REASON_KEYS
    reason = (data.get("reason") or "").strip()
    if rating == "useless":
        # 无用必须给一个枚举原因（含 other）：否则标完没法聚类；有用不带原因
        if reason not in FEEDBACK_REASON_KEYS:
            return jsonify({"status": "error",
                            "message": "无用原因必须是 %s 之一" % "/".join(FEEDBACK_REASON_KEYS)}), 400
    else:
        reason = ""

    from tools import trace_store
    anchor = data.get("trace") or {}
    ctx = {}
    if isinstance(anchor, dict):
        # 上下文消息里带了该轮锚点：直接按（日期, 轮次）取，不再用 query 猜轮次
        ctx = trace_store.find_turn_by_index(session_id, anchor.get("date"), anchor.get("turn_index"))
        if ctx and ctx.get("query") != query:
            ctx = {}          # 锚点与提交的问句对不上，不当锚点用
    if not ctx:
        ctx = trace_store.find_turn(session_id, query)
    if not ctx:
        # 没对上轮次：标注没地方落，如实告诉前端（也提醒可能是跨天/轮次已过久）
        logger.warning("[Feedback] no matching turn: session=%s query=%s", session_id, query[:40])
        return jsonify({"status": "no_turn", "message": "没找到对应的轮次（可能已超出回看天数）"}), 404
    marked = trace_store.feedback_of(session_id, ctx.get("date"), ctx.get("turn_index"))
    # 同一轮可以再来一次（改主意）：追加一条、不改历史，读侧按轮次取最后一条为准
    trace_store.feedback_turn(
        ctx, rating=rating,
        reason=reason[:40],
        note=(data.get("note") or "")[:200],
    )
    logger.info("[Feedback] turn=%s/%s rating=%s reason=%s session=%s%s",
                ctx.get("date"), ctx.get("turn_index"), rating,
                reason or "-", session_id, " (revise)" if marked else "")
    return jsonify({"status": "ok", "date": ctx.get("date"),
                    "turn_index": ctx.get("turn_index"), "updated": bool(marked)})


@app.route("/api/stats", methods=["GET"])
def stats():
    try:
        info = list_collections_info()
        return jsonify(info)
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
    if not base or not ext.endswith(_supported_exts):
        msg = f"Unsupported or unsafe file name. Supported: {', '.join(_supported_exts)}"
        logger.warning(f"[Ingest] {msg}: {upload.filename!r}")
        return jsonify({"status": "error", "message": msg}), 400

    # 先拿锁再落盘：两个并发同名上传会互相覆盖/删掉对方正在摄入的源文件；
    # 锁拿不到时文件还没写盘，直接 409 不留垃圾
    from tools.redis_store import acquire_lock, release_lock
    if not acquire_lock("lock:ingest"):
        return jsonify({"status": "busy", "message": "正在摄入知识库，请稍候"}), 409

    try:
        save_path = os.path.realpath(os.path.join(_upload_dir, base))
        if os.path.dirname(save_path) != os.path.realpath(_upload_dir):
            logger.warning(f"[Ingest] path escapes upload dir: {upload.filename!r}")
            return jsonify({"status": "error", "message": "Invalid file name."}), 400
        upload.save(save_path)
        logger.info(f"[Ingest] Saved upload: {save_path}")
        try:
            result = ingest_file(save_path)
            if result.get("status") == "ok":
                # 手动入库也要让索引侧跟上：重建 BM25、失效 models.pkl、重置检索器单例；
                # 快照一并作废——上传按 basename 替换会盖掉同名知识文件的 chunk，
                # 而磁盘上的原文件哈希没变，快照与索引从此对不上，作废让下轮热更新全量对齐
                from tools.hot_ingest import invalidate_snapshot, rebuild_sparse_index
                invalidate_snapshot()
                rebuild_sparse_index()
        finally:
            # 摄入完成后清理临时文件（只删自己刚命名的这个）
            try:
                os.remove(save_path)
            except OSError:
                pass
    finally:
        release_lock("lock:ingest")

    code = 200 if result.get("status") == "ok" else 500
    return jsonify(result), code


@app.route("/api/reset", methods=["POST"])
def reset():
    from tools.redis_store import acquire_lock, release_lock
    if not acquire_lock("lock:ingest"):
        return jsonify({"status": "busy", "message": "正在摄入知识库，请稍候"}), 409
    try:
        # 先失效快照再清空：两步之间中断时，热更新下一轮会按「全新增」重入，不会留下「空索引 + 快照称已同步」
        from tools.hot_ingest import invalidate_snapshot
        invalidate_snapshot()
        deleted = reset_collection()
        # 清库后立即失效全部索引缓存（BM25/pkl/检索器单例/型号名单）：
        # 否则进程内还留着旧知识，最长 30 分钟继续用已删内容作答
        from tools.hot_ingest import rebuild_sparse_index
        rebuild_sparse_index()
        return jsonify({"status": "ok", "deleted": deleted})
    except Exception as e:
        logger.error(f"[Reset] {e}")
        return jsonify({"status": "error", "message": _INTERNAL_ERROR_MSG}), 500
    finally:
        release_lock("lock:ingest")


@app.route("/api/chat/stop", methods=["POST"])
def chat_stop():
    """设置停止标志，中断当前会话的流式回答。"""
    data = request.get_json(silent=True) or {}
    session_id = ensure_session_id(data.get("session_id", ""))
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
    session_id = ensure_session_id(data.get("session_id", ""))
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
                    from tools.context_store import rollback_last_user_message

                    rollback_last_user_message(session_id)
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
        return jsonify({"status": "ok", "sessions": list_sessions()})
    except Exception as e:
        logger.error(f"[Sessions] {e}")
        return jsonify({"status": "error", "message": _INTERNAL_ERROR_MSG}), 500


@app.route("/api/sessions/<sid>/messages", methods=["GET"])
def session_messages(sid):
    """返回指定会话的完整消息（供前端加载历史对话）。"""
    from tools.context_store import get_recent
    if sid != ensure_session_id(sid):
        return jsonify({"status": "error", "message": "Invalid session id."}), 400
    try:
        msgs = get_recent(sid, n=1000)  # 文件保留全量，这里取足够多
        return jsonify({"status": "ok", "messages": msgs})
    except Exception as e:
        logger.error(f"[SessionMessages] {e}")
        return jsonify({"status": "error", "message": _INTERNAL_ERROR_MSG}), 500


@app.route("/api/sessions/<sid>", methods=["DELETE"])
def remove_session(sid):
    """删除指定会话。"""
    from tools.context_store import delete_session as _del_session
    if sid != ensure_session_id(sid):
        return jsonify({"status": "error", "message": "Invalid session id."}), 400
    try:
        ok = _del_session(sid)
        return jsonify({"status": "ok", "deleted": ok})
    except Exception as e:
        logger.error(f"[DeleteSession] {e}")
        return jsonify({"status": "error", "message": _INTERNAL_ERROR_MSG}), 500


@app.route("/api/config", methods=["GET"])
def agent_config():
    """返回当前的 agent 与检索配置，用于前端标识。"""
    try:
        agent_cfg = load_config("agent")
        behavior = agent_cfg.get("behavior", {})
        return jsonify({
            "status": "ok",
            "mode": "retrieval_only" if behavior.get("retrieval_only", True) else "rag_full",
            "mode_label": "RAG 检索模式" if behavior.get("retrieval_only", True) else "RAG+LLM 模式",
            "model": agent_cfg.get("llm", {}).get("model", "?"),
        })
    except Exception as e:
        logger.error(f"[Config] {e}")
        return jsonify({"status": "error", "message": _INTERNAL_ERROR_MSG}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    logger.info(f"[WebApp] Starting on http://localhost:{port}")

    # 启动热更新守护线程（每 30 分钟扫描一次 data/knowledge）
    from tools.hot_ingest import start_hot_ingest
    start_hot_ingest()

    app.run(host="0.0.0.0", port=port, debug=False)
