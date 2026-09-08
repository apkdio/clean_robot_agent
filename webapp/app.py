"""Flask Web 应用：知识库上传与向量化管理。

运行方式：
  python -m webapp.app          # 从项目根目录运行
  python webapp/app.py          # 或直接运行
然后打开 http://localhost:5050
"""

import os
import sys

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# 确保项目根目录和 tools/ 位于 sys.path 中。
# 需要 tools/ 是因为其中的模块使用了裸导入
# （例如 `from log_tool import`）而非包相对导入。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_DIR = os.path.join(_PROJECT_ROOT, "tools")
for _p in (_PROJECT_ROOT, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.agent import ask_stream
from tools.config_tool import load_config
from tools.context_store import ensure_session_id, list_sessions
from tools.log_tool import get_logger
from tools.vector_store import (
    ingest_data_dir,
    ingest_file,
    list_collections_info,
    reset_collection,
)

# 统一「裸导入」与「包导入」的模块身份。
# tools 内部模块用裸导入（from xxx import），webapp 用包导入（from tools.xxx import），
# 同一文件会被加载两次、模块级状态分裂（如 agent._hybrid_retriever 单例、redis_store 锁）。
# 这里把所有 tools 模块统一按包加载，并把裸模块名指向包实例，后续裸 import 命中同一实例。
import importlib as _importlib
_TOOL_MODULES = (
    "agent", "config_tool", "log_tool", "context_store", "metadata_extractor",
    "redis_store", "vector_store", "llm_tool", "prompts_tool", "hot_ingest",
    "hybrid_retriever", "sparse_retriever", "rrf_fusion", "entry_splitter",
    "file_tools", "path_tool", "intent_router",
)
for _name in _TOOL_MODULES:
    try:
        _pkg_module = _importlib.import_module(f"tools.{_name}")
        sys.modules[_name] = _pkg_module
    except ImportError:
        pass

logger = get_logger(name="webapp")

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

_rag_cfg = load_config("rag")
_upload_dir = os.path.join(str(_PROJECT_ROOT), "temp", "uploads")
os.makedirs(_upload_dir, exist_ok=True)

_supported_exts = tuple(_rag_cfg.get("supported_exts", [".txt", ".pdf"]))

# 停止回答标志（Redis + 内存降级）
_stop_flags = set()


def _set_stop(session_id: str):
    from tools.redis_store import get_redis
    r = get_redis()
    if r is not None:
        r.set(f"stop:{session_id}", "1", ex=120)
    _stop_flags.add(session_id)


def _is_stopped(session_id: str) -> bool:
    from tools.redis_store import get_redis
    r = get_redis()
    if r is not None:
        return bool(r.exists(f"stop:{session_id}"))
    return session_id in _stop_flags


def _clear_stop(session_id: str):
    from tools.redis_store import get_redis
    r = get_redis()
    if r is not None:
        r.delete(f"stop:{session_id}")
    _stop_flags.discard(session_id)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats", methods=["GET"])
def stats():
    try:
        info = list_collections_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f"[Stats] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/ingest", methods=["POST"])
def ingest():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part in request."}), 400

    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"status": "error", "message": "Empty filename."}), 400

    # 校验扩展名
    ext = os.path.splitext(upload.filename)[1].lower()
    if not ext.endswith(_supported_exts):
        msg = f"Unsupported file type: {ext}. Supported: {', '.join(_supported_exts)}"
        logger.warning(f"[Ingest] {msg}")
        return jsonify({"status": "error", "message": msg}), 400

    # 保存到 temp/uploads
    save_path = os.path.join(_upload_dir, upload.filename)
    upload.save(save_path)
    logger.info(f"[Ingest] Saved upload: {save_path}")

    try:
        # 摄入到向量库（摄入锁：与热更新/重置互斥）
        from tools.redis_store import acquire_lock, release_lock
        if not acquire_lock("lock:ingest"):
            return jsonify({"status": "busy", "message": "正在摄入知识库，请稍候"}), 409
        try:
            result = ingest_file(save_path)
        finally:
            release_lock("lock:ingest")
    finally:
        # 摄入完成后清理临时文件
        try:
            os.remove(save_path)
        except OSError:
            pass

    code = 200 if result.get("status") == "ok" else 500
    return jsonify(result), code


@app.route("/api/reset", methods=["POST"])
def reset():
    from tools.redis_store import acquire_lock, release_lock
    if not acquire_lock("lock:ingest"):
        return jsonify({"status": "busy", "message": "正在摄入知识库，请稍候"}), 409
    try:
        deleted = reset_collection()
        # 使热更新快照失效，以便下一个周期从头重新摄入
        from tools.hot_ingest import invalidate_snapshot
        invalidate_snapshot()
        return jsonify({"status": "ok", "deleted": deleted})
    except Exception as e:
        logger.error(f"[Reset] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        release_lock("lock:ingest")


@app.route("/api/chat/stop", methods=["POST"])
def chat_stop():
    """设置停止标志，中断当前会话的流式回答。"""
    data = request.get_json(silent=True) or {}
    session_id = ensure_session_id(data.get("session_id", ""))
    _set_stop(session_id)
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

    # 会话级锁：同一会话回答未完成时拒绝新请求（Redis SETNX + 本地降级）
    from tools.redis_store import acquire_lock
    lock_key = f"lock:session:{session_id}"
    if not acquire_lock(lock_key): # 方法内已经实现加锁逻辑
        return jsonify({"status": "busy", "message": "正在回答中，请稍候"}), 409

    logger.info(f"[Stream] query: {query[:60]}... session={session_id}")

    def generate():
        try:
            import json
            from tools.context_store import append_message, rollback_last_user_message
            from sops.base import get_last_recommend
            parts = []
            stopped = False
            try:
                for chunk in ask_stream(query, session_id, lng=lng, lat=lat):
                    if _is_stopped(session_id):
                        stopped = True
                        break
                    parts.append(chunk)
                    # JSON 编码，避免 chunk 内的换行符破坏 SSE 帧格式
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"[Stream] {e}")
                yield f"data: {json.dumps(f'[错误: {e}]', ensure_ascii=False)}\n\n"
            if stopped:
                # 停止：回滚这一轮（忽略进上下文），首句取消则会话不落地磁盘
                rollback_last_user_message(session_id)
                yield "data: [STOPPED]\n\n"
                return
            # 记录 assistant 完整回答 + 结构化推荐（供自由指代消解）
            full_answer = "".join(parts)
            append_message(session_id, "assistant", full_answer, models=get_last_recommend(session_id))
            # 若会话尚未生成标题（第一轮），则调用 LLM 生成并写入 meta
            try:
                from tools.context_store import get_session_title, set_session_title, generate_session_title
                if not get_session_title(session_id):
                    title = generate_session_title(query, full_answer)
                    set_session_title(session_id, title)
            except Exception as e:
                logger.warning(f"[Stream] Generate title failed: {e}")
            yield "data: [DONE]\n\n"
        finally:
            from tools.redis_store import release_lock
            release_lock(lock_key)
            _clear_stop(session_id)

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
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sessions/<sid>/messages", methods=["GET"])
def session_messages(sid):
    """返回指定会话的完整消息（供前端加载历史对话）。"""
    from tools.context_store import get_recent
    try:
        msgs = get_recent(sid, n=1000)  # 文件保留全量，这里取足够多
        return jsonify({"status": "ok", "messages": msgs})
    except Exception as e:
        logger.error(f"[SessionMessages] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sessions/<sid>", methods=["DELETE"])
def remove_session(sid):
    """删除指定会话。"""
    from tools.context_store import delete_session as _del_session
    try:
        ok = _del_session(sid)
        return jsonify({"status": "ok", "deleted": ok})
    except Exception as e:
        logger.error(f"[DeleteSession] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/ingest/batch", methods=["POST"])
def ingest_batch():
    """一次性摄入知识目录中所有受支持的文件。"""
    data_dir = os.path.join(_PROJECT_ROOT, "data", "knowledge")
    try:
        results = ingest_data_dir(data_dir)
        return jsonify({"status": "ok", "results": results})
    except Exception as e:
        logger.error(f"[BatchIngest] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/config", methods=["GET"])
def agent_config():
    """返回当前的 agent 与检索配置，用于前端标识。"""
    try:
        agent_cfg = load_config("agent")
        rag_cfg = load_config("rag")
        behavior = agent_cfg.get("behavior", {})
        retrieval = rag_cfg.get("retrieval", {})
        return jsonify({
            "status": "ok",
            "mode": "retrieval_only" if behavior.get("retrieval_only", True) else "rag_full",
            "mode_label": "RAG 检索模式" if behavior.get("retrieval_only", True) else "RAG+LLM 模式",
            "model": agent_cfg.get("llm", {}).get("model", "?"),
        })
    except Exception as e:
        logger.error(f"[Config] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    logger.info(f"[WebApp] Starting on http://localhost:{port}")

    # 启动热更新守护线程（每 30 分钟扫描一次 data/knowledge）
    from tools.hot_ingest import start_hot_ingest
    start_hot_ingest()

    app.run(host="0.0.0.0", port=port, debug=False)
