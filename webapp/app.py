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
from tools.config_tool import load_agent_config, load_rag_config
from tools.context_store import ensure_session_id, list_sessions
from tools.log_tool import get_logger
from tools.vector_store import (
    ingest_data_dir,
    ingest_file,
    list_collections_info,
    reset_collection,
)

logger = get_logger(name="webapp")

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

_rag_cfg = load_rag_config()
_upload_dir = os.path.join(_PROJECT_ROOT, "temp", "uploads")
os.makedirs(_upload_dir, exist_ok=True)

_supported_exts = tuple(_rag_cfg.get("supported_exts", [".txt", ".pdf"]))


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

    # 摄入到向量库
    result = ingest_file(save_path)

    # 摄入完成后清理临时文件
    try:
        os.remove(save_path)
    except OSError:
        pass

    code = 200 if result.get("status") == "ok" else 500
    return jsonify(result), code


@app.route("/api/reset", methods=["POST"])
def reset():
    try:
        deleted = reset_collection()
        # 使热更新快照失效，以便下一个周期从头重新摄入
        from tools.hot_ingest import invalidate_snapshot
        invalidate_snapshot()
        return jsonify({"status": "ok", "deleted": deleted})
    except Exception as e:
        logger.error(f"[Reset] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


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

    logger.info(f"[Stream] query: {query[:60]}... session={session_id}")

    def generate():
        import json
        from tools.context_store import append_message
        from sops.base import get_last_recommend
        parts = []
        try:
            for chunk in ask_stream(query, session_id):
                parts.append(chunk)
                # JSON 编码，避免 chunk 内的换行符破坏 SSE 帧格式
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"[Stream] {e}")
            yield f"data: {json.dumps(f'[错误: {e}]', ensure_ascii=False)}\n\n"
        # 记录 assistant 完整回答 + 结构化推荐（供自由指代消解）
        append_message(session_id, "assistant", "".join(parts), models=get_last_recommend(session_id))
        yield "data: [DONE]\n\n"

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
        agent_cfg = load_agent_config()
        rag_cfg = load_rag_config()
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
