"""Flask web app: knowledge base upload & vectorization management.

Run:
  python -m webapp.app          # from project root
  python webapp/app.py          # or directly
Then open http://localhost:5050
"""

import os
import sys

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# Ensure project root and tools/ are on sys.path.
# tools/ is needed because modules inside use bare imports
# (e.g. `from log_tool import`) rather than package-relative imports.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_DIR = os.path.join(_PROJECT_ROOT, "tools")
for _p in (_PROJECT_ROOT, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.agent import ask_stream
from tools.config_tool import load_agent_config, load_rag_config
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

    # Validate extension
    ext = os.path.splitext(upload.filename)[1].lower()
    if not ext.endswith(_supported_exts):
        msg = f"Unsupported file type: {ext}. Supported: {', '.join(_supported_exts)}"
        logger.warning(f"[Ingest] {msg}")
        return jsonify({"status": "error", "message": msg}), 400

    # Save to temp/uploads
    save_path = os.path.join(_upload_dir, upload.filename)
    upload.save(save_path)
    logger.info(f"[Ingest] Saved upload: {save_path}")

    # Ingest into vector store
    result = ingest_file(save_path)

    # Clean up temp file after ingestion
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
        return jsonify({"status": "ok", "deleted": deleted})
    except Exception as e:
        logger.error(f"[Reset] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Stream answer chunks via Server-Sent Events."""
    data = request.get_json(silent=True)
    if not data or "query" not in data:
        return jsonify({"status": "error", "message": "Missing 'query' in JSON body."}), 400
    query = data["query"].strip()
    if not query:
        return jsonify({"status": "error", "message": "Empty query."}), 400

    logger.info(f"[Stream] query: {query[:60]}...")

    def generate():
        try:
            for chunk in ask_stream(query):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.error(f"[Stream] {e}")
            yield f"data: [错误: {e}]\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/ingest/batch", methods=["POST"])
def ingest_batch():
    """Ingest all supported files from the data/ directory at once."""
    data_dir = os.path.join(_PROJECT_ROOT, "data")
    try:
        results = ingest_data_dir(data_dir)
        return jsonify({"status": "ok", "results": results})
    except Exception as e:
        logger.error(f"[BatchIngest] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/config", methods=["GET"])
def agent_config():
    """Return the current agent & retrieval config for the frontend badge."""
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
    app.run(host="0.0.0.0", port=port, debug=False)
