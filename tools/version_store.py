"""评测版本中台：代码/环境指纹、评测集账本、run manifest 与快照。

原则：内容哈希是事实，rev 是人显式声明；缺失显式告警；不接管业务发布。
"""
from __future__ import annotations

import hashlib
import json
import glob
import os
import shutil
import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime

from tools.config_tool import get_data_dir, load_config
from tools.log_tool import get_logger
from tools.path_tool import get_abs_path, get_project_root

logger = get_logger(name="version_store")
ROOT = get_project_root()
VERSIONS_REL = "data/versions"
REGISTRY_REL = "data/versions/registry.json"
RUNS_REL = "data/versions/runs"
EVALS_REL = "data/versions/evals"
SNAPSHOT_REL = "data/versions/exports"
REGISTRY_SCHEMA = 1

TRACKS = ("retrieval", "intent", "sop", "context")
ACTIVE_TRACKS = ("retrieval", "intent", "sop")
EVAL_PATHS = {
    "retrieval": "data/eval/retrieval/golden.jsonl",
    "intent": "data/eval/intent/golden.jsonl",
    "sop": "data/eval/sop/golden.jsonl",
    "context": "data/eval/context/golden.jsonl",
}
_CONFIG_FILES = ("config/rag.yaml", "config/chroma.yaml", "config/agent.yaml",
                 "config/context.yaml")
_PROMPT_DIR = "prompts"
_INTENT_HEAD = "data/bgm_model/classifier_head.pt"
_INTENT_DATASET = "data/datasets/intent_dataset.jsonl"
_ENTRY_KIND = "eval"
_registry_lock = threading.Lock()


def _abs(rel: str) -> str:
    return get_abs_path(rel)


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _sha8(path: str) -> str:
    return _sha256(path)[:8] if os.path.isfile(path) else "missing"


def _digest(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _short(value) -> str:
    return _digest(value)[:8]


def _pairs_digest(pairs) -> str:
    text = "\n".join("%s:%s" % (k, v) for k, v in sorted(pairs))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


_git_error = None


def _git(*args: str) -> str | None:
    global _git_error
    try:
        out = subprocess.run(["git", "-C", ROOT, *args], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=15)
        return out.stdout if out.returncode == 0 else None
    except Exception as exc:  # noqa: BLE001 - git 不可用是显式降级，由调用方落 warning
        if _git_error is None:
            _git_error = "%s: %s" % (type(exc).__name__, exc)
            logger.warning("[Version] git 不可用，代码版本降级为 unknown：%s", _git_error)
        return None


def _status_paths() -> list:
    text = _git("status", "--porcelain", "--untracked-files=all") or ""
    paths = []
    for line in text.splitlines():
        if len(line) <= 3:
            continue
        raw = line[3:].strip()
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        paths.append(raw)
    return sorted(set(paths))


def _dirty_digest(paths: list) -> str:
    """未提交内容摘要：tracked diff + 未跟踪文件内容；文件数变化不够。"""
    parts = [_git("diff", "--binary", "HEAD") or ""]
    for rel in paths:
        path = _abs(rel)
        if os.path.isdir(path):
            for base, _dirs, files in os.walk(path):
                for name in sorted(files):
                    full = os.path.join(base, name)
                    parts.append("%s:%s" % (_rel(full), _sha256(full)))
        elif os.path.isfile(path):
            parts.append("%s:%s" % (rel, _sha256(path)))
        else:
            parts.append("%s:deleted" % rel)
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _prompts_meta() -> dict:
    base = _abs(_PROMPT_DIR)
    if not os.path.isdir(base):
        return {"files": [], "per_file": {}, "fingerprint": "missing"}
    pairs = [("%s/%s" % (_PROMPT_DIR, name), _sha256(os.path.join(base, name)))
             for name in sorted(os.listdir(base))
             if os.path.isfile(os.path.join(base, name))]
    return {"files": [rel for rel, _ in pairs],
            "per_file": {rel: sha[:8] for rel, sha in pairs},
            "fingerprint": _pairs_digest(pairs) if pairs else "empty",
            "algo": "sha1-pairs-v1"}


def _config_meta() -> dict:
    pairs = [(rel, _sha256(_abs(rel))) for rel in _CONFIG_FILES if os.path.isfile(_abs(rel))]
    try:
        rag = load_config("rag") or {}
        ops = (load_config("agent") or {}).get("operations") or {}
        key = {"retrieval.score_threshold": rag["retrieval"]["score_threshold"],
               "retrieval.final_top_k": rag["retrieval"]["final_top_k"],
               "retrieval.domain_margin": rag["retrieval"].get("domain_margin", 0),
               "rerank.enabled": rag["rerank"]["enabled"],
               "rerank.score_threshold": rag["rerank"]["score_threshold"],
               "rerank.candidate_top_k": rag["rerank"]["candidate_top_k"],
               "chunk.chunk_size": rag["chunk"]["chunk_size"],
               "operations.scoring": ops.get("scoring"),
               "operations.floor": ops.get("floor"),
               "operations.gap": ops.get("gap")}
    except Exception:  # noqa: BLE001
        key = {}
    return {"files": [p for p, _ in pairs], "fingerprint": _pairs_digest(pairs),
            "key": key, "algo": "sha1-pairs-v1"}


def _kb_meta() -> dict:
    from tools.hot_ingest import kb_fingerprint, load_kb_metadata, metadata_path, scan_files

    kb_dir = get_data_dir()
    rel_dir = _rel(kb_dir)
    meta = load_kb_metadata()
    if meta:
        return {"dir": rel_dir, "fingerprint": meta.get("fingerprint"),
                "files": len(meta.get("file_list") or {}),
                "ts": meta.get("ts"), "source": _rel(metadata_path()),
                "algo": "kb-metadata-v2"}
    files = scan_files(kb_dir) if os.path.isdir(kb_dir) else {}
    return {"dir": rel_dir, "fingerprint": kb_fingerprint(files), "files": len(files),
            "source": "现场扫描", "algo": "kb-scan-v1"}


def _index_meta() -> dict:
    from tools.hot_ingest import load_kb_metadata, scan_files

    meta = load_kb_metadata()
    if not meta:
        return {"in_sync": None, "note": "无 kb_metadata.jsonl 记录"}
    indexed = {rel: (v or {}).get("sha256") for rel, v in (meta.get("file_list") or {}).items()}
    live = scan_files(get_data_dir())
    added = sorted(set(live) - set(indexed))
    removed = sorted(set(indexed) - set(live))
    changed = sorted(k for k in set(live) & set(indexed) if live[k] != indexed[k])
    return {"in_sync": not (added or removed or changed), "added": added,
            "removed": removed, "changed": changed, "ts": meta.get("ts")}


def _intent_meta() -> dict:
    return {"head": _sha8(_abs(_INTENT_HEAD)), "dataset": _sha8(_abs(_INTENT_DATASET))}


def _trigger_ver(sop: dict) -> str:
    guards = []
    for g in sop.get("guards") or []:
        fn = g.get("check") if isinstance(g, dict) else None
        guards.append(getattr(fn, "__name__", repr(fn)))
    payload = {"trigger": sop.get("trigger") or [], "weak_trigger": sop.get("weak_trigger") or [],
               "guards": guards, "enter_confirm": sop.get("enter_confirm")}
    return _short(payload)


def trigger_ver(sop: dict) -> str:
    """评测用行为定义指纹；与只管续跑兼容的 def_ver 分开。"""
    return _trigger_ver(sop)


def _sops_meta() -> dict:
    from sops.base import SOPS

    out = {}
    for sid in sorted(SOPS):
        sop = SOPS[sid]
        out[sid] = {"def_ver": sop.get("def_ver"), "trigger_ver": _trigger_ver(sop)}
    return out


def _models_meta() -> dict:
    def _get(cfg, *keys):
        cur = cfg
        for key in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    try:
        agent = load_config("agent") or {}
        chroma = load_config("chroma") or {}
        rag = load_config("rag") or {}
    except Exception:  # noqa: BLE001
        return {}
    return {"chat": _get(agent, "llm", "model"), "small": _get(agent, "llm", "small_model"),
            "embedding": _get(chroma, "embedding", "model"),
            "reranker": _get(rag, "rerank", "model")}


def collect_code() -> dict:
    commit_raw = _git("rev-parse", "--short", "HEAD")
    git_available = bool(commit_raw and commit_raw.strip())
    commit = (commit_raw or "unknown").strip()
    changed = _status_paths() if git_available else []
    content = _dirty_digest(changed) if changed else ""
    code = {"commit": commit, "dirty": bool(changed) if git_available else None,
            "changed": changed, "git_available": git_available,
            "content_digest": content[:8] if content else "",
            "prompts": _prompts_meta().get("fingerprint"),
            "deps": _sha8(_abs("requirements.txt"))}
    suffix = "+d%d#%s" % (len(changed), content[:8]) if changed else ""
    code["id"] = "code:app@%s%s" % (commit, suffix)
    return code


def _count_cases(path: str) -> int:
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                count += 1
                continue
            if not (isinstance(row, dict) and row.get("_meta")):
                count += 1
    return count


def collect_eval(track: str, path: str = None) -> dict:
    path = path or EVAL_PATHS.get(track)
    if not path:
        return {"track": track, "id": None, "missing": True}
    full = _abs(path)
    if not os.path.isfile(full):
        return {"track": track, "path": path, "id": None, "missing": True}
    sha = _sha256(full)
    sh = sha[:8]
    entry = _find_eval(track, sh, status="active")
    rev = entry.get("rev") if entry else None
    return {"track": track, "path": _rel(full), "sha256": sh, "cases": _count_cases(full),
            "rev": rev, "registered": bool(entry),
            "id": entry.get("key") if entry else None}


def collect_env() -> dict:
    return {"kb": _kb_meta(), "config": _config_meta(), "prompts": _prompts_meta(),
            "index": _index_meta(), "intent": _intent_meta(), "sops": _sops_meta(),
            "trace": getattr(__import__("tools.trace_store", fromlist=["TRACE_VERSION"]),
                             "TRACE_VERSION", None), "models": _models_meta()}


def _canonical(components: dict) -> str:
    return json.dumps(components, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def resolve() -> dict:
    code = collect_code()
    eval_components = {}
    warnings = []
    if not code.get("git_available"):
        warnings.append("git 不可用：代码版本降级为 unknown，不能判断工作区是否干净")
    for track in ACTIVE_TRACKS:
        info = collect_eval(track)
        eval_components[track] = info.get("id")
        if info.get("missing"):
            warnings.append("%s 评测集缺失" % track)
        elif not info.get("registered"):
            warnings.append("%s 评测集未登记（%s）" % (track, info.get("sha256")))
    eval_components["context"] = None
    ctx = dataset("context")
    if ctx.get("registered"):
        eval_components["context"] = ctx.get("id")
    components = {"code": code, "eval": eval_components, "env": collect_env()}
    digest = hashlib.sha1(_canonical(components).encode("utf-8")).hexdigest()[:8]
    return {"run_id": "run:" + digest, "components": components, "warnings": warnings}


# ── 账本 ─────────────────────────────────────────────────────────

def _registry_path() -> str:
    return _abs(REGISTRY_REL)


def _empty_registry() -> dict:
    return {"schema_v": REGISTRY_SCHEMA, "created_at": _now(), "entries": {}}


def _read_registry(create: bool = False) -> dict:
    path = _registry_path()
    if not os.path.isfile(path):
        data = _empty_registry()
        if create:
            _write_json(path, data)
        return data
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("entries", {})
    return data


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


@contextmanager
def _registry_guard():
    """进程内锁 + 尝试跨进程文件锁；锁不可用时仍保持进程内串行。"""
    with _registry_lock:
        lock_path = _registry_path() + ".lock"
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        fd = None
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            if fd is not None:
                try:
                    if os.name == "nt":
                        import msvcrt
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(fd, fcntl.LOCK_UN)
                except Exception:  # noqa: BLE001
                    pass
                os.close(fd)


def _find_eval(track: str, sha8: str = None, status: str = None) -> dict | None:
    for key, entry in (_read_registry().get("entries") or {}).items():
        if entry.get("kind") != _ENTRY_KIND or entry.get("track") != track:
            continue
        if sha8 and entry.get("sha256") != sha8:
            continue
        if status and entry.get("status") != status:
            continue
        out = dict(entry)
        out["key"] = key
        return out
    return None


def _eval_entry(track: str, rev: int, sha8: str, info: dict, note: str,
                status: str = "active", source_commit: str = "", archive_dir: str = None) -> dict:
    snapshot = "%s/%s/%d/%s" % (EVALS_REL, track, rev, os.path.basename(info["path"]))
    dst = _abs(snapshot)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(_abs(info["path"]), dst)
    entry = {"kind": _ENTRY_KIND, "track": track, "rev": rev, "sha256": sha8,
             "cases": info["cases"], "path": info["path"], "snapshot": snapshot,
             "updated_at": _now(), "status": status,
             "status_history": [{"status": status, "at": _now()}],
             "note": note or "", "source_commit": source_commit}
    if track == "sop":
        meta = _sops_meta()
        entry["sop_def_ver"] = {sid: item.get("def_ver") for sid, item in meta.items()}
        entry["sop_trigger_ver"] = {sid: item.get("trigger_ver") for sid, item in meta.items()}
    if archive_dir:
        entry["archive_dir"] = archive_dir
    return entry


def register_eval(track: str, note: str = "", path: str = None, rev: int = None,
                  status: str = "active", archive_dir: str = None) -> dict:
    if track not in TRACKS:
        raise ValueError("未知评测轨：%s" % track)
    info = collect_eval(track, path or EVAL_PATHS.get(track))
    if info.get("missing"):
        raise FileNotFoundError("评测集不存在：%s" % (path or EVAL_PATHS.get(track)))
    with _registry_guard():
        reg = _read_registry(create=True)
        entries = reg["entries"]
        key = _find_key(entries, track, info["sha256"],
                        status="active" if status == "active" else None)
        if key:
            return dict(entries[key], key=key)
        if rev is None:
            rev = max([e.get("rev", 0) for e in entries.values()
                       if e.get("kind") == _ENTRY_KIND and e.get("track") == track] or [0]) + 1
        if any(e.get("track") == track and e.get("rev") == rev for e in entries.values()):
            raise ValueError("%s rev=%s 已存在" % (track, rev))
        if status == "active":
            # 新 active 顶替旧 active：旧版本保留快照，转 archived 以便 diff/回溯
            for other in entries.values():
                if other.get("track") == track and other.get("status") == "active":
                    other["status"] = "archived"
                    other["updated_at"] = _now()
                    other.setdefault("status_history", []).append(
                        {"status": "archived", "at": _now(), "by": "superseded"})
        commit = (_git("rev-parse", "--short", "HEAD") or "unknown").strip()
        entry = _eval_entry(track, rev, info["sha256"], info, note, status, commit, archive_dir)
        key = "eval:%s@%d#%s" % (track, rev, info["sha256"])
        entries[key] = entry
        _write_json(_registry_path(), reg)
        return dict(entry, key=key)


def _find_key(entries: dict, track: str, sha8: str, status: str = None) -> str | None:
    for key, entry in entries.items():
        if entry.get("kind") == _ENTRY_KIND and entry.get("track") == track \
                and entry.get("sha256") == sha8 \
                and (status is None or entry.get("status") == status):
            return key
    return None


def bump_eval(track: str, note: str) -> dict:
    if not (note or "").strip():
        raise ValueError("bump-eval 必须写 note")
    latest = _latest(track)
    if not latest:
        raise ValueError("%s 没有 active 版本，先 register-eval" % track)
    info = collect_eval(track)
    if info.get("missing"):
        raise FileNotFoundError("评测集不存在：%s" % EVAL_PATHS.get(track))
    if info.get("sha256") == latest.get("sha256"):
        raise ValueError("%s 内容没有变化，不需要 bump" % track)
    return register_eval(track, note=note, rev=int(latest.get("rev", 0)) + 1)


def _latest(track: str) -> dict | None:
    entries = [dict(e, key=k) for k, e in (_read_registry().get("entries") or {}).items()
               if e.get("kind") == _ENTRY_KIND and e.get("track") == track and e.get("status") == "active"]
    return max(entries, key=lambda e: e.get("rev", 0)) if entries else None


def dataset(track: str, path: str = None) -> dict:
    info = collect_eval(track, path or EVAL_PATHS.get(track))
    if info.get("missing"):
        return {"track": track, "registered": False, "status": "missing", "warning": "评测集缺失"}
    entry = _find_eval(track, info["sha256"])
    if not entry:
        return {"track": track, "registered": False, "status": "unregistered",
                "sha256": info["sha256"], "warning": "未登记变更"}
    return {"track": track, "registered": entry.get("status") == "active",
            "status": entry.get("status"), "id": entry.get("key"),
            "rev": entry.get("rev"), "sha256": info["sha256"]}


def dataset_warning(track: str) -> str:
    """给单轨入口用的版本告警；空串表示已登记且 active。"""
    ds = dataset(track)
    if ds.get("registered"):
        return ""
    if ds.get("status") == "missing":
        return "%s 评测集缺失，无法核对版本" % track
    if ds.get("status") == "archived":
        return "%s 评测集已归档（非 active）" % track
    return "%s 评测集有未登记变更（sha=%s）" % (track, ds.get("sha256"))


def archive_eval(track: str, reason: str, archive_dir: str = None) -> dict:
    with _registry_guard():
        reg = _read_registry(create=False)
        keys = [k for k, e in reg.get("entries", {}).items()
                if e.get("track") == track and e.get("status") == "active"]
        if len(keys) != 1:
            raise ValueError("%s 活跃版本数=%d，无法归档" % (track, len(keys)))
        entry = reg["entries"][keys[0]]
        entry["status"] = "archived"
        entry["updated_at"] = _now()
        entry.setdefault("status_history", []).append({"status": "archived", "at": _now()})
        entry["note"] = reason or entry.get("note", "")
        if archive_dir:
            entry["archive_dir"] = archive_dir
        _write_json(_registry_path(), reg)
        return dict(entry, key=keys[0])


def require_clean() -> list:
    """正式基线前置：返回非空表示工作区不干净（含未跟踪文件）。"""
    if not _git("rev-parse", "--short", "HEAD"):
        raise RuntimeError("git 不可用，无法判断工作区是否干净")
    return _status_paths()


def baseline_ready(track: str) -> list:
    """正式基线前置检查：干净工作区 + 当前数据集已登记为 active。"""
    issues = []
    try:
        dirty = require_clean()
    except RuntimeError as exc:
        dirty = []
        issues.append(str(exc))
    if dirty:
        issues.append("工作区未提交：%s" % "、".join(dirty[:8]) + (" …" if len(dirty) > 8 else ""))
    ds = dataset(track)
    if not ds.get("registered"):
        issues.append("%s 数据集未登记/非 active：%s" % (track, ds.get("warning") or ds.get("status")))
    return issues


def _code_id_from_env(env: dict) -> str | None:
    code = env.get("code") or {}
    commit = code.get("commit")
    if not commit:
        return None
    changed = code.get("changed") or []
    suffix = "+d%d" % len(changed) if (code.get("dirty") or changed) else ""
    return "code:app@%s%s" % (commit, suffix)


def _eval_id_from_env(track: str, env: dict) -> str | None:
    ev = env.get("evalset") or {}
    sha = ev.get("sha256") or env.get("golden")
    if not sha:
        return None
    entry = _find_eval(track, str(sha)[:8])
    return entry.get("key") if entry else None


def _pin_env_from_components(env: dict) -> dict:
    def _fp(key):
        value = env.get(key)
        return value.get("fingerprint") if isinstance(value, dict) else value

    sops = {}
    for sid, item in (env.get("sops") or {}).items():
        sops[sid] = item.get("def_ver") if isinstance(item, dict) else item
    out = {"kb": _fp("kb"), "prompts": _fp("prompts"), "config": _fp("config"),
           "index": (env.get("index") or {}).get("in_sync"),
           "intent": env.get("intent"), "sops": sops,
           "trace": env.get("trace"), "models": env.get("models")}
    return out


def _pin_env_from_baseline(env: dict) -> dict:
    out = {}
    for key in ("kb", "prompts", "config", "index", "sops", "trace", "models"):
        value = env.get(key)
        if value is None:
            continue
        if key in ("kb", "prompts", "config") and isinstance(value, dict):
            value = value.get("fingerprint")
        if key == "index" and isinstance(value, dict):
            value = value.get("in_sync")
        if key == "sops":
            value = {sid: item.get("def_ver") if isinstance(item, dict) else item
                     for sid, item in value.items()}
        if value is not None:
            out[key] = value
    if env.get("head") or env.get("dataset"):
        out["intent"] = {"head": env.get("head"), "dataset": env.get("dataset")}
    return out


def pin_from_env(track: str, env: dict) -> dict:
    """从旧基线 env 生成 pinned（不重跑、不冒充当前代码）；未记录组件留空。"""
    code_id = _code_id_from_env(env)
    eval_id = _eval_id_from_env(track, env)
    eval_map = {track: eval_id} if eval_id else {}
    env_pin = _pin_env_from_baseline(env)
    run_digest = hashlib.sha1(_canonical({"code": code_id, "eval": eval_map, "env": env_pin})
                               .encode("utf-8")).hexdigest()[:8]
    return {"run": "run:" + run_digest, "code": code_id, "eval": eval_map, "env": env_pin}


def pin(track: str) -> dict:
    """当前 clean/dirty 运行的 pinned 块；写正式基线前已由 baseline_ready 拦截 dirty。"""
    resolved = resolve()
    comp = resolved["components"]
    return {"run": resolved["run_id"], "code": comp["code"]["id"],
            "eval": {track: comp["eval"].get(track)},
            "env": _pin_env_from_components(comp["env"])}


def migrate_baseline_payload(payload: dict, track: str) -> tuple[dict, bool]:
    """把旧 per_id 迁到 baseline_v:2；旧期望值不可补算，标记 expect_known=false。"""
    out = dict(payload or {})
    per_id = out.get("per_id") or {}
    changed = False
    migrated = {}
    known = True
    for item_id, value in per_id.items():
        if isinstance(value, dict):
            migrated[item_id] = value
            if value.get("expect_ref") == "legacy" or ("expect" in value and value.get("expect") is None):
                known = False
            continue
        changed = True
        if track in ("retrieval", "context"):
            migrated[item_id] = {"pass": bool(value), "expect_ref": "legacy"}
        else:
            migrated[item_id] = {"got": value, "expect": None}
        known = False
    out["per_id"] = migrated
    out["baseline_v"] = 2
    out["expect_known"] = bool(known and per_id)
    out["pinned"] = pin_from_env(track, out.get("env") or {})
    return out, changed


def check(strict: bool = False) -> list:
    issues = []
    reg = _read_registry()
    entries = reg.get("entries") or {}
    for key, entry in entries.items():
        path = _abs(entry.get("path") or "")
        if not path or not os.path.isfile(path):
            issues.append("%s 文件不存在：%s" % (key, entry.get("path")))
            continue
        check_path = path
        if entry.get("status") != "active" and entry.get("snapshot"):
            check_path = _abs(entry["snapshot"])
        if os.path.isfile(check_path) and _sha256(check_path)[:8] != entry.get("sha256"):
            issues.append("%s 文件 sha 与账本不一致" % key)
        snap = entry.get("snapshot")
        if snap and not os.path.isfile(_abs(snap)):
            issues.append("%s 缺少快照：%s" % (key, snap))
        if entry.get("status") == "active":
            p = entry.get("path")
            if not p:
                issues.append("%s 缺少 path" % key)
    active_tracks = list(ACTIVE_TRACKS)
    if os.path.isfile(_abs(EVAL_PATHS.get("context", ""))):
        active_tracks.append("context")
    for track in active_tracks:
        ds = dataset(track)
        if not ds.get("registered") and strict:
            issues.append("%s 未登记：%s" % (track, ds.get("warning", ds.get("status"))))
    active = {}
    for key, entry in entries.items():
        if entry.get("status") == "active":
            active.setdefault(entry.get("track"), []).append(key)
    for track, keys in active.items():
        if len(keys) > 1:
            issues.append("%s 同时有多个 active：%s" % (track, ", ".join(keys)))

    # 基线 pinned 必须指向账本里真实存在的版本
    baseline_paths = ["data/eval/retrieval/baseline.json", "data/eval/intent/baseline.json"]
    baseline_paths += glob.glob("data/eval/context/archive/*/baseline.json")
    for rel in baseline_paths:
        path = _abs(rel)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            issues.append("%s 基线不可读" % rel)
            continue
        pinned = payload.get("pinned")
        if not pinned:
            issues.append("%s 缺 pinned（未迁移 v2）" % rel)
            continue
        for track, eval_id in (pinned.get("eval") or {}).items():
            if eval_id and eval_id not in entries:
                issues.append("%s 的 pinned.%s 不在账本：%s" % (rel, track, eval_id))

    # SOP 数据集 _meta 与当前定义 / 账本三方对齐
    sop_path = _abs(EVAL_PATHS["sop"])
    if os.path.isfile(sop_path):
        try:
            with open(sop_path, encoding="utf-8") as f:
                meta = json.loads(f.readline() or "{}")
        except (OSError, ValueError):
            meta = {}
        cur = _sops_meta()
        for field, key in (("sop_def_ver", "def_ver"), ("trigger_ver", "trigger_ver")):
            want = meta.get(field) or {}
            actual = {sid: item.get(key) for sid, item in cur.items()}
            for sid, val in actual.items():
                if want.get(sid) != val:
                    issues.append("SOP %s %s 与当前定义不一致" % (sid, field))
        sop_entry = _latest("sop")
        if sop_entry:
            for field, key in (("sop_def_ver", "def_ver"), ("sop_trigger_ver", "trigger_ver")):
                if not sop_entry.get(field):
                    issues.append("SOP 账本条目缺 %s" % field)
    return issues


def check_warnings() -> list:
    """非 strict 模式的未登记/暂停提示；不混入硬错误。"""
    out = []
    if not collect_code().get("git_available"):
        out.append("git 不可用：代码版本降级为 unknown")
    active_tracks = list(ACTIVE_TRACKS)
    if os.path.isfile(_abs(EVAL_PATHS.get("context", ""))):
        active_tracks.append("context")
    out.extend(msg for track in active_tracks for msg in [dataset_warning(track)] if msg)
    return out


def diff_eval(track: str, rev_a: int, rev_b: int) -> dict:
    def _snapshot(rev):
        rel = "%s/%s/%d" % (EVALS_REL, track, rev)
        dir_path = _abs(rel)
        if not os.path.isdir(dir_path):
            raise FileNotFoundError("缺少 %s 快照" % rel)
        names = [n for n in os.listdir(dir_path) if n.endswith(".jsonl")]
        if not names:
            raise FileNotFoundError("快照没有 jsonl：%s" % rel)
        return os.path.join(dir_path, sorted(names)[0])

    def _rows(path):
        out = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and not row.get("_meta") and row.get("id"):
                    out[row["id"]] = row
        return out

    a, b = _rows(_snapshot(rev_a)), _rows(_snapshot(rev_b))
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = sorted(i for i in set(a) & set(b) if _digest(a[i]) != _digest(b[i]))
    return {"track": track, "rev_a": rev_a, "rev_b": rev_b,
            "added": added, "removed": removed, "changed": changed}


def compare_pinned(pinned: dict, resolved: dict = None) -> dict:
    """基线 pinned 与当前运行的组件级归因。"""
    current = resolved or resolve()
    comp = current["components"]
    cur_code = comp["code"]["id"]
    cur_eval = comp["eval"]
    cur_env = _pin_env_from_components(comp["env"])
    base_code = (pinned or {}).get("code")
    code_recorded = bool(base_code)
    base_eval = (pinned or {}).get("eval") or {}
    base_env = (pinned or {}).get("env") or {}
    env_changed = {}
    for key, base_value in base_env.items():
        if base_value is not None and base_value != cur_env.get(key):
            env_changed[key] = {"base": base_value, "current": cur_env.get(key)}
    eval_changed = {track: {"base": base_eval.get(track), "current": cur_eval.get(track)}
                    for track in set(base_eval) & set(cur_eval)
                    if base_eval.get(track) and base_eval.get(track) != cur_eval.get(track)}
    return {"code_recorded": code_recorded,
            "code_changed": (base_code != cur_code) if code_recorded else None,
            "code": {"base": base_code, "current": cur_code},
            "eval_changed": eval_changed, "env_changed": env_changed,
            "current": current}


def export(out: str = "data/versions/exports/version_snapshot.json") -> str:
    path = _abs(out)
    payload = {"exported_at": _now(), "registry": _read_registry(), "resolve": resolve()}
    _write_json(path, payload)
    return path


# ── run manifest ──────────────────────────────────────────────────

def begin_run(entry: str, resolved: dict = None) -> dict:
    resolved = resolved or resolve()
    started = _now()
    rec = {"run_id": resolved["run_id"], "status": "running", "started_at": started,
           "entry": entry, "code": resolved["components"]["code"],
           "eval": resolved["components"]["eval"], "env": resolved["components"]["env"],
           "warnings": resolved.get("warnings") or []}
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    path = _abs("%s/%s_%s.json" % (RUNS_REL, stamp, resolved["run_id"].split(":", 1)[-1]))
    _write_json(path, rec)
    return {"record": rec, "path": path}


def finish_run(handle: dict, status: str) -> str:
    rec = handle["record"]
    rec["status"] = status
    rec["finished_at"] = _now()
    _write_json(handle["path"], rec)
    return handle["path"]
