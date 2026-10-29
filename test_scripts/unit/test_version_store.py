"""版本中台测试：登记、快照、升版与 diff 的确定性契约。"""
import json
import os
import shutil
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
from tools import version_store as vs


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fixture():
    tmp = tempfile.mkdtemp(prefix="version_")
    old_abs, old_rel = vs._abs, vs._rel
    vs._abs = lambda rel: os.path.join(tmp, rel)
    vs._rel = lambda path: os.path.relpath(path, tmp).replace("\\", "/")
    path = os.path.join(tmp, "data/eval/retrieval/golden.jsonl")
    _write(path, [{"id": "R1", "query": "a"}, {"id": "R2", "query": "b"}])
    return tmp, old_abs, old_rel


def test_register_snapshot_and_bump():
    tmp, old_abs, old_rel = _fixture()
    try:
        entry = vs.register_eval("retrieval", note="首版")
        _assert_eq(entry["rev"], 1, "首次登记 rev=1")
        _assert_eq(entry["cases"], 2, "条数按 JSONL 统计")
        _assert(vs._abs(entry["snapshot"]) and os.path.isfile(vs._abs(entry["snapshot"])),
                "登记时生成快照")
        _assert(vs.dataset("retrieval")["registered"], "当前文件可反查")

        _write(vs._abs("data/eval/retrieval/golden.jsonl"),
               [{"id": "R1", "query": "a"}, {"id": "R2", "query": "b"},
                {"id": "R3", "query": "c"}])
        _assert_raises(ValueError, lambda: vs.bump_eval("retrieval", ""), "bump 必须写 note")
        entry2 = vs.bump_eval("retrieval", "新增 R3")
        _assert_eq(entry2["rev"], 2, "升版 rev 连续")
        _assert_eq(entry2["cases"], 3, "新版本条数")
        _assert(os.path.isfile(vs._abs(entry2["snapshot"])), "新版本快照")
    finally:
        vs._abs, vs._rel = old_abs, old_rel
        shutil.rmtree(tmp, ignore_errors=True)


def test_diff_reads_snapshots():
    tmp, old_abs, old_rel = _fixture()
    try:
        vs.register_eval("retrieval", note="首版")
        _write(vs._abs("data/eval/retrieval/golden.jsonl"),
               [{"id": "R1", "query": "a2"}, {"id": "R3", "query": "c"}])
        vs.bump_eval("retrieval", "改 R1、删 R2、加 R3")
        diff = vs.diff_eval("retrieval", 1, 2)
        _assert_eq(diff["added"], ["R3"], "新增 id")
        _assert_eq(diff["removed"], ["R2"], "删除 id")
        _assert_eq(diff["changed"], ["R1"], "期望/字段变更")
    finally:
        vs._abs, vs._rel = old_abs, old_rel
        shutil.rmtree(tmp, ignore_errors=True)


def test_trigger_ver_covers_selection_definition():
    from sops.base import is_brand, is_consulting

    sop = {"trigger": ["推荐"], "weak_trigger": ["预算"], "enter_confirm": "always",
           "guards": [{"check": is_consulting}], "steps": []}
    a = vs._trigger_ver(sop)
    sop["guards"] = [{"check": is_brand}]
    b = vs._trigger_ver(sop)
    _assert(a != b, "guards 变化会改变 trigger_ver（与 def_ver 分离）")


def test_migrate_old_per_id():
    payload = {"per_id": {"R1": True, "R2": False},
               "env": {"code": {"commit": "abc", "dirty": False, "changed": []},
                       "evalset": {"sha256": "deadbeef"}}}
    out, changed = vs.migrate_baseline_payload(payload, "retrieval")
    _assert(changed, "旧格式被识别")
    _assert_eq(out["baseline_v"], 2, "升到 baseline_v=2")
    _assert_eq(out["per_id"]["R1"], {"pass": True, "expect_ref": "legacy"}, "旧 bool 转 pass")
    _assert_eq(out["expect_known"], False, "旧基线期望值未知")
    _assert_eq(out["pinned"]["code"], "code:app@abc", "pinned 从旧 env 推导")


def test_context_can_be_reactivated():
    tmp, old_abs, old_rel = _fixture()
    try:
        _write(vs._abs("data/eval/context/golden.jsonl"), [{"id": "C1", "query": "x"}])
        entry = vs.register_eval("context", note="恢复归档", path="data/eval/context/golden.jsonl")
        _assert_eq(entry["status"], "active", "恢复后是新 active 版本")
        _assert_eq(vs.dataset("context")["registered"], True, "context 可被 resolve/dataset 识别")
        _assert_eq(vs._find_eval("context", entry["sha256"], status="active")["key"], entry["key"],
                   "按 active 反查")
    finally:
        vs._abs, vs._rel = old_abs, old_rel
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_git_is_explicit():
    old = vs._git
    vs._git = lambda *args: None
    try:
        code = vs.collect_code()
        _assert_eq(code["git_available"], False, "无 git 显式标记")
        _assert_eq(code["id"], "code:app@unknown", "无 git 不伪造 commit")
        _assert_raises(RuntimeError, lambda: vs.require_clean(), "干净检查不静默通过")
        _assert(vs.baseline_ready("retrieval"), "正式基线明确拒绝无 git 环境")
    finally:
        vs._git = old


def test_dataset_prefers_active_when_sha_shared():
    """归档与 active 同 sha 时（恢复场景），dataset 要认 active，不能报「已归档」。"""
    tmp, old_abs, old_rel = _fixture()
    try:
        vs.register_eval("retrieval", note="首版")
        vs.archive_eval("retrieval", "暂停")
        entry = vs.register_eval("retrieval", note="恢复同一份内容")
        ds = vs.dataset("retrieval")
        _assert_true(ds["registered"], "同 sha 有 active 时应判 registered")
        _assert_eq(ds["status"], "active", "状态取 active")
        _assert_eq(ds["rev"], entry["rev"], "rev 指向新的 active 版本")
    finally:
        vs._abs, vs._rel = old_abs, old_rel
        shutil.rmtree(tmp, ignore_errors=True)


def test_pin_env_carries_sop_trigger_ver():
    """pinned.env.sops 必须同时带 def 与 trigger，否则触发/门控变化在归因里看不见。"""
    env = vs.collect_env()
    pin_env = vs._pin_env_from_components(env)
    sops = pin_env.get("sops") or {}
    _assert_true(sops, "有 SOP 指纹")
    for sid, item in sops.items():
        _assert_true(isinstance(item, dict) and "def" in item and "trigger" in item,
                     f"{sid} 同时带 def/trigger")


def test_run_manifest_paths_unique():
    """同一秒内两次 begin_run 不能互相覆盖。"""
    tmp, old_abs, old_rel = _fixture()
    try:
        h1 = vs.begin_run("probe")
        h2 = vs.begin_run("probe")
        _assert_true(h1["path"] != h2["path"], "两次 run manifest 路径不同")
    finally:
        vs._abs, vs._rel = old_abs, old_rel
        shutil.rmtree(tmp, ignore_errors=True)


TESTS = [test_register_snapshot_and_bump, test_diff_reads_snapshots,
         test_trigger_ver_covers_selection_definition, test_migrate_old_per_id,
         test_context_can_be_reactivated, test_missing_git_is_explicit,
         test_dataset_prefers_active_when_sha_shared,
         test_pin_env_carries_sop_trigger_ver,
         test_run_manifest_paths_unique]


def run():
    return run_tests("版本中台测试（登记 / 快照 / 升版 / diff）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
