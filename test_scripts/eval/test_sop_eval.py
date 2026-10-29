"""第四轨：SOP 行为轨（确定性状态机，数据集驱动）。

只验证「进入 SOP 之后」的槽位与流程，不判路由、不判检索质量、不调 LLM：
  - 场景与断言在 data/eval/sop/golden.jsonl（一行一个场景；首行 `_meta` 带版本符号）；
  - `_meta.sop_def_ver` = 每个 SOP 的流程结构指纹（sops.base._def_ver）。SOP 流程改了
    指纹就会变——数据集必须同步更新并升 version，否则本轨直接报「数据集与主链路不一致」；
  - 需要外部依赖的路径用 `patches` 打桩（LLM 抽城市 / 故障检索），保证确定性：
    `llm_city` → service_point 的 LLM 兜底返回空；`repair_action` → 故障检索返回固定话术。
运行（不进 --e2e 也会跑，不依赖 Ollama）：
  .venv\\Scripts\\python.exe test_scripts/eval/test_sop_eval.py
"""
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
import sops
import sops.base
from test_scripts import trace_reader
from tools import pending_store, trace_store
from tools.context_store import delete_session

_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_GOLDEN_FILE = os.path.join(_ROOT, "data", "eval", "sop", "golden.jsonl")
_SUPPORTED_V = 1


def _reset(sid):
    sops.end_sop(sid)
    pending_store.close(sid)
    delete_session(sid)


def _models_ready():
    try:
        from tools.metadata_extractor import get_all_models
        return len(get_all_models()) > 0
    except Exception:
        return False


def _turn(sid, fn, query):
    """跑一轮并返回 (reply, flow)；flow 取自 trace 的 steps.sop.flow。"""
    trace_store.begin_turn(sid, query)
    try:
        reply = fn()
    finally:
        trace_store.end_turn()
    row = trace_reader.load(trace_store._path(sid))[-1]
    flow = ((row.get("steps") or {}).get("sop") or {}).get("flow") or []
    return reply, flow


def _reply_of(result):
    return result.reply if result else ""


def _load_dataset():
    if not os.path.isfile(_GOLDEN_FILE):
        return None, []
    with open(_GOLDEN_FILE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    meta = rows[0] if rows and rows[0].get("_meta") else {}
    return meta, [r for r in rows if not r.get("_meta")]


def _apply_patches(names):
    """给依赖外部模型的路径打桩；返回可用于复原的 (对象, 属性, 原值) 列表。"""
    import sops.service_point as sp

    saved = []
    if "llm_city" in names:
        saved.append((sp, "_extract_city_by_llm", sp._extract_city_by_llm))
        sp._extract_city_by_llm = lambda text: []
    if "repair_action" in names:
        for step in (sops.base.SOPS.get("repair") or {}).get("steps") or []:
            if step.get("type") == "action":
                saved.append((step, "action", step["action"]))
                step["action"] = lambda slots: {"answer": "（stub）故障排查结果"}
    return saved


def _restore(saved):
    for obj, attr, value in saved:
        if isinstance(obj, dict):
            obj[attr] = value
        else:
            setattr(obj, attr, value)


def _norm(value):
    return list(value) if isinstance(value, (list, tuple)) else value


def _check_turn(sid, expect, reply, flow, finished=None):
    problems = []
    state = sops.get_sop_state(sid) or {}
    if not state and finished:
        state = finished            # 已完成：槽位看「刚收尾」的那份快照
    active = sops.has_active_sop(sid)
    if "active" in expect and active != expect["active"]:
        problems.append(f"active 应为 {expect['active']}，实际 {active}")
    if "step" in expect and state.get("step") != expect["step"]:
        problems.append(f"step 应为 {expect['step']}，实际 {state.get('step')}")
    if "retry" in expect and int(state.get("retry_count") or 0) != expect["retry"]:
        problems.append(f"retry 应为 {expect['retry']}，实际 {state.get('retry_count')}")
    slots = state.get("slots") or {}
    for key, want in (expect.get("slots") or {}).items():
        got = slots.get(key)
        if _norm(got) != _norm(want):
            problems.append(f"槽位 {key} 应为 {want!r}，实际 {got!r}")
    if "flow" in expect:
        got_flow = [f.get("outcome") for f in flow]
        if got_flow != expect["flow"]:
            problems.append(f"flow 应为 {expect['flow']}，实际 {got_flow}")
    if expect.get("reply_nonempty") and not reply:
        problems.append("回复为空")
    return problems


def _run_case(case):
    """跑一个场景，返回问题清单（空 = 通过）。"""
    sid = "sopeval-" + case["id"].lower()
    _reset(sid)
    saved = _apply_patches(case.get("patches") or [])
    problems = []
    try:
        for i, turn in enumerate(case.get("turns") or []):
            if i == 0:
                entry = case.get("entry", "")
                reply, flow = _turn(
                    sid,
                    lambda: sops.start_sop(sid, case["sop"], entry, slots=case.get("slots")).reply,
                    turn.get("q") or entry)
            else:
                q = turn.get("q") or ""
                reply, flow = _turn(
                    sid, lambda q=q: _reply_of(sops.continue_sop(sid, q)), q)
            problems += _check_turn(sid, turn.get("expect") or {}, reply, flow,
                                    sops.pop_finished(sid) if not sops.has_active_sop(sid) else None)
        final = case.get("final") or {}
        if "active" in final and sops.has_active_sop(sid) != final["active"]:
            problems.append(f"最终 active 应为 {final['active']}")
    finally:
        _restore(saved)
        _reset(sid)
    return problems


def _check_version(meta):
    """数据集版本符号：流程结构（def_ver）与行为定义（trigger_ver）分别核对。"""
    _assert_eq(meta.get("v"), _SUPPORTED_V, f"数据集版本可读（v={meta.get('v')}）")
    cur = {sid: (sop or {}).get("def_ver") for sid, sop in sops.base.SOPS.items()}
    want = meta.get("sop_def_ver") or {}
    stale = {sid: {"数据集": want.get(sid), "当前": cur[sid]} for sid in cur if want.get(sid) != cur[sid]}
    _assert_eq(stale, {}, "SOP 流程结构未变；变了要先更新数据集并升 _meta.version")
    want_trigger = meta.get("trigger_ver")
    if want_trigger:
        from tools.version_store import trigger_ver
        cur_trigger = {sid: trigger_ver(sop) for sid, sop in sops.base.SOPS.items()}
        stale_trigger = {sid: {"数据集": want_trigger.get(sid), "当前": cur_trigger[sid]}
                         for sid in cur_trigger if want_trigger.get(sid) != cur_trigger[sid]}
        _assert_eq(stale_trigger, {}, "SOP 触发词/guards/进入门未变；变了要同步 trigger_ver")


def run():
    reset()
    try:
        from tools.version_store import dataset_warning
        warning = dataset_warning("sop")
        if warning:
            print(f"  [版本] ⚠ {warning}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [版本] ⚠ 版本检查失败：{type(exc).__name__}: {exc}")
    meta, cases = _load_dataset()
    if meta is None:
        _skip(f"未找到 SOP 评测集 {_GOLDEN_FILE}（本地数据，已 gitignore）")
        return stats()
    _check_version(meta)
    _assert(len(cases) > 0, f"评测集加载成功（{len(cases)} 个场景）")
    ids = [c.get("id") for c in cases]
    _assert(len(ids) == len(set(ids)), "场景 id 唯一")

    skipped = 0
    for case in cases:
        if case.get("requires_models") and not _models_ready():
            print(f"  SKIP  {case['id']}（型号数据不可用）")
            skipped += 1
            continue
        problems = _run_case(case)
        _assert(not problems, f"{case['id']}（{case.get('sop')}）"
                + ("：" + "；".join(problems) if problems else ""))
    if skipped:
        print(f"  跳过 {skipped} 个依赖型号数据的场景")
    return stats()


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
