"""行为观测点（`[Behavior]` tag）与评测判据的契约测试：不依赖 LLM / Chroma，纯规则。

覆盖：
  - `_BehaviourCapture` 只收 `[Behavior]` 行、且只留 tag（`[Shadow]` / 其它日志不误收）
  - handler 与主链路挂**同一个 logger 实例**（不受「模块双重导入」影响）
  - 单个 tag → 出口行为的映射覆盖了全部硬行为（硬行为都能从 tag 判出来）
  - `_check_case` 的硬行为四态：tag 命中 / 不一致 / 未捕获 / 未登记 tag
  - 本轨只判行为：`must_contain_any` / `must_not_contain` 不参与判定（2026-09-29）
  - 主链路 trace（`tools/trace_store.py`）：一轮一条记录、稳定核 + 开放层、落在隔离目录、best-effort
"""
import logging
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 与 test_scripts._runner 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
from tools import agent as agent_mod
from test_scripts.eval import test_context_eval as ctx_eval


def _capture(lines):
    """挂上 harness 的采集器，发完给定日志后返回收到的 tag 列表。"""
    capture = ctx_eval._BehaviourCapture()
    agent_mod.logger.addHandler(capture)
    try:
        for line in lines:
            agent_mod.logger.info(line)
    finally:
        agent_mod.logger.removeHandler(capture)
    return capture.tags


def test_capture_only_behavior_lines():
    tags = _capture([
        "[Behavior] carry_safety danger=火星",
        "[Behavior] exit_sop via=zero",
        "[Shadow] chain=kb_search agreed=True",
        "[Guard] danger word hit: 火星",
    ])
    _assert_eq(tags, ["carry_safety", "exit_sop"], "只收 [Behavior] 行，且 detail 不进 tag")


def test_capture_same_logger_instance():
    _assert_true(agent_mod.logger is logging.getLogger("agent"),
                 "handler 与主链路同一 logger 实例（同名 logger，不受双重导入影响）")
    _assert_eq(ctx_eval._BEHAVIOR_PREFIX, "[Behavior] ", "观测行前缀与 tools/agent.py::_log_behavior 一致")


def test_tag_mapping_covers_hard_behaviours():
    mapped = set(ctx_eval._TAG_TO_BEHAVIOUR.values())
    for beh in sorted(ctx_eval._HARD_BEHAVIOURS):
        _assert_in(beh, mapped, f"硬行为 {beh} 有对应 tag（回归门全靠它判）")
    _assert_in("stop_use_safety", ctx_eval._TAG_TO_BEHAVIOUR, "危险拦截 tag 已登记")
    _assert_eq(ctx_eval._TAG_TO_BEHAVIOUR["retrieve_answer"], "answer", "检索作答 → answer")


def test_check_case_hard_by_tag():
    exp = {"behaviour": "safety_carry"}
    _assert_eq(ctx_eval._check_case(exp, ["carry_safety"]), [],
               "tag 命中即通过（不再解析话术）")
    _assert_true(ctx_eval._check_case(exp, ["retrieve_answer"]),
                 "tag 不符 → 报问题")
    _assert_true(ctx_eval._check_case(exp, []),
                 "未捕获 tag → 报问题（出口缺观测点要暴露出来）")
    _assert_true(ctx_eval._check_case(exp, ["brand_new_tag"]),
                 "未登记 tag → 报问题")
    _assert_eq(ctx_eval._check_case({"behaviour": "exit_sop"}, ["exit_sop", "retrieve_answer"]), [],
               "一轮多 tag（退出带诉求）→ 任一命中即通过")
    _assert_eq(ctx_eval._check_case({"behaviour": "safety_carry"}, ["stop_use_safety", "carry_safety"]), [],
               "多 tag 里命中期望行为即通过")


def test_check_case_ignores_text_asserts():
    """本轨只看行为：数据里的 must_contain_any / must_not_contain 不参与判定（2026-09-29）。"""
    exp = {"behaviour": "answer", "must_not_contain": ["没查到"], "must_contain_any": ["滤网"]}
    _assert_eq(ctx_eval._check_case(exp, ["retrieve_answer"]), [],
               "行为一致即通过，哪怕回复撞了 must_not_contain")
    _assert_eq(ctx_eval._check_case({"behaviour": "safety_carry", "must_contain_any": ["火星"]},
                                    ["carry_safety"]), [],
               "行为一致即通过，哪怕没写 must_contain_any 的内容")


def test_check_case_soft_tag_hint():
    """软行为不判失败，但 tag 与期望行为对不上要进复核清单。"""
    _assert_eq(ctx_eval._check_case({"behaviour": "chitchat"}, ["chitchat"]), [],
               "tag 对得上 → 不进复核")
    _assert_true(ctx_eval._check_case({"behaviour": "chitchat"}, ["ask_clarify"]),
                 "tag 对不上 → 进复核")
    _assert_true(ctx_eval._check_case({"behaviour": "slot_filled"}, ["ask_clarify"]),
                 "期望 slot_filled 却停在 ask_clarify（槽位没提取）→ 进复核")
    _assert_eq(ctx_eval._check_case({"behaviour": "slot_filled"}, ["sop_step"]), [],
               "slot_filled 接受 sop_step（dev 映射表）")
    _assert_eq(ctx_eval._check_case({"behaviour": "scope_guard"}, ["refuse_offtopic"]), [],
               "scope_guard 接受 refuse_offtopic（dev 映射表）")
    _assert_true(ctx_eval._check_case({"behaviour": "scope_guard"}, ["chitchat"]),
                 "越界问题了走闲聊 → 进复核")


def test_accepted_tags_cover_mapping():
    """两张表不能漂移：_TAG_TO_BEHAVIOUR 里每个 tag 都要在 _ACCEPTED_TAGS 对应行为里被接受。"""
    for tag, beh in ctx_eval._TAG_TO_BEHAVIOUR.items():
        _assert_in(tag, ctx_eval._ACCEPTED_TAGS.get(beh, set()), f"{tag} 被 {beh} 接受")


def test_disable_shadow_restores():
    """评测隔离：跑用例期间影子探针被替掉，跑完能原样放回（否则污染同进程的后继模块）。"""
    original = agent_mod._shadow_probe
    restore = disable_shadow()
    _assert_true(agent_mod._shadow_probe is not original, "关闭后探针被替换成空操作")
    _assert(agent_mod._shadow_probe("q", "sid", "kb_search") is None, "被替换的探针不做事")
    restore()
    _assert_true(agent_mod._shadow_probe is original, "复原回调把原探针放回去")


def test_isolation_and_import_convention():
    """导入侧的隐性坑：测试隔离目录真的生效、项目模块不再出现裸名实例。"""
    import tools.context_store as tc

    _assert_in("test_context", tc._CONTEXT_DIR.replace("\\", "/"),
               "会话目录指向测试隔离目录 data/test_context")
    for name in ("context_store", "vector_store", "agent", "log_tool", "llm_tool"):
        _assert_true(name not in sys.modules,
                     f"{name} 未以裸名出现在 sys.modules（全项目统一完整导入）")


def test_trace_record_schema():
    """主链路 trace：一轮一条、稳定核 + 开放层、落在隔离目录（评测侧消费它的前提）。"""
    from tools import trace_store

    sid = "tracetest-schema"
    reset_trace(sid)
    trace_store.begin_turn(sid, "测试问题")
    trace_store.step("intent", intent="robot", margin=0.9)
    trace_store.step("retrieve", domain="repair", n_chunks=2,
                     chunks=[{"file": "a.md", "score": 0.5}])
    trace_store.add_ms("retrieve", time.perf_counter() - 0.01)
    trace_store.note_behavior("retrieve_answer", {"hits": 2})
    trace_store.end_turn()
    rec = read_last_turn(sid)

    _assert_true(rec, "一轮跑完落盘了一条记录")
    for field in ("v", "ts", "session_id", "turn_index", "query", "retry", "tag", "tags",
                  "detail", "steps", "ms", "aborted", "error"):
        _assert_in(field, rec, f"稳定核字段 {field} 在")
    _assert_eq(rec.get("tag"), "retrieve_answer", "tag 记最后一次行为")
    _assert_eq(((rec.get("steps") or {}).get("intent") or {}).get("intent"), "robot",
               "环节信息进 steps（加环节 = 加 key，不改结构）")
    _assert_true(isinstance(rec.get("steps"), dict) and isinstance(rec.get("detail"), dict),
                 "开放层（steps / detail）是 dict")
    _assert_true(isinstance((rec.get("ms") or {}).get("total"), int), "带本轮总耗时 ms.total")

    path = trace_store._path(sid).replace("\\", "/")
    _assert_in("trace_test", path, "trace 落在隔离目录（不污染真实 logs/trace/）")
    _assert("data/context" not in path, "trace 不写进 data/context（会被当会话读）")
    _assert_true(path.split("/")[-2].startswith(sid + "_"),
                 "目录名 = 会话 ID + 时间（与 context 命名一致）")
    _assert_true(path.split("/")[-1].startswith(sid + "_") and path.endswith(".jsonl"),
                 "文件名 = 会话 ID + 时间（与 context 命名一致）")

    reset_trace(sid)
    _assert_eq(read_last_turn(sid), {}, "reset_trace 之后读不到（文件与进程内计数都清了）")


def test_trace_best_effort():
    """trace 是旁路：没开轮次时各写入接口都静默返回、不抛异常（不能影响回答）。"""
    from tools import trace_store

    try:
        trace_store.step("intent", intent="x")
        trace_store.note_behavior("chitchat")
        trace_store.add_ms("retrieve", time.perf_counter())
        trace_store.note_error(RuntimeError("probe"))
        trace_store.end_turn()
        ok = True
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("    ", exc)
    _assert(ok, "无轮次时各写入接口不抛异常")


def test_trace_report_flags():
    """逐轮读数器（`test_scripts/trace_report.py`）的启发式标记与行渲染，纯函数直接测。"""
    from test_scripts import trace_report

    _assert_in("aborted 未跑完", trace_report._flags({"aborted": True}), "未跑完的轮会被标出")
    _assert_in("0 召回/兜底", trace_report._flags({"tag": "no_answer_fallback", "steps": {}}),
               "兜底轮会被标出")
    _assert_in("慢 45s", trace_report._flags({"ms": {"total": 45000}}), "超时阈值会标「慢」")
    _assert_eq(trace_report._flags({}), [], "平轮不给标记")
    line = trace_report._turn_lines({"turn_index": 2, "query": "那这个多少钱", "tag": "chitchat",
                                     "steps": {"intent": {"intent": "casual", "margin": 0.9}},
                                     "ms": {"total": 1200}})[0]
    _assert_in("tag=chitchat", line, "每轮行里带 tag")
    _assert_in("意图=casual(0.900)", line, "带意图与 margin")


def test_golden_schema_and_context():
    """评测集自身的自检：一行一条 case、归因取枚举、fixed 语义、上下文可达。

    放这里是为了**免跑 20 分钟才发现写错**（带 --e2e 的实跑很贵）。评测集是本地数据，缺失时跳过。
    """
    if not os.path.isfile(ctx_eval._GOLDEN_FILE):
        return
    rows = ctx_eval._load_golden()
    cases = [c for row in rows for c in row["cases"]]
    _assert_true(rows and cases, "评测集能读出会话与标注（%d 会话 / %d 条）" % (len(rows), len(cases)))
    bad = [(c.get("id"), f) for c in cases for f in ("id", "session_id", "query_index", "query",
                                                      "expect", "verdict") if f not in c]
    _assert_eq(bad, [], "每条 case 都有必需字段（id/session_id/query_index/query/expect/verdict）")

    from config.word_dict_config import FEEDBACK_REASON_KEYS

    bad = [c["id"] for c in cases if (c.get("type") or "") and c["type"] not in FEEDBACK_REASON_KEYS]
    _assert_eq(bad, [], "归因 type 都在 FEEDBACK_REASONS 内（与前端标注同一套 key）")
    bad = [c["id"] for c in cases if c.get("fixed") and c.get("verdict") != "fail"]
    _assert_eq(bad, [], "fixed（曾坏已修）的条目 verdict 都是 fail")
    missing = [row.get("session_id") for row in rows if not ctx_eval._turns_of(row)]
    _assert_eq(missing, [], "每个会话的上下文都取得到（真实会话或评测集自带 sessions/）")


def test_tag_text():
    _assert_eq(ctx_eval._tag_text(["carry_safety"]), "tag=carry_safety", "有 tag → 摘要")
    _assert_in("未捕获", ctx_eval._tag_text([]), "无 tag → 明说未捕获而非留空")


def test_feedback_reason_enum():
    """前端「无用」原因枚举的自检：键唯一、都带标签与归因层、保留兜底类「其他」且在末尾。"""
    from config.word_dict_config import (FEEDBACK_REASONS, FEEDBACK_REASON_KEYS,
                                         FEEDBACK_REASON_LABELS, FEEDBACK_REASON_LAYERS)

    keys = [r["key"] for r in FEEDBACK_REASONS]
    _assert_eq(len(keys), len(set(keys)), "枚举键唯一（存的是键，重复会让统计串味）")
    _assert_true(all(r.get("label") and r.get("layer") for r in FEEDBACK_REASONS),
                 "每条都有中文标签与归因层（标完即知往哪一层修）")
    _assert_eq(FEEDBACK_REASON_KEYS[-1], "other", "兜底类「其他」保留且排在末尾（新增类追加在它之前）")
    _assert_eq(set(FEEDBACK_REASON_LABELS), set(keys), "标签表与枚举一致")
    _assert_eq(set(FEEDBACK_REASON_LAYERS), set(keys), "归因层表与枚举一致")


TESTS = [
    test_capture_only_behavior_lines,
    test_capture_same_logger_instance,
    test_tag_mapping_covers_hard_behaviours,
    test_check_case_hard_by_tag,
    test_check_case_ignores_text_asserts,
    test_check_case_soft_tag_hint,
    test_accepted_tags_cover_mapping,
    test_disable_shadow_restores,
    test_isolation_and_import_convention,
    test_trace_record_schema,
    test_trace_best_effort,
    test_trace_report_flags,
    test_golden_schema_and_context,
    test_tag_text,
    test_feedback_reason_enum,
]


def run():
    reset()
    return run_tests("行为观测点（tag）与 trace 记录契约测试", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
