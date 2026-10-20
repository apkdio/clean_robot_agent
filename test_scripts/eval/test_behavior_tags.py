"""行为闭集（trace 的 branch）与评测判据的契约测试：不依赖 LLM / Chroma，纯规则。

覆盖：
  - 闭集只有一份：harness 的行为闭集取自 config/word_dict_config.py::BEHAVIOR_BY_TAG，无副本
  - 闭集覆盖全部硬行为（硬行为都能从 branch 判出来）
  - _check_case 的硬行为四态：branch 命中 / 不一致 / 未记（分「无观测」与「tag 未登记」）/ 闭集外
  - 本轨只判行为：must_contain_any / must_not_contain 不参与判定
  - 评测集里每条期望行为都在闭集内（或属两个无专属 branch 的软行为期望）
  - 主链路 trace（tools/trace_store.py）：一轮一条记录、分层块 + 兼容投影、落在隔离目录、best-effort
"""
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 与 test_scripts._runner 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
from tools import agent as agent_mod
from test_scripts.eval import test_context_eval as ctx_eval


def test_branch_closed_set_from_config():
    """出口闭集只有一份（config），harness 不自带副本——两处各存一张表迟早漂移。"""
    from config.word_dict_config import BEHAVIOR_BY_TAG

    _assert_eq(ctx_eval._BEHAVIOURS, set(BEHAVIOR_BY_TAG.values()),
               "行为闭集取自 BEHAVIOR_BY_TAG（记录侧与评测轨同一份）")
    _assert_true(not hasattr(ctx_eval, "_TAG_TO_BEHAVIOUR"),
                 "tag → 行为的副本已删净（判定不再反推 tag）")
    _assert_true(not hasattr(ctx_eval, "_BehaviourCapture"),
                 "日志行采集器已删净（判定读 trace 的 branch）")
    _assert_eq(set(ctx_eval._ACCEPTED_BRANCHES) - ctx_eval._BEHAVIOURS,
               ctx_eval._SOFT_BEHAVIOURS - ctx_eval._BEHAVIOURS,
               "接受表只多出两个没有专属 branch 的软行为期望（scope_guard / slot_filled）")


def test_branch_covers_behaviours():
    """硬行为都能从 branch 判出来；软行为的每个期望都有折算入口。"""
    from config.word_dict_config import BEHAVIOR_BY_TAG

    for beh in sorted(ctx_eval._HARD_BEHAVIOURS):
        _assert_in(beh, ctx_eval._BEHAVIOURS, f"硬行为 {beh} 在闭集内（回归门全靠它判）")
    _assert_in("stop_use_safety", BEHAVIOR_BY_TAG, "危险拦截 tag 已登记")
    _assert_eq(BEHAVIOR_BY_TAG["retrieve_answer"], "answer", "检索作答 → answer")
    _assert_eq(BEHAVIOR_BY_TAG["carry_safety"], "safety_carry", "安全承接 → safety_carry")
    for beh in sorted(ctx_eval._SOFT_BEHAVIOURS):
        _assert_true(ctx_eval._ACCEPTED_BRANCHES.get(beh), f"软行为 {beh} 有可接受的 branch（复核提示用）")


def test_check_case_hard_by_branch():
    exp = {"behaviour": "safety_carry"}
    _assert_eq(ctx_eval._check_case(exp, "safety_carry"), [], "branch 命中即通过（不再解析话术）")
    _assert_true(ctx_eval._check_case(exp, "answer"), "branch 不符 → 报问题")
    _assert_true(ctx_eval._check_case(exp, None), "未记 branch → 报问题（出口缺观测点要暴露出来）")
    _assert_true(ctx_eval._check_case(exp, None, "no_answer_fallback"),
                 "有 tag 却折不出 branch → 报问题（该 tag 未登记）")
    _assert_true(ctx_eval._check_case(exp, "brand_new_branch"), "闭集外的 branch → 报问题")
    _assert_eq(ctx_eval._check_case({"behaviour": "exit_sop"}, "exit_sop"), [], "流程类出口一一对应")
    _assert_true(ctx_eval._check_case({"behaviour": "exit_sop"}, "answer"),
                 "多 tag 的轮以唯一出口（最后那个）为准，不再「任一命中即算对」")


def test_check_case_ignores_text_asserts():
    """本轨只看行为：数据里的 must_contain_any / must_not_contain 不参与判定。"""
    exp = {"behaviour": "answer", "must_not_contain": ["没查到"], "must_contain_any": ["滤网"]}
    _assert_eq(ctx_eval._check_case(exp, "answer"), [],
               "行为一致即通过，哪怕回复撞了 must_not_contain")
    _assert_eq(ctx_eval._check_case({"behaviour": "safety_carry", "must_contain_any": ["火星"]},
                                    "safety_carry"), [],
               "行为一致即通过，哪怕没写 must_contain_any 的内容")


def test_check_case_soft_branch_hint():
    """软行为不判失败，但 branch 与期望行为对不上要进复核清单。"""
    _assert_eq(ctx_eval._check_case({"behaviour": "chitchat"}, "chitchat"), [],
               "branch 对得上 → 不进复核")
    _assert_true(ctx_eval._check_case({"behaviour": "chitchat"}, "no_answer_fallback"),
                 "对不上 → 进复核")
    _assert_true(ctx_eval._check_case({"behaviour": "slot_filled"}, "clarify"),
                 "期望 slot_filled 却停在 clarify（槽位没提取）→ 进复核")
    _assert_eq(ctx_eval._check_case({"behaviour": "slot_filled"}, "sop_step"), [],
               "slot_filled 接受 sop_step（流程内推进）")
    _assert_eq(ctx_eval._check_case({"behaviour": "slot_filled"}, "structured"), [],
               "slot_filled 也接受 structured（预算直出）")
    _assert_eq(ctx_eval._check_case({"behaviour": "scope_guard"}, "refuse"), [],
               "scope_guard 接受 refuse（越界由拒答分支接住）")
    _assert_true(ctx_eval._check_case({"behaviour": "scope_guard"}, "chitchat"),
                 "越界问题了走闲聊 → 进复核")
    _assert_true(ctx_eval._check_case({"behaviour": "clarify"}, None),
                 "软行为也没记出口 → 同样进复核（不静默当通过）")


def test_accepted_branches_cover_mapping():
    """闭集与接受表不能漂移：映射表里每个出口都要被它对应的行为接受（硬行为是相等比较）。"""
    from config.word_dict_config import BEHAVIOR_BY_TAG

    for tag, beh in BEHAVIOR_BY_TAG.items():
        _assert_in(beh, ctx_eval._ACCEPTED_BRANCHES.get(beh, set()), f"{tag} → {beh} 自我接受")


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
    """逐轮读数器（test_scripts/trace_report.py）的启发式标记与行渲染，纯函数直接测。"""
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

    放这里是为了免跑 20 分钟才发现写错（带 --e2e 的实跑很贵）。评测集是本地数据，缺失时跳过。
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
    outside = [c["id"] for c in cases
               if c["expect"].get("behaviour")
               not in (ctx_eval._BEHAVIOURS | ctx_eval._SOFT_BEHAVIOURS)]
    _assert_eq(outside, [], "每条期望行为都在闭集内（或属两个无专属 branch 的软行为期望）")


def test_branch_text():
    _assert_eq(ctx_eval._branch_text("carry_safety"), "branch=carry_safety", "有 branch → 摘要")
    _assert_in("未记", ctx_eval._branch_text(None), "没记 branch → 明说未记而非留空")


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


def test_trace_nesting():
    """分层块：记录与执行同构——节点有序可查、闭集受控、只存事实、两遍召回能读出来。"""
    from tools import trace_store
    from config.word_dict_config import BEHAVIOR_BY_TAG, TRACE_NODE_LABELS, TRACE_NODE_STATUSES

    def run(sid, fill, query="测试问题", reply="好的"):
        reset_trace(sid)
        trace_store.begin_turn(sid, query)
        fill()
        trace_store.end_turn(reply=reply)
        return read_last_turn(sid)

    def fill_answer():
        trace_store.route("inject", hit=False)
        trace_store.step("intent", intent="robot", margin=0.98,
                         intent_retry={"label": "robot", "margin": 0.5, "adopted": True})
        trace_store.step("rewrite", normalize={"changed": True, "text": "支持以旧换新吗？"})
        trace_store.step("rewrite", via="none", effective_query="支持以旧换新吗")
        trace_store.retrieval(
            passes=[{"pass": 1, "dense": 3, "sparse": 5, "fused": 5,
                     "rerank": {"in": 5, "out": 2, "degraded": False}}],
            twice_retrieval={"active": False, "kept": 0}, source="first",
            merge={"dense_only": 1, "sparse_only": 0, "both": 1, "final": 2})
        trace_store.add_ms("retrieve", time.perf_counter() - 0.01)
        trace_store.step("retrieve", domain="售后服务.txt", via="top1", n_chunks=2)
        trace_store.step("generate", model="qwen2.5:7b")
        trace_store.note_behavior("retrieve_answer", {"hits": 2})

    rec = run("tracetest-layer-a", fill_answer)
    for block in ("route", "path", "retrieval", "sop", "sop_gate", "shadow", "answer"):
        _assert_in(block, rec, f"分层块「{block}」在（读侧按存在性取）")
    _assert_true(isinstance(rec["path"], list) and isinstance(rec["route"], dict),
                 "route 是 dict、path 是有序列表")
    _assert_eq(rec["branch"], "answer", "branch = tag 折算后的唯一出口")
    _assert_eq(rec["answer"], {"kind": "answer", "model": "qwen2.5:7b", "chars": 2},
               "answer 汇总出口类型 / 模型 / 字数")
    _assert_eq([n["node"] for n in rec["path"]][:3],
               ["route.inject", "route.intent", "rewrite"],
               "path 按执行顺序（不再靠 dict 插入序）")
    retrieve_node = next(n for n in rec["path"] if n["node"] == "retrieve")
    _assert_true(retrieve_node.get("ms", 0) >= 10, "add_ms 的耗时落到对应节点自己身上")
    _assert_eq((rec["route"].get("intent") or {}).get("label"), "robot", "route.intent 记分类结果")
    _assert_eq((rec["steps"]["intent"] or {}).get("intent"), "robot", "投影 steps.intent 与分层块等价")
    _assert_eq((rec["steps"]["retrieve"] or {}).get("n_chunks"), 2, "投影 steps.retrieve 仍在")
    rw = (rec["retrieval"] or {}).get("rewrite") or {}
    _assert_eq((rw.get("normalize") or {}).get("changed"), True,
               "归一化与上下文改写共用一个 rewrite 块（先记 normalize）")
    _assert_eq(rw.get("via"), "none", "上下文改写的 via 没被归一化那次调用盖掉")
    _assert_eq([n["node"] for n in rec["path"]].count("rewrite"), 1,
               "rewrite 路径节点只落一个（同一步调两次不能落两个节点）")
    _assert_eq(((rec["route"].get("intent") or {}).get("retry") or {}).get("label"), "robot",
               "route.intent.retry 记两遍分类的第二遍")

    for node_rec in rec["path"]:
        _assert_in(node_rec["node"], TRACE_NODE_LABELS, f"节点 {node_rec['node']} 已登记")
        _assert_in(node_rec["status"], TRACE_NODE_STATUSES, f"状态 {node_rec['status']} 在闭集内")
    _assert_in(rec["branch"], set(BEHAVIOR_BY_TAG.values()), "branch 复用行为闭集（不另造词表）")

    def strings(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from strings(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from strings(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            yield path, obj

    nested = {k: rec[k] for k in ("route", "path", "retrieval", "sop", "sop_gate", "shadow", "answer")}
    prose = [(p, s[:20]) for p, s in strings(nested)
             if len(s) > 200 and not p.endswith((".query", ".effective_query"))]
    _assert_eq(prose, [], "分层块里没有长文本：散文不进记录，人话由读侧渲染")

    def fill_twice():
        trace_store.step("intent", intent="robot", margin=0.7)
        trace_store.retrieval(
            passes=[{"pass": 1, "dense": 2, "sparse": 6, "fused": 6,
                     "rerank": {"in": 6, "out": 6, "degraded": False}},
                    {"pass": 2, "dense": 5, "sparse": 9, "fused": 9,
                     "rerank": {"in": 9, "out": 3, "degraded": False}}],
            twice_retrieval={"active": True, "kept": 1, "trigger": {"top1": 0.183, "threshold": 0.2}},
            merge={"dense_only": 0, "sparse_only": 1, "both": 1, "final": 2})
        trace_store.step("retrieve", domain="售后服务.txt", via="top1", n_chunks=2)
        trace_store.note_behavior("retrieve_answer", {"hits": 2})

    ret = run("tracetest-layer-b", fill_twice, reply="保修两年")["retrieval"]
    _assert_eq(ret["twice_retrieval"]["active"], len(ret["passes"]) == 2,
               "twice_retrieval.active 与 passes 条数一致")
    _assert_true(ret["twice_retrieval"]["trigger"]["top1"] < ret["twice_retrieval"]["trigger"]["threshold"],
                 "触发值成对出现（top1 < threshold 才是它触发的事实依据）")
    mix = ret["merge"]
    _assert_eq(mix["dense_only"] + mix["sparse_only"] + mix["both"], mix["final"],
               "merge 分布之和 = final（最终候选的来源计数）")

    rec3 = run("tracetest-layer-c", lambda: trace_store.note_behavior("chitchat"),
               query="你是谁", reply="我是无尘")
    _assert_eq(rec3["path"][-1], {"node": "retrieve", "status": "skipped", "by": "chitchat"},
               "没检索的轮补一条 skipped 节点（谁让它跳过的）")
    for sid in ("tracetest-layer-a", "tracetest-layer-b", "tracetest-layer-c"):
        reset_trace(sid)


def test_feedback_one_per_turn():
    """同一轮只认一条标注：feedback_of 能按轮次查到已有标注（重复提交据此拒绝）。"""
    from tools import trace_store

    sid = "tracetest-feedback"
    reset_trace(sid)
    trace_store.begin_turn(sid, "标注去重")
    trace_store.note_behavior("retrieve_answer", {"hits": 1})
    trace_store.end_turn(reply="回复")
    rec = read_last_turn(sid)
    day, turn = rec["ts"][:10], rec["turn_index"]

    _assert_eq(trace_store.feedback_of(sid, day, turn), {}, "没标注时查不到")
    trace_store.feedback_turn({"session_id": sid, "date": day, "turn_index": turn,
                               "query": "标注去重", "tag": "retrieve_answer"}, rating="useful")
    _assert_eq((trace_store.feedback_of(sid, day, turn) or {}).get("rating"), "useful",
               "标了之后按轮次查得到（重复提交据此拒绝）")
    _assert_eq(trace_store.feedback_of(sid, day, turn + 1), {}, "别的轮次不受影响")
    _assert_eq((trace_store.find_turn_by_index(sid, day, turn) or {}).get("query"), "标注去重",
               "按锚点（日期 + 轮次）直接取到该轮，不用 query 猜")
    _assert_eq(trace_store.find_turn_by_index(sid, day, turn + 99), {}, "锚点对不上就没有")
    _assert_eq(trace_store.turn_anchor(), {}, "不在轮次里时锚点为空")
    trace_store.feedback_turn({"session_id": sid, "date": day, "turn_index": turn,
                               "query": "标注去重", "tag": "retrieve_answer"},
                              rating="useless", reason="wrong_route")
    eff = trace_store.effective_feedbacks(sid, day)
    _assert_eq((eff.get(turn) or {}).get("rating"), "useless",
               "同一轮多条标注时取最后一条（改过以后写的为准）")
    reset_trace(sid)


def test_context_sets_defined():
    """上下文轨子集的自检：集合非空、选出的 id 都在评测集里、smoke 与「必过」同口径。"""
    if not os.path.isfile(ctx_eval._GOLDEN_FILE):
        return
    rows = ctx_eval._load_golden()
    all_ids = {c["id"] for row in rows for c in row["cases"]}
    for name in sorted(ctx_eval._SETS):
        sub, desc = ctx_eval._select_sets(rows, [name])
        ids = [c["id"] for row in sub for c in row["cases"]]
        _assert_true(ids, f"集合 {name} 非空（{desc}）")
        _assert_eq(set(ids) - all_ids, set(), f"集合 {name} 选出的 id 都在评测集里")
    must = sorted(c["id"] for row in rows for c in row["cases"] if ctx_eval._is_must_pass_case(c))
    smoke, _d = ctx_eval._select_sets(rows, ["smoke"])
    _assert_eq(sorted(c["id"] for row in smoke for c in row["cases"]), must,
               "smoke 就是必过集合（与 _is_must_pass_case 同一口径）")


def test_context_set_filter_and_cost():
    """子集过滤与成本估算：id 前缀、未知集合报错、回放轮数不少于用例数、fast 确实更省。"""
    if not os.path.isfile(ctx_eval._GOLDEN_FILE):
        return
    rows = ctx_eval._load_golden()
    sub, desc = ctx_eval._select_sets(rows, [], ["CTX-20261015-*"])
    _assert_eq(len([c for r in sub for c in r["cases"]]), 3, "id 前缀 `CTX-20261015-*` 命中 3 条")
    _assert_in("3/", desc, "说明里带「命中/总数」")
    _assert_raises(ValueError, lambda: ctx_eval._select_sets(rows, ["nope"]), "未知集合要报错")
    n_cases = sum(len(r["cases"]) for r in rows)
    _assert_true(ctx_eval._replay_turns(rows) >= n_cases, "回放轮数不少于用例数（每条至少跑一轮）")
    quick, _d = ctx_eval._select_sets(rows, ["fast"])
    _assert_true(ctx_eval._replay_turns(quick) < ctx_eval._replay_turns(rows),
                 "fast 的回放成本低于全量")


def test_emitted_tags_and_nodes_are_registered():
    """静态登记核查：源码里实际写出的行为 tag / trace 节点与状态，必须在闭集表里登记。

    读写侧都容忍未登记的词（读侧回落原名、评测只报「tag 未登记」），所以要等运行时踩到才发现；
    这里直接扫 tools/ sops/ function_tools/ 的字面量——新增埋点忘了登记会当场红。
    """
    import re

    from config.word_dict_config import BEHAVIOR_BY_TAG, TRACE_NODE_LABELS, TRACE_NODE_STATUSES

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = {}
    for sub in ("tools", "sops", "function_tools"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".py"):
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    src[os.path.join(sub, fn)] = f.read()

    tags, nodes, statuses = set(), set(), set()
    for text in src.values():
        tags |= set(re.findall(r'_log_behavior\("([a-z_]+)"', text))
        nodes |= set(re.findall(r'trace_store\.node\("([a-z._]+)"', text))
        for line in text.splitlines():          # 状态与节点写在同一条调用里（本项目是一行）
            if "trace_store.node(" in line:
                statuses |= set(re.findall(r'status="([a-z]+)"', line))

    _assert_true(tags and nodes, "扫到了实际发出的 tag 与节点（否则是扫描逻辑失效，不是真没问题）")
    _assert_eq(sorted(tags - set(BEHAVIOR_BY_TAG)), [], "发出的行为 tag 都在 BEHAVIOR_BY_TAG 里登记")
    _assert_eq(sorted(nodes - set(TRACE_NODE_LABELS)), [], "写入的 trace 节点都在 TRACE_NODE_LABELS 里登记")
    _assert_eq(sorted(statuses - set(TRACE_NODE_STATUSES)), [], "写入的节点状态都在 TRACE_NODE_STATUSES 里")


def test_step_keys_registered():
    """每个 step("xxx") 都必须在 trace_store._MIRRORED_STEPS 里登记。

    没登记时只有 v2 投影（steps）有值，path 与 v3 分层块会静默缺失——扫调用点兜住。
    """
    import ast
    import pathlib

    from tools.trace_store import _MIRRORED_STEPS

    root = pathlib.Path(__file__).resolve().parents[2]
    found = {}
    for sub in ("tools", "sops", "webapp"):
        for py in (root / sub).rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                arg = node.args[0]
                if name == "step" and isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.setdefault(arg.value, str(py.relative_to(root)))
    missing = {k: v for k, v in found.items() if k not in _MIRRORED_STEPS}
    _assert_eq(missing, {}, "step 调用点全部已登记 mirror 分支")
    _assert(_MIRRORED_STEPS <= set(found) | {"sop_gate"},
            "登记表里没有僵尸键（sop_gate 只在 agent 侧条件调用）")


def test_route_nodes_written_once():
    """V10 契约：`route()` 自动落节点后，同一轮里每个 `route.*` 节点只能出现一次。

    inject/danger/emotion 由 `route()` 落，intent/ops 由 `_mirror_step` 落（`_ROUTE_MIRRORED` 例外）——
    任何一边重复写，路径读起来就像"判了两次"。以前靠人守，这里做机器检查。
    """
    from config.word_dict_config import TRACE_NODE_STATUSES
    from tools import trace_store

    sid = "tracetest-route-once"
    reset_trace(sid)
    trace_store.begin_turn(sid, "路由节点唯一性")
    trace_store.route("inject", hit=True)
    trace_store.route("danger", hit=False)
    trace_store.route("emotion", hit=True)
    trace_store.route("ops", veto="none")
    trace_store.step("intent", intent="robot", margin=0.9)
    trace_store.note_behavior("block_inject")
    trace_store.end_turn()
    rec = read_last_turn(sid)
    path = rec.get("path") or []
    nodes = [n.get("node") for n in path if str(n.get("node", "")).startswith("route.")]
    _assert_eq(len(nodes), len(set(nodes)), f"每个 route.* 节点只出现一次（实际 {nodes}）")
    for name in ("route.inject", "route.danger", "route.emotion", "route.intent"):
        _assert_in(name, nodes, f"{name} 落进 path")
    by = {n.get("node"): n for n in path}
    _assert_eq(by["route.emotion"].get("status"), "hit", "命中时 status=hit")
    _assert_eq(by["route.danger"].get("status"), "ok", "未命中时 status=ok")
    for name in ("route.inject", "route.danger", "route.emotion", "route.intent"):
        _assert_in(by[name].get("status"), TRACE_NODE_STATUSES, f"{name} 的状态在闭集内")
    _assert_not_in("route.ops", nodes, "ops 只落字段不落节点（避免与 mirror 重复）")
    _assert_eq(((rec.get("route") or {}).get("ops") or {}).get("veto"), "none",
               "ops 结论仍落 route 字段")
    reset_trace(sid)


def test_operation_keyphrase_coverage():
    """V11 覆盖率契约：操作注册表 / 关键句 / 打分配置三者不许单边漂移。

    "加词不加句"最容易发生：给操作补了触发词却忘了补关键句，打分永远过不了 floor，
    派发看着还在跑，新词其实进不了候选。键名写错更隐蔽——关键句静默不参与打分。
    """
    from config.word_dict_config import OPERATION_KEYPHRASES
    from tools import operations as ops

    _assert_eq([o for o in ops.ORDER if o not in ops.OPERATIONS], [], "ORDER 里每个操作都有谓词注册")
    _assert_eq(sorted(set(ops.OPERATIONS) - set(ops.ORDER)), [], "没有表外谓词注册")
    _assert_eq(sorted(set(OPERATION_KEYPHRASES) - set(ops.ORDER)), [],
               "关键句没有表外操作（键名写错会静默不参与打分）")
    short = {o: len(ops._phrases(o)) for o in ops.ORDER if len(ops._phrases(o)) < 4}
    _assert_eq(short, {}, "每个操作至少 4 句关键句（设计口径 4~6）")
    seen = {}
    for name in ops.ORDER:
        phrases = ops._phrases(name)
        _assert_eq(len(phrases), len(set(phrases)), f"{name} 组内关键句不重复")
        for p in phrases:
            seen.setdefault(p, []).append(name)
    _assert_eq({p: v for p, v in seen.items() if len(v) > 1}, {},
               "关键句不跨操作重复（重复会制造并列，gap 门失效）")
    cfg = ops._ops_cfg()
    if cfg:
        _assert_in(str(cfg.get("scoring", "")).lower(), ("off", "shadow", "on"),
                   "operations.scoring 取值在闭集内")
        for key in ("floor", "gap"):
            try:
                val = float(cfg.get(key))
            except (TypeError, ValueError):
                val = -1.0
            _assert_true(0.0 <= val <= 3.0, f"operations.{key} 是 0~3 的数值（当前 {cfg.get(key)!r}）")
    else:
        print("  [注意] 本机没有 operations 段配置，只校验了关键句覆盖率")


def test_operation_multi_hit_winner():
    """多命中裁决契约：一句话同时命中多个操作时，winner 由表序兜底（无分数/同分的降级路径）。

    V11 之后线上按分数裁决，ORDER 只在同分与降级时起作用——这里锁的正是这条兜底：
    改 ORDER、插新操作、放宽某个谓词，命中集合或 winner 变了会当场红。
    """
    from tools import operations as ops

    cases = [("那有没有2000以上的产品", "structured"),
             ("买个扫地机，顺便问问网点", "service_point"),
             ("有更便宜的吗", "followup")]
    for query, want in cases:
        hits = ops.probe(query)
        _assert_in(want, hits, f"「{query}」命中 {want}（实际 {hits}）")
        state, winner = ops.resolve(hits)
        _assert_eq((state, winner), ("win", want),
                   f"「{query}」按表序裁决 winner={want}（实际 {state}/{winner}）")
        tied = {o: 1.0 for o in hits}
        state, winner = ops.resolve(hits, scores=tied, floor=1.0, gap=0.0)
        _assert_eq((state, winner), ("win", want), f"「{query}」同分时回表序")


TESTS = [
    test_branch_closed_set_from_config,
    test_branch_covers_behaviours,
    test_check_case_hard_by_branch,
    test_check_case_ignores_text_asserts,
    test_check_case_soft_branch_hint,
    test_accepted_branches_cover_mapping,
    test_disable_shadow_restores,
    test_isolation_and_import_convention,
    test_trace_record_schema,
    test_trace_nesting,
    test_feedback_one_per_turn,
    test_trace_best_effort,
    test_trace_report_flags,
    test_golden_schema_and_context,
    test_context_sets_defined,
    test_context_set_filter_and_cost,
    test_emitted_tags_and_nodes_are_registered,
    test_route_nodes_written_once,
    test_branch_text,
    test_feedback_reason_enum,
    test_step_keys_registered,
    test_operation_keyphrase_coverage,
    test_operation_multi_hit_winner,
]


def run():
    reset()
    return run_tests("行为闭集（branch）与 trace 记录契约测试", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
