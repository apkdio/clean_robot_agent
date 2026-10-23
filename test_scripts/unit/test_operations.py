"""操作判别器测试：注册表完整性 + 入口判据 + 三态出口。

判据本身由现成实现保证，这里只测接线与三态是否符合设计。"""
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 与 test_scripts._runner 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *
from tools import operations as ops


def test_registry_shape():
    """注册表与表序一一对应，每条都有 predicate、allow 是 tuple 或 None。"""
    for name, (pred, allow, order) in ops.OPERATIONS.items():
        _assert(callable(pred), "%s 有 predicate" % name)
        _assert(allow is None or isinstance(allow, tuple), "%s 的 allow 是 tuple 或 None" % name)
        _assert_eq(order, ops.ORDER.index(name), "%s 的 order 与 ORDER 对齐" % name)
    _assert_eq(sorted(ops.OPERATIONS), sorted(ops.ORDER), "注册表与表序一一对应")


def test_probe_hits():
    """典型问法命中哪些操作；真判据比指代词表窄，断言以真判据为准。"""
    hit = ops.probe("离我最近的维修点")
    _assert_in("service_point", hit, "网点词 → service_point")
    _assert_in("sop", hit, "「维修」同时撞 SOP 弱触发（真实竞争，靠表序裁决）")
    _assert("date" not in hit, "date 已并入 find_models，不再是独立操作")
    casual = ops.probe("那你可以陪我玩过家家吗，班里这个游戏很火")
    _assert("model" not in casual, "闲聊句不命中 model（真判据比指代词表窄）")


def test_resolve_three_states():
    _assert_eq(ops.resolve(["service_point"], intent="robot"), ("win", "service_point"), "单命中 → win")
    _assert_eq(ops.resolve(["service_point", "sop"], intent="robot", margin=0.9),
               ("win", "service_point"), "多命中、未过阈值门 → 按表序裁决（与现链路一致）")
    _assert_eq(ops.resolve(["service_point", "sop"], intent="robot", margin=0.05),
               ("ambiguous", "service_point"), "多命中且低置信 → ambiguous（该问用户）")
    _assert_eq(ops.resolve(["model", "sop"], intent="casual", margin=0.9), ("none", None),
               "意图不允许 → none（假阳性被 allow 挡住）")
    _assert_eq(ops.resolve([], intent="robot"), ("none", None), "没有命中 → none")


def test_note_ops_why_none():
    """state=none 时 why 必须是 none（S-16）：intent=other/casual 的常态轮，
    被标成 order 会把「无可用操作」读成「按表序裁决过」。"""
    note = ops.note_ops("谢谢你的回答", intent="casual", margin=0.9)
    _assert_eq(note.get("state"), "none", "无可用操作 → state=none")
    _assert_eq(note.get("why"), "none", "why=none（不再误标 order）")
    saved = ops._ops_cfg
    ops._ops_cfg = lambda: {"scoring": "shadow"}
    try:
        _assert_eq(ops.note_ops("离我最近的维修点", intent="robot", margin=0.9).get("why"), "order",
                   "多命中且高置信（shadow）→ why=order")
    finally:
        ops._ops_cfg = saved


def test_note_ops_writes_route_and_returns_usable():
    """判定结果落 trace 的 route.ops，并把过 allow 门的候选交给主链路派发（shadow 口径）。"""
    from tools import trace_store
    from test_scripts import trace_reader
    saved = ops._ops_cfg
    ops._ops_cfg = lambda: {"scoring": "shadow"}      # 本用例锁的是 shadow 的派发顺序
    try:
        reset_trace("ops1")
        trace_store.begin_turn("ops1", "离我最近的维修点")
        note = ops.note_ops("离我最近的维修点", intent="robot", margin=0.9)
        trace_store.end_turn()
        casual = ops.note_ops("这两个有什么区别", intent="casual", margin=0.9)
    finally:
        ops._ops_cfg = saved
    rows = trace_reader.load(trace_store._path("ops1"))
    got = ((rows[0].get("route") or {}).get("ops") if rows else None) or {}
    _assert_eq(got.get("state"), "win", "落 state")
    _assert_eq(got.get("ops"), "service_point/sop/domain", "候选按表序（/ 连接）")
    _assert_eq(got.get("n"), 3, "落候选数")
    _assert_eq(got.get("winner"), "service_point", "落裁决结果")
    _assert_eq(note.get("usable"), ["service_point", "sop", "domain"],
               "shadow: usable 按表序给出")
    _assert_eq(casual.get("usable"), [], "casual 时没有可用操作")


def test_order_contract():
    """顺序契约：一句话同时命中多个操作时，winner 是谁必须稳定。

    改 ORDER、插新操作、改某个谓词的宽度，都会在这里当场红——比写在注释里可靠。
    期望值是**业务上该走哪条**，不是"当前恰好走哪条"；两者不一致就是缺陷。
    """
    cases = [
        # 预算/系列筛选都归 find_models
        ("那有没有2000以上的产品", "find_models"),
        # 网点 + SOP 弱触发竞争：网点更具体，排在 SOP 前
        ("买个扫地机，顺便问问网点", "service_point"),
        ("有更便宜的吗", "followup"),
        # 型号详情（单一命中）与系列枚举各走各的
        ("云顶X2多少钱", "model"),
        ("净白系列有哪些型号", "find_models"),
    ]
    for query, expected in cases:
        got = ops.resolve(ops.probe(query), intent="robot", margin=0.9)
        _assert_eq(got, ("win", expected), "%s → %s" % (query, expected))


def test_score_all_shape_and_determinism():
    """8 个操作都有分；同一 query 两次打分一致（可复现、可断言）。"""
    s1, st1 = ops.score_all("净白系列有哪些型号")
    s2, st2 = ops.score_all("净白系列有哪些型号")
    _assert_eq(set(s1), set(ops.OPERATIONS), "每个操作都有分")
    _assert_eq(st1, st2, "两次状态一致")
    _assert_eq({k: round(v, 3) for k, v in s1.items()}, {k: round(v, 3) for k, v in s2.items()},
               "同一 query 分数可复现")
    top = max(s1, key=s1.get)
    _assert_eq(top, "find_models", "系列/预算问句 find_models 分最高")


def test_score_all_degrades_without_dense():
    """embedding 不可用 → dense_degraded，BM25 + 强判据仍出分（不静默归零）。"""
    saved = ops._dense_scores
    ops._dense_scores = lambda q: None
    try:
        scores, status = ops.score_all("净白系列有哪些型号")
    finally:
        ops._dense_scores = saved
    _assert_eq(status, "dense_degraded", "状态标出降级")
    _assert(scores.get("find_models", 0) > 0, "BM25 + 强判据仍有分")


def test_resolve_by_scores():
    """分数裁决：按分排序、floor 过滤、gap 判 ambiguous、全不过 → none。"""
    scores = {"model": 1.8, "sop": 1.7, "domain": 0.2}
    _assert_eq(ops.resolve(["model", "sop", "domain"], intent="robot", scores=scores, floor=0.5),
               ("win", "model"), "按分排序（model 1.8 > sop 1.7）")
    _assert_eq(ops.resolve(["model", "sop"], intent="robot", scores=scores, floor=1.75),
               ("win", "model"), "floor 过滤掉 sop")
    _assert_eq(ops.resolve(["model", "sop"], intent="robot", scores=scores, floor=0.5, gap=0.3),
               ("ambiguous", "model"), "top1-top2 < gap → ambiguous")
    _assert_eq(ops.resolve(["domain"], intent="robot", scores={"domain": 0.2}, floor=0.5),
               ("none", None), "全不过 floor → none")


def test_note_ops_shadow_keeps_order():
    """scoring=shadow：分数落 trace，但 usable 仍按表序（派发行为不变）。"""
    from tools import trace_store
    from test_scripts import trace_reader

    saved = (ops.score_all, ops._ops_cfg)
    ops.score_all = lambda q: ({"service_point": 0.6, "sop": 1.9}, "ok")
    ops._ops_cfg = lambda: {"scoring": "shadow"}
    reset_trace("ops-shadow")
    trace_store.begin_turn("ops-shadow", "离我最近的维修点")
    try:
        note = ops.note_ops("离我最近的维修点", intent="robot", margin=0.9)
    finally:
        ops.score_all, ops._ops_cfg = saved
    trace_store.end_turn()
    _assert_eq(note["usable"][:2], ["service_point", "sop"], "shadow 仍按表序")
    got = (trace_reader.load(trace_store._path("ops-shadow"))[0].get("route") or {}).get("ops") or {}
    _assert_eq(got.get("scoring_status"), "ok", "trace 记打分状态")
    _assert_in("scores", got, "trace 记分数")
    _assert(got["scores"].startswith("sop="), "分数串按分数降序")


def test_note_ops_score_mode_orders_by_score():
    """scoring=on：usable/winner 改按分数，why 记为 score。"""
    saved_cfg, saved_score = ops._ops_cfg, ops.score_all
    ops._ops_cfg = lambda: {"scoring": "on", "floor": 0.5, "gap": 0.0}
    ops.score_all = lambda q: ({"service_point": 0.6, "sop": 1.9}, "ok")
    try:
        note = ops.note_ops("离我最近的维修点", intent="robot", margin=0.9)
    finally:
        ops._ops_cfg, ops.score_all = saved_cfg, saved_score
    _assert_eq(note["usable"], ["sop", "service_point"], "on 模式按分排序")
    _assert_eq(note["winner"], "sop", "winner 取最高分")
    _assert_eq(note["why"], "score", "why 记为 score")


def test_two_gates_score_then_vocab():
    """三道门：① 分数过阈值 ② 词表/结构命中 ③ 分差。①②各砍一类候选。"""
    saved = (ops.score_all, ops._ops_cfg)
    ops._ops_cfg = lambda: {"scoring": "on", "floor": 1.2, "gap": 0.10}
    # model 分数够但词表没命中；find_models 词表命中但分数不够
    ops.score_all = lambda q: ({"model": 2.0, "find_models": 0.10, "service_point": 1.9}, "ok")
    try:
        note = ops.note_ops("离我最近的维修点", intent="robot", margin=0.9)
    finally:
        ops.score_all, ops._ops_cfg = saved
    _assert("model" not in note["usable"], "分数够但词表没命中 → 第二道门砍掉")
    _assert("find_models" not in note["usable"], "词表命中但分数不够 → 第一道门砍掉")
    _assert_eq(note["usable"][0], "service_point", "两门都过的排第一")
    _assert_eq(note.get("n_candidates"), 2, "过第一道门的数量（model/service_point）")


def test_keyphrases_shape():
    """关键句表形状：6 个操作都在、每个至少 4 句、无空项（打分的最低前提）。

    数量口径与 eval 侧 test_operation_keyphrase_coverage 一致（设计 4~6 句）。"""
    from config.word_dict_config import OPERATION_KEYPHRASES

    _assert_eq(set(OPERATION_KEYPHRASES), set(ops.OPERATIONS), "每个操作都有关键句")
    for op, raw in OPERATION_KEYPHRASES.items():
        phrases = [p for p in raw.split("##") if p.strip()]
        _assert(len(phrases) >= 4, f"{op} 至少 4 条关键句（现 {len(phrases)}）")
        _assert(all(len(p.strip()) >= 2 for p in phrases), f"{op} 没有空/单字项")


def test_cfg_opts_defaults_match_template():
    """缺段/非法值回模板默认（on / 1.20 / 0.10），不能静默退回表序或宽松 floor。"""
    _assert_eq(ops._cfg_opts({}), ("on", 1.20, 0.10), "缺段 → 与模板一致")
    _assert_eq(ops._cfg_opts({"scoring": "ON", "floor": "1.5", "gap": 0.2}),
               ("on", 1.5, 0.2), "大小写与字符串数字都认")
    _assert_eq(ops._cfg_opts({"scoring": "bogus", "floor": "x", "gap": -1}),
               ("on", 1.20, 0.10), "非法值显式回默认")
    _assert_eq(ops._cfg_opts({"scoring": "off"})[0], "off", "off 是合法值")


def test_note_ops_missing_section_uses_defaults():
    """缺 operations 段时仍按分数派发（存量配置不漂移成纯表序）。"""
    saved = (ops.score_all, ops._ops_cfg)
    ops._ops_cfg = lambda: {}
    ops.score_all = lambda q: ({"service_point": 1.5, "sop": 1.9}, "ok")
    try:
        note = ops.note_ops("离我最近的维修点", intent="robot", margin=0.9)
    finally:
        ops.score_all, ops._ops_cfg = saved
    _assert_eq(note["why"], "score", "缺段也按分派发")
    _assert_eq(note["usable"][:2], ["sop", "service_point"], "按分排序（sop 1.9 > 1.5）")
    _assert_eq(note["scoring_status"], "ok", "分数照常落 trace")


TESTS = [test_registry_shape, test_probe_hits, test_resolve_three_states, test_note_ops_why_none,
         test_note_ops_writes_route_and_returns_usable, test_order_contract,
         test_score_all_shape_and_determinism, test_score_all_degrades_without_dense,
         test_resolve_by_scores, test_note_ops_shadow_keeps_order,
         test_note_ops_score_mode_orders_by_score, test_two_gates_score_then_vocab,
         test_cfg_opts_defaults_match_template, test_note_ops_missing_section_uses_defaults,
         test_keyphrases_shape]


def run():
    reset()
    return run_tests("操作判别器测试（注册表 / 入口判据 / 三态）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
