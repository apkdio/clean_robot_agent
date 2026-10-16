"""trace HTML 报告测试：合成一份最小 trace，验数据整形与渲染（纯规则，不调模型）。

重点盯两件会导致"页面空白"的事：① 旁路 chunks 是**一轮一条、片段在嵌套列表里**（曾按"一行一片"读，取到全空）；
② 无召回的轮次不该渲染片段区。
"""
import json
import os
import shutil
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *

from test_scripts import trace_reader as tr
from test_scripts import trace_report as html_report

SID = "11111111-2222-3333-4444-555566667777"
DAY = "20261015"
DATE = "2026-10-15"


def _turn(turn_index, query, branch, **extra):
    rec = {"v": 3, "ts": "2026-10-15T15:%02d:00" % turn_index, "session_id": SID,
           "turn_index": turn_index, "query": query, "branch": branch, "tag": branch,
           "detail": {"hits": 2},
           "route": {"intent": {"label": "robot", "margin": 0.9}},
           "retrieval": {"domain": {"value": "不染一尘售后服务.txt", "via": "top1"},
                         "rewrite": {"via": "none", "effective_query": query,
                                     "normalize": {"changed": turn_index == 2, "text": query}},
                         "passes": [{"pass": 1, "filter": {"file_name": "不染一尘售后服务.txt"},
                                     "dense": 3, "sparse": 5, "fused": 5,
                                     "rerank": {"in": 5, "out": 2, "degraded": False,
                                                "model": "BAAI/bge-reranker-v2-m3"}},
                                    {"pass": 2, "filter": None, "dense": 6, "sparse": 4, "fused": 8,
                                     "rerank": {"in": 8, "out": 3, "degraded": True}}],
                         "merge": {"dense_only": 1, "sparse_only": 0, "both": 1, "final": 2},
                         "twice_retrieval": {"active": False, "kept": 0}},
           "path": [{"node": "route.inject", "status": "ok", "out": {"hit": False}},
                    {"node": "route.intent", "status": "ok", "out": {"label": "robot", "margin": 0.9}},
                    {"node": "retrieve", "status": "ok", "ms": 12, "out": {"domain": "全库", "n": 2}}],
           "answer": {"kind": "answer", "model": "qwen2.5:7b", "chars": 40},
           "ms": {"retrieve": 900, "total": 5000}, "steps": {}}
    rec.update(extra)
    return rec


def _write(root, turns=None):
    """造一份最小 trace 目录：日期层 + 会话目录 + 主干 + chunks 旁路。"""
    d = os.path.join(root, DAY, "%s_%s_1500" % (SID, DAY))
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, "%s_%s_1500" % (SID, DAY))
    with open(stem + ".jsonl", "w", encoding="utf-8") as f:
        for rec in (turns or (_turn(1, "保修期是多久", "answer"),
                              _turn(2, "怎么设置定时清扫", "no_answer_fallback", shadow={"agree": False}))):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(stem + ".chunks.jsonl", "w", encoding="utf-8") as f:
        # 一轮一条记录，片段在嵌套列表里
        f.write(json.dumps({"v": 1, "ts": "2026-10-15T15:01:00", "session_id": SID, "turn_index": 1,
                            "query": "保修期是多久", "domain": "不染一尘售后服务.txt", "n_chunks": 1,
                            "chunks": [{"rank": 1, "file": "不染一尘售后服务.txt", "entry": "## 一、保修政策",
                                        "score": 0.078, "rerank": 0.078, "rrf": 0.033,
                                        "dense_rank": 1, "sparse_rank": 1, "chars": 8,
                                        "text": "整机保修 2 年"}]}, ensure_ascii=False) + "\n")
    return d


def test_chunks_are_flattened_per_turn():
    root = tempfile.mkdtemp(prefix="trace_html_")
    try:
        _write(root)
        chunks = tr.chunks_by_turn(tr.load(tr.session_files(root, SID + "_%s_1500" % DAY, kind="chunks",
                                                             date=DATE)[0]))
        got = chunks.get(1) or []
        _assert_eq(len(got), 1, "一轮一条记录 → 摊平成该轮的片段列表")
        _assert_eq(got[0].get("text"), "整机保修 2 年", "片段正文取到 text 字段")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_session_view_and_render():
    root = tempfile.mkdtemp(prefix="trace_html_")
    try:
        _write(root)
        name = "%s_%s_1500" % (SID, DAY)
        view = html_report.session_view(root, name, DATE, 1500)
        _assert_eq(view["n_turns"], 2, "两轮都在")
        _assert_eq(len(view["turns"][0]["chunks"]), 1, "第一轮有 1 片召回")
        _assert_eq(view["turns"][1]["chunks"], [], "第二轮无召回")
        _assert(view["turns"][1]["shadow_mismatch"], "影子不一致被识别")
        _assert(view["turns"][1]["norm"], "归一化改过被识别")

        page = html_report.render([view], DATE, root)
        _assert_in("整机保修 2 年", page, "片段正文渲染进页面（曾因读法错误空白）")
        _assert_in("一、保修政策", page, "片段条目名也渲染")
        _assert_in('data-branch="answer"', page, "轮次带出口 branch（供筛选）")
        _assert_in('data-no_recall="1"', page, "0 召回标记带上了")
        _assert_in("保修期是多久", page, "query 渲染")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_latest_date_and_empty_chunks():
    root = tempfile.mkdtemp(prefix="trace_html_")
    try:
        _write(root)
        _assert_eq(html_report.latest_date(root), DAY, "取最近有 trace 的日期")
        view = html_report.session_view(root, "%s_%s_1500" % (SID, DAY), DATE, 1500)
        _assert_eq(len(view["turns"][0]["chunks"]), 1, "有召回的轮次照常有片段")
        _assert_eq(view["turns"][1]["chunks"], [], "无召回的轮次是空列表，不报错")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_details_render_as_tables():
    """链路细节拆成表格：检索各遍逐列铺开，rewrite / merge / shadow 走键值表，不再吐原始 dict。"""
    root = tempfile.mkdtemp(prefix="trace_html_")
    try:
        _write(root)
        view = html_report.session_view(root, "%s_%s_1500" % (SID, DAY), DATE, 1500)
        page = html_report.render([view], DATE, root)
        _assert_in("检索各遍", page, "各遍召回有独立表头")
        _assert_in("<th>稠密</th><th>稀疏</th><th>融合</th>", page, "各遍按列展开")
        _assert_in("不染一尘售后服务.txt", page, "第一遍的域过滤进表格")
        _assert_in("<td>全库</td>", page, "第二遍撤域显示为全库")
        _assert_in("降级，未精排", page, "降级在各遍表里标出")
        _assert_in("hit=否", page, "path 的 out 拆成键=值（hit 说人话）")
        _assert_not_in("&#39;hit&#39;", page, "path 的 out 不再吐原始 dict")
        _assert_in("<caption>detail · 标签细节</caption>", page, "stable core 的 detail 进页面")
        _assert_in("<th>hits</th>", page, "detail 的命中数逐项列出")
        _assert_in("召回 2 片", page, "召回数取主干 detail.hits，不是数旁路片段")
        _assert_in("<caption>rewrite · 检索口径</caption>", page, "rewrite 是键值表")
        _assert_in("<th>normalize</th>", page, "归一化单独一行")
        _assert_in("<caption>merge · 两路合并</caption>", page, "merge 是键值表")
        _assert_in("<caption>shadow · 影子决策</caption>", page, "shadow 是键值表")
        _assert_in("<td>否</td>", page, "布尔值说人话（agree=False）")
        _assert_not_in("<p>rewrite: {", page, "不再输出原始 dict 串")
        _assert_not_in("<p>passes: [", page, "不再输出原始 dict 串")
        _assert_not_in("<caption>twice_retrieval · 撤域二次召回</caption>", page, "没触发二次召回的轮次不铺这张表")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_twice_retrieval_table():
    """撤域二次召回真触发时才出表，触发项（域内 top1 / 阈值 / 兜底池）逐项列出。"""
    root = tempfile.mkdtemp(prefix="trace_html_")
    try:
        rec = _turn(1, "撤域试试", "answer")
        rec["retrieval"]["twice_retrieval"] = {"active": True, "kept": 2,
                                               "trigger": {"top1": 0.11, "threshold": 0.2}}
        _write(root, [rec])
        view = html_report.session_view(root, "%s_%s_1500" % (SID, DAY), DATE, 1500)
        page = html_report.render([view], DATE, root)
        _assert_in("<caption>twice_retrieval · 撤域二次召回</caption>", page, "触发了就有这张表")
        _assert_in("0.11", page, "域内 top1 进表")
        _assert_in("threshold", page, "触发条件逐项列出，不再是一行 dict")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_recall_count_source():
    """召回数量优先读主干 detail.hits（0 也照实显示），没有它才退回旁路片段数。"""
    root = tempfile.mkdtemp(prefix="trace_html_")
    try:
        rec = _turn(1, "保修期是多久", "answer")
        rec.pop("detail")                      # 老记录 / 非检索标签没有 detail
        _write(root, [rec])
        view = html_report.session_view(root, "%s_%s_1500" % (SID, DAY), DATE, 1500)
        _assert_in("召回 1 片", html_report.render([view], DATE, root), "没有 detail.hits 时退回旁路片段数")

        rec2 = _turn(2, "怎么设置定时清扫", "no_answer_fallback")
        rec2["detail"] = {"hits": 0}
        _write(root, [rec2])
        view2 = html_report.session_view(root, "%s_%s_1500" % (SID, DAY), DATE, 1500)
        _assert_in("召回 0 片", html_report.render([view2], DATE, root), "0 召回照实显示，不被当成没这个字段")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_out_dir_layout_by_date():
    """默认落点按日期建一层目录（<base>/<YYYYMMDD>/trace_<YYYYMMDD>.html）；日期解析与根目录都可覆盖。"""
    root = tempfile.mkdtemp(prefix="trace_html_")
    base = tempfile.mkdtemp(prefix="trace_out_")
    try:
        _write(root)
        path, n_sessions, n_turns = html_report.write_report(
            root, DAY, html_report.default_out_path(base, DAY))
        _assert_eq(os.path.relpath(path, base), os.path.join(DAY, "trace_%s.html" % DAY),
                   "报告落在 <base>/<YYYYMMDD>/ 下")
        _assert_eq((n_sessions, n_turns), (1, 2), "回报会话数与轮数")
        with open(path, encoding="utf-8") as f:
            _assert_in("trace 报告", f.read(), "写出来的是那份页面")
        _assert_eq(html_report.resolve_day(root, "2026-10-15"), DAY, "带横线的日期归一成 YYYYMMDD")
        _assert_eq(html_report.resolve_day(root), DAY, "不给日期就取最近有 trace 的一天")
        _assert_eq(html_report.out_dir(base), base, "--out-dir 覆盖根目录")
        _assert_eq(html_report.out_dir(""), os.path.join(html_report._ROOT, "reports"),
                   "默认根目录是项目下 reports")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(base, ignore_errors=True)


TESTS = [
    test_chunks_are_flattened_per_turn,
    test_session_view_and_render,
    test_latest_date_and_empty_chunks,
    test_details_render_as_tables,
    test_twice_retrieval_table,
    test_recall_count_source,
    test_out_dir_layout_by_date,
]


def run():
    reset()
    return run_tests("trace HTML 报告测试（数据整形 / 渲染 / 边界）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
