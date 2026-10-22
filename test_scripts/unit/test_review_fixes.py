"""代码 review 集中修复的回归测试。

覆盖：依赖降级（redis 包缺失）/ 日期直出闸门 / 上传文件名净化 / 数据集构建不覆盖正式集 /
意图头降级与维度对账 / embedding 来源统一 / 本地锁与停止标志 TTL / 死接口删除 / 重答与历史口径。"""
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 与 test_scripts._runner 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *


# ──────────────────────────────────────────────────────────────
# 1. V-01：redis 包未安装时降级，不再首问即崩
# ──────────────────────────────────────────────────────────────

def test_redis_missing_package_degrades():
    import builtins
    from tools import redis_store as rs

    saved = (rs._redis, rs._enabled)
    rs._redis, rs._enabled = None, None
    real_import = builtins.__import__

    def _no_redis(name, *a, **k):
        if name == "redis":
            raise ImportError("No module named 'redis'")
        return real_import(name, *a, **k)

    builtins.__import__ = _no_redis
    try:
        _assert(rs.get_redis() is None, "包缺失 → 返回 None 走本地内存降级")
        _assert(rs._enabled is False, "探测失败后置 False，后续不再反复重试")
    finally:
        builtins.__import__ = real_import
        rs._redis, rs._enabled = saved


# ──────────────────────────────────────────────────────────────
# 2. S-02：日期直出闸门——时间词可解析但无选型信号时不直出
# ──────────────────────────────────────────────────────────────

def test_date_gate_needs_select_signal():
    from config.word_dict_config import DATE_SELECT_HINT_RE
    from tools.agent import _resolve_date_filter

    _assert(_resolve_date_filter("今年买的有保修吗") is not None, "问保修句的时间词本就可解析（闸门要挡的就是它）")
    _assert(not DATE_SELECT_HINT_RE.search("今年买的有保修吗"), "问保修句无选型信号 → 不直出")
    _assert(not DATE_SELECT_HINT_RE.search("最近半年拖布有异味怎么办"), "问故障句无选型信号 → 不直出")
    _assert(DATE_SELECT_HINT_RE.search("今年出的新款有哪些"), "选型问句有信号 → 照常直出")
    _assert(DATE_SELECT_HINT_RE.search("去年发布的机型推荐一款"), "推荐问句有信号 → 照常直出")


# ──────────────────────────────────────────────────────────────
# 3. W-01：上传文件名净化
# ──────────────────────────────────────────────────────────────

def test_upload_name_sanitized():
    from webapp.app import _safe_upload_name

    _assert_eq(_safe_upload_name("../../README.md"), "README.md", "相对路径穿越被剥掉")
    _assert_eq(_safe_upload_name("C:\\x\\知识.txt"), "知识.txt", "Windows 绝对路径被剥掉")
    _assert_eq(_safe_upload_name("不染一尘具体型号.txt"), "不染一尘具体型号.txt",
               "正常中文名保留（入库元数据与域路由按文件名匹配）")
    _assert(_safe_upload_name("..") == "", "纯点名为空 → 拒绝")
    _assert_eq(_safe_upload_name('a<b>c:"d|e?f*.txt'), "abcdef.txt", "文件系统非法字符被清（其余字符保留）")


# ──────────────────────────────────────────────────────────────
# 4. W-02：build 脚本默认写 _generated.jsonl，不覆盖已提交数据集
# ──────────────────────────────────────────────────────────────

def test_build_dataset_writes_generated_not_dataset():
    import json
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ds = os.path.join(root, "data", "datasets", "intent_dataset.jsonl")
    gen = os.path.join(root, "data", "datasets", "_generated.jsonl")
    _assert(os.path.isfile(ds), "前提：正式数据集在仓库里")
    before = open(ds, "rb").read()
    if os.path.isfile(gen):
        os.remove(gen)

    env = dict(os.environ)
    env["PYTHONPATH"] = root
    subprocess.run(
        [sys.executable, os.path.join(root, "intent_classifier_training", "build_intent_dataset.py")],
        capture_output=True, cwd=root, env=env, timeout=300,
    )
    _assert(os.path.isfile(gen), "默认输出 _generated.jsonl")
    _assert(open(ds, "rb").read() == before, "正式数据集未被覆盖")
    first = json.loads(open(gen, encoding="utf-8").readline())
    _assert("source" in first, "生成样本带 source 字段")


# ──────────────────────────────────────────────────────────────
# 5. S-09：意图推理异常降级为 unknown，不再把整轮打成 500
# ──────────────────────────────────────────────────────────────

def test_intent_inference_failure_degrades():
    import tools.intent_router as ir

    saved = ir._head, ir._labels, ir._head_unavailable
    try:
        # 构造一个"加载成功但推理抛异常"的头：embed 正常、head 前向必炸
        class _Boom:
            def __call__(self, *a, **k):
                raise RuntimeError("ollama down")

            def eval(self):
                return self

        class _Emb:
            @staticmethod
            def embed_documents(xs):
                return [[0.0] * 1024 for _ in xs]

        ir._head, ir._labels, ir._head_unavailable = _Boom(), ["robot", "other", "casual", "unknown"], False
        saved_emb = ir._get_emb
        ir._get_emb = lambda: _Emb()
        label, margin = ir.route_intent_with_margin("边刷多久换一次")
        _assert_eq(label, "unknown", "推理异常 → 降级 unknown")
        _assert_eq(margin, 0.0, "降级时 margin 记 0")
    finally:
        ir._head, ir._labels, ir._head_unavailable = saved
        ir._get_emb = saved_emb


# ──────────────────────────────────────────────────────────────
# 6. S-10：流式生成失败/零输出走兜底话术并留痕
# ──────────────────────────────────────────────────────────────

def test_stream_generate_failure_falls_back():
    from tools import agent as agent_mod
    from tools import trace_store
    from test_scripts import trace_reader

    reset_trace("s10")
    saved = agent_mod.stream_chat
    trace_store.begin_turn("s10", "边刷多久换一次")

    def _boom(messages, model="", temperature=0.3):
        raise RuntimeError("llm gone")
        yield  # pragma: no cover

    agent_mod.stream_chat = _boom
    try:
        out = list(agent_mod._stream_with_fallback([("human", "x")]))
        _assert(len(out) >= 1, "生成失败仍有回复（兜底话术）")
        _assert(any(rl in out[0] for rl in agent_mod.NO_ANSWER_REPLIES),
                "兜底话术来自 NO_ANSWER_REPLIES")
    finally:
        agent_mod.stream_chat = saved
        trace_store.end_turn()
    rows = trace_reader.load(trace_store._path("s10"))
    rec = rows[0] if rows else {}
    _assert(rec.get("error"), "稳定核 error 留异常痕")
    _assert(rec.get("branch") == "no_answer_fallback", "行为出口是兜底（不再假装 answer）")
    nodes = [n.get("node") for n in rec.get("path") or []]
    _assert_in("generate", nodes, "generate 节点存在")
    gen = next((n for n in rec.get("path") or [] if n.get("node") == "generate"), {})
    _assert_eq(gen.get("status"), "degraded", "generate 节点标 degraded")


# ──────────────────────────────────────────────────────────────
# 7. V-02：labels.json 带维度时按 meta 建头；维度不符显式降级而非静默错分
# ──────────────────────────────────────────────────────────────

def test_intent_head_builds_from_meta():
    import json
    import tempfile
    import tools.intent_router as ir

    with tempfile.TemporaryDirectory() as d:
        import torch
        # 新格式：维度 8（非默认），权重按它保存
        meta = {"labels": ["robot", "other"], "input_dim": 8, "embedding_model": "bge-m3"}
        with open(os.path.join(d, "labels.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        head = torch.nn.Linear(8, 2)
        torch.save(head.state_dict(), os.path.join(d, "classifier_head.pt"))

        saved = (ir._MODEL_DIR, ir._head, ir._labels, ir._head_unavailable)
        try:
            ir._MODEL_DIR = os.path.relpath(d).replace("\\", "/")
            ir._head, ir._labels, ir._head_unavailable = None, None, False
            h, labels = ir._load_model()
            _assert_eq(h.in_features, 8, "头按 labels.json 的 input_dim 构建（不再写死 1024）")
            _assert_eq(labels, ["robot", "other"], "标签从 meta 读")
        finally:
            ir._MODEL_DIR, ir._head, ir._labels, ir._head_unavailable = saved

        # 维度不符（权重 8 维、meta 声明 4 维）→ 显式降级，不抛、不静默错分
        with open(os.path.join(d, "labels.json"), "w", encoding="utf-8") as f:
            json.dump({"labels": ["robot", "other", "casual", "unknown"], "input_dim": 4}, f)
        saved = (ir._MODEL_DIR, ir._head, ir._labels, ir._head_unavailable)
        try:
            ir._MODEL_DIR = os.path.relpath(d).replace("\\", "/")
            ir._head, ir._labels, ir._head_unavailable = None, None, False
            h, labels = ir._load_model()
            _assert(h is None and labels is None, "维度不符 → (None, None) 走显式降级（unknown）")
        finally:
            ir._MODEL_DIR, ir._head, ir._labels, ir._head_unavailable = saved


# ──────────────────────────────────────────────────────────────
# 8. 尾巴收口：embedding 单一来源 / 本地锁 TTL / 停止标志 TTL / 死接口已删
# ──────────────────────────────────────────────────────────────

def test_embedding_source_unified():
    """embedding 与向量库共用 chroma.yaml 这一处来源（此前 intent_router 用内置默认，会漂移）。"""
    from tools.config_tool import load_config
    from tools.llm_tool import get_embedding_model

    cfg = (load_config("chroma") or {}).get("embedding") or {}
    m = get_embedding_model()
    _assert_eq(getattr(m, "model", None), cfg.get("model"),
               "留空时按 chroma.yaml 的 embedding.model 解析")


def test_embedding_client_cached():
    """同一配置复用同一个 embedding 客户端：避免每次构造都打一条 init 日志。"""
    from tools.llm_tool import get_embedding_model

    _assert(get_embedding_model() is get_embedding_model(), "重复调用返回同一实例")


def test_local_lock_has_ttl():
    """Redis 不可用时的本地降级锁也要过期（原来无 TTL，进程内会永久占住）。"""
    import time

    import tools.redis_store as rs

    saved = rs.get_redis
    rs.get_redis = lambda: None            # 强制走本地降级态
    try:
        _assert(rs.acquire_lock("t:lock", ttl=60), "首次获锁")
        _assert(not rs.acquire_lock("t:lock", ttl=60), "未过期时抢占失败")
        rs._local_locks["t:lock"] = time.time() - 1     # 手工置为已过期
        _assert(rs.acquire_lock("t:lock", ttl=60), "过期后可重新获锁")
        rs.release_lock("t:lock")
        _assert("t:lock" not in rs._local_locks, "释放后清掉")
    finally:
        rs.get_redis = saved
        rs._local_locks.pop("t:lock", None)


def test_stop_flag_has_ttl_local():
    """停止标志的本地降级态也要过期（无 TTL 的残留会误杀下一轮）。"""
    import time

    import tools.redis_store as rs
    from webapp import app as wa

    saved = rs.get_redis
    rs.get_redis = lambda: None
    sid = "s-stop-ttl"
    try:
        wa._stop_flags.pop(sid, None)
        _assert(not wa._is_stopped(sid), "未置位时不是停止态")
        wa._set_stop(sid)
        _assert(wa._is_stopped(sid), "置位后是停止态")
        wa._stop_flags[sid] = time.time() - 1           # 手工置为已过期
        _assert(not wa._is_stopped(sid), "过期后自动失效")
        _assert(sid not in wa._stop_flags, "过期项被顺手清掉")
    finally:
        rs.get_redis = saved
        wa._stop_flags.pop(sid, None)


def test_dead_batch_endpoint_removed():
    """前端零调用的 /api/ingest/batch 已删（文档同步：project_detail 的相应描述）。"""
    from webapp.app import app

    rules = {str(r) for r in app.url_map.iter_rules()}
    _assert(not any("ingest/batch" in r for r in rules), "废弃接口不在路由表里")


# ──────────────────────────────────────────────────────────────
# 5. L-02 / L-03：「重新回答」覆盖上一轮 + 当前句不进对话历史
# ──────────────────────────────────────────────────────────────

class _StubRetriever:
    """0 命中的检索桩：这两条只看落盘与历史拼串，不需要真检索（也就不拖慢单测）。"""

    def search_with_scores(self, query, filter=None, trace_out=None):
        return []


def _ask_with_stubs(query, sid, retry=False):
    """用桩件走一遍链路（不碰 Ollama / Chroma / 分类头）。"""
    import tools.intent_router as ir
    from tools import agent as agent_mod
    from tools.agent import ask_stream

    saved = (ir.route_intent_with_margin, agent_mod._get_retriever, agent_mod.stream_chat)
    ir.route_intent_with_margin = lambda q: ("robot", 0.9)
    agent_mod._get_retriever = lambda: _StubRetriever()
    agent_mod.stream_chat = lambda messages, model="", temperature=0.3, **kw: iter(["答案"])
    try:
        return "".join(ask_stream(query, sid, retry=retry)).strip()
    finally:
        ir.route_intent_with_margin, agent_mod._get_retriever, agent_mod.stream_chat = saved


def _reset_session(sid):
    from tools.context_store import delete_session

    delete_session(sid)


def _disk_messages(sid):
    """按落盘（jsonl）读消息，不走内存窗口 —— 断言的是持久化结果。"""
    from tools.context_store import get_recent

    return get_recent(sid, n=1000)


def test_retry_replaces_previous_turn():
    """「重新回答」＝同一句重发：覆盖上一轮，同一句不落第二遍、上一轮回答不再留在上下文。

    前端是就地替换同一个气泡，落盘也应当只有一个提问与一版回答；trace 仍留两轮（只增不改）。
    """
    from tools.context_store import append_message

    sid, q = "t-retry-cover", "边刷多久换一次"
    _reset_session(sid)
    _ask_with_stubs(q, sid)
    append_message(sid, "assistant", "第一版的回答")      # 模拟 webapp 流结束后的 assistant 落盘
    _ask_with_stubs(q, sid, retry=True)

    msgs = _disk_messages(sid)
    users = [m for m in msgs if m.get("role") == "user"]
    _assert_eq(len(users), 1, "同一句只落一遍")
    _assert_eq(users[0].get("content"), q, "留下的就是这句")
    _assert_eq(len([m for m in msgs if m.get("role") == "assistant"]), 0, "上一轮的回答已被撤掉")


def test_retry_without_matching_last_user_keeps_turns():
    """retry 标记但上一条不是这句话时不撤任何轮次（防误删更早的正常轮次）。"""
    from tools.context_store import append_message

    sid = "t-retry-keep"
    _reset_session(sid)
    _ask_with_stubs("边刷多久换一次", sid)
    append_message(sid, "assistant", "答案一")
    _ask_with_stubs("滤网多久换一次", sid, retry=True)

    msgs = _disk_messages(sid)
    _assert_eq(len([m for m in msgs if m.get("role") == "user"]), 2, "两句都留着")
    _assert_in("边刷多久换一次", [m.get("content") for m in msgs], "上一轮没被撤掉")


def test_history_block_excludes_current_query():
    """当前这句不进「对话历史」（Prompt 的「问题：」已单独给出），与 _select_history_turns 同口径。"""
    from tools.agent import _history_block
    from tools.context_store import append_message

    sid = "t-history-block"
    _reset_session(sid)
    append_message(sid, "user", "边刷多久换一次")
    append_message(sid, "assistant", "边刷建议每 3 个月换一次。")
    append_message(sid, "user", "滤网呢")                 # 本轮这句（ask_stream 开头就落了盘）

    block = _history_block(sid)
    _assert_in("边刷多久换一次", block, "上文的用户话仍在历史里")
    _assert_in("边刷建议每 3 个月", block, "上文的回答仍在历史里")
    _assert_not_in("滤网呢", block, "本轮这句不重复进历史")


# ──────────────────────────────────────────────────────────────
# 6. trace 语义：被停的轮记 aborted、重开 SOP 落节点
# ──────────────────────────────────────────────────────────────

def test_stopped_turn_marks_aborted():
    """被停的轮必须记 aborted=true：_ask_stream 静默 return 时迭代是"正常结束"的，
    只看循环是否抛异常会把停掉的轮记成 false，与 webapp 的回滚对不上。"""
    import tools.agent as am

    seen = {}
    saved_ask, saved_end = am._ask_stream, am.trace_store.end_turn
    am._ask_stream = lambda q, sid, lng=None, lat=None, retry=False, stop_check=None: iter(())
    am.trace_store.end_turn = lambda aborted=False, reply="": seen.update(aborted=aborted)
    try:
        list(am.ask_stream("x", "t-abort", stop_check=lambda: True))
        _assert(seen.get("aborted") is True, "被停的轮 → aborted=true")
        seen.clear()
        list(am.ask_stream("x", "t-abort", stop_check=lambda: False))
        _assert(seen.get("aborted") is False, "正常轮仍为 false")
    finally:
        am._ask_stream, am.trace_store.end_turn = saved_ask, saved_end


def test_reenter_sop_logs_node():
    """「重新回答」重开 SOP 时要落 sop.reenter：否则这一轮在 path 里看不出是重答重开的。"""
    import tools.agent as am
    import tools.context_store as cs
    import sops
    from test_scripts import trace_reader
    from tools import trace_store

    sid, q = "t-reenter-node", "帮我推荐一款扫地机器人"
    reset_trace(sid)
    trace_store.begin_turn(sid, q)        # 第 1 轮（被重发的那轮）
    trace_store.end_turn()
    trace_store.begin_turn(sid, q)        # 第 2 轮（本轮 = 重答）

    saved = (trace_store.find_turn, trace_store.latest_turn, trace_store.find_turn_by_index,
             am._enter_sop, cs.get_last_sop, sops.sop_needs_enter_confirm)
    trace_store.find_turn = lambda s, query: {"date": "2026-10-20", "turn_index": 1}
    trace_store.latest_turn = lambda s: {"date": "2026-10-20", "turn_index": 1}
    trace_store.find_turn_by_index = lambda s, d, i: {"sop": {"id": "purchase"}, "branch": "start_sop"}
    am._enter_sop = lambda sid, sop_id, query, **kw: "您的预算大概是多少？"
    cs.get_last_sop = lambda s: {"entry_query": q}
    sops.sop_needs_enter_confirm = lambda *a, **kw: False   # 本用例只测重开，不测进入门
    try:
        taken, reply = am._reenter_sop_on_retry(sid, q)
    finally:
        (trace_store.find_turn, trace_store.latest_turn, trace_store.find_turn_by_index,
         am._enter_sop, cs.get_last_sop, sops.sop_needs_enter_confirm) = saved
    _assert(taken, "重开被接管")
    _assert_in("预算", reply, "回到 SOP 第一步")
    trace_store.end_turn()
    rows = trace_reader.load(trace_store._path(sid))
    nodes = [n.get("node") for n in ((rows[-1] if rows else {}).get("path") or [])]
    _assert_in("sop.reenter", nodes, "path 里有 sop.reenter")
    reset_trace(sid)


def test_latest_turn_across_days():
    """latest_turn 取「最后写入」而不是「当天最大 turn_index」——跨天后它是昨天那轮。

    直接按日期目录铺两天的文件并固定 mtime：`_today` 不控制目录命名，
    用 begin_turn 模拟会让用例跟着真实日期走（曾在换日后误红）。
    """
    import json
    import os

    from tools import trace_store

    sid = "t-cross-day"
    reset_trace(sid)
    root = trace_store._root_dir()
    files = []
    for day, query in (("20261019", "昨天最后一轮"), ("20261020", "今天第一轮")):
        d = os.path.join(root, day, "%s_%s_0900" % (sid, day))
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "%s_%s_0900.jsonl" % (sid, day))
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({"ts": "%s-09:00:00" % day[:4] + "-" + day[4:6] + "-" + day[6:],
                                "turn_index": 1, "query": query}, ensure_ascii=False) + "\n")
        files.append(p)
    os.utime(files[0], (1000, 1000))
    os.utime(files[1], (2000, 2000))       # 第二天最后写入
    last = trace_store.latest_turn(sid)
    _assert_eq(last.get("date"), "2026-10-20", "取最后写入的那天")
    _assert_eq(last.get("query"), "今天第一轮", "取最后写入的那条")
    reset_trace(sid)


def test_reenter_across_days():
    """跨天后「重新回答」仍能重开：判据是「上一轮就是被重发的那轮」，不是 turn_index 相差 1。"""
    import tools.agent as am
    import tools.context_store as cs
    import sops
    from tools import trace_store

    sid, q = "t-reenter-cross", "帮我推荐一款扫地机器人"
    reset_trace(sid)
    trace_store.begin_turn(sid, q)                  # 当前轮
    saved = (trace_store.latest_turn, trace_store.find_turn, trace_store.find_turn_by_index,
             am._enter_sop, cs.get_last_sop, sops.sop_needs_enter_confirm)
    trace_store.latest_turn = lambda s: {"date": "2026-10-19", "turn_index": 7}
    trace_store.find_turn = lambda s, query: {"date": "2026-10-19", "turn_index": 7}
    trace_store.find_turn_by_index = lambda s, d, i: {"sop": {"id": "purchase"}, "branch": "start_sop"}
    am._enter_sop = lambda sid, sop_id, query, **kw: "您的预算大概是多少？"
    cs.get_last_sop = lambda s: {"entry_query": q}
    sops.sop_needs_enter_confirm = lambda *a, **kw: False   # 本用例只测跨天重开
    try:
        taken, reply = am._reenter_sop_on_retry(sid, q)
    finally:
        (trace_store.latest_turn, trace_store.find_turn, trace_store.find_turn_by_index,
         am._enter_sop, cs.get_last_sop, sops.sop_needs_enter_confirm) = saved
        reset_trace(sid)
    _assert(taken, "昨天最后一轮 + 今天重答 → 仍重开")
    _assert_in("预算", reply, "回到 SOP 第一步")


TESTS = [test_redis_missing_package_degrades, test_date_gate_needs_select_signal,
         test_upload_name_sanitized, test_build_dataset_writes_generated_not_dataset,
         test_intent_inference_failure_degrades, test_stream_generate_failure_falls_back,
         test_embedding_client_cached,
         test_intent_head_builds_from_meta, test_embedding_source_unified,
         test_local_lock_has_ttl, test_stop_flag_has_ttl_local, test_dead_batch_endpoint_removed,
         test_retry_replaces_previous_turn, test_retry_without_matching_last_user_keeps_turns,
         test_history_block_excludes_current_query, test_stopped_turn_marks_aborted,
         test_reenter_sop_logs_node, test_latest_turn_across_days, test_reenter_across_days]


def run():
    reset()
    return run_tests("代码 review 集中修复回归（依赖降级 / 日期闸门 / 上传净化 / 上下文与重答口径）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
