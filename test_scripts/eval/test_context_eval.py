"""多轮对话上下文评测：回放真实会话前缀 → 断言当前轮的出口行为（评测集 data/eval/context/golden.jsonl）。

评测集 data/eval/context/golden.jsonl（本地数据，已 gitignore；从真实会话整理，
结构与用法见同目录 README.md）。一行 = 一条标注（case），上下文按 session_id 现取。

harness 对每条 case：回放前缀重建上下文与 SOP 状态，再发 case.query，
按 expect.behaviour（出口行为）判定——读主链路 trace 记录的 branch（本轮唯一出口）。
数据里的 must_contain_any / must_not_contain 保留不动但暂不参与判定：
话术断言会随文案漂移，而「行为对不对」与「话里有没有某个词」是两回事——后者属内容级，归检索轨与人工复核。

判定口径：branch 是记录侧按 config/word_dict_config.py::BEHAVIOR_BY_TAG 折算出的唯一出口
（[Behavior] tag → 行为），取值构成闭集、可断言。不按日志行判定：一轮可能有多个 tag，
真正结束这一轮的是最后一个出口。

**判别器澄清（2026-10-21 起）**：v2.8 的操作判别器按分数选出候选，多个候选的出口不同时先问用户
选 1/2（trace 落 `ops.clarify` 节点，`branch=clarify`）。澄清这一轮按**候选出口**判：
候选操作的出口集合包含期望行为即通过；SOP 弱触发进入门（「不是 → 直接回答」）同理，
RAG 召回（answer / no_answer_fallback）在澄清里一律算「其他」选项。没进澄清的轮次仍按 `branch` 判
（执行体推进到哪就是哪）——注意「单候选」不等于「没澄清」：单候选的弱触发 SOP 也会先过进入门。

两类断言的分工：
  - 硬行为（safety_alert / safety_carry / refuse / no_answer_fallback / structured / answer）
    → 进回归门。
  - 软行为（chitchat / clarify / slot_filled / scope_guard）→ 一律不判失败：它们本来就没有
    唯一正确答案（“闲聊该怎么回”），所以只输出「软行为复核清单」（本次回复 + 偏离点）交人工 / LLM 复核；
    branch 与期望行为对不上时也列进清单（同一份复核，依旧不判失败）。

通过语义（xfail 风格，仅对硬行为生效）：
  - verdict=pass 或 fixed: true → 必过；失败 = 回归（硬失败）。
  - 其余（未修 badcase）→ 通过记 xpass（已修复），失败记 xfail（仍存在）；都不算回归。

运行（需 Ollama + Chroma）：
  .venv\\Scripts\\python.exe test_scripts/eval/test_context_eval.py --e2e
  # 基线快照：跑完存一份 / 与已存的逐条比对（形态与检索轨、意图轨一致）
  .venv\\Scripts\\python.exe test_scripts/eval/test_context_eval.py --e2e --save-baseline
  .venv\\Scripts\\python.exe test_scripts/eval/test_context_eval.py --e2e --compare
  # 子集：全量一次约 55 分钟（回放成本按会话累积），改一处只想验一小片时用
  # 注意：子集跑不能 --save-baseline（会把子集当成全量基线）
  .venv\\Scripts\\python.exe test_scripts/eval/test_context_eval.py --list-sets
  .venv\\Scripts\\python.exe test_scripts/eval/test_context_eval.py --e2e --set smoke --compare
  .venv\\Scripts\\python.exe test_scripts/eval/test_context_eval.py --e2e --only CTX-20261015-01,CTX-20261009-*
"""
import json
import os
import shutil
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 与 test_scripts._runner 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
from test_scripts._runner import *
from config.word_dict_config import BEHAVIOR_BY_TAG, TRACE_NODE_LABELS

_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))   # 上两级：test_scripts/eval/ → 项目根
_GOLDEN_FILE = os.path.join(_ROOT, "data", "eval", "context", "golden.jsonl")
_REVIEW_DIR = os.path.join(_ROOT, "data", "eval", "output")   # 复核文件：<YYYYMMDD>/<HHMM>.md
# 基线快照（与检索轨 / 意图轨同一形态：聚合计数 + ids + per_id + env）
_BASELINE_FILE = os.path.join(_ROOT, "data", "eval", "context", "baseline.json")
_PREV_BASELINE_FILE = os.path.join(_ROOT, "data", "eval", "context", "baseline.prev.json")

# 唯一还靠文本的地方：回放「孤儿危险告警轮」时用它认出那条 assistant 回复（见 _replay_prefix）
_DANGER_ALERT_MARK = "请立即停止使用"

# 出口行为闭集 = 记录侧 trace 的 branch 取值（含流程类的 4 个），在 config/word_dict_config.py 登记。
# 行为轨只认这一份表，不再自带副本——两处各存一张是会漂移的。
_BEHAVIOURS = set(BEHAVIOR_BY_TAG.values())

# 有唯一正确出口的硬行为 → 断言「本轮 branch == 期望行为」
_HARD_BEHAVIOURS = {"safety_alert", "safety_carry", "refuse",
                    "no_answer_fallback", "structured", "answer"}
# 没有唯一正确答案的软行为 → 不判失败；branch 对不上只进复核清单
# （scope_guard / slot_filled 两个期望没有专属 branch，只从期望侧出现）
_SOFT_BEHAVIOURS = {"chitchat", "clarify", "slot_filled", "scope_guard"}

# 期望行为 → 可接受的 branch（软行为的「行为对得上吗」提示用；硬行为判定是相等比较）。
# 两个没有专属 branch 的期望按 project_detail.md §4.13「与存量 behaviour 的映射」折算：
# 越界应由拒答分支接住（scope_guard）；槽位提取的证据是流程内推进或预算直出（slot_filled）。
_ACCEPTED_BRANCHES = {b: {b} for b in _BEHAVIOURS}
_ACCEPTED_BRANCHES["scope_guard"] = {"refuse"}
_ACCEPTED_BRANCHES["slot_filled"] = {"sop_step", "structured"}

# ── 判别器出口（2026-10-21）──────────────────────────────────────
# 操作 → 它在链路里可能落到的出口行为。澄清那一轮 branch 只能是 clarify，
# 但用户是在这些候选出口里选，所以判据要看候选出口是否覆盖期望行为。
# 单候选时不改判：执行体照常推进，branch 就是它的出口。
_OPS_EXITS = {
    "followup": {"structured"},                                  # 追问筛选 → 直出列表
    "model": {"structured"},                                     # 型号详情 / 对比 → 直出
    "series": {"structured"},                                    # 系列枚举 → 直出
    "service_point": {"structured", "sop_step", "start_sop"},    # 有坐标直出；否则进网点 SOP
    "sop": {"start_sop", "sop_step", "clarify"},                 # 强触发直接进；弱触发先过确认门
    "date": {"structured", "answer", "no_answer_fallback"},      # 日期闸门不过 → 落 RAG
    "structured": {"structured", "answer", "no_answer_fallback"},
    "domain": {"answer", "no_answer_fallback"},                  # 知识问答 = RAG
}
# RAG（知识问答）在澄清里算「其他」选项，但只有下面这些澄清真把它列出来：
# SOP 进入门的「不是 → 直接回答」、判别器澄清的「其他」；退出门只有「退出 / 继续」。
_RAG_EXITS = {"answer", "no_answer_fallback"}
_RAG_CLARIFY_KINDS = {"confirm_enter", "confirm_enter_retry",
                      "ops_clarify", "ops_clarify_retry"}

# 缺触发轮时的危险占位消息（含危险词，供 agent._window_danger_word 取词）
_DANGER_PLACEHOLDER = "（漏电告警轮：原始用户消息未落盘）"


_SESS_DIR = os.path.join(os.path.dirname(_GOLDEN_FILE), "sessions")   # 评测集自带的（手工/合成）会话


_CONTEXT_ROOT = None      # 真实会话目录（惰性解析，见 _real_context_root）


def _real_context_root() -> str:
    """真实会话目录。不能用 context_store._CONTEXT_DIR：评测进程里 _runner 为隔离被测写入
    把它指到了 data/test_context，而评测集要读的是真实会话（契约测试会当场报出读不到）。
    """
    global _CONTEXT_ROOT
    if _CONTEXT_ROOT is None:
        from tools.path_tool import get_abs_path   # 与 context_store 同一个路径助手，但不读 CONTEXT_DIR

        _CONTEXT_ROOT = get_abs_path("data/context")
    return _CONTEXT_ROOT


def _session_file(session_id: str) -> str:
    """按 session_id 找会话文件：评测集自带的 sessions/ 优先，再在真实会话目录里找。

    目录布局与命名都兼容：<day>/<stem>.jsonl 与根下 <stem>.jsonl；stem 有 <sid>（旧）
    与 <sid>_<YYYYMMDD>_<HHMM>（新）两种。
    """
    own = os.path.join(_SESS_DIR, "%s.jsonl" % session_id)
    if os.path.isfile(own):
        return own
    root = _real_context_root()
    if not os.path.isdir(root):
        return ""

    def _match(stem: str) -> bool:
        return stem == session_id or stem.startswith(session_id + "_")

    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            for fn in sorted(os.listdir(full)):
                if fn.endswith(".jsonl") and _match(fn[:-6]):
                    return os.path.join(full, fn)
        elif entry.endswith(".jsonl") and _match(entry[:-6]):
            return full
    return ""


def _turns_of(row: dict) -> list:
    """会话消息：老行自带 turns 就用它，否则按 session_id 现读会话文件（行里不再内嵌上下文）。"""
    if row.get("turns"):
        return row["turns"]
    path = _session_file(row.get("session_id") or "")
    if not path:
        return []
    msgs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    msgs.append(json.loads(line))
                except ValueError:
                    continue
    return msgs


def _load_golden():
    """读评测集：一行一条 case，按 session_id 分组返回（回放按会话走）。

    兼容两种行形态：新形态只带 session_id + query_index（上下文按 session_id 现取，行里不内嵌）；
    旧形态（迁移前，见 golden.prev.jsonl）一行一个会话，自带 turns 与 cases[]。
    """
    with open(_GOLDEN_FILE, encoding="utf-8") as f:
        text = f.read()
    decoder = json.JSONDecoder()
    groups, order, idx = {}, [], 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, idx = decoder.raw_decode(text, idx)
        sid = obj.get("session_id")
        cases = obj["cases"] if "cases" in obj else [obj]
        if sid not in groups:
            groups[sid] = {"session_id": sid, "turns": obj.get("turns") or [], "cases": []}
            order.append(sid)
        groups[sid]["cases"].extend(cases)
    return [groups[s] for s in order]


def _validate_golden(rows) -> None:
    """评测集自检：归因取值、fixed 语义、会话上下文可达——写错当场报出来，别等跑完 20 分钟。"""
    from config.word_dict_config import FEEDBACK_REASON_KEYS

    cases = [c for row in rows for c in row["cases"]]
    bad = [c["id"] for c in cases if (c.get("type") or "") and c["type"] not in FEEDBACK_REASON_KEYS]
    _assert(not bad, "归因 type 取值都在 FEEDBACK_REASONS 内" + (f"（越界：{bad}）" if bad else ""))
    bad = [c["id"] for c in cases if c.get("fixed") and c.get("verdict") != "fail"]
    _assert(not bad, "fixed（曾坏已修）的条目 verdict 都是 fail" + (f"（越界：{bad}）" if bad else ""))
    missing = [row.get("session_id") for row in rows if not _turns_of(row)]
    _assert(not missing, "每条 case 的会话上下文都可取" + (f"（缺：{missing}）" if missing else ""))


def _branch_text(branch) -> str:
    """给人看的出口摘要；没记到 branch 是要报出来的事实，不是空白。"""
    return "branch=" + (branch or "未记")


def _ops_clarify_ops(record: dict) -> list:
    """本轮若走了操作澄清，返回候选操作（top1/top2，按展示顺序）；否则 []。"""
    for node in reversed((record or {}).get("path") or []):
        if node.get("node") == "ops.clarify":
            facts = node.get("in") or {}
            opts = [facts.get("top1"), facts.get("top2")]
            return [o for o in opts if o and o != "-"]
    return []


def _clarify_covers(behaviour, record, branch=None) -> str:
    """澄清是否覆盖期望行为：候选操作的出口命中，或 RAG 作为「其他」选项。

    两类澄清都算：判别器 ops.clarify（选项 = 两个操作）与 SOP 弱触发的进入门
    （选项 = 进入引导 / 「不是，直接回答」）。RAG 召回（answer / no_answer_fallback）
    在澄清里一律算「其他」选项——用户随时可以选它。
    """
    if not behaviour:
        return ""
    opts = _ops_clarify_ops(record)
    hits = [op for op in opts if behaviour in (_OPS_EXITS.get(op) or set())]
    if hits:
        return "澄清选项 %s 的出口包含 %s" % ("/".join(hits), behaviour)
    cur = branch if branch is not None else (record or {}).get("branch")
    kind = ((record or {}).get("detail") or {}).get("kind")
    if (behaviour in _RAG_EXITS and cur == "clarify"
            and (kind is None or kind in _RAG_CLARIFY_KINDS)):
        return "澄清里「其他」= RAG，可覆盖 %s" % behaviour
    return ""


def _check_case(expect: dict, branch, tag=None, record=None) -> list:
    """返回问题清单（空 = 通过）。判据只有出口行为（读 trace 的 branch）。

    判别器澄清轮另算：候选出口覆盖期望行为即通过（见 _clarify_covers 与文首口径）。
    must_contain_any / must_not_contain 暂不参与判定：它们判的是“话里有没有某个词”，
    与“这一轮被处置成哪类行为”是两回事（内容级问题归检索轨与人工复核）。
    """
    problems = []
    beh = expect.get("behaviour")
    if beh in _HARD_BEHAVIOURS:
        if not branch:
            # 两种缺法要分开报：没记 tag = 出口漏了观测点；有 tag 但折不出 branch = tag 没登记
            problems.append("本轮 trace 未记出口："
                            + ("tag=%s 未登记在 BEHAVIOR_BY_TAG" % tag if tag
                               else "本轮无 [Behavior] 观测（出口漏了观测点）"))
        elif branch not in _BEHAVIOURS:
            problems.append(f"未登记的出口 branch={branch}（应在 BEHAVIOR_BY_TAG 内登记）")
        elif branch != beh and not _clarify_covers(beh, record, branch):
            problems.append(f"出口行为应为 {beh}，实际 {_branch_text(branch)}")
    elif beh:
        # 软行为不判对错，但「行为对不上」是有价值的复核信号：只列进复核清单，不参与失败判定。
        accepted = _ACCEPTED_BRANCHES.get(beh) or set()
        if not branch:
            problems.append(f"期望行为 {beh}，本轮未记出口（软行为，仅供复核参考）")
        elif branch not in accepted and not _clarify_covers(beh, record, branch):
            problems.append(f"期望行为 {beh}，实际 {_branch_text(branch)}（软行为，仅供复核参考）")
    return problems


def _ask(sid: str, query: str):
    """发一轮，返回 (回复, 本轮 trace 记录)。

    判定读记录里的 branch（唯一出口），失败归因读 path / route / retrieval——同一份记录，
    不再是两处证据。end_turn() 在本轮生成器消费完时落盘，所以返回时记录已在。
    """
    from tools.agent import ask_stream

    reply = "".join(ask_stream(query, sid)).strip()
    return reply, read_last_turn(sid)


def _replay_prefix(sid: str, turns: list, lo: int, hi: int):
    """回放 turns[lo:hi] 重建上下文与状态。

    user 轮真跑一遍（驱动 SOP 状态、落盘、按内容打 danger 标记），其回复用当前系统生成的
    结果落盘（而非数据里的历史 actual）——否则上一条被修复的回复不会影响下一条的承接判定。
    assistant 轮已被上一条 user 轮的生成结果代表，跳过。

    例外：危险告警轮可能缺触发它的 user 消息（见数据 anomalies，当时的 guard 在落盘前
    return），此时补一条带 danger 标记的占位 user 消息，否则后续「安全承接」无从触发。
    """
    from tools.context_store import append_message
    for idx in range(lo, hi):
        t = turns[idx]
        if t["role"] == "user":
            reply, _record = _ask(sid, t["content"])
            append_message(sid, "assistant", reply)
            continue
        prev_is_user = idx > 0 and turns[idx - 1]["role"] == "user"
        if not prev_is_user and t["content"].startswith(_DANGER_ALERT_MARK):
            append_message(sid, "user", _DANGER_PLACEHOLDER, danger=True)
            append_message(sid, "assistant", t["content"])


def _is_must_pass_case(case: dict) -> bool:
    """该 case 是否必过：硬行为，且「采集时就是 pass」或「曾坏已修（fixed）」。"""
    return (case["expect"].get("behaviour") in _HARD_BEHAVIOURS
            and ((case.get("verdict") == "pass") or bool(case.get("fixed"))))


def _run_row(row: dict) -> list:
    import sops.base
    from tools.context_store import append_message, delete_session
    from tools.pending_store import close as clear_pending

    sid = "ctxeval-" + row["session_id"]
    sops.base._sessions = {}
    # 澄清状态统一在 pending_store：
    # Redis 键 + 内存降级都要清，否则同一 sid 上一行的残留会污染下一行。
    clear_pending(sid)
    # 主链路 trace（tools/trace_store.py）也按 sid 落盘，同一 sid 跨行复用——先清掉上轮的落盘与
    # 进程内计数，否则 turn_index 会接着前一次运行数、文件也会累积。
    reset_trace(sid)
    # 上一次运行若被中断，会话文件会留在盘上（收尾才删）——开头也删一次，让中断后的重跑自愈。
    delete_session(sid)

    turns = _turns_of(row)
    if not turns:
        # 取不到上下文就别回放（否则等于换了另一个用例）；逐条报出来
        return [{"case": c, "reply": "<NO SESSION>",
                 "problems": ["找不到会话文件，无法回放上下文"],
                 "hard": c["expect"].get("behaviour") in _HARD_BEHAVIOURS,
                 "must_pass": False, "branch": None, "tag": None, "trace": ""} for c in row["cases"]]
    results = []
    replayed = 0
    for case in sorted(row["cases"], key=lambda c: c["query_index"]):
        qi = case["query_index"]
        try:
            _replay_prefix(sid, turns, replayed, qi)
            reply, record = _ask(sid, case["query"])
            append_message(sid, "assistant", reply)
            problems = _check_case(case["expect"], record.get("branch"),
                                   record.get("tag"), record=record)
        except Exception as exc:  # noqa: BLE001
            reply = f"<EXCEPTION {type(exc).__name__}: {exc}>"
            record = {}
            problems = [f"执行异常：{type(exc).__name__}: {exc}"]
        if record.get("aborted"):
            # 评测里正常不会出现（生成器必被消费完）；真出现说明这轮被截断，行为判定不可信
            problems.append("本轮 trace 标记 aborted（回答未跑完，判定不可信）")
        hard = case["expect"].get("behaviour") in _HARD_BEHAVIOURS
        must_pass = _is_must_pass_case(case)
        beh = case["expect"].get("behaviour")
        via = (_clarify_covers(beh, record) if record and record.get("branch") != beh else "")
        # 有问题的条目附上本轮 trace 摘要（主链路落的结构化环节记录），省得回头 grep 日志
        results.append({"case": case, "reply": reply, "problems": problems,
                        "hard": hard, "must_pass": must_pass,
                        "branch": record.get("branch"), "tag": record.get("tag"), "via": via,
                        "trace": _trace_brief(record) if problems else ""})
        replayed = qi + 1

    delete_session(sid)
    clear_pending(sid)
    sops.base._sessions = {}
    return results


def _reason_text(key) -> str:
    """归因展示：wrong_route「答非所问」；空值 = 正例基线。与前端 feedback 同一套 key。"""
    if not key:
        return "正例"
    try:
        from config.word_dict_config import FEEDBACK_REASON_LABELS

        return "%s「%s」" % (key, FEEDBACK_REASON_LABELS.get(key, ""))
    except Exception:  # noqa: BLE001
        return str(key)


def _trace_brief(record: dict) -> str:
    """把一条 trace 记录压成一行人读摘要（失败归因用）；没有 trace 返回空串。

    主读 v3 分层块（route / retrieval / path），缺失时回落到 v2 投影 steps——
    存量 v2 记录也看得懂。
    """
    if not record:
        return ""
    steps = record.get("steps") or {}
    retrieval = record.get("retrieval") or {}
    bits = []
    old_intent = steps.get("intent") or {}
    intent = record.get("route", {}).get("intent") or {
        "label": old_intent.get("intent"), "margin": old_intent.get("margin")}
    if intent.get("label"):
        margin = intent.get("margin")
        bits.append("意图=%s%s" % (intent["label"],
                                  "(%.3f)" % float(margin) if isinstance(margin, (int, float)) else ""))
    rewrite = retrieval.get("rewrite") or steps.get("rewrite") or {}
    if rewrite and rewrite.get("via") not in (None, "none"):
        bits.append("改写=%s→「%s」" % (rewrite.get("via"), rewrite.get("effective_query")))
    domain = retrieval.get("domain")
    domain = domain.get("value") if isinstance(domain, dict) else (steps.get("retrieve") or {}).get("domain")
    n_chunks = (steps.get("retrieve") or {}).get("n_chunks")
    if domain or n_chunks is not None:
        bits.append("检索=%s/%s条" % (domain or "全库", n_chunks if n_chunks is not None else "?"))
    if len(retrieval.get("passes") or []) > 1:
        bits.append("两遍召回")
    if record.get("path"):
        bits.append("路径=" + "→".join(_node_text(n) for n in record["path"][-4:]))
    ms = record.get("ms") or {}
    if ms:
        bits.append("耗时ms=" + "/".join("%s:%s" % (k, v) for k, v in sorted(ms.items())))
    return " · ".join(bits)


def _node_text(node: dict) -> str:
    """路径节点渲染：登记表里的中文名 + 非默认状态（命中 / 跳过一眼可读）。"""
    name = node.get("node") or "?"
    label = TRACE_NODE_LABELS.get(name, name)
    status = node.get("status") or "ok"
    return label if status == "ok" else "%s[%s]" % (label, status)


def _trace_version() -> str:
    """主链路 trace 的记录版本（tools/trace_store.py::TRACE_VERSION），写进报告便于日后对齐。"""
    try:
        from tools import trace_store

        return "v%s" % trace_store.TRACE_VERSION
    except Exception:  # noqa: BLE001
        return "?"


def _print_detail(results):
    print("── 硬行为（有 branch 可判定 → 进回归门）──")
    for r in (x for x in results if x["hard"]):
        c = r["case"]
        if not r["problems"]:
            mark = "PASS" if r["must_pass"] else "xpass"
        else:
            mark = "REGRESS" if r["must_pass"] else "xfail"
        kind = "必过" if r["must_pass"] else "badcase"
        print(f"  [{mark:<7}] {c['id']} ({kind}/{_reason_text(c.get('type'))}) {c['query']}"
              f"   [{_branch_text(r['branch'])}]"
              + (f"  · 澄清覆盖：{r['via']}" if r.get("via") else ""))
        if r["problems"]:
            print(f"              ↳ {'；'.join(r['problems'])}")
            if r.get("trace"):
                print(f"              · 本轮环节：{r['trace']}")

    print("── 软行为（无唯一正确答案 → 只出复核清单，不判失败）──")
    for r in (x for x in results if not x["hard"]):
        c = r["case"]
        if r["problems"]:
            mark = "⚠需复核"
        elif c.get("verdict") == "fail":
            mark = "✓已转通过"
        else:
            mark = "· 未偏离"
        print(f"  [{mark:<7}] {c['id']} ({_reason_text(c.get('type'))}) {c['query']}"
              f"   [{_branch_text(r['branch'])}]"
              + (f"  · 澄清覆盖：{r['via']}" if r.get("via") else ""))
        if r["problems"]:
            print(f"              ↳ {'；'.join(r['problems'])}")
            if r.get("trace"):
                print(f"              · 本轮环节：{r['trace']}")


def _review_path():
    """复核文件路径：data/eval/output/<YYYYMMDD>/<HHMM>.md（同一分钟重复跑则加 -2 / -3）。"""
    day_dir = os.path.join(_REVIEW_DIR, time.strftime("%Y%m%d"))
    os.makedirs(day_dir, exist_ok=True)
    stamp = time.strftime("%H%M")
    path = os.path.join(day_dir, stamp + ".md")
    n = 2
    while os.path.exists(path):
        path = os.path.join(day_dir, f"{stamp}-{n}.md")
        n += 1
    return path


def _write_review(rows, results, hard_results, soft_results, must_pass, regressions):
    """把软行为复核清单落盘——复核是人工 / LLM 的活，得能拿出去看。

    软行为没有唯一正确答案，所以文件里只摆事实：期望行为 + 实际 branch + 本次回复 + 偏离点 + 归档判据。
    """
    need_review = [r for r in soft_results if r["problems"]]
    out = [
        "# 上下文轨 · 软行为复核清单",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 评测集：`data/eval/context/golden.jsonl`"
        f"（{len(rows)} 会话 / {len(results)} 条标注：硬 {len(hard_results)} · 软 {len(soft_results)}）",
        f"- trace 记录：{_trace_version()}（`tools/trace_store.py`，每轮一条；有问题的条目附「本轮环节」）",
        f"- 硬行为（回归门）：必过 {len(must_pass) - len(regressions)}/{len(must_pass)} 通过"
        + (f"，**{len(regressions)} 条回归需处理**" if regressions else ""),
        f"- 软行为：**{len(need_review)}/{len(soft_results)} 条行为与期望不符**（不判失败，按下表复核）",
        "",
        "> 软行为本来就没有“必须包含某句话”的正确答案，所以只列事实，判断留给人 / LLM。",
        "",
    ]
    for r in soft_results:
        c, exp = r["case"], r["case"]["expect"]
        if r["problems"]:
            mark = "⚠ 需复核"
        elif c.get("verdict") == "fail":
            mark = "✓ 已转通过"
        else:
            mark = "· 未偏离"
        out += [f"## {mark} {c['id']}（{_reason_text(c.get('type'))}）", "",
                f"- 用户：{c['query']}",
                f"- 期望行为：{exp.get('behaviour')}",
                f"- 实际 branch：{_branch_text(r['branch'])}",
                f"- 本次回复：{r['reply']}"]
        if r["problems"]:
            out.append(f"- 偏离点：{'；'.join(r['problems'])}")
        if r.get("trace"):
            out.append(f"- 本轮环节：{r['trace']}")
        if c.get("note"):
            out.append(f"- 归档判据：{c['note']}")
        out.append("")
    path = _review_path()
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return path


# ──────────────────────────────────────────────────────────────
# 基线快照：把「这次跑出来是什么样」落成可比对的 JSON（与检索轨 / 意图轨同一形态）
# ──────────────────────────────────────────────────────────────

def _env_meta() -> dict:
    """本轨的环境指纹：复用检索轨那套（代码 / 知识库 / 配置 / 索引），

    再补本轨特有的三项——分类头与训练集（意图判错会直接改变出口行为）、trace 记录版本。
    """
    from test_scripts.eval import test_retrieval_eval as retr
    from tools import trace_store

    env = retr._env_meta()
    try:
        from test_scripts.eval import test_intent_eval as intent

        head = intent._env_meta()
        env["intent"] = {"head": head.get("head"), "dataset": head.get("dataset")}
    except Exception as exc:  # noqa: BLE001 - 取不到不该拦住评测，但降级必须显式（AGENTS.md 准则 3）
        print(f"  [环境] 分类头 / 训练集指纹取不到（{type(exc).__name__}: {exc}）→ env.intent 留空")
        env["intent"] = {}
    env["trace_version"] = getattr(trace_store, "TRACE_VERSION", None)
    return env


def _baseline_payload(rows, results) -> dict:
    """基线内容：聚合计数 + ids + per_id（每条「与期望行为是否一致」）+ env。

    per_id 的口径：硬行为比 branch 是否等于期望、软行为比是否落在接受集内，
    取不到出口（没记 branch / 未登记）也算不一致。软行为那条只是参考，不当门禁。
    """
    hard = [r for r in results if r["hard"]]
    soft = [r for r in results if not r["hard"]]
    must = [r for r in hard if r["must_pass"]]
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {
            "sessions": len(rows), "cases": len(results),
            "hard": len(hard), "soft": len(soft),
            "must_pass_passed": sum(1 for r in must if not r["problems"]),
            "must_pass_total": len(must),
            "xfail": sum(1 for r in hard if not r["must_pass"] and r["problems"]),
            "xpass": sum(1 for r in hard if not r["must_pass"] and not r["problems"]),
            "soft_deviated": sum(1 for r in soft if r["problems"]),
        },
        "ids": [r["case"]["id"] for r in results],
        "per_id": {r["case"]["id"]: not r["problems"] for r in results},
        "env": _env_meta(),
    }


def _save_baseline(rows, results) -> None:
    payload = _baseline_payload(rows, results)
    os.makedirs(os.path.dirname(_BASELINE_FILE), exist_ok=True)
    rotated = os.path.isfile(_BASELINE_FILE)
    if rotated:
        shutil.copyfile(_BASELINE_FILE, _PREV_BASELINE_FILE)
    with open(_BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    c = payload["counts"]
    print(f"\n[基线] 已写入 {os.path.relpath(_BASELINE_FILE, _ROOT)}（{c['cases']} 条："
          f"必过 {c['must_pass_passed']}/{c['must_pass_total']}、未修仍失败 {c['xfail']}、"
          f"已转通过 {c['xpass']}、软行为偏离 {c['soft_deviated']}）")
    if rotated:
        print(f"       上一版留存为 {os.path.relpath(_PREV_BASELINE_FILE, _ROOT)}")


def _print_id_list(label, ids) -> None:
    if not ids:
        print(f"{label}：无")
    elif len(ids) <= 12:
        print(f"{label}：{'、'.join(ids)}")
    else:
        print(f"{label}：{'、'.join(ids[:12])} …（共 {len(ids)} 条）")


def _print_env_compare(env, base_env) -> None:
    """核对基线采集时的代码 / 知识库 / 配置 / 索引 / 分类头 / trace——结论漂了先看这里。"""
    if not base_env:
        print("  基线未记录环境（旧格式）—— 跑一次 --save-baseline 即可核对")
        return
    from test_scripts.eval import test_retrieval_eval as retr

    code, base_code = env.get("code") or {}, base_env.get("code") or {}
    print(f"  代码    : {retr._code_label(code)}  vs  基线 {retr._code_label(base_code)}")
    kb, base_kb = env.get("kb") or {}, base_env.get("kb") or {}
    print(f"  知识库  : {kb.get('files')} 文件 fp={kb.get('fingerprint')}"
          f"  vs  基线 {base_kb.get('files')} 文件 fp={base_kb.get('fingerprint')}"
          + ("" if kb.get("fingerprint") == base_kb.get("fingerprint") else "  ⚠ 已变"))
    cfg, base_cfg = env.get("config") or {}, base_env.get("config") or {}
    print(f"  配置    : fp={cfg.get('fingerprint')}  vs  基线 fp={base_cfg.get('fingerprint')}"
          + ("" if cfg.get("fingerprint") == base_cfg.get("fingerprint") else "  ⚠ 已变"))
    print(f"  索引    : {retr._index_label(env.get('index') or {})}"
          f"  vs  基线 {retr._index_label(base_env.get('index') or {})}")
    it, base_it = env.get("intent") or {}, base_env.get("intent") or {}
    print(f"  分类头  : {it.get('head')}  vs  基线 {base_it.get('head')}"
          + ("" if it.get("head") == base_it.get("head") else "  ⚠ 已变（意图判错会直接改出口）"))
    print(f"  训练集  : {it.get('dataset')}  vs  基线 {base_it.get('dataset')}"
          + ("" if it.get("dataset") == base_it.get("dataset") else "  ⚠ 已变"))
    print(f"  trace   : v{env.get('trace_version')}  vs  基线 v{base_env.get('trace_version')}")


def _compare_baseline(rows, results) -> None:
    """与基线逐条比 + 核对环境。

    分类与检索轨一致，但「回归」只认必过条目：必过变红才是红灯；
    未修 badcase 从「已转通过」退回失败单列「退步」——是信息，不是门禁失败。
    """
    if not os.path.isfile(_BASELINE_FILE):
        print(f"\n[基线] 无基线快照（{os.path.relpath(_BASELINE_FILE, _ROOT)}）"
              f"——跑一次 --save-baseline 才会有")
        return
    with open(_BASELINE_FILE, encoding="utf-8") as f:
        base = json.load(f)
    old = base.get("per_id") or {}
    cur = {r["case"]["id"]: (not r["problems"], bool(r["must_pass"])) for r in results}
    regress = sorted(i for i, (ok, must) in cur.items() if must and old.get(i) is True and not ok)
    back = sorted(i for i, (ok, must) in cur.items() if not must and old.get(i) is True and not ok)
    improve = sorted(i for i, (ok, _m) in cur.items() if old.get(i) is False and ok)
    added = sorted(i for i in cur if i not in old)
    missing = sorted(i for i in (base.get("ids") or []) if i not in cur)

    print("\n[基线环境]")
    print(f"  基线采集于 {base.get('created_at', '?')}")
    _print_env_compare(_env_meta(), base.get("env") or {})
    print("[基线逐条]（按 id 比「与期望行为是否一致」）")
    _print_id_list("  回归（必过条目变红）", regress)
    _print_id_list("  退步（未修 badcase 由通过退回失败）", back)
    _print_id_list("  改善（由不一致转为一致）", improve)
    _print_id_list("  新增（基线无记录，首次）", added)
    _print_id_list("  未跑（基线有、本次没跑）", missing)


# ──────────────────────────────────────────────────────────────
# 子集：全量一次要按 21 个会话各回放前缀（约 120+ 轮真跑），改一处只想验一小片时按集合跑
# ──────────────────────────────────────────────────────────────

def _arg_value(flag: str) -> str:
    """取 --flag value 的值；没给返回空串。"""
    if flag in sys.argv and len(sys.argv) > sys.argv.index(flag) + 1:
        return sys.argv[sys.argv.index(flag) + 1]
    return ""


def _latest_batch() -> str:
    """评测集里最新的那批日期（id 形如 CTX-<YYYYMMDD>-<序号>）。"""
    dates = [i.split("-")[1] for i in _all_case_ids() if i.count("-") >= 2]
    return max(dates) if dates else ""


def _all_case_ids() -> list:
    if not os.path.isfile(_GOLDEN_FILE):
        return []
    return [c.get("id", "") for row in _load_golden() for c in row["cases"]]


# 命名集合：按 case 自身可核实的字段筛（期望行为 / 必过语义 / 位置 / 批次），不靠人工维护清单
_SET_DESC = {
    "smoke": "必过集合（回归门）——只想确认「没把基线弄红」时跑它",
    "safety": "期望安全出口（safety_alert / safety_carry）",
    "answer": "期望作答类出口（answer / structured / no_answer_fallback / refuse）",
    "soft": "软行为（chitchat / clarify / slot_filled / scope_guard）",
    "fast": "位置最靠前的用例（回放少、最省时间）",
    "recent": "最新一批（id 日期最大的那批）",
}
_SETS = {
    "smoke": _is_must_pass_case,
    "safety": lambda c: c["expect"].get("behaviour") in ("safety_alert", "safety_carry"),
    "answer": lambda c: c["expect"].get("behaviour")
    in ("answer", "structured", "no_answer_fallback", "refuse"),
    "soft": lambda c: c["expect"].get("behaviour") in _SOFT_BEHAVIOURS,
    "fast": lambda c: c["query_index"] <= 4,
    "recent": lambda c: c.get("id", "").startswith("CTX-%s-" % _latest_batch()),
}


def _replay_turns(rows) -> int:
    """这套用例要真跑多少轮：按会话累积（同一会话只回放到最后一个 case 的位置）。"""
    total = 0
    for row in rows:
        cases = sorted(row["cases"], key=lambda c: c["query_index"])
        if not cases:
            continue
        hi = cases[-1]["query_index"]
        total += 1 + sum(1 for t in _turns_of(row)[:hi] if t.get("role") == "user")
    return total


def _select_sets(rows, names=(), ids=()):
    """按集合名 / id（支持 前缀*）过滤；两者都给就取交集。返回 (rows, 说明文字)。"""
    bad = set(names) - set(_SETS)
    if bad:
        raise ValueError("未知集合：%s（可用：%s）"
                         % ("、".join(sorted(bad)), "、".join(sorted(_SETS))))

    def keep(case):
        if names and not any(_SETS[n](case) for n in names):
            return False
        if ids:
            cid = case.get("id", "")
            if not any(cid == i or (i.endswith("*") and cid.startswith(i[:-1])) for i in ids):
                return False
        return True

    out, kept = [], 0
    for row in rows:
        cases = [c for c in row["cases"] if keep(c)]
        kept += len(cases)
        if cases:
            out.append(dict(row, cases=cases))
    total = sum(len(r["cases"]) for r in rows)
    bits = []
    if names:
        bits.append("集合 " + "、".join(sorted(names)))
    if ids:
        bits.append("id " + "、".join(sorted(ids)))
    return out, "%s（%d/%d 条，回放 %d 轮）" % (" + ".join(bits), kept, total, _replay_turns(out))


def _print_sets(rows) -> None:
    """列出可用集合与各自的规模 / 回放成本——挑集合时先看这个。"""
    total = sum(len(r["cases"]) for r in rows)
    print(f"可用集合（全量 {total} 条 / 回放 {_replay_turns(rows)} 轮）：")
    for name in sorted(_SETS):
        sub, desc = _select_sets(rows, [name])
        print(f"  {name:<8} {desc}")
        print(f"           {_SET_DESC[name]}")
    print("\n用法：--e2e --set smoke --compare ｜ --e2e --only CTX-20261015-01,CTX-20261009-*")
    print("注意：子集跑不能 --save-baseline（会把子集当成全量基线）。")


def run():
    reset()
    if "--list-sets" in sys.argv:            # 纯数据：不用 --e2e 也能看
        if os.path.isfile(_GOLDEN_FILE):
            _print_sets(_load_golden())
        else:
            print(f"未找到评测集 {_GOLDEN_FILE}（本地数据目录 data/eval，已 gitignore）")
        return stats()
    if not E2E:
        _skip("多轮上下文评测需要 Ollama + Chroma（加 --e2e 运行）")
        return stats()
    if not os.path.isfile(_GOLDEN_FILE):
        _skip(f"未找到评测集 {_GOLDEN_FILE}（本地数据目录 data/eval，已 gitignore）")
        return stats()

    rows = _load_golden()
    _assert(len(rows) > 0, f"评测集加载成功（{len(rows)} 个会话）")
    _validate_golden(rows)

    # 子集：--set / --only 过滤。基线只跟子集比（未跑桶会列出没选的），存基线会被拦
    subset_desc = ""
    if "--set" in sys.argv or "--only" in sys.argv:
        try:
            rows, subset_desc = _select_sets(
                rows,
                [s.strip() for s in _arg_value("--set").split(",") if s.strip()],
                [s.strip() for s in _arg_value("--only").split(",") if s.strip()])
        except ValueError as exc:
            _assert(False, f"子集参数非法：{exc}")
            return stats()
        if not rows:
            _assert(False, f"子集没命中任何用例（{subset_desc}）")
            return stats()
        _assert(True, f"子集命中用例：{subset_desc}")
        print(f"  [子集] {subset_desc}")

    # 影子探针在这条轨上是纯开销：多跑一次 FC（实测每轮 +75~125s），且它只落日志、不改被测行为。
    # 跑用例期间关掉，算力让给被测主链路；跑完即恢复（run_tests 同进程还会跑后面的模块）。
    restore_shadow = disable_shadow()
    print("  [隔离] 影子探针已关闭：算力让给被测主链路（见 _runner.disable_shadow）")
    results = []
    try:
        for row in rows:
            results.extend(_run_row(row))
    finally:
        restore_shadow()

    _print_detail(results)

    print()
    print("=" * 72)
    print("多轮上下文评测汇总")
    print("=" * 72)
    hard_results = [r for r in results if r["hard"]]
    soft_results = [r for r in results if not r["hard"]]
    must_pass = [r for r in hard_results if r["must_pass"]]
    regressions = [r for r in must_pass if r["problems"]]
    xfail = [r for r in hard_results if not r["must_pass"] and r["problems"]]
    xpass = [r for r in hard_results if not r["must_pass"] and not r["problems"]]
    need_review = [r for r in soft_results if r["problems"]]

    print(f"  会话 {len(rows)} 个 / 标注 {len(results)} 条（硬行为 {len(hard_results)} · 软行为 {len(soft_results)}）")
    print(f"  trace 记录版本：{_trace_version()}（失败归因取自它；逐轮时间线用 test_scripts/trace_report.py 看）")
    try:                                    # 索引与知识库对不上时，任何轨都要喊一声（不只是 --compare）
        from test_scripts.eval import test_retrieval_eval as _retr

        _retr._print_index_warning()
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ 索引状态检查没跑起来（{type(exc).__name__}: {exc}）")
    from collections import Counter

    by_reason = Counter((r["case"].get("type") or "") for r in hard_results if r["problems"])
    if by_reason:
        print("  失败归因分布（第一个出错环节）："
              + "、".join(f"{_reason_text(k)}×{v}" for k, v in by_reason.most_common()))
    print(f"  【硬行为·回归门】必过 {len(must_pass) - len(regressions)}/{len(must_pass)} 通过"
          f"；未修 badcase 仍失败 {len(xfail)}、已转通过 {len(xpass)}")
    if xpass:
        print("                  已转通过：" + "、".join(r["case"]["id"] for r in xpass))
    covered = [r for r in results if r.get("via")]
    if covered:
        print(f"  判别器澄清覆盖目标出口：{len(covered)} 条（"
              + "、".join(r["case"]["id"] for r in covered) + "）")
    print(f"  【软行为·复核清单】{len(need_review)}/{len(soft_results)} 条行为与期望不符 → 需人工 / LLM 复核（不算失败）")
    print("=" * 72)

    if need_review:
        print("\n── 软行为复核清单（软行为无唯一正确答案，下面只摆事实，请人工判断）──")
        for r in need_review:
            c, exp = r["case"], r["case"]["expect"]
            want = []
            if exp.get("must_contain_any"):
                want.append("应含 " + " / ".join(exp["must_contain_any"]))
            if exp.get("must_not_contain"):
                want.append("不应含 " + " / ".join(exp["must_not_contain"]))
            want_text = "；".join(want) if want else "（未写 must / must_not）"
            print(f"  {c['id']}  「{c['query']}」")
            print(f"      期望要点：{want_text}")
            print(f"      本次回复：{r['reply']}")
            print(f"      偏离点：{'；'.join(r['problems'])}")

    review_path = _write_review(rows, results, hard_results, soft_results, must_pass, regressions)
    print(f"\n[复核清单] 已写入 {os.path.relpath(review_path, _ROOT)}")

    if "--save-baseline" in sys.argv:
        if subset_desc:
            # AGENTS.md 准则 3：不许静默换一套口径——子集存成基线会让下一个人以为那是全量
            _assert(False, "子集跑不能存基线（会把子集当成全量基线）——去掉 --set / --only 再存")
        else:
            _save_baseline(rows, results)
    if "--compare" in sys.argv:
        if subset_desc:
            print(f"\n[基线] 本次是子集（{subset_desc}）：未跑桶里列的是没选的用例，属正常")
        _compare_baseline(rows, results)

    if regressions:
        for r in regressions:
            _assert(False, f"回归 {r['case']['id']}「{r['case']['query']}」→ " + "；".join(r["problems"]))
    else:
        _assert(True, f"硬行为必过条目全部通过（{len(must_pass)}/{len(must_pass)}）")

    return stats()


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
