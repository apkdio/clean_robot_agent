"""多轮对话上下文评测：回放真实会话前缀 → 断言当前轮的出口行为（P0-3；评测集 data/eval/context/golden.jsonl）。

评测集 `data/eval/context/golden.jsonl`（**本地数据，已 gitignore**；从真实会话整理，
结构与用法见同目录 `README.md`）。一行 = 一条标注（case），上下文按 `session_id` 现取。

harness 对每条 case：回放前缀重建上下文与 SOP 状态，再发 `case.query`，
按 `expect.behaviour`（出口行为）判定——读主链路 trace 记录的 `branch`（本轮唯一出口）。
数据里的 `must_contain_any` / `must_not_contain` 保留不动但**暂不参与判定**：
话术断言会随文案漂移，而「行为对不对」与「话里有没有某个词」是两回事——后者属内容级，归检索轨与人工复核。

判定口径：`branch` 是记录侧按 `config/word_dict_config.py::BEHAVIOR_BY_TAG` 折算出的**唯一出口**
（`[Behavior]` tag → 行为），取值构成闭集、可断言。不按日志行判定：一轮可能有多个 tag，
真正结束这一轮的是最后一个出口。

两类断言的分工：
  - **硬行为**（`safety_alert` / `safety_carry` / `refuse` / `no_answer_fallback` / `structured` / `answer`）
    → 进**回归门**。
  - **软行为**（`chitchat` / `clarify` / `slot_filled` / `scope_guard`）→ **一律不判失败**：它们本来就没有
    唯一正确答案（“闲聊该怎么回”），所以只输出「软行为复核清单」（本次回复 + 偏离点）交人工 / LLM 复核；
    `branch` 与期望行为对不上时也列进清单（同一份复核，依旧不判失败）。

通过语义（xfail 风格，仅对硬行为生效）：
  - `verdict=pass` 或 `fixed: true` → **必过**；失败 = 回归（硬失败）。
  - 其余（未修 badcase）→ 通过记 xpass（已修复），失败记 xfail（仍存在）；都不算回归。

运行（需 Ollama + Chroma）：
  .venv\\Scripts\\python.exe test_scripts/eval/test_context_eval.py --e2e
"""
import json
import os
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

# 唯一还靠文本的地方：回放「孤儿危险告警轮」时用它认出那条 assistant 回复（见 _replay_prefix）
_DANGER_ALERT_MARK = "请立即停止使用"

# 出口行为闭集 = 记录侧 trace 的 `branch` 取值（含流程类的 4 个），在 config/word_dict_config.py 登记。
# 行为轨只认这一份表，不再自带副本——两处各存一张是会漂移的。
_BEHAVIOURS = set(BEHAVIOR_BY_TAG.values())

# 有唯一正确出口的硬行为 → 断言「本轮 branch == 期望行为」
_HARD_BEHAVIOURS = {"safety_alert", "safety_carry", "refuse",
                    "no_answer_fallback", "structured", "answer"}
# 没有唯一正确答案的软行为 → 不判失败；branch 对不上只进复核清单
# （`scope_guard` / `slot_filled` 两个期望没有专属 branch，只从期望侧出现）
_SOFT_BEHAVIOURS = {"chitchat", "clarify", "slot_filled", "scope_guard"}

# 期望行为 → 可接受的 branch（软行为的「行为对得上吗」提示用；硬行为判定是相等比较）。
# 两个没有专属 branch 的期望按 project_detail.md §4.13「与存量 behaviour 的映射」折算：
# 越界应由拒答分支接住（scope_guard）；槽位提取的证据是流程内推进或预算直出（slot_filled）。
_ACCEPTED_BRANCHES = {b: {b} for b in _BEHAVIOURS}
_ACCEPTED_BRANCHES["scope_guard"] = {"refuse"}
_ACCEPTED_BRANCHES["slot_filled"] = {"sop_step", "structured"}

# 缺触发轮时的危险占位消息（含危险词，供 agent._window_danger_word 取词）
_DANGER_PLACEHOLDER = "（漏电告警轮：原始用户消息未落盘）"


_SESS_DIR = os.path.join(os.path.dirname(_GOLDEN_FILE), "sessions")   # 评测集自带的（手工/合成）会话


_CONTEXT_ROOT = None      # 真实会话目录（惰性解析，见 _real_context_root）


def _real_context_root() -> str:
    """真实会话目录。**不能用 `context_store._CONTEXT_DIR`**：评测进程里 `_runner` 为隔离被测写入
    把它指到了 `data/test_context`，而评测集要读的是**真实**会话（契约测试会当场报出读不到）。
    """
    global _CONTEXT_ROOT
    if _CONTEXT_ROOT is None:
        from tools.path_tool import get_abs_path   # 与 `context_store` 同一个路径助手，但不读 CONTEXT_DIR

        _CONTEXT_ROOT = get_abs_path("data/context")
    return _CONTEXT_ROOT


def _session_file(session_id: str) -> str:
    """按 session_id 找会话文件：评测集自带的 `sessions/` 优先，再在真实会话目录里找。

    目录布局与命名都兼容：`<day>/<stem>.jsonl` 与根下 `<stem>.jsonl`；stem 有 `<sid>`（旧）
    与 `<sid>_<YYYYMMDD>_<HHMM>`（新）两种。
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
    """会话消息：老行自带 `turns` 就用它，否则按 session_id 现读会话文件（行里不再内嵌上下文）。"""
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
    """读评测集：**一行一条 case**，按 `session_id` 分组返回（回放按会话走）。

    兼容两种行形态：新形态只带 `session_id` + `query_index`（上下文按 session_id 现取，行里不内嵌）；
    旧形态（迁移前，见 `golden.prev.jsonl`）一行一个会话，自带 `turns` 与 `cases[]`。
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
    """评测集自检：归因取值、`fixed` 语义、会话上下文可达——写错当场报出来，别等跑完 20 分钟。"""
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


def _check_case(expect: dict, branch, tag=None) -> list:
    """返回问题清单（空 = 通过）。**判据只有出口行为**（读 trace 的 `branch`）。

    `must_contain_any` / `must_not_contain` 暂不参与判定：它们判的是“话里有没有某个词”，
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
        elif branch != beh:
            problems.append(f"出口行为应为 {beh}，实际 {_branch_text(branch)}")
    elif beh:
        # 软行为不判对错，但「行为对不上」是有价值的复核信号：只列进复核清单，不参与失败判定。
        accepted = _ACCEPTED_BRANCHES.get(beh) or set()
        if not branch:
            problems.append(f"期望行为 {beh}，本轮未记出口（软行为，仅供复核参考）")
        elif branch not in accepted:
            problems.append(f"期望行为 {beh}，实际 {_branch_text(branch)}（软行为，仅供复核参考）")
    return problems


def _ask(sid: str, query: str):
    """发一轮，返回 (回复, 本轮 trace 记录)。

    判定读记录里的 `branch`（唯一出口），失败归因读 `path` / `route` / `retrieval`——同一份记录，
    不再是两处证据。`end_turn()` 在本轮生成器消费完时落盘，所以返回时记录已在。
    """
    from tools.agent import ask_stream

    reply = "".join(ask_stream(query, sid)).strip()
    return reply, read_last_turn(sid)


def _replay_prefix(sid: str, turns: list, lo: int, hi: int):
    """回放 turns[lo:hi] 重建上下文与状态。

    user 轮真跑一遍（驱动 SOP 状态、落盘、按内容打 danger 标记），其回复用**当前系统**生成的
    结果落盘（而非数据里的历史 actual）——否则上一条被修复的回复不会影响下一条的承接判定。
    assistant 轮已被上一条 user 轮的生成结果代表，跳过。

    例外：危险告警轮可能缺触发它的 user 消息（见数据 `anomalies`，当时的 guard 在落盘前
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


def _run_row(row: dict) -> list:
    import sops.base
    from tools.context_store import append_message, delete_session
    from tools.pending_store import clear as clear_pending

    sid = "ctxeval-" + row["session_id"]
    sops.base._sessions = {}
    # 待确认状态统一在 pending_store：
    # Redis 键 + 内存降级都要清，否则同一 sid 上一行的残留会污染下一行。
    clear_pending(sid)
    # 主链路 trace（tools/trace_store.py）也按 sid 落盘，同一 sid 跨行复用——先清掉上轮的落盘与
    # 进程内计数，否则 `turn_index` 会接着前一次运行数、文件也会累积。
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
            problems = _check_case(case["expect"], record.get("branch"), record.get("tag"))
        except Exception as exc:  # noqa: BLE001
            reply = f"<EXCEPTION {type(exc).__name__}: {exc}>"
            record = {}
            problems = [f"执行异常：{type(exc).__name__}: {exc}"]
        if record.get("aborted"):
            # 评测里正常不会出现（生成器必被消费完）；真出现说明这轮被截断，行为判定不可信
            problems.append("本轮 trace 标记 aborted（回答未跑完，判定不可信）")
        hard = case["expect"].get("behaviour") in _HARD_BEHAVIOURS
        must_pass = hard and ((case.get("verdict") == "pass") or bool(case.get("fixed")))
        # 有问题的条目附上本轮 trace 摘要（主链路落的结构化环节记录），省得回头 grep 日志
        results.append({"case": case, "reply": reply, "problems": problems,
                        "hard": hard, "must_pass": must_pass,
                        "branch": record.get("branch"), "tag": record.get("tag"),
                        "trace": _trace_brief(record) if problems else ""})
        replayed = qi + 1

    delete_session(sid)
    clear_pending(sid)
    sops.base._sessions = {}
    return results


def _reason_text(key) -> str:
    """归因展示：`wrong_route「答非所问」`；空值 = 正例基线。与前端 feedback 同一套 key。"""
    if not key:
        return "正例"
    try:
        from config.word_dict_config import FEEDBACK_REASON_LABELS

        return "%s「%s」" % (key, FEEDBACK_REASON_LABELS.get(key, ""))
    except Exception:  # noqa: BLE001
        return str(key)


def _trace_brief(record: dict) -> str:
    """把一条 trace 记录压成一行人读摘要（失败归因用）；没有 trace 返回空串。

    主读 v3 分层块（`route` / `retrieval` / `path`），缺失时回落到 v2 投影 `steps`——
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
    """主链路 trace 的记录版本（`tools/trace_store.py::VERSION`），写进报告便于日后对齐。"""
    try:
        from tools import trace_store

        return "v%s" % trace_store.VERSION
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
              f"   [{_branch_text(r['branch'])}]")
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
              f"   [{_branch_text(r['branch'])}]")
        if r["problems"]:
            print(f"              ↳ {'；'.join(r['problems'])}")
            if r.get("trace"):
                print(f"              · 本轮环节：{r['trace']}")


def _review_path():
    """复核文件路径：`data/eval/output/<YYYYMMDD>/<HHMM>.md`（同一分钟重复跑则加 -2 / -3）。"""
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
    """把**软行为复核清单**落盘——复核是人工 / LLM 的活，得能拿出去看。

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


def run():
    reset()
    if not E2E:
        _skip("多轮上下文评测需要 Ollama + Chroma（加 --e2e 运行）")
        return stats()
    if not os.path.isfile(_GOLDEN_FILE):
        _skip(f"未找到评测集 {_GOLDEN_FILE}（本地数据目录 data/eval，已 gitignore）")
        return stats()

    rows = _load_golden()
    _assert(len(rows) > 0, f"评测集加载成功（{len(rows)} 个会话）")
    _validate_golden(rows)

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
    from collections import Counter

    by_reason = Counter((r["case"].get("type") or "") for r in hard_results if r["problems"])
    if by_reason:
        print("  失败归因分布（第一个出错环节）："
              + "、".join(f"{_reason_text(k)}×{v}" for k, v in by_reason.most_common()))
    print(f"  【硬行为·回归门】必过 {len(must_pass) - len(regressions)}/{len(must_pass)} 通过"
          f"；未修 badcase 仍失败 {len(xfail)}、已转通过 {len(xpass)}")
    if xpass:
        print("                  已转通过：" + "、".join(r["case"]["id"] for r in xpass))
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

    if regressions:
        for r in regressions:
            _assert(False, f"回归 {r['case']['id']}「{r['case']['query']}」→ " + "；".join(r["problems"]))
    else:
        _assert(True, f"硬行为必过条目全部通过（{len(must_pass)}/{len(must_pass)}）")

    return stats()


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
