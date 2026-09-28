"""多轮对话上下文评测：回放真实会话前缀 → 断言当前轮的出口行为（P0-3；评测集 data/eval/context/golden.jsonl）。

评测集 `data/eval/context/golden.jsonl`（**本地数据，已 gitignore**；由 dev-agent 从真实会话整理，
结构与用法见同目录 `README.md`）。一行 = 一个会话：`turns`（全部消息）+ `cases`（标注条目）。

harness 对每条 case：回放 `turns[:query_index]` 重建上下文与 SOP 状态，再发 `case.query`，
按 `expect.behaviour`（出口行为）+ `must_contain_any` / `must_not_contain` 断言。

出口行为的判定方式（2026-09-28 起）：**不解析话术**，改读主链路的 `[Behavior]` tag
（`tools/agent.py::_log_behavior`，每个出口一行）——harness 用一个 logging handler 在**进程内**收集，
判「tag 映射出的行为」是否等于 `expect.behaviour`。话术会随文案漂移，tag 是契约。

两类断言的分工（2026-09-24 明确）：
  - **硬行为**（`safety_alert` / `safety_carry` / `refuse` / `no_answer_fallback` / `structured` / `answer`，
    都有对应的 `[Behavior]` tag）→ 进**回归门**。
  - **软行为**（`chitchat` / `clarify` / `slot_filled` / `scope_guard`）→ **一律不判失败**：它们本来就没有
    唯一正确答案（“闲聊该怎么回”），所以只输出「软行为复核清单」（本次回复 + 偏离点）交人工 / LLM 复核；
    `[Behavior]` tag 与期望行为对不上时也列进清单（同一份复核，依旧不判失败）。

通过语义（xfail 风格，仅对硬行为生效）：
  - `verdict=pass` 或 `type=regression_fixed` → **必过**；失败 = 回归（硬失败）。
  - 其余（未修 badcase）→ 通过记 xpass（已修复），失败记 xfail（仍存在）；都不算回归。

运行（需 Ollama + Chroma）：
  .venv\\Scripts\\python.exe test_scripts/eval/test_context_eval.py --e2e
"""
import json
import logging
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_SCRIPT_DIR) not in sys.path:   # test_scripts/：供 `_runner` 导入
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))
from _runner import *

_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))   # 上两级：test_scripts/eval/ → 项目根
_GOLDEN_FILE = os.path.join(_ROOT, "data", "eval", "context", "golden.jsonl")
_REVIEW_DIR = os.path.join(_ROOT, "data", "eval", "output")   # 复核文件：<YYYYMMDD>/<HHMM>.md

# 唯一还靠文本的地方：回放「孤儿危险告警轮」时用它认出那条 assistant 回复（见 _replay_prefix）
_DANGER_ALERT_MARK = "请立即停止使用"

# 行为观测点：主链路每个出口记一行 `[Behavior] <tag> [k=v ...]`（tools/agent.py::_log_behavior）
_BEHAVIOR_PREFIX = "[Behavior] "

# `[Behavior]` tag → 本评测集的出口行为闭集（expect.behaviour 的取值）。
# 软行为侧没有专属 tag 的（slot_filled / scope_guard）只从期望侧出现，不会被映射到。
_TAG_TO_BEHAVIOUR = {
    "stop_use_safety": "safety_alert",
    "carry_safety": "safety_carry",
    "refuse_offtopic": "refuse",
    "no_answer_fallback": "no_answer_fallback",
    "structured_answer": "structured",
    "retrieve_answer": "answer",
    "chitchat": "chitchat",
    "ask_clarify": "clarify",
    # 下面三个不折算成别的行为：它们既不是「检索作答」也不是「闲聊」，
    # 出现即说明这一轮走了流程，不该被任何期望值悄悄满足。
    "start_sop": "start_sop",
    "sop_step": "sop_step",
    "exit_sop": "exit_sop",
    "block_inject": "block_inject",
}

# 有 tag 可判定的硬行为 → 断言「映射出的行为 == 期望行为」
_HARD_BEHAVIOURS = {"safety_alert", "safety_carry", "refuse",
                    "no_answer_fallback", "structured", "answer"}
# 文本无法可靠区分的软行为 → 不做行为判定，只由 must / must_not + 下面的 tag 提示进复核
_SOFT_BEHAVIOURS = {"chitchat", "clarify", "slot_filled", "scope_guard"}

# 期望行为 → 可接受的 tag（软行为的「行为对得上吗」提示用；硬行为判定走 _TAG_TO_BEHAVIOUR）。
# 两条无专属 tag 的按 notes/BEHAVIOR_TAGS_0928.md §五 折算：
# 越界应由拒答分支接住（scope_guard）；槽位提取的证据是流程内推进或预算直出（slot_filled）。
_ACCEPTED_TAGS = {
    "safety_alert": {"stop_use_safety"},
    "safety_carry": {"carry_safety"},
    "refuse": {"refuse_offtopic"},
    "no_answer_fallback": {"no_answer_fallback"},
    "structured": {"structured_answer"},
    "answer": {"retrieve_answer"},
    "chitchat": {"chitchat"},
    "clarify": {"ask_clarify"},
    "scope_guard": {"refuse_offtopic"},
    "slot_filled": {"sop_step", "structured_answer"},
    # 流程类：没有对应的旧标签（行为轨用例会直接用这些期望值），一对一同名
    "start_sop": {"start_sop"},
    "sop_step": {"sop_step"},
    "exit_sop": {"exit_sop"},
    "block_inject": {"block_inject"},
}

# 缺触发轮时的危险占位消息（含危险词，供 agent._window_danger_word 取词）
_DANGER_PLACEHOLDER = "（漏电告警轮：原始用户消息未落盘）"


def _load_golden():
    """读评测集：兼容严格 JSONL（一行一条）与缩进拼接的多对象 JSON。"""
    with open(_GOLDEN_FILE, encoding="utf-8") as f:
        text = f.read()
    decoder = json.JSONDecoder()
    rows, idx = [], 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, idx = decoder.raw_decode(text, idx)
        rows.append(obj)
    return rows


class _BehaviourCapture(logging.Handler):
    """收集本轮主链路发出的 `[Behavior]` 行（行为观测点），供断言直接读 tag。"""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.tags = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - 观测点解析失败不该打断评测
            return
        if msg.startswith(_BEHAVIOR_PREFIX):
            self.tags.append(msg[len(_BEHAVIOR_PREFIX):].split(" ", 1)[0])


def _tag_text(tags: list) -> str:
    """给人看的 tag 摘要；一轮没有 tag 是要报出来的事实，不是空白。"""
    return "tag=" + "/".join(tags) if tags else "tag=未捕获"


def _check_case(reply: str, expect: dict, tags: list) -> list:
    """返回问题清单（空 = 通过）。硬行为读 tag，软行为只由 must / must_not 承担。"""
    problems = []
    for s in expect.get("must_not_contain") or []:
        if s in reply:
            problems.append(f"不应出现「{s}」")
    mca = expect.get("must_contain_any") or []
    if mca and not any(s in reply for s in mca):
        problems.append("应至少包含 " + " / ".join(mca) + " 之一")
    beh = expect.get("behaviour")
    if beh in _HARD_BEHAVIOURS:
        if not tags:
            problems.append(f"本轮未捕获 [Behavior] 行（期望出口行为 {beh}）")
        else:
            unknown = [t for t in tags if t not in _TAG_TO_BEHAVIOUR]
            got = {_TAG_TO_BEHAVIOUR[t] for t in tags if t in _TAG_TO_BEHAVIOUR}
            if unknown:
                problems.append("未登记的 [Behavior] tag：" + " / ".join(unknown))
            elif beh not in got:
                problems.append(f"出口行为应为 {beh}，实际 {_tag_text(tags)}"
                                f"（→ {' / '.join(sorted(got))}）")
        # 任一 tag 命中即算对：`exit_sop via=exit_word_with_request` 之后本轮还会继续
        # 处理这一句（拿它当新问题重新理解），两种期望都算这一轮做对了。
    elif beh and tags:
        # 软行为（chitchat / clarify / slot_filled / scope_guard）不判对错，但「行为对不上」
        # 是有价值的复核信号——例如期望 slot_filled 却停在 ask_clarify：槽位并没提取，
        # 只是被进入确认门挡在了前面。只列进复核清单，不参与失败判定。
        accepted = _ACCEPTED_TAGS.get(beh) or set()
        if accepted and not (set(tags) & accepted):
            problems.append(f"期望行为 {beh}，实际 {_tag_text(tags)}（软行为，仅供复核参考）")
    return problems


def _ask(sid: str, query: str):
    """发一轮，返回 (回复, 本轮 [Behavior] tag 列表)。

    handler 挂在 `tools.agent.logger` 上（`get_logger` 走 `logging.getLogger(name)`，全局同名），
    因此与主链路是同一个 logger 实例，不受模块双重导入影响。
    """
    from tools import agent as agent_module
    from tools.agent import ask_stream

    capture = _BehaviourCapture()
    agent_module.logger.addHandler(capture)
    try:
        reply = "".join(ask_stream(query, sid)).strip()
    finally:
        agent_module.logger.removeHandler(capture)
    return reply, list(capture.tags)


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
            reply, _tags = _ask(sid, t["content"])
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
    # 待确认状态（退出确认门 / 网点问城市等）2026-09-28 起统一在 pending_store：
    # Redis 键 + 内存降级都要清，否则同一 sid 上一行的残留会污染下一行。
    clear_pending(sid)

    turns = row["turns"]
    results = []
    replayed = 0
    for case in sorted(row["cases"], key=lambda c: c["query_index"]):
        qi = case["query_index"]
        try:
            _replay_prefix(sid, turns, replayed, qi)
            reply, tags = _ask(sid, case["query"])
            append_message(sid, "assistant", reply)
            problems = _check_case(reply, case["expect"], tags)
        except Exception as exc:  # noqa: BLE001
            reply = f"<EXCEPTION {type(exc).__name__}: {exc}>"
            tags = []
            problems = [f"执行异常：{type(exc).__name__}: {exc}"]
        hard = case["expect"].get("behaviour") in _HARD_BEHAVIOURS
        must_pass = hard and ((case.get("verdict") == "pass") or (case.get("type") == "regression_fixed"))
        results.append({"case": case, "reply": reply, "problems": problems,
                        "hard": hard, "must_pass": must_pass, "tags": tags})
        replayed = qi + 1

    delete_session(sid)
    clear_pending(sid)
    sops.base._sessions = {}
    return results


def _print_detail(results):
    print("── 硬行为（有 [Behavior] tag 可判定 → 进回归门）──")
    for r in (x for x in results if x["hard"]):
        c = r["case"]
        if not r["problems"]:
            mark = "PASS" if r["must_pass"] else "xpass"
        else:
            mark = "REGRESS" if r["must_pass"] else "xfail"
        kind = "必过" if r["must_pass"] else "badcase"
        print(f"  [{mark:<7}] {c['id']} ({kind}/{c.get('type')}) {c['query']}"
              f"   [{_tag_text(r['tags'])}]")
        if r["problems"]:
            print(f"              ↳ {'；'.join(r['problems'])}")

    print("── 软行为（无唯一正确答案 → 只出复核清单，不判失败）──")
    for r in (x for x in results if not x["hard"]):
        c = r["case"]
        if r["problems"]:
            mark = "⚠需复核"
        elif c.get("verdict") == "fail":
            mark = "✓已转通过"
        else:
            mark = "· 未偏离"
        print(f"  [{mark:<7}] {c['id']} ({c.get('type')}) {c['query']}"
              f"   [{_tag_text(r['tags'])}]")
        if r["problems"]:
            print(f"              ↳ {'；'.join(r['problems'])}")


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

    软行为没有唯一正确答案，所以文件里只摆事实：期望要点 + 本次回复 + 偏离点 + 归档判据。
    """
    need_review = [r for r in soft_results if r["problems"]]
    out = [
        "# 上下文轨 · 软行为复核清单",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 评测集：`data/eval/context/golden.jsonl`"
        f"（{len(rows)} 会话 / {len(results)} 条标注：硬 {len(hard_results)} · 软 {len(soft_results)}）",
        f"- 硬行为（回归门）：必过 {len(must_pass) - len(regressions)}/{len(must_pass)} 通过"
        + (f"，**{len(regressions)} 条回归需处理**" if regressions else ""),
        f"- 软行为：**{len(need_review)}/{len(soft_results)} 条偏离期望要点或行为对不上**（不判失败，按下表复核）",
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
        want = []
        if exp.get("must_contain_any"):
            want.append("应含 " + " / ".join(exp["must_contain_any"]))
        if exp.get("must_not_contain"):
            want.append("不应含 " + " / ".join(exp["must_not_contain"]))
        out += [f"## {mark} {c['id']}（{c.get('type')}）", "",
                f"- 用户：{c['query']}",
                f"- 期望要点：{'；'.join(want) if want else '（未写 must / must_not）'}",
                f"- 行为 tag：{_tag_text(r['tags'])}（期望行为 {c['expect'].get('behaviour')}，软行为不判定）",
                f"- 本次回复：{r['reply']}"]
        if r["problems"]:
            out.append(f"- 偏离点：{'；'.join(r['problems'])}")
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
    print(f"  【硬行为·回归门】必过 {len(must_pass) - len(regressions)}/{len(must_pass)} 通过"
          f"；未修 badcase 仍失败 {len(xfail)}、已转通过 {len(xpass)}")
    if xpass:
        print("                  已转通过：" + "、".join(r["case"]["id"] for r in xpass))
    print(f"  【软行为·复核清单】{len(need_review)}/{len(soft_results)} 条偏离期望要点或行为对不上 → 需人工 / LLM 复核（不算失败）")
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
