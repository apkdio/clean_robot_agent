"""多轮对话上下文评测：回放真实会话前缀 → 断言当前轮的出口行为（P0-3；评测集 data/eval/context/golden.jsonl）。

评测集 `data/eval/context/golden.jsonl`（**本地数据，已 gitignore**；由 dev-agent 从真实会话整理，
结构与用法见同目录 `README.md`）。一行 = 一个会话：`turns`（全部消息）+ `cases`（标注条目）。

harness 对每条 case：回放 `turns[:query_index]` 重建上下文与 SOP 状态，再发 `case.query`，
按 `expect.behaviour`（出口行为）+ `must_contain_any` / `must_not_contain` 断言。

两类断言的分工（2026-09-24 明确）：
  - **硬行为**（`safety_alert` / `safety_carry` / `refuse` / `no_answer_fallback` / `structured` / `answer`，
    话术可判定）→ 进**回归门**。
  - **软行为**（`chitchat` / `clarify` / `slot_filled` / `scope_guard`）→ **一律不判失败**：它们本来就没有
    唯一正确答案（“闲聊该怎么回”），所以只输出「软行为复核清单」（本次回复 + 偏离点）交人工 / LLM 复核。

通过语义（xfail 风格，仅对硬行为生效）：
  - `verdict=pass` 或 `type=regression_fixed` → **必过**；失败 = 回归（硬失败）。
  - 其余（未修 badcase）→ 通过记 xpass（已修复），失败记 xfail（仍存在）；都不算回归。

运行（需 Ollama + Chroma）：
  .venv\\Scripts\\python.exe test_scripts/eval/test_context_eval.py --e2e
"""
import json
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

# 出口行为的文本标记（与 tools/agent.py 的话术一一对应）
_DANGER_ALERT_MARK = "请立即停止使用"           # 危险拦截话术
_SAFETY_CARRY_MARK = "安全隐患"                 # 安全承接话术
_CARRY_PREFIX = "您前面提到的"
_REFUSE_MARK = "专属助手"                       # 领域外拒答
_FALLBACK_MARKS = ("没查到", "没找到", "暂时还没")  # 兜底话术
_STRUCTURED_MARKS = ("型号信息如下", "在您预算内", "有以下型号",
                     "符合您预算和时间要求", "该时间段内发布")
_SCOPE_GUARD_MARKS = ("只能回答", "产品相关", "不涉及", "不便提供", "无法提供")

# 文本可可靠区分的硬行为 → 断言「判定行为 == 期望行为」
_HARD_BEHAVIOURS = {"safety_alert", "safety_carry", "refuse",
                    "no_answer_fallback", "structured", "answer"}
# 文本无法可靠区分的软行为 → 只靠 must_contain_any / must_not_contain 断言
_SOFT_BEHAVIOURS = {"chitchat", "clarify", "slot_filled", "scope_guard"}

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


def _detect_behaviour(reply: str) -> str:
    """把回复归入出口行为闭集（标记法，按确定性从高到低判定）。"""
    if _DANGER_ALERT_MARK in reply:
        return "safety_alert"
    if _SAFETY_CARRY_MARK in reply or reply.startswith(_CARRY_PREFIX):
        return "safety_carry"
    if _REFUSE_MARK in reply:
        return "refuse"
    if any(m in reply for m in _STRUCTURED_MARKS):
        return "structured"
    if any(m in reply for m in _FALLBACK_MARKS):
        return "no_answer_fallback"
    if any(m in reply for m in _SCOPE_GUARD_MARKS):
        return "scope_guard"
    return "answer"


def _check_case(reply: str, expect: dict) -> list:
    """返回问题清单（空 = 通过）。"""
    problems = []
    for s in expect.get("must_not_contain") or []:
        if s in reply:
            problems.append(f"不应出现「{s}」")
    mca = expect.get("must_contain_any") or []
    if mca and not any(s in reply for s in mca):
        problems.append("应至少包含 " + " / ".join(mca) + " 之一")
    beh = expect.get("behaviour")
    if beh in _HARD_BEHAVIOURS:
        got = _detect_behaviour(reply)
        if got != beh:
            problems.append(f"出口行为应为 {beh}，实际判定为 {got}")
    # 软行为（chitchat / slot_filled / scope_guard / clarify）不做行为文本判定，
    # 由 must / must_not 承担——见 data/eval/context/README.md「没有唯一正确答案」一节。
    return problems


def _ask(sid: str, query: str) -> str:
    from tools.agent import ask_stream
    return "".join(ask_stream(query, sid)).strip()


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
            append_message(sid, "assistant", _ask(sid, t["content"]))
            continue
        prev_is_user = idx > 0 and turns[idx - 1]["role"] == "user"
        if not prev_is_user and t["content"].startswith(_DANGER_ALERT_MARK):
            append_message(sid, "user", _DANGER_PLACEHOLDER, danger=True)
            append_message(sid, "assistant", t["content"])


def _run_row(row: dict) -> list:
    import sops.base
    from tools.agent import _pending_exits, _pending_service
    from tools.context_store import append_message, delete_session

    sid = "ctxeval-" + row["session_id"]
    sops.base._sessions = {}
    _pending_exits.clear()
    _pending_service.clear()

    turns = row["turns"]
    results = []
    replayed = 0
    for case in sorted(row["cases"], key=lambda c: c["query_index"]):
        qi = case["query_index"]
        try:
            _replay_prefix(sid, turns, replayed, qi)
            reply = _ask(sid, case["query"])
            append_message(sid, "assistant", reply)
            problems = _check_case(reply, case["expect"])
        except Exception as exc:  # noqa: BLE001
            reply = f"<EXCEPTION {type(exc).__name__}: {exc}>"
            problems = [f"执行异常：{type(exc).__name__}: {exc}"]
        hard = case["expect"].get("behaviour") in _HARD_BEHAVIOURS
        must_pass = hard and ((case.get("verdict") == "pass") or (case.get("type") == "regression_fixed"))
        results.append({"case": case, "reply": reply, "problems": problems,
                        "hard": hard, "must_pass": must_pass})
        replayed = qi + 1

    delete_session(sid)
    sops.base._sessions = {}
    return results


def _print_detail(results):
    print("── 硬行为（话术可判定 → 进回归门）──")
    for r in (x for x in results if x["hard"]):
        c = r["case"]
        if not r["problems"]:
            mark = "PASS" if r["must_pass"] else "xpass"
        else:
            mark = "REGRESS" if r["must_pass"] else "xfail"
        tag = "必过" if r["must_pass"] else "badcase"
        print(f"  [{mark:<7}] {c['id']} ({tag}/{c.get('type')}) {c['query']}")
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
        print(f"  [{mark:<7}] {c['id']} ({c.get('type')}) {c['query']}")
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
        f"- 软行为：**{len(need_review)}/{len(soft_results)} 条偏离期望要点**（不判失败，按下表复核）",
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

    results = []
    for row in rows:
        results.extend(_run_row(row))

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
    print(f"  【软行为·复核清单】{len(need_review)}/{len(soft_results)} 条偏离期望要点 → 需人工 / LLM 复核（不算失败）")
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
