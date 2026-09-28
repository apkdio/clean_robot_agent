"""行为观测点（`[Behavior]` tag）与评测判据的契约测试：不依赖 LLM / Chroma，纯规则。

覆盖：
  - `_BehaviourCapture` 只收 `[Behavior]` 行、且只留 tag（`[Shadow]` / 其它日志不误收）
  - handler 与主链路挂**同一个 logger 实例**（不受「模块双重导入」影响）
  - 单个 tag → 出口行为的映射覆盖了全部硬行为（硬行为都能从 tag 判出来）
  - `_check_case` 的硬行为四态：tag 命中 / 不一致 / 未捕获 / 未登记 tag
  - 软行为不看 tag，仍只由 must / must_not 兜
"""
import logging
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_SCRIPTS_DIR, os.path.join(_SCRIPTS_DIR, "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _runner import *
from tools import agent as agent_mod
import test_context_eval as ctx_eval


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
    _assert_eq(ctx_eval._check_case("随便什么话术", exp, ["carry_safety"]), [],
               "tag 命中即通过（不再解析话术）")
    _assert_true(ctx_eval._check_case("随便什么话术", exp, ["retrieve_answer"]),
                 "tag 不符 → 报问题")
    _assert_true(ctx_eval._check_case("随便什么话术", exp, []),
                 "未捕获 tag → 报问题（出口缺观测点要暴露出来）")
    _assert_true(ctx_eval._check_case("随便什么话术", exp, ["brand_new_tag"]),
                 "未登记 tag → 报问题")
    _assert_eq(ctx_eval._check_case("随便什么话术", {"behaviour": "exit_sop"},
                                    ["exit_sop", "retrieve_answer"]), [],
               "一轮多 tag（退出带诉求）→ 任一命中即通过")
    _assert_true(ctx_eval._check_case("我没查到相关资料",
                                      {"behaviour": "safety_alert"}, ["carry_safety"]),
                 "硬行为不一致时 must / must_not 的问题一并报出")


def test_check_case_soft_tag_hint():
    """软行为不判失败，但 tag 与期望行为对不上要进复核清单。"""
    _assert_eq(ctx_eval._check_case("随便怎么答", {"behaviour": "chitchat"}, ["chitchat"]), [],
               "tag 对得上 → 不进复核")
    _assert_true(ctx_eval._check_case("随便怎么答", {"behaviour": "chitchat"}, ["ask_clarify"]),
                 "tag 对不上 → 进复核")
    _assert_true(ctx_eval._check_case("2000左右", {"behaviour": "slot_filled"}, ["ask_clarify"]),
                 "期望 slot_filled 却停在 ask_clarify（槽位没提取）→ 进复核")
    _assert_eq(ctx_eval._check_case("预算已收到", {"behaviour": "slot_filled"}, ["sop_step"]), [],
               "slot_filled 接受 sop_step（dev 映射表）")
    _assert_eq(ctx_eval._check_case("只答产品相关", {"behaviour": "scope_guard"}, ["refuse_offtopic"]), [],
               "scope_guard 接受 refuse_offtopic（dev 映射表）")
    _assert_true(ctx_eval._check_case("你好呀", {"behaviour": "scope_guard"}, ["chitchat"]),
                 "越界问题了走闲聊 → 进复核")
    _assert_true(ctx_eval._check_case("我没查到相关资料",
                                      {"behaviour": "chitchat", "must_not_contain": ["没查到"]},
                                      ["chitchat"]),
                 "软行为仍由 must_not 兜")


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


def test_tag_text():
    _assert_eq(ctx_eval._tag_text(["carry_safety"]), "tag=carry_safety", "有 tag → 摘要")
    _assert_in("未捕获", ctx_eval._tag_text([]), "无 tag → 明说未捕获而非留空")


TESTS = [
    test_capture_only_behavior_lines,
    test_capture_same_logger_instance,
    test_tag_mapping_covers_hard_behaviours,
    test_check_case_hard_by_tag,
    test_check_case_soft_tag_hint,
    test_accepted_tags_cover_mapping,
    test_disable_shadow_restores,
    test_tag_text,
]


def run():
    reset()
    return run_tests("行为观测点 tag 与判据契约测试", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
