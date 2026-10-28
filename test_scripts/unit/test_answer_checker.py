"""答案校验测试：确定性抽取、召回核对、用户原话豁免与 trace 留痕。"""
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *

from tools.answer_checker import check_answer
from tools import trace_store


_MODELS = ["不染一尘云顶 X2", "不染一尘云顶 X1 Pro"]
_CHUNKS = [
    "型号：不染一尘云顶 X2\n参考价：5999 元\n保修期：2 年",
]


def test_found_entities():
    r = check_answer("云顶X2 参考价 5999 元，保修期 2 年。", _CHUNKS, "云顶X2 多少钱",
                     model_names=_MODELS)
    _assert_eq(r["checked"], ["model:云顶X2", "price:5999", "warranty:2年"],
               "型号 / 金额 / 保修时长都抽出来")
    _assert_eq(r["missing"], [], "召回片段里都找得到")
    _assert_eq(r["exempt"], ["model:云顶X2"], "用户问到的型号按原话豁免")


def test_missing_hard_facts():
    r = check_answer("云顶 X1 Pro 参考价 6999 元，保修 3 年。", _CHUNKS, "推荐一款",
                     model_names=_MODELS)
    _assert_eq(r["checked"], ["model:云顶X1Pro", "price:6999", "warranty:3年"],
               "错型号 / 编造金额 / 编造保修年限都进入核对")
    _assert_eq(sorted(r["missing"]), ["model:云顶X1Pro", "price:6999", "warranty:3年"],
               "片段里找不到的三类硬事实都标 missing")


def test_unknown_model_suffix_not_truncated():
    r = check_answer("云顶 X1 Max 参考价 5999 元。", _CHUNKS, "推荐一款",
                     model_names=_MODELS)
    _assert_eq(r["checked"], ["price:5999"], "「云顶 X1 Max」不被截成已知的「云顶 X1」")
    _assert_eq(r["unchecked"], ["model:云顶X1Max"],
               "带虚构后缀的完整 token 记 unchecked，不再静默套到已知型号")


def test_off_catalog_model_unchecked():
    r = check_answer("云顶 X9 参考价 5999 元。", _CHUNKS, "推荐一款",
                     model_names=_MODELS)
    _assert_eq(r["unchecked"], ["model:云顶X9"], "目录外型号暴露成 unchecked")
    _assert_eq(r["missing"], [], "目录外型号无法判断真假，不直接判缺失")


def test_known_model_with_suffix():
    chunks = ["型号：不染一尘云顶 X1 Pro\n参考价：6999 元"]
    r = check_answer("云顶 X1 Pro 卖 6999 元。", chunks, "", model_names=_MODELS)
    _assert_eq(r["checked"], ["model:云顶X1Pro", "price:6999"],
               "目录内带后缀型号整体命中")
    _assert_eq(r["unchecked"], [], "已知后缀不会误进 unchecked")


def test_query_exemption():
    r = check_answer("您预算 3000 元，推荐云顶 X2，保修 2 年。", _CHUNKS, "预算3000元 云顶X2",
                     model_names=_MODELS)
    _assert_eq(r["missing"], [], "用户原话里的预算/型号不判缺失")
    _assert_eq(sorted(r["exempt"]), ["model:云顶X2", "price:3000"],
               "豁免项单独列出，便于区分真的核对过与跳过")


def test_no_generic_number_false_positive():
    r = check_answer("有 3 个档位，2026 年发布。", ["普通召回片段"], "", model_names=_MODELS)
    _assert_eq(r["checked"], [], "不带货币单位/保修语境的数字不当作硬事实")
    _assert_eq(r["missing"], [], "没有核对项就不产生缺失")


def test_normalize_price_and_month():
    r = check_answer("价格 5,999 元，保修 12 个月。",
                     ["参考价：5999 元\n保修期：12 个月"], "", model_names=[])
    _assert_eq(r["checked"], ["price:5999", "warranty:12个月"], "千分位与「月」归一")
    _assert_eq(r["missing"], [], "归一后能对上")


def test_trace_verify_record():
    sid = "answertest-verify"
    reset_trace(sid)
    trace_store.begin_turn(sid, "云顶 X2 保修多久")
    trace_store.note_behavior("retrieve_answer")
    trace_store.verify(["model:云顶X2", "price:5999"], ["price:5999"],
                       action="log", exempt=["model:云顶X2"])
    trace_store.end_turn("该型号保修 2 年。")
    rec = read_last_turn(sid)
    _assert_eq(rec["answer"]["verify"], {
        "checked": ["model:云顶X2", "price:5999"],
        "missing": ["price:5999"],
        "exempt": ["model:云顶X2"],
        "action": "log",
    }, "完整校验结果落在 answer.verify")
    node = [n for n in rec["path"] if n["node"] == "answer.verify"]
    _assert_eq(len(node), 1, "路径里有一个答案校验节点")
    _assert_eq(node[0]["status"], "miss", "有缺失时节点状态是 miss")
    _assert_eq(node[0]["out"]["checked"], 2, "节点只放计数与短摘要")
    from config.word_dict_config import TRACE_NODE_LABELS
    _assert_in("answer.verify", TRACE_NODE_LABELS, "答案校验节点已登记")
    reset_trace(sid)


def test_trace_verify_degraded():
    sid = "answertest-verify-degraded"
    reset_trace(sid)
    trace_store.begin_turn(sid, "q")
    trace_store.note_behavior("retrieve_answer")
    trace_store.verify([], [], skipped=["model_catalog_unavailable"])
    trace_store.end_turn("答案")
    rec = read_last_turn(sid)
    _assert_eq(rec["answer"]["verify"]["skipped"], ["model_catalog_unavailable"],
               "降级原因进 answer.verify")
    node = next(n for n in rec["path"] if n["node"] == "answer.verify")
    _assert_eq(node["status"], "degraded", "型号目录不可用显式标降级")
    reset_trace(sid)


def test_trace_verify_unchecked():
    sid = "answertest-verify-unchecked"
    reset_trace(sid)
    trace_store.begin_turn(sid, "q")
    trace_store.note_behavior("retrieve_answer")
    trace_store.verify([], [], unchecked=["model:云顶X9"])
    trace_store.end_turn("答案")
    rec = read_last_turn(sid)
    _assert_eq(rec["answer"]["verify"]["unchecked"], ["model:云顶X9"],
               "目录外型号进 answer.verify")
    node = next(n for n in rec["path"] if n["node"] == "answer.verify")
    _assert_eq(node["out"]["unchecked_n"], 1, "节点摘要里有 unchecked 计数")
    reset_trace(sid)


def test_stream_verify_only_on_complete():
    from tools import agent as agent_mod

    saved = agent_mod.stream_chat
    captured = []

    def _stream(*args, **kwargs):  # noqa: ARG001
        yield "完整回答"

    try:
        agent_mod.stream_chat = _stream
        out = "".join(agent_mod._stream_with_fallback([], verify=lambda text: captured.append(text)))
        _assert_eq(out, "完整回答", "流式内容原样输出")
        _assert_eq(captured, ["完整回答"], "完整生成后调用校验")

        captured.clear()
        out = "".join(agent_mod._stream_with_fallback(
            [], stop_check=lambda: True, verify=lambda text: captured.append(text)))
        _assert_eq(out, "", "停止后不输出半截内容")
        _assert_eq(captured, [], "停止的轮次不校验半截答案")
    finally:
        agent_mod.stream_chat = saved


def test_warranty_context_only():
    """只抽保修语境下的时长；续航/寿命不算保修事实。"""
    r = check_answer("续航 3 年，保修 2 年。", ["续航 3 年，保修 2 年。"], "", model_names=[])
    _assert_eq(r["checked"], ["warranty:2年"], "只抽保修语境的 2 年")


def test_price_requires_currency_unit():
    """没有货币单位/「元」「块」的数字不当作价格。"""
    r = check_answer("价格 2999，有 3 个档位。", ["价格 2999"], "", model_names=[])
    _assert_eq(r["checked"], [], "裸数字不抽")


def test_chinese_numeral_not_extracted():
    """已知边界：中文数字（保修两年）暂不抽取；支持后更新本用例。"""
    r = check_answer("保修两年。", ["保修两年。"], "", model_names=[])
    _assert_eq(r["checked"], [], "中文数字不产生核对项，也不误报缺失")


TESTS = [
    test_found_entities,
    test_missing_hard_facts,
    test_unknown_model_suffix_not_truncated,
    test_off_catalog_model_unchecked,
    test_known_model_with_suffix,
    test_query_exemption,
    test_no_generic_number_false_positive,
    test_normalize_price_and_month,
    test_warranty_context_only,
    test_price_requires_currency_unit,
    test_chinese_numeral_not_extracted,
    test_trace_verify_record,
    test_trace_verify_degraded,
    test_trace_verify_unchecked,
    test_stream_verify_only_on_complete,
]


def run():
    reset()
    return run_tests("答案校验（确定性抽取 / 核对 / trace）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
