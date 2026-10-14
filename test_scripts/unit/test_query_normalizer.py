"""query 入口归一化测试：标点 / 空格 / 全半角 / 同义词，外加幂等与"不删实词"两条性质。

实词比对要两边都滤掉标点与空白；同义映射是有向替换，比对前先按同一张表映射原句。
"""
import os
import re
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根：供 tools.* 导入
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *

CASES = [
    "怎么设置定时清扫。。",
    "　怎么，，设置勿扰模式！",
    "定时 清扫 怎么用",
    "净界 P1 多少钱",
    "云顶X2 怎么样?",
    "ＡＰＰ在哪里下载",
    "１２３元的机器人好用吗",
    "3.5 米和 1,000 元是什么概念",
    "扫托机器人怎么用",
    "怎么设置定时清扫",              # 已经干净的
    "机器人不动了怎么办？",
]


def _norm():
    from tools.query_normalizer import normalize
    return normalize


def test_punct_and_space():
    n = _norm()
    _assert_eq(n("怎么设置定时清扫。。"), "怎么设置定时清扫。", "重复标点折叠")
    _assert_eq(n("　怎么，，设置勿扰模式！"), "怎么，设置勿扰模式！", "全角空格 / 重复逗号 / 标点两侧空格")
    _assert_eq(n("定时 清扫 怎么用"), "定时清扫怎么用", "中文之间的空格去掉（语音输入常见）")
    _assert_eq(n("净界 P1 多少钱"), "净界 P1 多少钱", "中文与英文之间的空格保留")
    _assert_eq(n("云顶X2 怎么样?"), "云顶X2 怎么样？", "半角问号归一成全角")


def test_fullwidth_and_numbers():
    n = _norm()
    _assert_eq(n("ＡＰＰ在哪里下载"), "APP在哪里下载", "全角字母→半角")
    _assert_eq(n("１２３元的机器人好用吗"), "123元的机器人好用吗", "全角数字→半角")
    _assert_eq(n("3.5 米和 1,000 元是什么概念"), "3.5 米和 1,000 元是什么概念", "数字之间的小数点 / 千分位不动")


def test_synonyms():
    n = _norm()
    _assert_eq(n("扫托机器人怎么用"), "扫拖机器人怎么用", "错别字映射：扫托→扫拖")


def test_keeps_semantic_marks():
    for q in ("怎么设置定时清扫呢？", "可以用吗", "充不进电吧", "没电了怎么办"):
        _assert_eq(_norm()(q), q, f"语义标记与实词不动：{q}")


def test_idempotent_and_no_word_loss():
    from config.word_dict_config import QUERY_PUNCT, QUERY_SYNONYMS
    n = _norm()
    drop = re.compile("[" + re.escape(QUERY_PUNCT) + r"\s]")
    fullwidth = re.compile(r"[\uff10-\uff19\uff21-\uff3a\uff41-\uff5a]")

    def words(s):
        # 原句先按归一化宣布过的两类改写映射一遍，再滤标点空白：剩下的必须逐字相同
        s = fullwidth.sub(lambda m: chr(ord(m.group()) - 0xFEE0), s)
        for src, dst in QUERY_SYNONYMS.items():
            s = s.replace(src, dst)
        return drop.sub("", s)

    for q in CASES:
        one = n(q)
        _assert_eq(n(one), one, f"幂等：{q}")
        _assert_eq(words(one), words(q), f"只动标点/空白/全半角/同义词，不删实词：{q}")


TESTS = [
    test_punct_and_space,
    test_fullwidth_and_numbers,
    test_synonyms,
    test_keeps_semantic_marks,
    test_idempotent_and_no_word_loss,
]


def run():
    reset()
    return run_tests("query 归一化测试（标点 / 空格 / 全半角 / 同义词 / 幂等）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
