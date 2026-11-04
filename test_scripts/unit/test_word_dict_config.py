"""词表配置契约：顶层常量只能定义一次。

`word_dict_config` 是纯配置模块，重复赋值会被 Python 静默覆盖，前一份永远读不到。
这里用 AST 扫顶层 Assign，发现同名即失败，避免改字面量时改到死副本。
"""
import ast
import os
import sys
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:   # 项目根
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *


def _top_level_assignments(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def test_no_duplicate_top_level_assignments():
    path = Path(__file__).resolve().parents[2] / "config" / "word_dict_config.py"
    names = _top_level_assignments(path)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    _assert_eq(duplicates, [], "word_dict_config 顶层常量重复定义：%s" % duplicates)


def test_empty_replies_single_source():
    from config.word_dict_config import MODEL_QUERY_EMPTY_REPLIES

    _assert_true(bool(MODEL_QUERY_EMPTY_REPLIES), "0 结果话术不能为空")
    _assert_true(all(isinstance(x, str) and x.strip() for x in MODEL_QUERY_EMPTY_REPLIES),
                 "0 结果话术必须是非空字符串")


TESTS = [test_no_duplicate_top_level_assignments, test_empty_replies_single_source]


def run():
    reset()
    return run_tests("词表配置契约（顶层唯一赋值）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
