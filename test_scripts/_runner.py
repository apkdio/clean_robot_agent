"""测试公共基础设施：路径注入 + 断言 + 计数汇总 + e2e 门控。

所有 test_*.py 模块通过 ``from _runner import *`` 复用本文件提供的：
  - 路径注入：把项目根目录和 tools/ 加入 sys.path，保证
    ``import tools/sops/config/function_tools`` 以及 tools 内部的
    裸 import（``from config_tool import ...``）都能正确解析。
  - 断言函数：_assert / _assert_eq / _assert_true / _assert_false /
    _assert_in / _assert_not_in / _assert_raises。
  - 计数与汇总：reset / run_tests / summary。
  - e2e 门控：E2E 标志（由命令行 ``--e2e`` 决定）与 @e2e 装饰器；
    依赖 Ollama / Chroma / torch 的集成用例默认跳过，加 ``--e2e`` 才执行。

约定：
  - 每个测试模块定义一个 ``run() -> (passed, total, skipped)``，既可独立
    运行，也可被 run_tests.py 聚合。
  - 纯规则用例默认运行（快、确定性）；集成/端到端用例用 @e2e 标注。

运行解释器：项目依赖 torch 等库只装在 .venv 里，请用
  .venv\\Scripts\\python.exe run_tests.py [--e2e]
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Windows 控制台默认 GBK，强制 stdout/stderr 用 UTF-8，避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

E2E = "--e2e" in sys.argv

_total = 0
_passed = 0
_skipped = 0
_failed = []  # list[(name, detail)]


def reset():
    global _total, _passed, _skipped, _failed
    _total = _passed = _skipped = 0
    _failed = []


def _record(name, ok, detail=""):
    global _total, _passed
    _total += 1
    if ok:
        _passed += 1
    else:
        _failed.append((name, detail))


def _skip(reason="需要 --e2e（依赖 Ollama/Chroma/torch）"):
    global _skipped
    _skipped += 1
    print(f"  SKIP {reason}")


def _assert(cond, msg=""):
    ok = bool(cond)
    _record(msg or "断言", ok)
    print(("  OK   " if ok else "  FAIL ") + msg)


def _assert_eq(actual, expected, msg=""):
    ok = actual == expected
    _record(msg or "相等断言", ok, f"期望={expected!r} 实际={actual!r}")
    print(("  OK   " if ok else "  FAIL ") + msg + ("" if ok else f"  [期望={expected!r} 实际={actual!r}]"))


def _assert_true(actual, msg=""):
    ok = bool(actual)
    _record(msg or "真值断言", ok, f"实际={actual!r}")
    print(("  OK   " if ok else "  FAIL ") + msg + ("" if ok else f"  [实际={actual!r}]"))


def _assert_false(actual, msg=""):
    ok = not actual
    _record(msg or "假值断言", ok, f"实际={actual!r}")
    print(("  OK   " if ok else "  FAIL ") + msg + ("" if ok else f"  [实际={actual!r}]"))


def _assert_in(content, container, msg=""):
    ok = content in container
    _record(msg or f"包含断言 '{content}'", ok, f"未找到 '{content}'")
    print(("  OK   " if ok else "  FAIL ") + msg + ("" if ok else f"  [未找到 '{content}']"))


def _assert_not_in(content, container, msg=""):
    ok = content not in container
    _record(msg or f"不包含断言 '{content}'", ok, f"不应出现 '{content}'")
    print(("  OK   " if ok else "  FAIL ") + msg + ("" if ok else f"  [不应出现 '{content}']"))


def _assert_raises(exc_type, fn, msg=""):
    try:
        fn()
    except exc_type:
        _record(msg or f"应抛出 {exc_type.__name__}", True)
        print("  OK   " + msg)
    except Exception as exc:  # noqa: BLE001
        _record(msg or f"应抛出 {exc_type.__name__}", False, f"抛出了 {type(exc).__name__}: {exc}")
        print(f"  FAIL {msg}  [抛出 {type(exc).__name__}: {exc}]")
    else:
        _record(msg or f"应抛出 {exc_type.__name__}", False, "未抛出异常")
        print(f"  FAIL {msg}  [未抛出 {exc_type.__name__}]")


def e2e(reason="需要 --e2e（依赖 Ollama/Chroma/torch）"):
    """集成/端到端用例装饰器：未加 --e2e 时跳过。"""
    def deco(fn):
        def wrapper(*args, **kwargs):
            if not E2E:
                _skip(reason)
                return
            return fn(*args, **kwargs)
        wrapper.__name__ = getattr(fn, "__name__", "e2e")
        return wrapper
    return deco


def run_tests(title, tests):
    """顺序执行 tests（零参 callable 列表），单个异常不中断后续。"""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    for fn in tests:
        name = getattr(fn, "__name__", str(fn))
        print(f"\n── {name} ──")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            _record(f"{name}（异常）", False, f"{type(exc).__name__}: {exc}")
            print(f"  FAIL 抛出异常 {type(exc).__name__}: {exc}")
    return summary(title)


def summary(title):
    passed, total = _passed, _total
    print()
    print("=" * 72)
    print(f"[{title}] 通过 {passed}/{total}", end="")
    if _skipped:
        print(f"，跳过 {_skipped}", end="")
    if _failed:
        print(f"，失败 {len(_failed)} 项：")
        for name, detail in _failed:
            print(f"  ✗ {name}" + (f"  → {detail}" if detail else ""))
    else:
        print("，无失败")
    print("=" * 72)
    return passed, total, _skipped


__all__ = [
    "E2E", "reset", "run_tests", "summary", "e2e", "_skip",
    "_assert", "_assert_eq", "_assert_true", "_assert_false",
    "_assert_in", "_assert_not_in", "_assert_raises",
]
