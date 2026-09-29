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
"""

import json
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

# 测试隔离：把会话/上下文文件写到 data/test_context(_metadata)，不污染真实会话数据。
# context_store 在 import 时读取这两个变量，而 _runner 总先于其它模块被导入，
# 故能保证首次加载时就指向测试目录（已显式设置时不覆盖）。
os.environ.setdefault("CONTEXT_DIR", "data/test_context")
os.environ.setdefault("CONTEXT_META_DIR", "data/test_context_metadata")
os.environ.setdefault("TRACE_DIR", "temp/trace_test")   # trace 也隔离，避免污染真实 logs/trace/

# 上一段那句「_runner 总先于其它模块被导入」是**隐含约定**，破了会静默失效
_EARLY_IMPORTED = [m for m in ("tools.context_store", "context_store",
                               "tools.vector_store", "vector_store") if m in sys.modules]
if _EARLY_IMPORTED:
    raise RuntimeError(
        "测试隔离未生效：%s 已在 _runner 之前被导入——测试会话会被写进真实数据目录。"
        "请把 `from _runner import *` 放到所有 tools / sops 导入之前。"
        % "、".join(_EARLY_IMPORTED)
    )

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


def stats():
    """返回当前计数 (passed, total, skipped)，供自定义 run() 模块读取。"""
    return _passed, _total, _skipped


def disable_shadow():
    """评测期间关掉主链路的影子探针（`tools/agent.py::_shadow_probe`）。

    影子只落日志、不参与被测行为，却要额外跑一次 LLM 决策（实测每轮 +75~125s），
    白白占住主链路要用的算力。返回 restore 回调，请在 `finally` 里复现——“run_tests.py
    在同一个进程里顺序跑多个模块，漏复现会污染后面的模块。
    """
    from tools import agent as agent_mod

    original = agent_mod._shadow_probe
    agent_mod._shadow_probe = lambda *args, **kwargs: None
    return lambda: setattr(agent_mod, "_shadow_probe", original)


# 双重导入（裸名 + `tools.` 前缀）会加载出两份带状态的模块，两者都能 import 成功。
# 项目里踩过：context_store 两份实例 → 喂给 LLM 的历史缺 assistant 一侧。
_WATCHED_MODULES = ("context_store", "vector_store", "redis_store", "log_tool", "llm_tool",
                    "config_tool", "path_tool", "agent", "intent_router", "metadata_extractor")


def check_module_duality():
    """返回「同名两份实例」的模块名：`sys.modules` 里裸名与 `tools.` 前缀都存在且不同一。

    只报告不报错：已知 `llm_tool` / `log_tool` 是幂等/缓存类，影响低；
    但带可变状态的模块（`context_store` / `vector_store`）分裂会真的出错。
    """
    dup = []
    for name in _WATCHED_MODULES:
        bare, prefixed = sys.modules.get(name), sys.modules.get("tools." + name)
        if bare is not None and prefixed is not None and bare is not prefixed:
            dup.append(name)
    return dup


def reset_trace(session_id: str) -> None:
    """清掉某个会话的 trace：删当天落盘文件 + 清 `trace_store` 的进程内计数。

    评测会跨行复用同一个 sid，不清就会：① 文件跨次运行累积；② `turn_index` 接着上次的行数数
    （`_seq` 只在首次种入）。trace 不是判定依据，这里全 best-effort；若将来内部结构改名，
    这里只会静默失效——真要长期依赖，应让 `trace_store` 提供一个公开的 reset。
    """
    try:
        import trace_store

        path = trace_store._path(session_id)
        if os.path.isfile(path):
            os.remove(path)
        for key in [k for k in list(trace_store._seq) if k[0] == session_id]:
            trace_store._seq.pop(key, None)
        trace_store._last.pop(session_id, None)
    except Exception:  # noqa: BLE001
        pass


def read_last_turn(session_id: str) -> dict:
    """读某会话当天 trace 的最后一条记录（评测失败归因用）；读不到返回 `{}`。"""
    try:
        import trace_store

        last = {}
        with open(trace_store._path(session_id), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except ValueError:
                        continue
        return last
    except Exception:  # noqa: BLE001
        return {}


__all__ = [
    "E2E", "reset", "run_tests", "summary", "stats", "e2e", "_skip", "disable_shadow",
    "check_module_duality", "reset_trace", "read_last_turn",
    "_assert", "_assert_eq", "_assert_true", "_assert_false",
    "_assert_in", "_assert_not_in", "_assert_raises",
]
