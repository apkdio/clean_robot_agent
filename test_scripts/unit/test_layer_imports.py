"""分层 import 契约：S8-5 收口后，底层不得反向依赖 service。

这不是完整的 CSM 约束；Controller 只依赖 Service 的收紧留给 8-A。
"""
import ast
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *

_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_LOWER_LAYERS = ("repo", "gateway", "sops", "tools")


def _imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
            if node.module in ("service", "sops", "tools", "repo", "gateway"):
                names.extend(node.module + "." + a.name for a in node.names)
    return names


def _service_imports(layer):
    root = os.path.join(_ROOT, layer)
    hits = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            for name in _imports(path):
                if name == "service" or name.startswith("service."):
                    hits.append("%s:%s" % (os.path.relpath(path, _ROOT), name))
    return sorted(hits)


def test_lower_layers_do_not_import_service():
    violations = []
    for layer in _LOWER_LAYERS:
        violations.extend(_service_imports(layer))
    _assert_eq(violations, [], "repo/gateway/sops/tools 不得反向依赖 service")


def test_compatibility_shells_removed():
    for name in ("agent", "chat_service", "context_store", "llm_tool", "redis_store", "reranker"):
        path = os.path.join(_ROOT, "tools", name + ".py")
        _assert_false(os.path.exists(path), "迁移薄壳应已删除：tools/%s.py" % name)


def test_service_entrypoints_importable():
    import service.agent
    import service.chat_service

    _assert(callable(service.agent.ask_stream), "service.agent.ask_stream 可调用")
    _assert(callable(service.chat_service.stream_turn), "service.chat_service.stream_turn 可调用")


TESTS = [
    test_lower_layers_do_not_import_service,
    test_compatibility_shells_removed,
    test_service_entrypoints_importable,
]


def run():
    reset()
    return run_tests("分层 import 契约", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
