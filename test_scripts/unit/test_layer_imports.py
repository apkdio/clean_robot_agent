"""分层 import 契约：底层不得反向依赖 service，Controller 只能走 service。

Controller 的共享例外是 `utils.dto` 与 `utils.log_tool`；其余 utils 不允许直连。
"""
import ast
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(os.path.dirname(_SCRIPT_DIR)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

from test_scripts._runner import *

_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_LOWER_LAYERS = ("gateway", "mapper", "utils")
_CONTROLLER_ROOT = os.path.join(_ROOT, "webapp")
_CONTROLLER_ALLOWED_UTILS = ("utils.dto", "utils.log_tool")


def _imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
            if node.module in ("service", "sops", "utils", "repo", "gateway"):
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
    _assert_eq(violations, [], "gateway/mapper/utils 不得反向依赖 service")


_LAYER_RULES = {
    "utils": ("utils", "config"),
    "gateway": ("gateway", "utils", "config"),
    "mapper": ("mapper", "gateway", "utils", "config"),
    "service": ("service", "mapper", "gateway", "utils", "function_tools", "config"),
    "webapp": ("service", "utils.dto", "utils.log_tool"),
}


def _allowed(name, prefixes):
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _layer_violations(layer, prefixes):
    hits = []
    root = os.path.join(_ROOT, layer)
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            for name in _imports(path):
                top = name.split(".", 1)[0]
                if top in _LAYER_RULES and not _allowed(name, prefixes):
                    hits.append("%s:%s" % (os.path.relpath(path, _ROOT), name))
    return sorted(hits)


def test_layer_direction_matrix():
    """按最终定稿检查所有层的允许 import，不只检查反向依赖。"""
    violations = []
    for layer, prefixes in _LAYER_RULES.items():
        violations.extend(_layer_violations(layer, prefixes))
    _assert_eq(violations, [], "各层 import 必须符合定稿方向矩阵")


def _controller_files():
    return [
        os.path.join(dirpath, fn)
        for dirpath, _, files in os.walk(_CONTROLLER_ROOT)
        for fn in files
        if fn.endswith(".py")
    ]


def _controller_violations():
    hits = []
    for path in _controller_files():
        for name in _imports(path):
            forbidden = (
                name == "repo" or name.startswith("repo.")
                or name == "gateway" or name.startswith("gateway.")
                or name == "mapper" or name.startswith("mapper.")
                or name == "sops" or name.startswith("sops.")
                or name == "service.sops" or name.startswith("service.sops.")
                or name == "config" or name.startswith("config.")
                or (name == "tools" or name == "utils" or
                    (name.startswith("utils.") and
                     not any(name == p or name.startswith(p + ".")
                             for p in _CONTROLLER_ALLOWED_UTILS)))
            )
            if forbidden:
                hits.append("%s:%s" % (os.path.relpath(path, _ROOT), name))
    return sorted(hits)


def test_controller_only_imports_service_and_shared_tools():
    """Controller 的业务依赖只能走 service；共享 DTO/log 是显式例外。"""
    _assert(_controller_files(), "扫到 webapp 源文件（否则规则是空转）")
    _assert_eq(_controller_violations(), [],
               "webapp 不得直接依赖 repo/gateway/mapper/service.sops/config 或业务型 utils")


def test_compatibility_shells_removed():
    for name in ("agent", "chat_service", "context_store", "llm_tool", "redis_store", "reranker"):
        path = os.path.join(_ROOT, "tools", name + ".py")
        _assert_false(os.path.exists(path), "迁移薄壳应已删除：tools/%s.py" % name)
    _assert_false(os.path.exists(os.path.join(_ROOT, "tools")),
                  "tools/ 已归 utils，不应保留旧目录")
    _assert_false(os.path.exists(os.path.join(_ROOT, "repo")),
                  "repo/ 已归 mapper，不应保留旧目录")
    _assert_false(os.path.exists(os.path.join(_ROOT, "sops")),
                  "sops/ 已归 service.sops，不应保留旧目录")
    legacy = [m for m in sys.modules
              if m in ("tools", "repo") or m.startswith(("tools.", "repo."))]
    _assert_eq(legacy, [], "运行期不得加载旧 tools/repo 模块，避免双实例")


def test_no_duplicate_loaded_module_files():
    from test_scripts.import_guard import duplicate_module_files
    _assert_eq(duplicate_module_files(_ROOT), {},
               "同一物理文件不得以多个模块名加载")


def test_service_entrypoints_importable():
    import service.agent
    import service.chat_service

    _assert(callable(service.agent.ask_stream), "service.agent.ask_stream 可调用")
    _assert(callable(service.chat_service.stream_turn), "service.chat_service.stream_turn 可调用")


TESTS = [
    test_lower_layers_do_not_import_service,
    test_layer_direction_matrix,
    test_controller_only_imports_service_and_shared_tools,
    test_compatibility_shells_removed,
    test_no_duplicate_loaded_module_files,
    test_service_entrypoints_importable,
]


def run():
    reset()
    return run_tests("分层 import 契约", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
