"""运行期模块身份检查：同一物理文件不得以多个模块名同时加载。"""
from __future__ import annotations

import os
import sys
from collections import defaultdict


def duplicate_module_files(root: str) -> dict:
    """返回 {真实文件路径: [模块名...]}，只统计 root 下的第一方模块。"""
    root = os.path.realpath(root)
    by_file = defaultdict(list)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        real = os.path.realpath(path)
        if not (real == root or real.startswith(root + os.sep)):
            continue
        if os.sep + "__pycache__" + os.sep in real:
            continue
        by_file[real].append(name)
    return {path: sorted(names) for path, names in by_file.items() if len(names) > 1}


def assert_no_duplicate_module_files(root: str) -> None:
    duplicates = duplicate_module_files(root)
    if duplicates:
        detail = "; ".join("%s -> %s" % (path, ", ".join(names))
                           for path, names in sorted(duplicates.items()))
        raise RuntimeError("检测到同一文件被多个模块名加载：" + detail)
