"""把旧基线迁到 baseline_v:2：补 pinned、升级 per_id，不重跑评测。

旧 per_id 没有期望值，迁移后 expect_known=false；从下一次干净 --save-baseline 起才启用
「期望变更」判断。
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from tools import version_store as vs  # noqa: E402


def _targets():
    rows = [("retrieval", "data/eval/retrieval/baseline.json"),
            ("intent", "data/eval/intent/baseline.json")]
    for path in glob.glob("data/eval/context/archive/*/baseline.json"):
        rows.append(("context", path))
    return rows


def main() -> int:
    changed_count = 0
    for track, rel in _targets():
        path = vs._abs(rel)
        if not os.path.isfile(path):
            print("[迁移] 跳过（不存在）：%s" % rel)
            continue
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("baseline_v") == 2 and payload.get("pinned") and "--refresh" not in sys.argv:
            print("[迁移] 已是 v2：%s（加 --refresh 可重算 pinned）" % rel)
            continue
        migrated, changed = vs.migrate_baseline_payload(payload, track)
        if not changed and migrated.get("baseline_v") == payload.get("baseline_v") \
                and migrated.get("pinned") == payload.get("pinned"):
            print("[迁移] 无需变更：%s" % rel)
            continue
        backup = path + ".v1.bak"
        if not os.path.isfile(backup):
            shutil.copyfile(path, backup)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(migrated, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
        changed_count += 1
        print("[迁移] %s（per_id %d 条，expect_known=%s）" %
              (rel, len(migrated.get("per_id") or {}), migrated.get("expect_known")))
    print("[迁移] 完成，变更 %d 个基线；旧文件保留为 *.v1.bak" % changed_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
