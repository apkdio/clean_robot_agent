"""版本中台 CLI：查看/校验/登记评测集与导出快照。"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from tools import version_store as vs  # noqa: E402


def _short_eval(value):
    return value.split("@", 1)[-1] if value else "-"


def _show(args) -> int:
    data = vs.resolve()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    comp = data["components"]
    code = comp["code"]
    print("[版本] %s" % code["id"])
    print("       eval " + " / ".join("%s@%s" % (k, _short_eval(v))
                                      for k, v in comp["eval"].items()
                                      if not (k == "context" and not v)))
    archived = vs._find_eval("context", status="archived")
    if archived and not comp["eval"].get("context"):
        print("       context@%s#%s（archived）" % (archived.get("rev"), archived.get("sha256")))
    env = comp["env"]
    models = env.get("models") or {}
    print("       env kb#%s · prompts#%s · config#%s · trace v%s · models %s" % (
        (env.get("kb") or {}).get("fingerprint"),
        (env.get("prompts") or {}).get("fingerprint"),
        (env.get("config") or {}).get("fingerprint"),
        env.get("trace"),
        "/".join(str(models.get(k) or "-") for k in ("chat", "small", "embedding", "reranker"))))
    print("       run %s" % data["run_id"])
    for warning in data.get("warnings") or []:
        print("       ⚠ %s" % warning)
    return 0


def _check(args) -> int:
    issues = vs.check(strict=args.strict)
    if not args.strict:
        for warning in vs.check_warnings():
            print("[版本] ⚠ %s" % warning)
    if not issues:
        print("[版本] check 通过")
        return 0
    for issue in issues:
        print("[版本] ✗ %s" % issue)
    return 1


def _register(args) -> int:
    entry = vs.register_eval(args.track, note=args.note or "", path=args.path,
                             rev=args.rev, status="archived" if args.archived else "active",
                             archive_dir=args.archive_dir)
    print("[版本] 已登记 %s rev=%s sha=%s cases=%s" %
          (entry["track"], entry["rev"], entry["sha256"], entry["cases"]))
    return 0


def _bump(args) -> int:
    entry = vs.bump_eval(args.track, args.note or "")
    print("[版本] 已升版 %s rev=%s sha=%s" % (entry["track"], entry["rev"], entry["sha256"]))
    return 0


def _archive(args) -> int:
    entry = vs.archive_eval(args.track, args.reason or "", archive_dir=args.archive_dir)
    print("[版本] 已归档 %s rev=%s" % (entry["track"], entry["rev"]))
    return 0


def _diff(args) -> int:
    diff = vs.diff_eval(args.track, args.rev_a, args.rev_b)
    print(json.dumps(diff, ensure_ascii=False, indent=2))
    return 0


def _export(args) -> int:
    path = vs.export(args.out)
    print("[版本] 已导出 %s" % path)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="评测版本中台")
    sub = ap.add_subparsers(dest="cmd", required=True)

    show = sub.add_parser("show")
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=_show)

    check = sub.add_parser("check")
    check.add_argument("--strict", action="store_true")
    check.set_defaults(func=_check)

    reg = sub.add_parser("register-eval")
    reg.add_argument("track", choices=vs.TRACKS)
    reg.add_argument("--note", default="")
    reg.add_argument("--path")
    reg.add_argument("--rev", type=int)
    reg.add_argument("--archived", action="store_true")
    reg.add_argument("--archive-dir")
    reg.set_defaults(func=_register)

    bump = sub.add_parser("bump-eval")
    bump.add_argument("track", choices=vs.TRACKS)
    bump.add_argument("--note", required=True)
    bump.set_defaults(func=_bump)

    archive = sub.add_parser("archive-eval")
    archive.add_argument("track", choices=vs.TRACKS)
    archive.add_argument("--reason", required=True)
    archive.add_argument("--archive-dir")
    archive.set_defaults(func=_archive)

    diff = sub.add_parser("diff-eval")
    diff.add_argument("track", choices=vs.TRACKS)
    diff.add_argument("rev_a", type=int)
    diff.add_argument("rev_b", type=int)
    diff.set_defaults(func=_diff)

    export = sub.add_parser("export")
    export.add_argument("--out", default="data/versions/exports/version_snapshot.json")
    export.set_defaults(func=_export)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print("[版本] %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
