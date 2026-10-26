"""评测入口：只跑 eval/ 下的数据集评测轨（检索质量 / 多轮行为 / 意图分类 / SOP 状态机 + tag 契约自检）。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # 项目根

from test_scripts._runner import E2E, disable_shadow  # noqa: E402
from tools import version_store as vs  # noqa: E402

# ── tag 契约自检（纯规则，先跑：判据错了后面的评测跑得再久也没意义）──────
from test_scripts.eval import test_behavior_tags  # noqa: E402

# ── 数据集评测轨（eval/）─────────────────────────────────────────
from test_scripts.eval import test_retrieval_eval  # noqa: E402
from test_scripts.eval import test_context_eval  # noqa: E402
from test_scripts.eval import test_intent_eval  # noqa: E402
from test_scripts.eval import test_sop_eval  # noqa: E402


MODULES = [
    ("行为观测点 tag 契约（自检）", test_behavior_tags),
    ("检索质量评测", test_retrieval_eval),
    ("多轮行为评测", test_context_eval),
    ("意图分类评测", test_intent_eval),
    ("SOP 行为轨", test_sop_eval),
]


def _short_eval(value):
    return value.split("@", 1)[-1] if value else "-"


def _print_version(resolved):
    comp = resolved["components"]
    print("[版本] %s · run %s" % (comp["code"]["id"], resolved["run_id"]))
    print("       eval " + " / ".join("%s@%s" % (k, _short_eval(v))
                                      for k, v in (comp["eval"] or {}).items()))
    env = comp.get("env") or {}
    intent = env.get("intent") or {}
    sops = env.get("sops") or {}
    print("       env kb#%s · prompts#%s · config#%s · intent %s/%s · sops#%s · trace v%s · models %s" % (
        (env.get("kb") or {}).get("fingerprint"),
        (env.get("prompts") or {}).get("fingerprint"),
        (env.get("config") or {}).get("fingerprint"),
        intent.get("head"), intent.get("dataset"),
        ",".join("%s:%s" % (sid, (item or {}).get("def_ver"))
                 for sid, item in sorted(sops.items())) or "-",
        env.get("trace"), env.get("models") or "-"))
    for warning in resolved.get("warnings") or []:
        print("       ⚠ %s" % warning)


def main():
    # 评测隔离：影子探针只落日志、不参与被测行为，却要额外跑一次 FC（实测每轮 +75~125s）。
    # 整个评测进程关掉，算力让给被测主链路；进程结束即失效，不影响正常运行时的影子观测。
    disable_shadow()
    strict = "--strict-versions" in sys.argv
    resolved = None
    try:
        resolved = vs.resolve()
        _print_version(resolved)
    except Exception as exc:  # noqa: BLE001 - 版本采集失败不能吞掉评测，但必须显式可见
        print("[版本] ⚠ 采集失败：%s: %s（继续评测，结果不带 run manifest）" %
              (type(exc).__name__, exc))
    if strict and resolved and resolved.get("warnings"):
        handle = vs.begin_run("run_tests --strict-versions", resolved)
        vs.finish_run(handle, "blocked")
        print("[版本] --strict-versions：存在未登记/缺失的版本，终止评测")
        return 1
    handle = vs.begin_run("run_tests%s" % (" --e2e" if E2E else ""), resolved) if resolved else None

    status = "error"
    try:
        grand_p = grand_t = grand_s = 0
        for name, mod in MODULES:
            try:
                p, t, s = mod.run()
            except Exception as exc:  # noqa: BLE001 - 一个模块崩了（如评测集文件坏了）不该让整份报告消失
                print(f"\n[!] {name} 执行异常：{type(exc).__name__}: {exc}")
                p, t, s = 0, 1, 0
            grand_p += p
            grand_t += t
            grand_s += s

        print()
        print("#" * 72)
        if grand_t:
            print(f"# 总计：{grand_p}/{grand_t} 通过，跳过 {grand_s}", end="")
            if grand_p == grand_t:
                print("  ✔ 通过")
            else:
                print(f"  ✘ {grand_t - grand_p} 项失败")
        else:
            print(f"# 未执行任何用例（{grand_s} 个模块被跳过）")
        print("#" * 72)
        if not E2E:
            print("⚠ 未加 --e2e：三条依赖外部模型/检索的数据集评测轨跳过；SOP 行为轨仍会运行。")
            print("  加 --e2e 才是完整评测（需 Ollama + Chroma + reranker）。")
        if grand_t == 0 or grand_s >= 3:
            # 「数据集评测轨全跳过」不等于「评测通过」：退出码要能区分（review §4.6）。
            # 无 --e2e、或评测集/分类头文件缺失时都落这里——只跑了纯规则自检，不能算验证过。
            print("# 退出码 2：数据集评测轨全部跳过（没验证 ≠ 验证通过）")
            status = "skipped"
            code = 2
        else:
            status = "passed" if grand_p == grand_t else "failed"
            code = 0 if grand_p == grand_t else 1
        return code
    finally:
        if handle:
            vs.finish_run(handle, status)
            print("[版本] run manifest: %s" % handle["path"])


if __name__ == "__main__":
    code = main()
    sys.exit(code)
