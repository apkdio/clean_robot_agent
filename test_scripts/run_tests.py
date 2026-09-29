"""评测入口：只跑 `eval/` 下的数据集评测轨（检索质量 / 多轮行为 / 意图分类 + tag 契约自检）。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in (_HERE, os.path.join(_HERE, "eval")):
    sys.path.insert(0, _sub)

from _runner import E2E, check_module_duality, disable_shadow  # noqa: E402

# ── tag 契约自检（纯规则，先跑：判据错了后面的评测跑得再久也没意义）──────
import test_behavior_tags    # noqa: E402

# ── 数据集评测轨（eval/）─────────────────────────────────────────
import test_retrieval_eval   # noqa: E402
import test_context_eval     # noqa: E402
import test_intent_eval      # noqa: E402


MODULES = [
    ("行为观测点 tag 契约（自检）", test_behavior_tags),
    ("检索质量评测", test_retrieval_eval),
    ("多轮行为评测", test_context_eval),
    ("意图分类评测", test_intent_eval),
]


def main():
    # 评测隔离：影子探针只落日志、不参与被测行为，却要额外跑一次 FC（实测每轮 +75~125s）。
    # 整个评测进程关掉，算力让给被测主链路；进程结束即失效，不影响正常运行时的影子观测。
    disable_shadow()
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

    dup = check_module_duality()
    if dup:
        print(f"\n⚠ 同名模块两份实例（裸名 + `tools.` 前缀）：{'、'.join(dup)}")
        print("  幂等/缓存类影响低；带可变状态的模块分裂会真的出错——新增模块时统一用 `tools.` 前缀。")

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
        print("⚠ 未加 --e2e：三条数据集评测轨全部跳过，本次只跑了纯规则自检。")
        print("  加 --e2e 才是完整评测（需 Ollama + Chroma + reranker）。")
    sys.exit(0 if grand_p == grand_t else 1)


if __name__ == "__main__":
    main()
