"""聚合运行所有分模块测试（每个 test_*.py 也可独立运行，约定见 _runner.py）。

默认只跑纯规则用例（快、确定性）；加 --e2e 追加需要 Ollama / Chroma / torch 的
集成用例（意图分类、多轮实战对话、检索质量评测、多轮上下文评测）。
模块清单见下方 import 与 TESTS 列表。解释器用 .venv\\Scripts\\python.exe。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _runner import E2E  # noqa: E402

import test_intent           # noqa: E402
import test_purchase_sop     # noqa: E402
import test_repair_sop       # noqa: E402
import test_context          # noqa: E402
import test_metadata         # noqa: E402
import test_function_tools   # noqa: E402
import test_retrieval        # noqa: E402
import test_agent_guards     # noqa: E402
import test_dialogue         # noqa: E402
import test_retrieval_eval   # noqa: E402
import test_context_eval     # noqa: E402


MODULES = [
    ("意图分类", test_intent),
    ("选购 SOP", test_purchase_sop),
    ("故障排查 SOP", test_repair_sop),
    ("上下文存储", test_context),
    ("结构化提取", test_metadata),
    ("Function Tools", test_function_tools),
    ("检索", test_retrieval),
    ("Agent 前置防护", test_agent_guards),
    ("多轮实战对话", test_dialogue),
    ("检索质量评测", test_retrieval_eval),
    ("多轮上下文评测", test_context_eval),
]


def main():
    grand_p = grand_t = grand_s = 0
    for name, mod in MODULES:
        p, t, s = mod.run()
        grand_p += p
        grand_t += t
        grand_s += s

    print()
    print("#" * 72)
    print(f"# 总计：{grand_p}/{grand_t} 通过，跳过 {grand_s}", end="")
    if grand_p == grand_t:
        print("  ✔ 全部通过")
    else:
        print(f"  ✘ {grand_t - grand_p} 项失败")
    print("#" * 72)
    if not E2E:
        print("提示：追加 --e2e 可运行意图分类与多轮对话等集成/端到端用例。")
    sys.exit(0 if grand_p == grand_t else 1)


if __name__ == "__main__":
    main()
