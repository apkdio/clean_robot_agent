"""聚合运行所有分模块测试。

用法（请用 .venv 解释器，torch 等依赖只装在 .venv）：
  .venv\\Scripts\\python.exe run_tests.py            # 运行全部单元测试（快、确定性）
  .venv\\Scripts\\python.exe run_tests.py --e2e      # 追加集成/端到端（意图分类、多轮对话、SOP 全流程）

等价于逐个运行（每个模块都可独立跑）：
  .venv\\Scripts\\python.exe test_intent.py [--e2e]
  .venv\\Scripts\\python.exe test_purchase_sop.py
  ...

模块清单：
  - 意图分类      test_intent.py        （需 --e2e：torch + Ollama）
  - 选购 SOP      test_purchase_sop.py
  - 故障排查 SOP  test_repair_sop.py
  - 上下文存储    test_context.py
  - 结构化提取    test_metadata.py
  - Function Tools test_function_tools.py
  - 检索          test_retrieval.py
  - Agent 前置防护 test_agent_guards.py
  - 多轮实战对话  test_dialogue.py      （需 --e2e：Ollama + Chroma）
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
