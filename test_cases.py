"""系统测试用例：意图分类 + 端到端问答。

运行:
  python test_cases.py            # 意图分类测试（快）
  python test_cases.py --e2e      # 意图分类 + 端到端抽样测试
"""

import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "tools"))

# (query, 预期意图)
TEST_CASES = [
    # ── robot：扫地机器人领域（30 条）──
    ("边刷多久换一次", "robot"),
    ("扫地机器人不回充怎么办", "robot"),
    ("吸力下降怎么办", "robot"),
    ("HEPA滤网多久清洗一次", "robot"),
    ("拖地有水痕怎么办", "robot"),
    ("首次使用扫地机器人需要做什么", "robot"),
    ("如何设置定时清扫", "robot"),
    ("预算1000以内扫地机器人推荐", "robot"),
    ("预算1500以内有什么机器人推荐", "robot"),
    ("预算2000以内有什么机器人", "robot"),
    ("主刷多久更换一次", "robot"),
    ("机器人找不到充电座怎么办", "robot"),
    ("APP无法连接机器人怎么办", "robot"),
    ("建图不完整怎么解决", "robot"),
    ("电池续航下降怎么办", "robot"),
    ("扫地机器人有异响怎么办", "robot"),
    ("水箱漏水怎么办", "robot"),
    ("边角位置清扫不干净怎么办", "robot"),
    ("地毯上的灰尘清理不彻底怎么办", "robot"),
    ("宠物毛发怎么清理", "robot"),
    ("木地板拖地有什么注意事项", "robot"),
    ("机器人跨越门槛失败怎么办", "robot"),
    ("尘盒多久清理一次", "robot"),
    ("拖布怎么清洗", "robot"),
    ("机器人可以恢复出厂设置吗", "robot"),
    ("最近半年发布的产品里有什么1000以内的扫地机器人", "robot"),
    ("最近半年发布的扫地机器人推荐", "robot"),
    ("近三个月发布的机器人有哪些", "robot"),
    ("2025年三月有什么发布的机器人", "robot"),
    ("2025年发布的扫地机器人有哪些", "robot"),

    # ── other：领域外（10 条）──
    ("有没有飞机卖", "other"),
    ("有没有2000以内的电动车推荐", "other"),
    ("手机哪个牌子好", "other"),
    ("空调不制冷了怎么办", "other"),
    ("电脑卡顿怎么办", "other"),
    ("汽车保养多久一次", "other"),
    ("洗衣机漏水怎么办", "other"),
    ("冰箱有异味怎么办", "other"),
    ("电视怎么联网", "other"),
    ("耳机推荐一下", "other"),

    # ── casual：闲聊（8 条）──
    ("你好", "casual"),
    ("谢谢", "casual"),
    ("在吗", "casual"),
    ("你是谁", "casual"),
    ("你能做什么", "casual"),
    ("再见", "casual"),
    ("早上好", "casual"),
    ("辛苦了", "casual"),

    # ── unknown：模糊（7 条）──
    ("这个东西怎么样", "unknown"),
    ("好用吗", "unknown"),
    ("推荐一下", "unknown"),
    ("多少钱", "unknown"),
    ("靠谱吗", "unknown"),
    ("值得买吗", "unknown"),
    ("有没有好点的", "unknown"),
]

# 端到端抽样（每类挑代表性的）
E2E_SAMPLE = [
    "边刷多久换一次",
    "预算1000以内扫地机器人推荐",
    "最近半年发布的产品里有什么1000以内的扫地机器人",
    "2025年三月有什么发布的机器人",
    "有没有飞机卖",
    "你好",
    "这个东西怎么样",
]


def test_intent():
    from intent_router import route_intent

    print("=" * 70)
    print("意图分类测试（共 %d 条）" % len(TEST_CASES))
    print("=" * 70)

    correct = 0
    errors = []
    for query, expected in TEST_CASES:
        actual = route_intent(query)
        ok = (actual == expected)
        if ok:
            correct += 1
        else:
            errors.append((query, expected, actual))
        mark = "OK " if ok else "FAIL"
        print(f"  {mark} {actual:9s} | 预期 {expected:9s} | {query}")

    print()
    print("准确率: %d/%d = %.1f%%" % (correct, len(TEST_CASES), correct / len(TEST_CASES) * 100))
    if errors:
        print("\n误判清单:")
        for q, e, a in errors:
            print(f"  FAIL {q}  预期={e}  实际={a}")
    return correct, len(TEST_CASES)


def test_e2e():
    from agent import ask_stream

    print()
    print("=" * 70)
    print("端到端抽样测试（%d 条）" % len(E2E_SAMPLE))
    print("=" * 70)

    for q in E2E_SAMPLE:
        answer = ''.join(ask_stream(q))
        print(f"\n【{q}】")
        print("  " + answer[:200].replace("\n", "\n  "))


if __name__ == "__main__":
    test_intent()
    if "--e2e" in sys.argv:
        test_e2e()
