"""
系统测试用例：扫地机器人 Agent 的意图分类 + 端到端问答。

支持的意图类别：
    - robot   ：扫地机器人领域问题
    - other   ：明确属于其他领域的问题
    - casual  ：问候、闲聊、能力咨询等
    - unknown ：上下文不足、指代不明、意图模糊

运行方式：
    python test_cases.py            # 仅运行意图分类测试（快）
    python test_cases.py --e2e      # 意图分类 + 端到端抽样测试
"""

import os
import sys
import time
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "tools"))


# ============================================================
# 意图分类测试集
# 格式：(用户问题, 预期意图)
# 每个类别 30 条，其中包含少量笔误、口语表达和不规范表达。
# ============================================================
TEST_CASES = [
    # --------------------------------------------------------
    # robot：扫地机器人领域（30 条）
    # --------------------------------------------------------
    ("边刷多久换一次", "robot"),
    ("主刷多久需要更换", "robot"),
    ("HEPA滤网多久清洗一次", "robot"),
    ("尘盒多久倒一次比较好", "robot"),
    ("拖布应该怎么清洗", "robot"),
    ("扫地机器人不回充怎么办", "robot"),
    ("机器人找不到充电座怎么办", "robot"),
    ("扫地机器人充不进电是什么原因", "robot"),
    ("吸力变小了怎么办", "robot"),
    ("扫地机器人有异响怎么办", "robot"),
    ("拖地后地板上有水痕怎么办", "robot"),
    ("水箱漏水怎么处理", "robot"),
    ("机器人跨不过门槛怎么办", "robot"),
    ("边角位置总是扫不干净", "robot"),
    ("地毯上的毛发吸不干净怎么办", "robot"),
    ("家里有宠物，扫地机器人怎么清理猫毛", "robot"),
    ("木地板可以用扫地机器人拖吗", "robot"),
    ("APP连不上扫地机器人怎么办", "robot"),
    ("扫地机器人联网失败怎么解决", "robot"),
    ("建图不完整怎么办", "robot"),
    ("地图乱了能重新建图吗", "robot"),
    ("机器人可以恢复出厂设置吗", "robot"),
    ("怎么设置每天定时清扫", "robot"),
    ("扫地机器人怎么设置禁区", "robot"),
    ("预算1000元以内推荐什么扫地机器人", "robot"),
    ("1500以内有没有性价比高的扫地机器人", "robot"),
    ("2000元左右的扫拖机器人推荐", "robot"),
    ("最近半年发布的扫地机器人有哪些", "robot"),
    ("2025年3月发布了哪些扫地机器人", "robot"),
    # 笔误 / 不规范表达
    ("扫地机不回充咋办", "robot"),
    ("吸力下降了怎莫办", "robot"),
    ("扫地机器人连不上wifii", "robot"),

    # --------------------------------------------------------
    # other：明确领域外问题（30 条）
    # --------------------------------------------------------
    ("有没有飞机卖", "other"),
    ("2000元以内的电动车推荐一下", "other"),
    ("手机哪个牌子比较好", "other"),
    ("空调不制冷了怎么办", "other"),
    ("电脑卡顿怎么解决", "other"),
    ("汽车多久保养一次", "other"),
    ("洗衣机漏水怎么办", "other"),
    ("冰箱有异味怎么处理", "other"),
    ("电视怎么连接无线网络", "other"),
    ("耳机推荐一下", "other"),
    ("笔记本电脑开不了机怎么办", "other"),
    ("热水器打不着火是什么原因", "other"),
    ("路由器总是断网怎么办", "other"),
    ("手机电池不耐用了怎么办", "other"),
    ("电饭煲煮饭夹生怎么办", "other"),
    ("打印机无法打印怎么处理", "other"),
    ("洗碗机洗不干净怎么办", "other"),
    ("空气净化器滤芯多久换一次", "other"),
    ("相机镜头怎么清洁", "other"),
    ("投影仪画面模糊怎么办", "other"),
    ("新能源汽车冬天续航下降正常吗", "other"),
    ("蓝牙耳机连不上手机", "other"),
    ("智能手表怎么绑定手机", "other"),
    ("咖啡机不出水怎么办", "other"),
    ("燃气灶火苗很小怎么办", "other"),
    ("帮我查一下北京明天的天气", "other"),
    ("附近有什么好吃的餐厅", "other"),
    ("怎么学习Python编程", "other"),
    ("帮我写一份简历", "other"),
    # 笔误 / 不规范表达
    ("空条不制冷咋回事", "other"),
    ("手几充不进电怎么办", "other"),
    ("洗衣机lou水了", "other"),

    # --------------------------------------------------------
    # casual：闲聊、问候、能力咨询（30 条）
    # --------------------------------------------------------
    ("你好", "casual"),
    ("嗨", "casual"),
    ("哈喽", "casual"),
    ("在吗", "casual"),
    ("你是谁", "casual"),
    ("你能做什么", "casual"),
    ("你可以帮我解决什么问题", "casual"),
    ("谢谢", "casual"),
    ("感谢", "casual"),
    ("辛苦了", "casual"),
    ("再见", "casual"),
    ("拜拜", "casual"),
    ("早上好", "casual"),
    ("中午好", "casual"),
    ("晚上好", "casual"),
    ("今天过得怎么样", "casual"),
    ("你今天忙吗", "casual"),
    ("你是真人还是机器人", "casual"),
    ("你会说英语吗", "casual"),
    ("你能聊天吗", "casual"),
    ("讲个笑话", "casual"),
    ("给我说一句鼓励的话", "casual"),
    ("你叫什么名字", "casual"),
    ("你是哪个公司的", "casual"),
    ("我只是来看看", "casual"),
    ("没事了", "casual"),
    ("好的", "casual"),
    # 笔误 / 不规范表达
    ("你hao", "casual"),
    ("谢xie", "casual"),
    ("在不在呀", "casual"),

    # --------------------------------------------------------
    # unknown：上下文不足、指代不明、意图不清（30 条）
    # --------------------------------------------------------
    ("这个东西怎么样", "unknown"),
    ("好用吗", "unknown"),
    ("推荐一下", "unknown"),
    ("多少钱", "unknown"),
    ("靠谱吗", "unknown"),
    ("值得买吗", "unknown"),
    ("有没有好点的", "unknown"),
    ("这个怎么用", "unknown"),
    ("那个是什么", "unknown"),
    ("为什么不行", "unknown"),
    ("怎么回事", "unknown"),
    ("能修吗", "unknown"),
    ("可以退吗", "unknown"),
    ("有货吗", "unknown"),
    ("什么时候到", "unknown"),
    ("在哪里买", "unknown"),
    ("有没有便宜一点的", "unknown"),
    ("哪个更好", "unknown"),
    ("这个和那个有什么区别", "unknown"),
    ("能不能用", "unknown"),
    ("需要换吗", "unknown"),
    ("怎么设置", "unknown"),
    ("为什么这么贵", "unknown"),
    ("有什么推荐的吗", "unknown"),
    ("帮我看看这个", "unknown"),
    ("我该选哪个", "unknown"),
    ("这个型号可以吗", "unknown"),
    # 笔误 / 不规范表达
    ("这玩意咋样", "unknown"),
    ("贵不贵呀", "unknown"),
    ("这东西hao用吗", "unknown"),
]


# ============================================================
# 端到端测试集
#
# 端到端测试不建议对 LLM 的完整自然语言回答做严格字符串匹配，
# 因为模型输出可能因措辞不同而变化。
#
# 这里主要验证：
# 1. 意图路由是否符合预期；
# 2. ask_stream 是否正常返回内容；
# 3. 调用过程中是否发生异常；
# 4. 响应耗时是否可接受（仅打印，不强制失败）。
# ============================================================
E2E_SAMPLE = [
    ("边刷多久换一次", "robot"),
    ("扫地机不回充咋办", "robot"),
    ("预算1000元以内推荐什么扫地机器人", "robot"),
    ("最近半年发布的扫地机器人有哪些", "robot"),
    ("扫地机器人APP连不上怎么办", "robot"),
    ("空调不制冷了怎么办", "other"),
    ("手机哪个牌子比较好", "other"),
    ("你好", "casual"),
    ("你能做什么", "casual"),
    ("谢xie", "casual"),
    ("这个东西怎么样", "unknown"),
    ("多少钱", "unknown"),
    ("这玩意咋样", "unknown"),
]


def test_intent():
    """运行全部意图分类测试。"""
    from tools.intent_router import route_intent

    print("=" * 80)
    print(f"意图分类测试（共 {len(TEST_CASES)} 条）")
    print("=" * 80)

    correct = 0
    errors = []
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for query, expected in TEST_CASES:
        try:
            actual = route_intent(query)
            ok = actual == expected
        except Exception as exc:
            actual = f"EXCEPTION: {type(exc).__name__}"
            ok = False

        category_stats[expected]["total"] += 1
        if ok:
            correct += 1
            category_stats[expected]["correct"] += 1
        else:
            errors.append((query, expected, actual))

        mark = "OK  " if ok else "FAIL"
        print(f"{mark} 实际={actual:<12} 预期={expected:<8} 问题：{query}")

    total = len(TEST_CASES)
    accuracy = correct / total * 100 if total else 0

    print("\n" + "=" * 80)
    print(f"总准确率：{correct}/{total} = {accuracy:.1f}%")
    print("-" * 80)
    print("各类别统计：")

    for category in ("robot", "other", "casual", "unknown"):
        stat = category_stats[category]
        cat_total = stat["total"]
        cat_correct = stat["correct"]
        cat_accuracy = cat_correct / cat_total * 100 if cat_total else 0
        print(
            f"  {category:<8} "
            f"{cat_correct:>2}/{cat_total:<2} "
            f"准确率：{cat_accuracy:>5.1f}%"
        )

    if errors:
        print("\n误判清单：")
        print("-" * 80)
        for query, expected, actual in errors:
            print(f"  问题：{query}")
            print(f"  预期：{expected}，实际：{actual}")
            print()
    else:
        print("\n全部意图分类测试通过。")

    return correct, total, errors


def test_e2e():
    """运行端到端抽样测试。"""
    from tools.intent_router import route_intent
    from tools.agent import ask_stream

    print("\n" + "=" * 80)
    print(f"端到端抽样测试（共 {len(E2E_SAMPLE)} 条）")
    print("=" * 80)

    passed = 0
    failed_cases = []

    for query, expected_intent in E2E_SAMPLE:
        print(f"\n【用户问题】{query}")
        print(f"【预期意图】{expected_intent}")

        start_time = time.time()

        try:
            # 先验证路由结果
            actual_intent = route_intent(query)
            intent_ok = actual_intent == expected_intent

            # 再验证 Agent 是否能正常流式返回内容
            answer = "".join(ask_stream(query)).strip()
            elapsed = time.time() - start_time

            answer_ok = bool(answer)
            ok = intent_ok and answer_ok

            print(f"【实际意图】{actual_intent}")
            print(f"【分类结果】{'OK' if intent_ok else 'FAIL'}")
            print(f"【回答状态】{'OK' if answer_ok else 'FAIL（空回答）'}")
            print(f"【耗时】{elapsed:.2f} 秒")
            print("【回答预览】")
            print("  " + (answer[:300] if answer else "(无内容)").replace("\n", "\n  "))

            if ok:
                passed += 1
            else:
                failed_cases.append({
                    "query": query,
                    "expected_intent": expected_intent,
                    "actual_intent": actual_intent,
                    "answer": answer,
                })

        except Exception as exc:
            elapsed = time.time() - start_time
            print(f"【结果】FAIL")
            print(f"【耗时】{elapsed:.2f} 秒")
            print(f"【异常】{type(exc).__name__}: {exc}")

            failed_cases.append({
                "query": query,
                "expected_intent": expected_intent,
                "actual_intent": "EXCEPTION",
                "answer": str(exc),
            })

    total = len(E2E_SAMPLE)
    print("\n" + "=" * 80)
    print(f"端到端测试结果：{passed}/{total} 通过")

    if failed_cases:
        print("\n端到端失败清单：")
        for item in failed_cases:
            print("-" * 80)
            print(f"问题：{item['query']}")
            print(f"预期意图：{item['expected_intent']}")
            print(f"实际意图：{item['actual_intent']}")
            print(f"回答/异常：{item['answer'][:200]}")
    else:
        print("全部端到端抽样测试通过。")

    return passed, total, failed_cases


if __name__ == "__main__":
    _, _, intent_errors = test_intent()

    if "--e2e" in sys.argv:
        test_e2e()
