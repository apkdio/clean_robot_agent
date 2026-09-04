"""意图分类测试：route_intent 四分类（robot / other / casual / unknown）。

覆盖：
  - 常规表达（每类约 30 条，含口语/笔误）
  - 极端泛化（错别字 / 全拼音 / 中英混杂 / 描述性指代）

依赖：torch（本地分类头）+ Ollama bge-m3（embedding），加 ``--e2e`` 运行。

运行：
  .venv\\Scripts\\python.exe test_intent.py --e2e
"""
import sys
from collections import defaultdict
from _runner import *


# ──────────────────────────────────────────────────────────────
# 测试数据：(用户问题, 预期意图)
# ──────────────────────────────────────────────────────────────

CASES = [
    # ── robot：扫地机器人领域 ──
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
    ("扫地机不回充咋办", "robot"),
    ("吸力下降了怎莫办", "robot"),
    ("扫地机器人连不上wifii", "robot"),
    # ── other：明确领域外 ──
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
    ("空条不制冷咋回事", "other"),
    ("手几充不进电怎么办", "other"),
    ("洗衣机lou水了", "other"),
    # ── casual：闲聊 / 问候 / 能力咨询 ──
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
    ("你hao", "casual"),
    ("谢xie", "casual"),
    ("在不在呀", "casual"),
    # ── unknown：上下文不足 / 指代不明 ──
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
    ("这玩意咋样", "unknown"),
    ("贵不贵呀", "unknown"),
    ("这东西hao用吗", "unknown"),
]

EXTREME = [
    # ── robot 极端变体 ──
    ("扫地鸡器人不会回充了", "robot"),
    ("机气人边刷不转了咋办", "robot"),
    ("拖地厚地面有水银怎么办", "robot"),
    ("西力下降很厉害怎么修", "robot"),
    ("尘合多久到一次", "robot"),
    ("saodijiqiren zenme huichong", "robot"),
    ("边刷多久换 yici", "robot"),
    ("jiqiren xili xiajiang le", "robot"),
    ("有没有1000以内的 saodiji", "robot"),
    ("扫地robot的HEPA filter多久换一次", "robot"),
    ("robot的battery续航下降怎么办", "robot"),
    ("cleaning robot 有没有1000以内的", "robot"),
    ("我家sweeper的app连不上了", "robot"),
    ("我家那个圆盘自动扫灰的机器不动了", "robot"),
    ("那个会自己跑自己拖地的小玩意儿要清理吗", "robot"),
    # ── other 极端变体 ──
    ("有没有2000以内的 electric bike", "other"),
    ("kongtiao 不制冷了", "other"),
    ("手几充不进电了", "other"),
    ("洗一机漏水了", "other"),
    ("帮我写 ge jianli", "other"),
    # ── casual 极端变体 ──
    ("ni hao", "casual"),
    ("你hao", "casual"),
    ("内啥，随便问问", "casual"),
    ("3q", "casual"),
    # ── unknown 极端变体 ──
    ("这个行不行", "unknown"),
    ("那到底咋弄", "unknown"),
    ("是不是要换新的了", "unknown"),
    ("就这", "unknown"),
    ("能行不", "unknown"),
    ("这玩意儿突然不干活了", "unknown"),
]


def _run_set(label, cases):
    """跑一组用例，返回 (正确数, 总数)；误判只打印、不计入断言（分类器为统计模型）。"""
    from tools.intent_router import route_intent
    stats = defaultdict(lambda: {"total": 0, "ok": 0})
    for q, exp in cases:
        try:
            act = route_intent(q)
        except Exception as exc:  # noqa: BLE001
            act = f"EXC:{type(exc).__name__}"
        ok = act == exp
        stats[exp]["total"] += 1
        if ok:
            stats[exp]["ok"] += 1
        if not ok:
            print(f"     ✗ {q!r}: 期望={exp} 实际={act}")
    for cat in ("robot", "other", "casual", "unknown"):
        s = stats[cat]
        if s["total"]:
            print(f"     [{cat}] {s['ok']}/{s['total']} = {s['ok'] / s['total'] * 100:.1f}%")
    total_ok = sum(s["ok"] for s in stats.values())
    total_n = sum(s["total"] for s in stats.values())
    if total_n:
        print(f"     [{label}小计] {total_ok}/{total_n} = {total_ok / total_n * 100:.1f}%")
    return total_ok, total_n


@e2e("意图分类需要 torch + Ollama bge-m3")
def test_intent_normal():
    ok, n = _run_set("常规用例", CASES)
    _assert(n > 0 and ok / n >= 0.90, f"常规准确率 {ok}/{n} >= 90%")


@e2e("意图分类需要 torch + Ollama bge-m3")
def test_intent_extreme():
    ok, n = _run_set("极端泛化用例", EXTREME)
    _assert(n > 0 and ok / n >= 0.75, f"极端泛化准确率 {ok}/{n} >= 75%")


TESTS = [test_intent_normal, test_intent_extreme]


def run():
    reset()
    return run_tests("意图分类测试（robot/other/casual/unknown）", TESTS)


if __name__ == "__main__":
    passed, total, skipped = run()
    sys.exit(0 if passed == total else 1)
