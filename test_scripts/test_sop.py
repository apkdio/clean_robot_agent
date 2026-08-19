"""SOP 测试脚本：基础设施单元测试 + 选购 SOP 端到端流程。

运行:
  python test_sop.py              # 运行全部测试
  python test_sop.py --e2e        # 全部测试 + 端到端（需 Chroma + 知识库已入库）
"""

import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

# ──────────────────────────────────────────────────────────────────
# 测试数据
# ──────────────────────────────────────────────────────────────────

_MOCK_MODELS: List[Dict[str, Any]] = [
    {"name": "米家扫拖机器人 M20",    "price": 899,  "suction": "2800Pa", "navigation": "LDS 激光", "obstacle": "红外"},
    {"name": "追觅 D10s",            "price": 1099, "suction": "3200Pa", "navigation": "LDS 激光", "obstacle": "红外"},
    {"name": "美的 i5 Pro",          "price": 799,  "suction": "3000Pa", "navigation": "视觉导航", "obstacle": "红外"},
    {"name": "海尔 HSR-1",           "price": 599,  "suction": "2600Pa", "navigation": "LDS 激光", "obstacle": "无"},
    {"name": "云米 VXVC12",          "price": 999,  "suction": "3100Pa", "navigation": "LDS 激光", "obstacle": "红外"},
    {"name": "石头 T7 Lite",         "price": 1299, "suction": "2500Pa", "navigation": "LDS 激光", "obstacle": "红外"},
    {"name": "科沃斯 T30 Mini",      "price": 2299, "suction": "4000Pa", "navigation": "LDS 激光", "obstacle": "结构光"},
    {"name": "石头 P10",             "price": 2699, "suction": "5500Pa", "navigation": "LDS 激光", "obstacle": "结构光"},
    {"name": "追觅 S20",            "price": 2499, "suction": "6000Pa", "navigation": "LDS 激光", "obstacle": "结构光"},
    {"name": "云鲸 J4",             "price": 2899, "suction": "4500Pa", "navigation": "LDS 激光", "obstacle": "结构光"},
    {"name": "米家全能扫拖 M30",     "price": 1999, "suction": "5000Pa", "navigation": "LDS 激光", "obstacle": "结构光"},
    {"name": "360 S9",             "price": 1799, "suction": "3800Pa", "navigation": "LDS 激光", "obstacle": "结构光"},
    {"name": "米家扫拖 M10 Lite",   "price": 699,  "suction": "2700Pa", "navigation": "LDS 激光", "obstacle": "红外"},
    {"name": "追觅 D9",             "price": 849,  "suction": "2900Pa", "navigation": "LDS 激光", "obstacle": "红外"},
    {"name": "石头 Q5",             "price": 1499, "suction": "5000Pa", "navigation": "LDS 激光", "obstacle": "红外"},
    {"name": "科沃斯 N8 纯扫版",    "price": 1199, "suction": "3600Pa", "navigation": "LDS 激光", "obstacle": "红外"},
    {"name": "科沃斯 X5 Pro",       "price": 3999, "suction": "8500Pa", "navigation": "dToF",     "obstacle": "3D 结构光"},
    {"name": "石头 G20S",           "price": 4299, "suction": "11000Pa","navigation": "dToF",     "obstacle": "3D 结构光"},
    {"name": "追觅 X40 Pro",        "price": 4599, "suction": "12000Pa","navigation": "dToF",     "obstacle": "3D 结构光"},
    {"name": "云鲸 J5",             "price": 3899, "suction": "7000Pa", "navigation": "dToF",     "obstacle": "3D 结构光"},
    {"name": "石头 P20 Pro",        "price": 3499, "suction": "10000Pa","navigation": "dToF",     "obstacle": "3D 结构光"},
    {"name": "科沃斯 X8 Pro Omni",  "price": 5999, "suction": "13000Pa","navigation": "dToF",     "obstacle": "3D 结构光"},
    {"name": "石头 V20",            "price": 5299, "suction": "12500Pa","navigation": "dToF",     "obstacle": "3D 结构光"},
    {"name": "云鲸 J5 Max",         "price": 4999, "suction": "8000Pa", "navigation": "dToF",     "obstacle": "3D 结构光"},
    {"name": "追觅 X30 宠物版",     "price": 4199, "suction": "11500Pa","navigation": "dToF",     "obstacle": "3D 结构光"},
    {"name": "石头 Qrevo MaxV 宠物版", "price": 4399, "suction": "10500Pa","navigation": "dToF", "obstacle": "3D 结构光"},
    {"name": "科沃斯 X5 Pro Max",   "price": 4799, "suction": "10000Pa","navigation": "dToF",     "obstacle": "3D 结构光"},
    {"name": "石头 S8 MaxV Ultra",  "price": 5199, "suction": "12000Pa","navigation": "dToF",     "obstacle": "3D 结构光"},
    {"name": "iRobot Roomba j9+",   "price": 5699, "suction": "7500Pa", "navigation": "视觉",     "obstacle": "3D 结构光"},
    {"name": "戴森 360 Vis Nav",    "price": 6999, "suction": "11000Pa","navigation": "视觉 SLAM","obstacle": "3D 结构光"},
    {"name": "浦桑尼克 M9",         "price": 1599, "suction": "4200Pa", "navigation": "视觉导航", "obstacle": "结构光"},
    {"name": "科沃斯 N20 Pro",      "price": 2199, "suction": "4800Pa", "navigation": "LDS 激光", "obstacle": "结构光"},
    {"name": "追觅 S40 Ultra",      "price": 5499, "suction": "13500Pa","navigation": "dToF",     "obstacle": "3D 结构光"},
]


# ──────────────────────────────────────────────────────────────────
# 提取器（与 purchase.py 逻辑一致，自包含免依赖）
# ──────────────────────────────────────────────────────────────────

def _mock_extract_budget(text: str) -> Tuple[int | None, int | None] | None:
    """提取预算：(min_price, max_price)。区间优先，单一上限次之。

    与 purchase._extract_budget 逻辑对齐，但不必依赖项目模块。
    """
    # 区间："1000-2000" / "1000到2000" / "1000~2000"
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-–~到]\s*(\d+(?:\.\d+)?)", text)
    if m:
        return (int(float(m.group(1))), int(float(m.group(2))))

    # 单一上限："1000以内" / "1000以下" / "1000左右" / "1000块"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:元|块|块钱)?\s*(?:以内|以下|之内|左右)", text)
    if m:
        return (None, int(float(m.group(1))))

    # 中文数字："一千以内" / "两千左右"
    m = re.search(r"([零一二两三四五六七八九十百千万]+)\s*(?:以内|以下|之内|左右)", text)
    if m:
        c = _mock_cn_to_int(m.group(1))
        if c is not None:
            return (None, c)

    return None


def _mock_cn_to_int(s: str) -> int | None:
    """简单中文数字转整数。"""
    num_map = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
               "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    unit_map = {"十": 10, "百": 100, "千": 1000, "万": 10000}

    total = 0
    section = 0
    number = 0
    for ch in s:
        if ch in num_map:
            number = num_map[ch]
        elif ch in unit_map:
            unit = unit_map[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
        else:
            return None
    return total + section + number


def _mock_extract_has_pet(text: str) -> bool | None:
    """提取是否有宠物。"""
    if re.search(r"没|无|不养|没有|没养", text):
        return False
    if re.search(r"有|养|猫|狗|宠物", text):
        return True
    return None


def _mock_format_model_line(info: Dict[str, Any]) -> str:
    """格式化一个型号到一行文本。"""
    specs = []
    if info.get("suction"):
        specs.append(f"吸力 {info['suction']}")
    if info.get("navigation"):
        nav = info["navigation"]
        if not nav.endswith("导航"):
            nav += "导航"
        specs.append(nav)
    if info.get("obstacle"):
        specs.append(f"{info['obstacle']}避障")
    spec_str = "、".join(specs)
    line = f"- **{info['name']}**"
    if spec_str:
        line += f"：{spec_str}"
    if info.get("price") is not None:
        line += f"，参考价 {info['price']} 元"
    return line


def _mock_search(slots: Dict[str, Any]) -> Dict[str, Any]:
    """模拟 search_by_filter：从 _MOCK_MODELS 按预算过滤。"""
    budget = slots.get("budget")
    if not budget:
        return {"count": 0, "list": ""}

    min_price, max_price = budget
    matched = []
    for m in _MOCK_MODELS:
        p = m["price"]
        if max_price is not None and p > max_price:
            continue
        if min_price is not None and p < min_price:
            continue
        matched.append(m)

    lines = [_mock_format_model_line(m) for m in matched]
    return {"count": len(matched), "list": "\n".join(lines)}


# ──────────────────────────────────────────────────────────────────
# 测试 SOP 定义
# ──────────────────────────────────────────────────────────────────

_TEST_SOP = {
    "id": "test_sop",
    "trigger": ["测试版", "试验场景", "试一下", "模拟选购"],
    "steps": [
        {
            "id": "ask_budget",
            "type": "ask",
            "slot": "budget",
            "ask": "好呀，先了解一下您的预算大概是多少呢？",
            "retry": "预算我没太听清，能说个具体数字吗？比如「1000以内」「2000元左右」。",
            "extract": lambda text, slots: _mock_extract_budget(text),
        },
        {
            "id": "ask_pet",
            "type": "ask",
            "slot": "has_pet",
            "ask": "家里有养宠物吗？（有猫狗的话，我会优先推荐防毛发缠绕的机型）",
            "retry": "这个我没听懂哦～家里有养猫狗等宠物吗？回复「有」或「没有」就可以啦。",
            "extract": lambda text, slots: _mock_extract_has_pet(text),
        },
        {
            "id": "do_search",
            "type": "action",
            "action": _mock_search,
        },
        {
            "id": "reply",
            "type": "reply",
            "template": (
                "根据您的需求，共找到 {count} 款符合条件的机器人：\n\n"
                "{list}\n\n"
                "需要我帮您对比其中某两款吗？"
            ),
            "fallback": "抱歉，出了一点小问题，请重新提问～",
        },
    ],
}

# 测试 fallback 场景的 SOP（模板故意缺字段，验证 fallback 触发）
_TEST_SOP_BAD_TEMPLATE = {
    "id": "test_sop_bad_template",
    "trigger": ["坏模板"],
    "steps": [
        {
            "id": "ask_name",
            "type": "ask",
            "slot": "name",
            "ask": "你叫什么？",
            "retry": "请说名字。",
            "extract": lambda text, slots: text.strip() or None,
        },
        {
            "id": "reply",
            "type": "reply",
            "template": "你好，{name}，{missing_field}！",     # missing_field 不存在
            "fallback": "抱歉，出了一点小问题，请重新提问～",
        },
    ],
}


# ──────────────────────────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────────────────────────

_total = 0
_passed = 0
_failed: List[Tuple[str, str]] = []


def _assert(cond: bool, msg: str):
    global _total, _passed, _failed
    _total += 1
    if cond:
        _passed += 1
        print(f"  OK  {msg}")
    else:
        _failed.append((msg, "断言失败"))
        print(f"  FAIL {msg}")


def _assert_eq(a, b, msg: str):
    global _total, _passed, _failed
    _total += 1
    if a == b:
        _passed += 1
        print(f"  OK  {msg}")
    else:
        _failed.append((msg, f"期望={b!r}, 实际={a!r}"))
        print(f"  FAIL {msg}  期望={b!r}  实际={a!r}")


def _cleanup():
    """清除 SOP 会话状态（导入 base 模块后直接操作 _session）。"""
    import sops.base
    sops.base._session = None


# ================================================================
# 1. 注册与匹配
# ================================================================

def test_register():
    """测试注册 SOP 到全局 SOPS 表"""
    import sops.base
    sops.base.register(_TEST_SOP)
    _assert("test_sop" in sops.base.SOPS, "SOP 已注册到 SOPS 表")
    _assert(sops.base.SOPS["test_sop"] is _TEST_SOP, "注册的 SOP 与原始对象一致")


def test_register_duplicate():
    """测试重复注册覆盖"""
    import sops.base
    sops.base.register(_TEST_SOP)
    sops.base.register(_TEST_SOP_BAD_TEMPLATE)
    _assert("test_sop" in sops.base.SOPS, "第一个 SOP 仍存在")
    _assert("test_sop_bad_template" in sops.base.SOPS, "第二个 SOP 已注册")


def test_match_sop_hit():
    """测试 trigger 关键词命中"""
    import sops.base
    _assert_eq(sops.base.match_sop("测试版"), "test_sop", "命中「测试版」")
    _assert_eq(sops.base.match_sop("试一下"), "test_sop", "命中「试一下」")
    _assert_eq(sops.base.match_sop("模拟选购"), "test_sop", "命中「模拟选购」")
    _assert(sops.base.match_sop("帮我推荐一款") is not None, "真实 trigger 也会命中 purchase SOP")


def test_match_sop_miss():
    """测试 trigger 未命中"""
    import sops.base
    _assert(sops.base.match_sop("你好呀") is None, "闲聊不触发 SOP")
    _assert(sops.base.match_sop("边刷多久换一次") is None, "故障维护不触发 SOP")
    _assert(sops.base.match_sop("今天天气怎么样") is None, "领域外不触发 SOP")


# ================================================================
# 2. 会话生命周期
# ================================================================

def test_start_sop():
    """测试启动 SOP：第一轮返回 ask 文本，不尝试提取"""
    _cleanup()
    import sops.base
    reply, done = sops.base.start_sop("test_sop", "帮忙推荐一下")
    _assert_eq(reply, "好呀，先了解一下您的预算大概是多少呢？", "返回 ask 文本")
    _assert(not done, "done=False，SOP 未结束")
    _assert(sops.base.has_active_sop(), "有活跃会话")
    _cleanup()


def test_continue_sop_success():
    """测试继续 SOP：提取成功并推进到下一步"""
    _cleanup()
    import sops.base
    sops.base.start_sop("test_sop", "随便")
    reply, done = sops.base.continue_sop("1000以内")
    _assert_eq(reply, "家里有养宠物吗？（有猫狗的话，我会优先推荐防毛发缠绕的机型）", "推进到 ask_pet")
    _assert(not done, "done=False")
    _cleanup()


def test_continue_sop_extract_fail():
    """测试继续 SOP：提取失败，触发 retry"""
    _cleanup()
    import sops.base
    sops.base.start_sop("test_sop", "随便")
    reply, done = sops.base.continue_sop("不知道")  # 无法提取预算
    _assert_eq(reply, "预算我没太听清，能说个具体数字吗？比如「1000以内」「2000元左右」。", "触发 retry 话术")
    _assert(not done, "done=False，会话未结束")
    _cleanup()


def test_continue_sop_no_session():
    """测试无活跃会话时 continue_sop 返回 None"""
    _cleanup()
    import sops.base
    _assert(sops.base.continue_sop("随便") is None, "无会话时返回 None")


def test_end_sop():
    """测试主动结束会话"""
    _cleanup()
    import sops.base
    sops.base.start_sop("test_sop", "随便")
    _assert(sops.base.has_active_sop(), "启动后有会话")
    sops.base.end_sop()
    _assert(not sops.base.has_active_sop(), "结束会话后无活跃会话")
    _assert(sops.base.continue_sop("随便") is None, "结束后 continue 返回 None")


def test_has_active_sop():
    """测试 has_active_sop 状态"""
    _cleanup()
    import sops.base
    _assert(not sops.base.has_active_sop(), "初始无会话")
    sops.base.start_sop("test_sop", "随便")
    _assert(sops.base.has_active_sop(), "启动后有会话")
    sops.base.end_sop()
    _assert(not sops.base.has_active_sop(), "结束后无会话")


# ================================================================
# 3. 完整流程
# ================================================================

def test_full_flow():
    """测试完整 SOP 流程：ask_budget → ask_pet → do_search → reply"""
    _cleanup()
    import sops.base
    sops.base.start_sop("test_sop", "帮我推荐")

    # 第 1 轮：给预算
    reply, done = sops.base.continue_sop("1000以内")
    _assert(not done, "预算收集后未结束")
    _assert("宠物" in reply, "下一步问宠物")
    _assert(sops.base.has_active_sop(), "会话仍在活跃")

    # 第 2 轮：给宠物信息
    reply, done = sops.base.continue_sop("没有宠物")
    _assert(done, "SOP 结束")
    _assert("共找到" in reply, "回复包含统计")
    _assert("米家扫拖机器人 M20" in reply, "回复包含 899 元以内型号")
    _assert("美的 i5 Pro" in reply, "回复包含 799 元型号")
    _assert("米家扫拖 M10 Lite" in reply, "回复包含 699 元型号")
    _assert(not sops.base.has_active_sop(), "会话已结束")
    _cleanup()


def test_full_flow_with_range():
    """测试区间预算的完整流程"""
    _cleanup()
    import sops.base
    sops.base.start_sop("test_sop", "帮我推荐")

    reply, done = sops.base.continue_sop("2000-3000")
    _assert(not done, "预算收集后未结束")

    reply, done = sops.base.continue_sop("有宠物")
    _assert(done, "SOP 结束")
    _assert("共找到" in reply, "回复包含统计")
    _assert("科沃斯 T30 Mini" in reply, "回复包含区间内型号")
    _assert("石头 P10" in reply, "回复包含区间内型号")
    _assert(not sops.base.has_active_sop(), "会话已结束")
    _cleanup()


def test_full_flow_no_pet():
    """测试无宠物场景的完整流程"""
    _cleanup()
    import sops.base
    sops.base.start_sop("test_sop", "随便")

    sops.base.continue_sop("1000以内")
    reply, done = sops.base.continue_sop("没有")
    _assert(done, "SOP 结束")
    _assert("共找到" in reply, "即使无宠物也能正常推荐")
    _cleanup()


# ================================================================
# 4. 边界场景
# ================================================================

def test_extract_budget():
    """测试预算提取器各分支"""
    # 区间
    _assert_eq(_mock_extract_budget("1000-2000"), (1000, 2000), "区间：1000-2000")
    _assert_eq(_mock_extract_budget("1000到2000"), (1000, 2000), "区间：1000到2000")
    _assert_eq(_mock_extract_budget("1000~2000"), (1000, 2000), "区间：1000~2000")
    # 单一上限
    _assert_eq(_mock_extract_budget("1000以内"), (None, 1000), "上限：1000以内")
    _assert_eq(_mock_extract_budget("1000左右"), (None, 1000), "上限：1000左右")
    _assert_eq(_mock_extract_budget("1000元以内"), (None, 1000), "上限：1000元以内")
    _assert_eq(_mock_extract_budget("1000块以内"), (None, 1000), "上限：1000块以内")
    _assert_eq(_mock_extract_budget("1000元以下"), (None, 1000), "上限：1000元以下")
    # 中文数字
    _assert_eq(_mock_extract_budget("一千以内"), (None, 1000), "中文：一千以内")
    _assert_eq(_mock_extract_budget("两千左右"), (None, 2000), "中文：两千左右")
    _assert_eq(_mock_extract_budget("三千以内"), (None, 3000), "中文：三千以内")
    # 无效
    _assert(_mock_extract_budget("不知道") is None, "无效输入返回 None")
    _assert(_mock_extract_budget("随便") is None, "无关输入返回 None")
    _assert(_mock_extract_budget("") is None, "空字符串返回 None")


def test_extract_has_pet():
    """测试宠物提取器各分支"""
    _assert(_mock_extract_has_pet("有猫") is True, "有猫 → True")
    _assert(_mock_extract_has_pet("有狗") is True, "有狗 → True")
    _assert(_mock_extract_has_pet("养了一只猫") is True, "养猫 → True")
    _assert(_mock_extract_has_pet("有宠物") is True, "有宠物 → True")
    _assert(_mock_extract_has_pet("没有") is False, "没有 → False")
    _assert(_mock_extract_has_pet("没养") is False, "没养 → False")
    _assert(_mock_extract_has_pet("无") is False, "无 → False")
    _assert(_mock_extract_has_pet("不养宠物") is False, "不养 → False")
    _assert(_mock_extract_has_pet("不知道") is None, "不知道 → None")
    _assert(_mock_extract_has_pet("") is None, "空字符串 → None")


def test_retry_cycle():
    """测试连续提取失败后最终成功"""
    _cleanup()
    import sops.base
    sops.base.start_sop("test_sop", "随便")

    # 第 1 次失败
    reply, done = sops.base.continue_sop("不知道")
    _assert(not done, "第 1 次失败后未结束")
    _assert("预算我没太听清" in reply, "第 1 次失败触发 retry")

    # 第 2 次失败
    reply, done = sops.base.continue_sop("随便")
    _assert(not done, "第 2 次失败后未结束")
    _assert("预算我没太听清" in reply, "第 2 次失败仍触发 retry")

    # 第 3 次成功
    reply, done = sops.base.continue_sop("1500以内")
    _assert(not done, "预算成功收集后未结束")
    _assert("宠物" in reply, "推进到 ask_pet")

    # 宠物提取失败
    reply2, done2 = sops.base.continue_sop("随便说说")
    _assert(not done2, "宠物提取失败后未结束")
    _assert("宠物" in reply2, "宠物 retry 话术")

    # 宠物提取成功
    reply3, done3 = sops.base.continue_sop("有猫")
    _assert(done3, "全部集齐，SOP 结束")
    _assert("共找到" in reply3, "回复包含统计")
    _cleanup()


def test_no_matching_sop():
    """测试无匹配 SOP 时正常流程"""
    _cleanup()
    import sops.base
    _assert(sops.base.match_sop("今天天气怎么样") is None, "领域外不匹配任何 SOP")


def test_mock_search_empty():
    """测试检索无结果"""
    result = _mock_search({"budget": (None, 50)})
    _assert_eq(result["count"], 0, "预算 50 以内无型号")
    _assert_eq(result["list"], "", "列表为空")


def test_mock_search_all():
    """测试检索全部型号"""
    result = _mock_search({"budget": (None, 99999)})
    _assert_eq(result["count"], len(_MOCK_MODELS), "不限预算返回全部型号")
    _assert("米家扫拖机器人 M20" in result["list"], "包含第一个型号")
    _assert("科沃斯 N8 纯扫版" in result["list"], "包含最后一个型号")


def test_mock_search_range():
    """测试区间检索"""
    result = _mock_search({"budget": (1500, 3000)})
    _assert(result["count"] > 0, "1500-3000 区间有结果")
    # 验证边界：价格在区间内
    for m in _MOCK_MODELS:
        if 1500 <= m["price"] <= 3000:
            _assert(m["name"] in result["list"], f"区间内型号 {m['name']} 应出现")
        elif m["price"] < 1500 or m["price"] > 3000:
            _assert(m["name"] not in result["list"], f"区间外型号 {m['name']} 不应出现")


# ================================================================
# 5. 异常防御
# ================================================================

def test_template_fallback():
    """测试模板缺失字段时触发 fallback"""
    _cleanup()
    import sops.base
    sops.base.register(_TEST_SOP_BAD_TEMPLATE)
    sops.base.start_sop("test_sop_bad_template", "测试")
    reply, done = sops.base.continue_sop("张三")
    _assert(done, "SOP 结束")
    _assert_eq(reply, "抱歉，出了一点小问题，请重新提问～", "触发 fallback 话术")


def test_unknown_sop_id():
    """测试从已注册的 SOP 发起错误的 start_sop（防御性）"""
    _cleanup()
    import sops.base
    try:
        # 注册一个 SOP
        sops.base.register(_TEST_SOP)
        # 尝试启动不存在的 SOP（但 base._start 不会做校验）
        sops.base._start("nonexistent_sop")
        reply, done = sops.base._run(None)
        # 此时 _run 会尝试访问 SOPS["nonexistent_sop"]，期待 KeyError
        _assert(False, "应该抛出 KeyError")
    except KeyError:
        _assert(True, "不存在的 sop_id 抛出 KeyError")
    except Exception as e:
        _assert(False, f"意外异常: {e}")
    _cleanup()


# ================================================================
# 6. 端到端测试（需 Chroma + 知识库已入库，可选）
# ================================================================

def test_e2e_purchase_sop():
    """端到端：实际的 purchase SOP 三轮对话。

    需要 Chroma 向量库已有数据，否则跳过。
    """
    import sops.base
    from sops import purchase  # 确保 purchase SOP 已注册

    # 检查 Chroma 是否有数据
    try:
        from tools.vector_store import list_collections_info
        info = list_collections_info()
        if info.get("chunk_count", 0) == 0:
            print("  SKIP 端到端：向量库为空（跳过）")
            return
    except Exception:
        print("  SKIP 端到端：Chroma 连接失败（跳过）")
        return

    _cleanup()

    # 第 1 轮：触发
    sops.base.start_sop("purchase", "帮我推荐一款")
    _assert(sops.base.has_active_sop(), "purchase SOP 已启动")  # 注：故意写错，应为 has_active_sop

    # 第 2 轮：给预算
    reply, done = sops.base.continue_sop("1000以内")
    _assert(not done, "预算收集后未结束")

    # 第 3 轮：给宠物 + 生成结果
    reply, done = sops.base.continue_sop("没有宠物")
    _assert(done, "SOP 结束")
    _assert("共找到" in reply, "回复包含统计")
    _assert(not sops.base.has_active_sop(), "会话已结束")
    _cleanup()


# ================================================================
# 主入口
# ================================================================

def run_all():
    global _total, _passed, _failed
    _total = 0
    _passed = 0
    _failed = []

    print("=" * 70)
    print("SOP 基础设施测试")
    print("=" * 70)

    # 1. 注册与匹配
    print("\n── 注册与匹配 ──")
    test_register()
    test_register_duplicate()
    test_match_sop_hit()
    test_match_sop_miss()

    # 2. 会话生命周期
    print("\n── 会话生命周期 ──")
    test_start_sop()
    test_continue_sop_success()
    test_continue_sop_extract_fail()
    test_continue_sop_no_session()
    test_end_sop()
    test_has_active_sop()

    # 3. 完整流程
    print("\n── 完整流程 ──")
    test_full_flow()
    test_full_flow_with_range()
    test_full_flow_no_pet()

    # 4. 边界场景
    print("\n── 边界场景 ──")
    test_extract_budget()
    test_extract_has_pet()
    test_retry_cycle()
    test_no_matching_sop()
    test_mock_search_empty()
    test_mock_search_all()
    test_mock_search_range()

    # 5. 异常防御
    print("\n── 异常防御 ──")
    test_template_fallback()
    test_unknown_sop_id()

    # 统计
    print()
    print("=" * 70)
    print(f"结果: {_passed}/{_total} 通过", end="")
    if _failed:
        print(f", {len(_failed)} 失败:")
        for name, reason in _failed:
            print(f"  FAIL {name}: {reason}")
    else:
        print()
    print("=" * 70)
    return _passed, _total


def run_e2e():
    print("\n" + "=" * 70)
    print("端到端测试（可选，需 Chroma）")
    print("=" * 70)
    test_e2e_purchase_sop()
    print("端到端测试完成。")
    print()


if __name__ == "__main__":
    total_tests = 0
    total_passed = 0

    _cleanup()

    # 先注册测试 SOP
    import sops.base
    sops.base.register(_TEST_SOP)

    passed, total = run_all()
    total_passed += passed
    total_tests += total

    if "--e2e" in sys.argv:
        # 注册 purchase SOP 用于端到端
        import sops.purchase  # noqa: F401
        run_e2e()

    _cleanup()

    # 退出码
    exit_code = 0 if total_passed == total_tests else 1
    sys.exit(exit_code)