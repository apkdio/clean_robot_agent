"""构建意图分类数据集（综合版）。

覆盖目标（基于测试发现）：
  1. 预算推荐类 robot 样本（"预算XX以内推荐"）—— 之前缺失
  2. 专业术语（HEPA/LDS/dToF/VSLAM）—— 之前覆盖不足
  3. 跨领域歧义消解（漏水/电池）—— 之前存在歧义
  4. 扩充的 other/casual/unknown 种子词表

输出：data/datasets/intent_dataset.jsonl
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from tools.entry_splitter import split_numbered_entries
from tools.file_tools import extract_file

_KNOWLEDGE_DIR = "data/knowledge"
_OUT_PATH = "data/datasets/intent_dataset.jsonl"


def _extract_title(entry_text: str) -> str | None:
    first_line = entry_text.split("\n")[0].strip()
    m = re.match(r"^\d+\.\s+\*\*(.+?)\*\*", first_line)
    if m:
        return m.group(1).strip()
    m = re.match(r"^\d+\.\s+(.+?)[；;。]", entry_text)
    if m:
        return m.group(1).strip()
    return None


def _to_question(title: str) -> str:
    title = title.strip()
    if title.endswith("？") or title.endswith("?"):
        return title
    if title.startswith("故障现象："):
        phenomenon = re.split(r"[；;，,]", title.replace("故障现象：", ""))[0]
        return f"{phenomenon}怎么办？"
    if "选购" in title or "购买" in title or "选择" in title:
        return "扫地机器人应该怎么选购？"
    if "维护" in title or "保养" in title or "清洁" in title:
        return "扫地机器人应该怎么维护保养？"
    if re.search(r"[A-Za-z0-9]", title) and "：" not in title and len(title) < 20:
        return f"{title}怎么样？"
    if "：" in title:
        key = title.split("：")[0]
        return f"{key}应该注意什么？"
    return f"{title}怎么办？"


def collect_robot_samples() -> list[str]:
    samples = []
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _KNOWLEDGE_DIR)
    for fname in sorted(os.listdir(root)):
        fp = os.path.join(root, fname)
        if not os.path.isfile(fp):
            continue
        docs = extract_file(fp)
        for doc in docs:
            for entry in split_numbered_entries(doc.page_content):
                title = _extract_title(entry.page_content)
                if title and len(title) >= 4:
                    q = _to_question(title)
                    if q:
                        samples.append(q)
    return samples


# ── 预算推荐类 robot 样本（针对测试误判补充）─────────────────────────────

_ROBOT_BUDGET = [
    "预算1000以内扫地机器人推荐", "预算1500以内有什么扫地机器人推荐",
    "预算2000以内有什么机器人", "1000元以下扫地机器人推荐",
    "1500以内扫地机器人推荐", "2000以内机器人推荐",
    "一千以内有什么扫地机器人", "三千以内扫拖一体机器人推荐",
    "5000元左右扫地机器人推荐", "预算3000以内扫地机器人",
    "2000以下扫地机器人推荐", "1000块以内的机器人",
    "3000以内的扫地机器人", "万元以下的扫地机器人推荐",
    "预算5000以内有什么扫地机器人", "1500元以内的扫拖一体机器人",
    "两千以内推荐什么机器人", "3000元以下机器人推荐",
    "5000以内的扫拖机器人", "10000以内扫地机器人推荐",
    "预算2000以内推荐", "预算1000以内推荐机器人",
    "2000元以内推荐", "1000以内有什么机器人推荐",
    "3000以内推荐扫地机器人", "1500以内买什么机器人",
    "2000块以内机器人", "5000以内推荐",
]

# ── 专业术语类 robot 样本（针对 HEPA/LDS 等缩写补充）────────────────────

_ROBOT_TERMS = [
    "HEPA滤网多久更换一次", "HEPA滤网可以清洗吗", "HEPA滤网怎么清洗",
    "LDS激光导航和视觉导航哪个好", "LDS激光导航有什么优势",
    "dToF导航技术是什么", "dToF导航和LDS有什么区别",
    "VSLAM和LDS的区别", "VSLAM视觉导航怎么样",
    "激光导航的机器人有什么优势", "3D结构光避障是什么",
    "扫地机器人吸力多少Pa合适", "滤网多久换一次",
    "边刷和主刷有什么区别", "滚刷和胶刷哪个好",
    "机器人导航用激光还是视觉", "dToF测距是什么",
    "HEPA滤网等级怎么看", "扫地机器人的导航方式有哪些",
    "激光雷达导航的机器人推荐",
]

# ── 边界歧义类 robot 样本（水箱/电池/滤网，消解跨领域歧义）───────────────

_ROBOT_BOUNDARY = [
    # 水箱类（"漏水"与热水器/水管/洗衣机歧义）
    "水箱漏水怎么办", "水箱怎么清洗", "水箱多久加一次水", "水箱容量一般多大",
    "水箱漏水怎么处理", "拖地水箱不出水怎么办", "水箱有异味怎么办", "水箱怎么拆下来",
    "水箱没水了怎么办", "水箱加满水能拖多久",
    # 电池类（"电池"与手机/电动车歧义）
    "电池续航下降怎么办", "电池续航时间变短", "电池多久需要更换", "电池老化怎么办",
    "电池充不进电怎么办", "电池续航多久", "机器人电池能用几年", "电池鼓包了怎么办",
    "扫地机器人电池续航下降", "机器人电池续航变短", "扫地机电池不耐用怎么办",
    "扫地机器人电池老化", "机器人电池充不进电",
    # 滤网/HEPA类（"HEPA"英文缩写）
    "HEPA滤网多久清洗一次", "HEPA滤网需要更换吗", "HEPA滤网怎么拆", "滤网清洗频率",
    "高效滤网多久换", "HEPA过滤网在哪里", "滤网脏了怎么清理", "滤网水洗还是更换",
]

# ── 时间类 robot 样本（"最近半年发布"等新品查询，配合日期工具调用）──────

_ROBOT_TIME = [
    "最近半年发布的产品有哪些", "最近半年发布的扫地机器人推荐",
    "最近半年发布的产品里有什么1000以内的扫地机器人", "近三个月发布的机器人有哪些",
    "今年发布的扫地机器人推荐", "最近半年有什么新机器人",
    "近半年上市的扫地机器人", "今年新出的扫地机器人推荐",
    "最近半年发布的产品里有什么2000以内的机器人", "近一个月发布的扫地机器人",
    "去年发布的扫地机器人有哪些", "最近三个月出的新机型",
    "今年有哪些新发布的扫地机器人", "最近半年发布的新品扫地机器人",
    "近一年上市的扫地机器人推荐", "最近半年出了哪些新款",
]

# ── other：领域外（扩充，含跨领域歧义消解）──────────────────────────────

_OTHER = [
    # 出行/交通工具
    "有没有飞机卖", "电动车多少钱", "电动车怎么选", "电瓶车充电安全吗", "汽车推荐",
    "汽车保养多久一次", "汽车保养项目有哪些", "汽车多久保养一次", "汽车维修要多少钱",
    "汽车轮胎多久换一次", "自行车推荐", "摩托车怎么选", "火车票怎么买",
    "电动车电池多久换一次", "电动车续航多少公里",
    # 手机/数码
    "手机哪个牌子好", "手机推荐", "手机屏幕碎了怎么修", "手机卡顿怎么办", "手机电池不耐用",
    "手机充电慢怎么办", "电脑怎么选", "笔记本推荐", "电脑卡顿怎么办", "电脑蓝屏怎么解决",
    "平板推荐", "平板电脑哪款好", "耳机推荐", "耳机有杂音怎么办", "蓝牙耳机连不上",
    "相机推荐", "单反怎么选", "智能手表推荐", "手表怎么调时间",
    # 大家电
    "空调不制冷了", "空调加氟多少钱", "冰箱有异味怎么办", "冰箱除霜方法", "洗衣机漏水",
    "洗衣机怎么用", "洗衣机不脱水", "热水器不出热水", "热水器漏水怎么办",
    "电视怎么联网", "电视黑屏了", "微波炉怎么选", "油烟机怎么清洗", "洗碗机推荐",
    "消毒柜怎么选", "水管漏水怎么处理",
    # 厨房小家电
    "电饭煲推荐", "电磁炉不加热", "烤箱怎么用", "空气炸锅推荐", "咖啡机推荐",
    "豆浆机怎么清洗", "破壁机推荐",
    # 生活电器
    "加湿器推荐", "空气净化器有用吗", "取暖器推荐", "电风扇不转", "吸尘器推荐",
    "除湿机怎么选", "净水器推荐", "投影仪推荐", "音响怎么选", "台灯推荐",
    "充电宝推荐", "汽车电池多久换一次", "笔记本电脑电池续航",
]

# ── casual：闲聊（扩充）─────────────────────────────────────────────────

_CASUAL = [
    "你好", "您好", "你好啊", "你好呀", "嗨", "哈喽", "哈喽哈喽", "hello", "hi", "hey",
    "在吗", "在不在", "在不在线", "有人吗", "在吗在吗", "嗨你好",
    "谢谢", "感谢", "多谢", "谢谢啦", "辛苦了", "太感谢了",
    "再见", "拜拜", "拜", "晚安", "早安", "午安", "早上好", "中午好", "晚上好", "晚上好呀",
    "你是谁", "你叫什么", "你叫什么名字", "你能做什么", "你会什么", "你能干嘛",
    "介绍一下你自己", "介绍一下", "你擅长什么", "你会哪些", "你都会点什么",
]

# ── unknown：模糊（扩充）─────────────────────────────────────────────────

_UNKNOWN = [
    "这个怎么样", "这个好用吗", "好用吗", "好用不", "推荐一下", "给我推荐一个", "哪个好",
    "哪个更好", "怎么样", "买哪个", "有没有便宜的", "性价比高的", "多少钱", "贵不贵",
    "靠谱吗", "这个靠谱吗", "值得买吗", "值不值得", "行不行", "好使吗", "能用吗", "能用不",
    "效果如何", "效果怎么样", "给点建议", "帮我看下", "帮我看一下", "这个可以吗",
    "有没有好点的", "有没有更好的", "好不好用", "好用不好用", "好不好", "质量怎么样",
    "划算吗", "性价比怎么样", "选哪个", "买哪个好", "哪个牌子好", "推荐个", "来个推荐",
    "给我说说", "讲讲", "说说看", "了解下", "这个咋样", "咋样", "怎么样啊", "好用么",
    "贵么", "值得么", "行么", "中不中", "这个怎么样啊", "那这个呢", "它好用吗",
]


def main():
    robot = collect_robot_samples()
    print(f"robot 知识库样本: {len(robot)}")
    print(f"robot 预算推荐补充: {len(_ROBOT_BUDGET)}")
    print(f"robot 专业术语补充: {len(_ROBOT_TERMS)}")
    print(f"robot 边界歧义补充: {len(_ROBOT_BOUNDARY)}")
    print(f"robot 时间类补充: {len(_ROBOT_TIME)}")

    rows = []
    for t in robot:
        rows.append({"text": t, "label": "robot"})
    for t in _ROBOT_BUDGET:
        rows.append({"text": t, "label": "robot"})
    for t in _ROBOT_TERMS:
        rows.append({"text": t, "label": "robot"})
    for t in _ROBOT_BOUNDARY:
        rows.append({"text": t, "label": "robot"})
    for t in _ROBOT_TIME:
        rows.append({"text": t, "label": "robot"})
    for t in _OTHER:
        rows.append({"text": t, "label": "other"})
    for t in _CASUAL:
        rows.append({"text": t, "label": "casual"})
    for t in _UNKNOWN:
        rows.append({"text": t, "label": "unknown"})

    seen, unique = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique.append(r)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _OUT_PATH)
    with open(out, "w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    dist = Counter(r["label"] for r in unique)
    print(f"总样本: {len(unique)}  分布: {dict(dist)}")
    print(f"已写入: {out}")


if __name__ == "__main__":
    main()
