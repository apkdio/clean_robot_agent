"""答案校验：核对 RAG 答案里的硬事实能否在召回片段里找到。

只做确定性抽取与字面核对，不调 LLM、不做单位换算、不判断语义蕴含：
型号对已知目录，金额取带货币单位的数字，年限只取保修/质保语境下的时长。
结果只供日志和 trace 使用，不阻塞流式输出。
"""

import re

from config.word_dict_config import ANSWER_PRICE_RE, ANSWER_WARRANTY_RE, SERIES_LIST
from function_tools.model_tool import match_key


def _compact(text: str) -> str:
    """去空白与千分位逗号，保留数字/字母/中文，供字面匹配。"""
    return re.sub(r"[\s,，]+", "", text or "")


def _contains_num(text: str, num: str) -> bool:
    """数字按完整 token 匹配，避免 26 命中 2026。"""
    return bool(re.search(r"(?<!\d)%s(?!\d)" % re.escape(num), _compact(text)))


def _model_candidates(answer: str, known: set) -> tuple:
    """返回 (已知型号, 目录外型号样 token)。

    先按系列前缀抓完整 token，再只把 token 整体拿去对目录；
    这样「云顶 X1 Max」不会被截成已知的「云顶 X1」蒙混过关。
    """
    flat = _compact(answer)
    candidates = []
    prefixes = [match_key(s) for s in SERIES_LIST if match_key(s)]
    if prefixes:
        pattern = re.compile("(?:%s)[A-Za-z0-9]+" % "|".join(
            re.escape(p) for p in sorted(prefixes, key=len, reverse=True)))
        for m in pattern.finditer(flat):
            token = m.group(0)
            if token not in candidates:
                candidates.append(token)
    # 目录里若有不符合系列前缀的型号，仍用右边界单独兜底
    for key in sorted(known, key=len, reverse=True):
        if key and key not in candidates and re.search(
                re.escape(key) + r"(?![A-Za-z0-9])", flat):
            candidates.append(key)

    found, unknown = [], []
    for token in candidates:
        (found if token in known else unknown).append(token)
    return found, unknown


def _warranty_entities(answer: str) -> list:
    """抽保修/质保语境下的数字 + 单位；“月”统一成“个月”，不做年/月换算。"""
    found = []
    for m in ANSWER_WARRANTY_RE.finditer(answer or ""):
        num = m.group(1) or m.group(3)
        unit = m.group(2) or m.group(4)
        if not num or not unit:
            continue
        if unit == "月":
            unit = "个月"
        entity = "%s%s" % (num, unit)
        if entity not in found:
            found.append(entity)
    return found


def _price_entities(answer: str) -> list:
    found = []
    for m in ANSWER_PRICE_RE.finditer(answer or ""):
        num = (m.group(1) or "").replace(",", "")
        if num and num not in found:
            found.append(num)
    return found


def check_answer(answer: str, chunks, query: str = "",
                 model_names=None) -> dict:
    """核对答案里的硬事实，返回 {checked, missing, exempt, unchecked, skipped}。

    checked 是抽到的实体全集，missing 是召回片段里找不到且用户原话也没提到的部分；
    unchecked 是像型号、但目录里没有的 token（无法判断真假，只暴露出来，不判缺失）。
    model_names 显式传空列表表示不核对型号；传 None 时从型号目录读取。
    """
    answer = answer or ""
    query = query or ""
    context = "\n".join(
        (c.page_content if hasattr(c, "page_content") else str(c)) or ""
        for c in (chunks or [])
    )
    skipped = []

    if model_names is None:
        try:
            from function_tools.model_tool import get_all_model_names
            model_names = get_all_model_names()
        except Exception:  # noqa: BLE001
            model_names = []
            skipped.append("model_catalog_unavailable")

    checked, missing, exempt, unchecked = [], [], [], []
    known = {match_key(n) for n in (model_names or []) if match_key(n)}

    if known:
        known_found, unknown_found = _model_candidates(answer, known)
    else:
        known_found, unknown_found = [], []
    for key in known_found:
        entity = "model:%s" % key
        checked.append(entity)
        if key in _compact(query):
            exempt.append(entity)       # 用户自己说的型号，不算答案新造
        elif key not in _compact(context):
            missing.append(entity)

    for token in unknown_found:
        unchecked.append("model:%s" % token)

    for num in _price_entities(answer):
        entity = "price:%s" % num
        checked.append(entity)
        if _contains_num(query, num):
            exempt.append(entity)       # 复述用户预算不算缺失
        elif not _contains_num(context, num):
            missing.append(entity)

    for duration in _warranty_entities(answer):
        entity = "warranty:%s" % duration
        checked.append(entity)
        if _compact(duration) in _compact(query):
            exempt.append(entity)
        elif _compact(duration) not in _compact(context):
            missing.append(entity)

    return {"checked": checked, "missing": missing, "exempt": exempt,
            "unchecked": unchecked, "skipped": skipped}
