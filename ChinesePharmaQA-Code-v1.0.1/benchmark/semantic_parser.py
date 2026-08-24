from __future__ import annotations

import re
from dataclasses import dataclass


MULTI_REAGENT_RE = re.compile(
    r"(?:两种试剂|"
    r"(?:同时|既)[^。！？]*?(?:需要|使用|涉及|用到|用)[^。！？]*?(?:和|与|及)[^。！？]*|"
    r"既[^。！？]*?(?:涉及|使用|需要|用到)[^。！？]*?又[^。！？]*)"
)

REAGENT_INTENT_RE = re.compile(
    r"试剂|试液|溶液|缓冲液|指示剂|对照品|标准品|滴定液|用到|涉及|需要"
)
INSTRUMENT_INTENT_RE = re.compile(
    r"仪器|色谱|光度|光谱|原子吸收|旋光|折光|滴定|显微|"
    r"崩解时限|澄清度|重金属|砷盐|微生物|细菌内毒素|不溶性微粒|"
    r"溶出|水分|炽灼|残留溶剂|熔点|含量均匀度|限度检查|"
    r"(?:引用|采用)[^。！？]{0,12}(?:检测|测定|检验|方法|法)|"
    r"(?:检测|分析|测定|检查)方法"
)
PROPERTY_INTENT_RE = re.compile(r"性状|外观|颜色|色泽|粉末|结晶|颗粒|液体|混悬")
VOLUME_INTENT_RE = re.compile(r"第一部|第二部|一部|二部")
FIRST_VOLUME_RE = re.compile(r"第一部|一部")
SECOND_VOLUME_RE = re.compile(r"第二部|二部")


@dataclass(frozen=True)
class Candidate:
    category: str
    value: str
    description: str


def candidate_description(category: str, value: str) -> str:
    if category == "reagent":
        return f"检测过程中使用{value}作为试剂或溶液"
    if category == "instrument":
        return f"使用{value}进行药品质量检验或分析检测"
    if category == "property":
        return f"药品的性状表现为{value}，外观是{value}"
    if category == "volume":
        return f"{value}收录的药品"
    raise ValueError(f"unsupported candidate category: {category}")


def build_candidates(vocab: dict[str, list[str]]) -> list[Candidate]:
    candidates = []
    for category in ("reagent", "instrument", "property"):
        candidates.extend(
            Candidate(category, value, candidate_description(category, value))
            for value in vocab.get(category, [])
        )
    candidates.extend(
        Candidate("volume", value, candidate_description("volume", value))
        for value in ("一部", "二部")
    )
    return candidates


def _top_candidates(
    question: str,
    scores: dict[str, list[tuple[str, float]]],
    category: str,
) -> list[tuple[str, float]]:
    ranked = scores.get(category, [])
    if not ranked:
        return []
    intent_patterns = {
        "reagent": REAGENT_INTENT_RE,
        "instrument": INSTRUMENT_INTENT_RE,
        "property": PROPERTY_INTENT_RE,
        "volume": VOLUME_INTENT_RE,
    }
    multi_reagent = bool(MULTI_REAGENT_RE.search(question))
    if category == "reagent":
        # A method name can occur in a phrase such as "需要使用气相色谱法".
        # In that case "使用" describes the method, not an additional reagent.
        reagent_intent = bool(REAGENT_INTENT_RE.search(question))
        method_intent = bool(INSTRUMENT_INTENT_RE.search(question))
        if not (multi_reagent or reagent_intent):
            return []
        if method_intent and not multi_reagent and not re.search(
            r"试剂|试液|溶液|缓冲液|指示剂|对照品|标准品|滴定液", question
        ):
            return []
    elif not intent_patterns[category].search(question):
        return []
    if category == "instrument" and multi_reagent:
        return []
    if category == "reagent" and multi_reagent:
        return ranked[:2]
    return ranked[:1]


def _explicit_volume(question: str) -> str | None:
    """Use an unambiguous part expression before semantic ranking."""
    has_first = bool(FIRST_VOLUME_RE.search(question))
    has_second = bool(SECOND_VOLUME_RE.search(question))
    if has_first and not has_second:
        return "一部"
    if has_second and not has_first:
        return "二部"
    return None


def parse_with_scores(
    question: str,
    candidates: list[Candidate],
    candidate_vectors,
    query_vector,
) -> tuple[dict, dict[str, list[tuple[str, float]]]]:
    scores: dict[str, list[tuple[str, float]]] = {
        "reagent": [],
        "instrument": [],
        "property": [],
        "volume": [],
    }
    similarities = candidate_vectors @ query_vector
    for candidate, score in zip(candidates, similarities.tolist()):
        scores[candidate.category].append((candidate.value, round(float(score), 6)))
    for category in scores:
        scores[category].sort(key=lambda item: (-item[1], item[0]))

    volume_candidates = _top_candidates(question, scores, "volume")
    selected = {
        "reagents": [value for value, _ in _top_candidates(question, scores, "reagent")],
        "instruments": [value for value, _ in _top_candidates(question, scores, "instrument")],
        "properties": [value for value, _ in _top_candidates(question, scores, "property")],
        "volume": _explicit_volume(question)
        or (volume_candidates or [(None, 0.0)])[0][0],
    }
    return selected, scores


def predict_from_parsed(parsed: dict, indexes: dict) -> set[str]:
    sets = []
    for value in parsed.get("reagents", []):
        sets.append(indexes["reagent"].get(value, set()))
    for value in parsed.get("instruments", []):
        sets.append(indexes["instrument"].get(value, set()))
    for value in parsed.get("properties", []):
        sets.append(indexes["property"].get(value, set()))
    if parsed.get("volume"):
        sets.append(indexes["volume"].get(parsed["volume"], set()))
    if not sets:
        return set()
    result = set(sets[0])
    for values in sets[1:]:
        result &= values
    return result


def parsed_constraint_values(parsed: dict) -> set[str]:
    values = set(parsed.get("reagents", []))
    values |= set(parsed.get("instruments", []))
    values |= set(parsed.get("properties", []))
    if parsed.get("volume"):
        values.add(parsed["volume"])
    return values
