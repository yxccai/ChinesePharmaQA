"""Reagent-name boundary matching used by ChPSetQA."""
from __future__ import annotations

import re
from source_text_matching import INSPECTION_SECTIONS, matching_text, matches


DISTINCT_NAMES = {
    '高氯酸': r'高氯酸(?:盐|钠|钾|锂|铵|镁)',
    '铁氰化钾': r'亚铁氰化钾',
    '硫酸钠': r'硫代硫酸钠',
    '硝酸': r'亚硝酸(?:钠|钾|盐)|硝酸(?:银|钾|钠|铵|镁|盐|铅|锌|钙|汞|钡|铜)',
    '草酸': r'甘草酸|氯化草酸铵|草酸(?:铵|钾|钠|钙|盐)',
}
_PATTERNS = {name: re.compile(expression) for name, expression in DISTINCT_NAMES.items()}


def has_reagent_name(text: str, canonical: str) -> bool:
    text, canonical = matching_text(text), matching_text(canonical)
    pattern = _PATTERNS.get(canonical)
    blocked = [(m.start(), m.end()) for m in pattern.finditer(text)] if pattern else []
    return any(
        not any(start <= occurrence.start() and occurrence.end() <= end for start, end in blocked)
        for occurrence in re.finditer(re.escape(canonical), text)
    )


def matches_with_boundaries(view: dict, field: str, canonical: str) -> bool:
    if field != 'reagent':
        return matches(view, field, canonical)
    return any(
        has_reagent_name(view['normalized_sections'].get(section, ''), canonical)
        for section in INSPECTION_SECTIONS
    )
