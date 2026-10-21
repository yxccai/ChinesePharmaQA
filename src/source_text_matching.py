"""Field matching and explicit pharmacopoeial chapter references."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

INSPECTION_SECTIONS = ('鉴别', '检查', '含量测定', '浸出物')
REFERENCE_PATTERN = re.compile(r'通则(\d{4}(?:第[一二三四五六七八九十\d]+法[12]?)?)')


def matching_text(value: str) -> str:
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', value))


def parse_sections(raw_text: str) -> dict[str, str]:
    headings = list(re.finditer(r'【([^】]+)】', raw_text))
    sections: dict[str, list[str]] = defaultdict(list)
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(raw_text)
        sections[heading.group(1)].append(raw_text[heading.end():end].strip())
    return {key: '\n'.join(values) for key, values in sections.items()}


def chapter_key(number: str) -> str:
    match = re.search(r'\d{4}(?:第[一二三四五六七八九十\d]+法[12]?)?', matching_text(number))
    return match.group() if match else ''


def build_chapter_registry(rules: list[dict]) -> dict[str, list[str]]:
    names: dict[str, set[str]] = defaultdict(set)
    for rule in rules:
        key = chapter_key(str(rule.get('number', '')))
        if key:
            names[key].add(str(rule.get('name', '')))
    return {key: sorted(value) for key, value in names.items()}


def reference_names(key: str, registry: dict[str, list[str]]) -> list[str]:
    """Resolve an exact reference, never borrow a sibling method's title."""
    if key in registry:
        return registry[key]
    if '法' in key and key.endswith(('1', '2')) and key[:-1] in registry:
        return registry[key[:-1]]
    return registry.get(key[:4], [])


def build_matching_view(record: dict, registry: dict[str, list[str]]) -> dict:
    sections = parse_sections(record['text'])
    existing = record['metadata'].get('general_rules', [])
    references = sorted({m.group(1) for m in REFERENCE_PATTERN.finditer(matching_text(record['text']))})
    names = {str(rule.get('name', '')) for rule in existing}
    for reference in references:
        names.update(reference_names(reference, registry))
    return {
        'id': record['_id'],
        'name': record['title'],
        'volume': record['metadata']['volume'],
        'raw_sections': sections,
        'normalized_sections': {key: matching_text(value) for key, value in sections.items()},
        'linked_names': sorted(names),
        'normalized_linked_names': sorted({matching_text(name) for name in names}),
        'explicit_references': references,
    }


def matches(view: dict, field: str, canonical: str) -> bool:
    if field == 'volume':
        return view['volume'] == canonical
    needle = matching_text(canonical)
    if field == 'property':
        return needle in view['normalized_sections'].get('性状', '')
    if field in ('reagent', 'instrument'):
        if any(needle in view['normalized_sections'].get(key, '') for key in INSPECTION_SECTIONS):
            return True
        return field == 'instrument' and any(needle in name for name in view['normalized_linked_names'])
    raise ValueError(f'Unknown constraint field: {field}')
