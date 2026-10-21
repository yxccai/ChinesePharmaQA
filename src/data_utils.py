from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


INSPECTION_FIELDS = ("identification", "tests", "assay", "extractives")
SECTION_MAP = {
    "definition": "描述",
    "description": "性状",
    "identification": "鉴别",
    "tests": "检查",
    "assay": "含量测定",
    "extractives": "浸出物",
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalize_corpus_record(record: dict) -> dict:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"record {record.get('_id')} has invalid metadata")
    sections = metadata.get("sections")
    if not isinstance(sections, dict):
        raise ValueError(f"record {record.get('_id')} has invalid metadata.sections")

    rules = []
    for rule in metadata.get("general_rules", []):
        if not isinstance(rule, dict):
            continue
        rules.append(
            {
                "number": str(rule.get("number", "")).strip(),
                "name": str(rule.get("name", "")).strip(),
            }
        )

    output = {
        "id": str(record.get("_id", "")).strip(),
        "name": str(record.get("title", "")).strip(),
        "volume": str(metadata.get("volume", "")).strip(),
        "edition": str(metadata.get("edition", "")).strip(),
        "raw_text": str(record.get("text", "")),
        "general_rules": rules,
    }
    for normalized, source in SECTION_MAP.items():
        value = sections.get(source, "")
        output[normalized] = "" if value is None else str(value).strip()
    return output


def load_normalized_corpus(dataset_root: Path) -> list[dict]:
    source = dataset_root / "data" / "corpus.jsonl"
    return [normalize_corpus_record(record) for record in read_jsonl(source)]










def concept_id(field: str, canonical: str) -> str:
    return f"{field}::{canonical}"


def constraints_to_concept_ids(constraints: dict) -> list[str]:
    output: list[str] = []
    for key in ("reagent", "reagent1", "reagent2"):
        if key in constraints:
            output.append(concept_id("reagent", str(constraints[key])))
    for key in ("instrument", "property", "volume"):
        if key in constraints:
            output.append(concept_id(key, str(constraints[key])))
    if not output:
        raise ValueError(f"constraints contain no executable fields: {constraints}")
    return output
