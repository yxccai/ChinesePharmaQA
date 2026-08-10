from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from rebuild_qrels import build_rows, read_qrels


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPECTED = {'corpus': 5138, 'fusion': 1065, 'single': 200, 'qrels': 48335}
FIELD_ALIASES = {"检查项": "检查", "功能主治": "功能与主治"}


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path.name} line {line_number}: {error}") from error
    return rows


def private_use(value):
    text = json.dumps(value, ensure_ascii=False)
    return any(
        0xE000 <= ord(char) <= 0xF8FF
        or 0xF0000 <= ord(char) <= 0xFFFFD
        or 0x100000 <= ord(char) <= 0x10FFFD
        for char in text
    )


def unique(rows, key, label):
    values = [row[key] for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}")


def main():
    corpus = load_jsonl(DATA / "corpus.jsonl")
    fusion = load_jsonl(DATA / "fusion_queries.jsonl")
    single = load_jsonl(DATA / "single_drug_qa.jsonl")
    qrels = read_qrels(DATA / "fusion_qrels.tsv")

    actual = {
        "corpus": len(corpus),
        "fusion": len(fusion),
        "single": len(single),
        "qrels": len(qrels),
    }
    if actual != EXPECTED:
        raise ValueError(f"Count mismatch: expected={EXPECTED}, actual={actual}")

    unique(corpus, "_id", "corpus ID")
    unique(corpus, "title", "corpus title")
    unique(fusion, "_id", "fusion query ID")
    unique(single, "_id", "single-drug ID")
    if [row["_id"] for row in corpus] != [f"cpqa-c-{index:05d}" for index in range(1, len(corpus) + 1)]:
        raise ValueError("Unexpected corpus ID sequence")
    if [row["_id"] for row in fusion] != [f"cpqa-fq-{index:05d}" for index in range(1, len(fusion) + 1)]:
        raise ValueError("Unexpected fusion query ID sequence")
    if [row["_id"] for row in single] != [f"cpqa-sq-{index:05d}" for index in range(1, len(single) + 1)]:
        raise ValueError("Unexpected single-drug ID sequence")

    corpus_by_id = {row["_id"]: row for row in corpus}
    query_ids = {row["_id"] for row in fusion}
    qrel_pairs = [(query_id, corpus_id) for query_id, corpus_id, _ in qrels]
    if len(qrel_pairs) != len(set(qrel_pairs)):
        raise ValueError("Duplicate qrels")
    if any(query_id not in query_ids or corpus_id not in corpus_by_id for query_id, corpus_id in qrel_pairs):
        raise ValueError("Broken qrels reference")

    qrel_counts = Counter(query_id for query_id, _, _ in qrels)
    if any(qrel_counts[query_id] == 0 for query_id in query_ids):
        raise ValueError("Fusion query without qrels")

    rebuilt = build_rows(corpus, fusion)
    if rebuilt != qrels:
        raise ValueError("Bundled qrels do not exactly match deterministic reconstruction")

    for row in single:
        record = corpus_by_id.get(row["corpus_id"])
        if record is None:
            raise ValueError(f"Broken single-drug reference: {row['_id']}")
        if row["drug"] != record["title"]:
            raise ValueError(f"Single-drug title mismatch: {row['_id']}")
        field = FIELD_ALIASES.get(row["field"], row["field"])
        if row["answer"] != record["metadata"]["sections"].get(field):
            raise ValueError(f"Single-drug answer mismatch: {row['_id']}")

    payload = {"corpus": corpus, "fusion": fusion, "single": single}
    if private_use(payload):
        raise ValueError("Private-use Unicode detected")
    serialized = json.dumps(payload, ensure_ascii=False)
    if '"source"' in serialized or '"verified"' in serialized:
        raise ValueError("Internal source/verified field detected")
    forbidden_markers = ("盩", "秓", "岀", "頢", "獉", "磺胺甲唑", "异唑", "PE-:")
    if any(marker in serialized for marker in forbidden_markers):
        raise ValueError("Residual OCR marker detected")
    if re.search(
        r"聚乙二醇\s*200\s*\(|(?<=\d)m1(?=[^0-9A-Za-z]|$)|\]\s*-\s*2\s*-\s*唑烷\s*酮",
        serialized,
    ):
        raise ValueError("Residual OCR pattern detected")

    print(json.dumps({"status": "ok", **actual}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
