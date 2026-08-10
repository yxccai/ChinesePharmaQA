from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MATCH_FIELDS = ("鉴别", "检查", "含量测定", "浸出物")


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_text(record):
    sections = record["metadata"]["sections"]
    return " ".join(sections.get(field, "") for field in MATCH_FIELDS)


def instrument_text(record):
    rules = " ".join(
        item.get("name", "")
        for item in record["metadata"].get("general_rules", [])
    )
    return f"{rules} {record_text(record)}"


def matches(record, constraints):
    text = record_text(record)
    for key, value in constraints.items():
        if key in {"reagent", "reagent1", "reagent2"} and value not in text:
            return False
        if key == "instrument" and value not in instrument_text(record):
            return False
        if key == "property" and value not in record["metadata"]["sections"].get("性状", ""):
            return False
        if key == "volume" and value != record["metadata"]["volume"]:
            return False
    return True


def build_rows(corpus, queries):
    rows = []
    for query in queries:
        for record in corpus:
            if matches(record, query["constraints"]):
                rows.append((query["_id"], record["_id"], 1))
    return rows


def read_qrels(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"Unexpected qrels header: {header}")
        for line in handle:
            if line.strip():
                query_id, corpus_id, score = line.rstrip("\n").split("\t")
                rows.append((query_id, corpus_id, int(score)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    corpus = load_jsonl(DATA / "corpus.jsonl")
    queries = load_jsonl(DATA / "fusion_queries.jsonl")
    rebuilt = build_rows(corpus, queries)

    if args.output:
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("query-id\tcorpus-id\tscore\n")
            for query_id, corpus_id, score in rebuilt:
                handle.write(f"{query_id}\t{corpus_id}\t{score}\n")
        print(f"Wrote {len(rebuilt)} qrels to {args.output}")
        return

    bundled = read_qrels(DATA / "fusion_qrels.tsv")
    if rebuilt != bundled:
        rebuilt_set = set(rebuilt)
        bundled_set = set(bundled)
        raise SystemExit(
            f"qrels mismatch: rebuilt={len(rebuilt)}, bundled={len(bundled)}, "
            f"missing={len(rebuilt_set - bundled_set)}, extra={len(bundled_set - rebuilt_set)}"
        )
    print(f"OK: exactly rebuilt {len(rebuilt)} qrels for {len(queries)} fusion queries")


if __name__ == "__main__":
    main()
