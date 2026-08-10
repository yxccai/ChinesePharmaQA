from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


VOLUME_FULL = {
    "一部": "中国药典第一部（中药）",
    "二部": "中国药典第二部（化学药）",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_release(dataset_root: Path) -> tuple[list[dict], list[dict]]:
    data_dir = dataset_root / "data"
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "fusion_queries.jsonl")
    id_to_title = {item["_id"]: item["title"] for item in corpus}
    qrels: dict[str, list[str]] = defaultdict(list)
    with (data_dir / "fusion_qrels.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            qrels[row["query-id"]].append(row["corpus-id"])

    prepared = []
    for item in queries:
        gold_ids = qrels[item["_id"]]
        prepared.append(
            {
                "id": item["_id"],
                "question": item["text"],
                "type": item["type"],
                "constraints": item["constraints"],
                "answer_ids": gold_ids,
                "answer": [id_to_title[corpus_id] for corpus_id in gold_ids],
            }
        )
    return corpus, prepared


def load_eval_corpus(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if len(records) != 5138:
        raise ValueError(f"expected 5,138 evaluation records, found {len(records)}")
    return records


def build_vocab(queries: list[dict]) -> dict[str, list[str]]:
    values = {"reagent": set(), "instrument": set(), "property": set()}
    for query in queries:
        constraints = query["constraints"]
        for key in ("reagent", "reagent1", "reagent2"):
            if key in constraints:
                values["reagent"].add(constraints[key])
        for key in ("instrument", "property"):
            if key in constraints:
                values[key].add(constraints[key])
    return {key: sorted(items, key=lambda value: (-len(value), value)) for key, items in values.items()}


def tokenize(text: str) -> list[str]:
    text = str(text).lower()
    cjk = [char for char in text if "一" <= char <= "鿿"]
    bigrams = [cjk[index] + cjk[index + 1] for index in range(len(cjk) - 1)]
    alnum = re.findall(r"[a-z0-9]+", text)
    return cjk + bigrams + alnum


def build_bm25(records: list[dict], fields: list[str], k1: float = 1.5, b: float = 0.75) -> dict:
    documents = []
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for record in records:
        parts = []
        for field in fields:
            if field == "通则引用":
                parts.extend(
                    str(rule.get("name", ""))
                    for rule in record.get(field, [])
                    if isinstance(rule, dict)
                )
            else:
                parts.append(str(record.get(field, "")))
        term_frequency = Counter(tokenize(" ".join(parts)))
        documents.append(
            {
                "name": record["名称"],
                "tf": term_frequency,
                "length": sum(term_frequency.values()),
            }
        )

    average_length = sum(item["length"] for item in documents) / len(documents)
    for doc_index, document in enumerate(documents):
        for token, count in document["tf"].items():
            postings[token].append((doc_index, count))
    return {
        "documents": documents,
        "postings": postings,
        "df": {token: len(items) for token, items in postings.items()},
        "average_length": average_length,
        "count": len(documents),
        "k1": k1,
        "b": b,
    }


def bm25_search(index: dict, query: str, k: int) -> list[str]:
    scores: dict[int, float] = defaultdict(float)
    for token in set(tokenize(query)):
        if token not in index["df"]:
            continue
        document_frequency = index["df"][token]
        inverse_document_frequency = math.log(
            (index["count"] - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0
        )
        for doc_index, term_frequency in index["postings"][token]:
            document_length = index["documents"][doc_index]["length"]
            denominator = term_frequency + index["k1"] * (
                1.0 - index["b"]
                + index["b"] * document_length / index["average_length"]
            )
            scores[doc_index] += (
                inverse_document_frequency
                * term_frequency
                * (index["k1"] + 1.0)
                / denominator
            )
    ranked = sorted(scores.items(), key=lambda item: -item[1])[:k]
    return [index["documents"][doc_index]["name"] for doc_index, _ in ranked]


def build_structured(records: list[dict], vocab: dict[str, list[str]]) -> dict[str, dict[str, set[str]]]:
    indexes: dict[str, dict[str, set[str]]] = {
        "reagent": defaultdict(set),
        "instrument": defaultdict(set),
        "property": defaultdict(set),
        "volume": defaultdict(set),
    }
    for record in records:
        name = record["名称"]
        inspection_text = " ".join(
            str(record.get(field, "")) for field in ("鉴别", "检查", "含量测定", "浸出物")
        )
        rule_text = " ".join(
            str(rule.get("name", ""))
            for rule in record.get("通则引用", [])
            if isinstance(rule, dict)
        )
        property_text = str(record.get("性状", ""))
        for value in vocab["reagent"]:
            if value in inspection_text:
                indexes["reagent"][value].add(name)
        for value in vocab["instrument"]:
            if value in inspection_text or value in rule_text:
                indexes["instrument"][value].add(name)
        if property_text not in ("", "未找到", "不适用"):
            for value in vocab["property"]:
                if value in property_text:
                    indexes["property"][value].add(name)
        indexes["volume"][record.get("来源", "")].add(name)
    return indexes


def structured_predict(query: dict, indexes: dict[str, dict[str, set[str]]]) -> set[str]:
    constraints = query["constraints"]
    sets = []
    for key in ("reagent", "reagent1", "reagent2"):
        if key in constraints:
            sets.append(indexes["reagent"].get(constraints[key], set()))
    for key in ("instrument", "property", "volume"):
        if key in constraints:
            sets.append(indexes[key].get(constraints[key], set()))
    if not sets:
        return set()
    result = set(sets[0])
    for values in sets[1:]:
        result &= values
    return result


def parse_query(question: str, vocab: dict[str, list[str]]) -> dict:
    parsed = {"reagents": [], "instruments": [], "properties": [], "volume": None}
    for code, full_name in VOLUME_FULL.items():
        if full_name in question:
            parsed["volume"] = code
            break
    if parsed["volume"] is None:
        if "第一部" in question:
            parsed["volume"] = "一部"
        elif "第二部" in question:
            parsed["volume"] = "二部"

    for vocab_key, output_key in (
        ("property", "properties"),
        ("reagent", "reagents"),
        ("instrument", "instruments"),
    ):
        remaining = question
        for value in vocab[vocab_key]:
            if value in remaining:
                parsed[output_key].append(value)
                remaining = remaining.replace(value, "\x00" * len(value), 1)
    return parsed


def e2e_predict(
    question: str,
    indexes: dict[str, dict[str, set[str]]],
    vocab: dict[str, list[str]],
) -> set[str]:
    parsed = parse_query(question, vocab)
    sets = []
    for value in parsed["reagents"]:
        sets.append(indexes["reagent"].get(value, set()))
    for value in parsed["instruments"]:
        sets.append(indexes["instrument"].get(value, set()))
    for value in parsed["properties"]:
        sets.append(indexes["property"].get(value, set()))
    if parsed["volume"]:
        sets.append(indexes["volume"].get(parsed["volume"], set()))
    if not sets:
        return set()
    result = set(sets[0])
    for values in sets[1:]:
        result &= values
    return result


def full_scan_predict(question: str, records: list[dict], vocab: dict[str, list[str]]) -> set[str]:
    parsed = parse_query(question, vocab)
    if not any((parsed["reagents"], parsed["instruments"], parsed["properties"], parsed["volume"])):
        return set()
    result = set()
    for record in records:
        inspection_text = " ".join(
            str(record.get(field, "")) for field in ("鉴别", "检查", "含量测定", "浸出物")
        )
        rule_text = " ".join(
            str(rule.get("name", ""))
            for rule in record.get("通则引用", [])
            if isinstance(rule, dict)
        )
        property_text = str(record.get("性状", ""))
        if any(value not in inspection_text for value in parsed["reagents"]):
            continue
        if any(value not in inspection_text and value not in rule_text for value in parsed["instruments"]):
            continue
        if any(value not in property_text for value in parsed["properties"]):
            continue
        if parsed["volume"] and record.get("来源", "") != parsed["volume"]:
            continue
        result.add(record["名称"])
    return result


def parsed_constraint_values(parsed: dict) -> set[str]:
    values = set(parsed["reagents"]) | set(parsed["instruments"]) | set(parsed["properties"])
    if parsed["volume"]:
        values.add(parsed["volume"])
    return values


def gold_constraint_values(query: dict) -> set[str]:
    return set(query["constraints"].values())


def metric(prediction: set[str] | list[str], gold: set[str] | list[str], extra_false_positives: int = 0) -> dict:
    prediction_set = set(prediction)
    gold_set = set(gold)
    true_positive = len(prediction_set & gold_set)
    predicted_count = len(prediction_set) + extra_false_positives
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / len(gold_set) if gold_set else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "jaccard": round(true_positive / (predicted_count + len(gold_set) - true_positive), 4)
        if predicted_count + len(gold_set) - true_positive
        else 1.0,
        "exact_match": 1.0
        if extra_false_positives == 0 and prediction_set == gold_set
        else 0.0,
    }


def aggregate(items: list[dict]) -> dict:
    keys = ("precision", "recall", "f1", "jaccard", "exact_match")
    if not items:
        return {key: 0.0 for key in keys}
    return {key: round(sum(item[key] for item in items) / len(items), 4) for key in keys}


def aggregate_by_type(items: list[dict]) -> dict:
    output = {}
    for query_type in "ABCD":
        subset = [item for item in items if item["type"] == query_type]
        output[query_type] = {"n": len(subset), **aggregate(subset)}
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
