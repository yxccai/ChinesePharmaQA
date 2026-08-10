from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from common import (
    aggregate,
    aggregate_by_type,
    bm25_search,
    build_bm25,
    build_structured,
    build_vocab,
    e2e_predict,
    full_scan_predict,
    gold_constraint_values,
    load_eval_corpus,
    load_release,
    metric,
    parse_query,
    parsed_constraint_values,
    sha256_file,
    structured_predict,
    write_json,
)


K_VALUES = (10, 20, 50, 100, 200, 500, 1000)


def evaluate_predictions(queries: list[dict], predictor) -> tuple[dict, dict, list[dict]]:
    items = []
    for query in queries:
        prediction = set(predictor(query))
        gold = set(query["answer"])
        items.append(
            {
                "id": query["id"],
                "type": query["type"],
                "question": query["question"],
                "gold": len(gold),
                "pred": len(prediction),
                **metric(prediction, gold),
            }
        )
    return aggregate(items), aggregate_by_type(items), items


def evaluate_ranked(queries: list[dict], index: dict) -> dict:
    ranked = [bm25_search(index, query["question"], max(K_VALUES)) for query in queries]
    output = {}
    for k in K_VALUES:
        items = []
        for query, names in zip(queries, ranked):
            prediction = set(names[:k])
            gold = set(query["answer"])
            items.append(
                {
                    "id": query["id"],
                    "type": query["type"],
                    "question": query["question"],
                    "gold": len(gold),
                    "pred": len(prediction),
                    **metric(prediction, gold),
                }
            )
        output[k] = {
            "overall": aggregate(items),
            "by_type": aggregate_by_type(items),
            "items": items,
            "ranked": ranked,
        }
    return output


def first_only(query: dict, indexes: dict) -> set[str]:
    constraints = query["constraints"]
    if query["type"] in "ABC":
        key = {"A": "reagent", "B": "instrument", "C": "property"}[query["type"]]
        return indexes[key].get(constraints.get(key, ""), set())
    for key in ("reagent", "reagent1", "instrument", "property"):
        if key in constraints:
            table = "reagent" if key.startswith("reagent") else key
            return indexes[table].get(constraints[key], set())
    return set()


def or_predict(query: dict, indexes: dict) -> set[str]:
    if query["type"] != "D":
        return structured_predict(query, indexes)
    constraints = query["constraints"]
    sets = []
    for key in ("reagent", "reagent1", "reagent2"):
        if key in constraints:
            sets.append(indexes["reagent"].get(constraints[key], set()))
    for key in ("instrument", "property", "volume"):
        if key in constraints:
            sets.append(indexes[key].get(constraints[key], set()))
    result = set()
    for values in sets:
        result |= values
    return result


def replace_component(
    query: dict,
    indexes: dict,
    bm25_top50: dict[str, set[str]],
    component: str,
) -> set[str]:
    constraints = query["constraints"]
    if query["type"] in "ABC":
        key = {"A": "reagent", "B": "instrument", "C": "property"}[query["type"]]
        if key == component:
            return set(bm25_top50[query["id"]])
        return structured_predict(query, indexes)

    sets = []
    for key, table in (
        ("reagent", "reagent"),
        ("reagent1", "reagent"),
        ("reagent2", "reagent"),
        ("instrument", "instrument"),
        ("property", "property"),
        ("volume", "volume"),
    ):
        if key not in constraints:
            continue
        if table == component:
            sets.append(set(bm25_top50[query["id"]]))
        else:
            sets.append(indexes[table].get(constraints[key], set()))
    if not sets:
        return set()
    result = set(sets[0])
    for values in sets[1:]:
        result &= values
    return result


def controlled_condition(queries: list[dict], predictor) -> dict:
    items = []
    for query in queries:
        values = metric(set(predictor(query)), set(query["answer"]))
        items.append({"id": query["id"], "type": query["type"], **values})
    return {"n": len(items), **aggregate(items), "by_type": aggregate_by_type(items)}


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_dataset = script.parents[2] / "ChinesePharmaQA"
    parser = argparse.ArgumentParser(description="Reproduce classic ChinesePharmaQA baselines.")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument(
        "--eval-corpus",
        type=Path,
        default=script.parent / "resources" / "eval_corpus.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script.parent / "results" / "classic_results.json",
    )
    parser.add_argument("--skip-full-scan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    public_corpus, queries = load_release(args.dataset_root)
    records = load_eval_corpus(args.eval_corpus)
    public_names = {record["title"] for record in public_corpus}
    eval_names = {record["名称"] for record in records}
    if public_names != eval_names:
        raise ValueError("public corpus and frozen evaluation view do not contain the same names")

    vocab = build_vocab(queries)
    original_fields = ["名称", "性状", "鉴别", "检查", "含量测定"]
    full_fields = original_fields + ["浸出物", "通则引用"]
    original = build_bm25(records, original_fields)
    full_fields_index = build_bm25(records, full_fields)
    structured = build_structured(records, vocab)

    summary: dict[str, dict] = {}
    per_item: dict[str, list[dict]] = {}

    random.seed(42)
    all_names = [record["名称"] for record in records]
    for k in (10, 50):
        key = f"Random_K{k}"
        overall, by_type, items = evaluate_predictions(
            queries,
            lambda _query, k=k: random.sample(all_names, min(k, len(all_names))),
        )
        summary[key] = {**overall, "by_type": by_type}
        per_item[key] = items

    ranked_outputs = {}
    for label, index in (("Original", original), ("FullFields", full_fields_index)):
        ranked_outputs[label] = evaluate_ranked(queries, index)
        for k in K_VALUES:
            key = f"BM25_{label}_K{k}"
            result = ranked_outputs[label][k]
            summary[key] = {**result["overall"], "by_type": result["by_type"]}
            per_item[key] = result["items"]

    for k in (200, 500, 1000):
        key = f"BM25_Filter_K{k}"
        items = []
        ranked = ranked_outputs["Original"][k]["ranked"]
        for query, names in zip(queries, ranked):
            prediction = set(names[:k]) & structured_predict(query, structured)
            gold = set(query["answer"])
            items.append(
                {
                    "id": query["id"],
                    "type": query["type"],
                    "question": query["question"],
                    "gold": len(gold),
                    "pred": len(prediction),
                    **metric(prediction, gold),
                }
            )
        summary[key] = {**aggregate(items), "by_type": aggregate_by_type(items)}
        per_item[key] = items

    overall, by_type, items = evaluate_predictions(
        queries, lambda query: structured_predict(query, structured)
    )
    summary["OracleStructured"] = {**overall, "by_type": by_type}
    per_item["OracleStructured"] = items

    parser_accuracy = {"exact": 0, "partial": 0, "fail": 0}
    parser_by_type = {
        query_type: {"exact": 0, "partial": 0, "fail": 0} for query_type in "ABCD"
    }
    for query in queries:
        gold_values = gold_constraint_values(query)
        predicted_values = parsed_constraint_values(parse_query(query["question"], vocab))
        status = "exact" if gold_values == predicted_values else "partial" if gold_values & predicted_values else "fail"
        parser_accuracy[status] += 1
        parser_by_type[query["type"]][status] += 1

    start = time.perf_counter()
    overall, by_type, items = evaluate_predictions(
        queries,
        lambda query: e2e_predict(query["question"], structured, vocab),
    )
    summary["E2E_Rule"] = {
        **overall,
        "by_type": by_type,
        "query_time_ms_total": round((time.perf_counter() - start) * 1000, 1),
    }
    per_item["E2E_Rule"] = items

    if not args.skip_full_scan:
        start = time.perf_counter()
        overall, by_type, items = evaluate_predictions(
            queries,
            lambda query: full_scan_predict(query["question"], records, vocab),
        )
        summary["FullCorpusScan"] = {
            **overall,
            "by_type": by_type,
            "query_time_ms_total": round((time.perf_counter() - start) * 1000, 1),
        }
        per_item["FullCorpusScan"] = items

    bm25_top50 = {
        query["id"]: set(names[:50])
        for query, names in zip(queries, ranked_outputs["Original"][50]["ranked"])
    }

    ablation = {
        "first_only": controlled_condition(queries, lambda query: first_only(query, structured)),
        "or_logic": controlled_condition(queries, lambda query: or_predict(query, structured)),
        "no_reagent": controlled_condition(
            queries, lambda query: replace_component(query, structured, bm25_top50, "reagent")
        ),
        "no_instrument": controlled_condition(
            queries, lambda query: replace_component(query, structured, bm25_top50, "instrument")
        ),
        "no_property": controlled_condition(
            queries, lambda query: replace_component(query, structured, bm25_top50, "property")
        ),
        "full_structured": controlled_condition(
            queries, lambda query: structured_predict(query, structured)
        ),
    }

    output = {
        "records": len(records),
        "n_fusion": len(queries),
        "eval_corpus_sha256": sha256_file(args.eval_corpus),
        "summary": summary,
        "parser_accuracy": {"overall": parser_accuracy, "by_type": parser_by_type},
        "ablation": ablation,
        "per_item": per_item,
    }
    write_json(args.output, output)
    for key in ("BM25_Original_K50", "BM25_FullFields_K50", "E2E_Rule"):
        print(key, summary[key])
    if "FullCorpusScan" in summary:
        print("FullCorpusScan", summary["FullCorpusScan"])
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
