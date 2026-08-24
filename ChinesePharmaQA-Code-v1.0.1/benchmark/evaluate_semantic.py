from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from common import (
    aggregate,
    aggregate_by_type,
    build_structured,
    build_vocab,
    gold_constraint_values,
    load_eval_corpus,
    load_release,
    metric,
    write_json,
)
from semantic_parser import (
    build_candidates,
    parsed_constraint_values,
    parse_with_scores,
    predict_from_parsed,
)


DEFAULT_REVISION = "79e7739b6ab944e86d6171e44d24c997fc1e0116"


def hash_texts(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description="Evaluate BGE semantic constraint parsing.")
    parser.add_argument("--dataset-root", type=Path, default=script.parents[2] / "ChinesePharmaQA")
    parser.add_argument(
        "--eval-corpus",
        type=Path,
        default=script.parent / "resources" / "eval_corpus.json",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=script.parent / "results" / "semantic_results.json",
    )
    return parser.parse_args()


def error_types(parsed: dict, gold_constraints: set[str], vocab: dict[str, list[str]]) -> list[str]:
    """Summarize parser errors without changing the set-level evaluation."""
    value_to_category = {
        value: category
        for category, values in vocab.items()
        for value in values
    }
    predicted = set(parsed_constraint_values(parsed))
    missing = gold_constraints - predicted
    extra = predicted - gold_constraints
    errors: set[str] = set()
    for value in missing:
        errors.add(f"missing_{value_to_category.get(value, 'constraint')}")
    for value in extra:
        errors.add(f"extra_{value_to_category.get(value, 'constraint')}")
    missing_categories = {value_to_category.get(value) for value in missing}
    extra_categories = {value_to_category.get(value) for value in extra}
    for category in missing_categories & extra_categories:
        if category:
            errors.add(f"wrong_{category}")
    property_missing = [value for value in missing if value_to_category.get(value) == "property"]
    property_extra = [value for value in extra if value_to_category.get(value) == "property"]
    if any(a in b or b in a for a in property_missing for b in property_extra):
        errors.add("property_parent_child_confusion")
    return sorted(errors)


def main() -> None:
    args = parse_args()
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit("Semantic evaluation requires torch and sentence-transformers.") from error

    public_corpus, queries = load_release(args.dataset_root)
    records = load_eval_corpus(args.eval_corpus)
    public_names = {record["title"] for record in public_corpus}
    eval_names = {record["名称"] for record in records}
    if public_names != eval_names:
        raise ValueError("public corpus and frozen evaluation view do not contain the same names")

    vocab = build_vocab(queries)
    indexes = build_structured(records, vocab)
    candidates = build_candidates(vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"device": device}
    if not Path(args.model).exists():
        model_kwargs["revision"] = args.revision
    model = SentenceTransformer(args.model, **model_kwargs)

    candidate_texts = [candidate.description for candidate in candidates]
    query_texts = [query["question"] for query in queries]
    encode_kwargs = {
        "batch_size": args.batch_size if device == "cuda" else min(args.batch_size, 8),
        "show_progress_bar": True,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
    }
    candidate_vectors = model.encode(candidate_texts, **encode_kwargs).astype("float32")
    query_vectors = model.encode(query_texts, **encode_kwargs).astype("float32")

    items = []
    for query, query_vector in zip(queries, query_vectors):
        parsed, scores = parse_with_scores(
            query["question"], candidates, candidate_vectors, query_vector
        )
        prediction = predict_from_parsed(parsed, indexes)
        gold = set(query["answer"])
        predicted_constraints = parsed_constraint_values(parsed)
        gold_constraints = gold_constraint_values(query)
        constraint_status = (
            "exact"
            if predicted_constraints == gold_constraints
            else "partial"
            if predicted_constraints & gold_constraints
            else "fail"
        )
        items.append(
            {
                "id": query["id"],
                "type": query["type"],
                "question": query["question"],
                "parsed": parsed,
                "gold_constraints": sorted(gold_constraints),
                "predicted_constraints": sorted(predicted_constraints),
                "constraint_status": constraint_status,
                "error_types": error_types(parsed, gold_constraints, vocab),
                "top_scores": {category: values[:5] for category, values in scores.items()},
                "gold_count": len(gold),
                "pred_count": len(prediction),
                **metric(prediction, gold),
            }
        )

    overall = aggregate(items)
    by_type = aggregate_by_type(items)
    exact_constraints = sum(item["constraint_status"] == "exact" for item in items)
    partial_constraints = sum(item["constraint_status"] == "partial" for item in items)
    failed_constraints = sum(item["constraint_status"] == "fail" for item in items)
    constraint_summary = {
        "exact": exact_constraints,
        "partial": partial_constraints,
        "fail": failed_constraints,
        "exact_rate": round(exact_constraints / len(items), 4) if items else 0.0,
    }

    write_json(
        args.output,
        {
            "description": "BGE semantic parsing followed by structured set execution",
            "parser_protocol": {
                "candidate_source": "released global constraint vocabulary",
                "candidate_description": "category-specific Chinese natural-language descriptions",
                "similarity": "normalized BGE inner product",
                "selection": "intent-gated top-1 per active category; top-2 for explicit multi-reagent questions; no similarity threshold",
                "empty_behavior": "return an empty set when no category gate activates",
                "execution": "set intersection over field indexes",
            },
            "model": args.model,
            "revision": None if Path(args.model).exists() else args.revision,
            "device": device,
            "n_queries": len(queries),
            "n_records": len(records),
            "candidate_text_sha256": hash_texts(candidate_texts),
            "query_text_sha256": hash_texts(query_texts),
            "candidate_counts": {
                category: sum(candidate.category == category for candidate in candidates)
                for category in ("reagent", "instrument", "property", "volume")
            },
            "overall": overall,
            "by_type": by_type,
            "constraint_summary": constraint_summary,
            "items": items,
        },
    )
    print("overall", overall)
    print("constraint_summary", constraint_summary)
    print("saved", args.output)


if __name__ == "__main__":
    main()
