from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import (
    VOLUME_FULL,
    aggregate,
    build_structured,
    build_vocab,
    e2e_predict,
    gold_constraint_values,
    load_eval_corpus,
    load_release,
    metric,
    parse_query,
    parsed_constraint_values,
    write_json,
)


TEMPLATES_A = [
    "{r}在哪些药品的质量标准中有使用？",
    "请列出检验时需要用到{r}试液的药品",
    "须使用{r}进行检验的药品包括哪些品种？",
    "用{r}作为检验试剂的药品有哪些种？",
    "{r}被哪些药品的鉴别或含量测定方法所采用？",
]
TEMPLATES_B = [
    "{i}被用于哪些药品的质量检验？",
    "哪些品种的质量标准中引用了{i}的检测方法？",
    "通过{i}来进行检测的药品有哪些？",
    "采用{i}进行鉴别或含量测定的品种有哪些？",
]
TEMPLATES_C = [
    "性状描述为{p}的药品有哪些种？",
    "药典中外观记录为{p}的药物有哪些？",
    "外观呈{p}的药品品种有哪些？",
    "请列出性状项中含有{p}描述的药品",
]
TEMPLATES_D_RR = [
    "检验时同时需要{r1}和{r2}两种试剂的药品有哪些种？",
    "{r1}和{r2}同时出现在哪些药品的质量标准中？",
    "须同时使用{r1}及{r2}进行检测的品种有哪些？",
]
TEMPLATES_D_VR = [
    "{v}收录的品种中，检验需要{r}的有哪些？",
    "{v}中用{r}进行检验的药品包括哪些？",
    "在{v}的质量标准里，涉及{r}使用的品种有哪些？",
]
TEMPLATES_D_VI = [
    "{v}收录的品种里，采用{i}检测的有哪些？",
    "{v}中通则引用了{i}方法的品种有哪些？",
]
TEMPLATES_D_PI = [
    "外观为{p}且通则中引用了{i}的品种有哪些？",
    "性状是{p}、同时使用{i}检验的品种有哪些？",
]
TEMPLATES_D_PR = [
    "性状为{p}并且检验用到{r}的品种有哪些？",
    "外观{p}的、检验也需要{r}的药品有哪些？",
]
TEMPLATES_D_IR = [
    "采用{i}且需要{r}的品种有哪些？",
    "{i}检验中同时涉及{r}使用的品种有哪些？",
]
TEMPLATES_D_VP = [
    "{v}中，外观为{p}的品种有哪些？",
    "在{v}里，性状描述含有{p}的药品有哪些？",
]
TEMPLATES_D_VRR = [
    "{v}的品种中，同时需要{r1}和{r2}检验的有哪些？",
    "在{v}里，既用{r1}又用{r2}的品种有哪些？",
]


def generate_two(query: dict) -> list[str]:
    constraints = query["constraints"]
    query_type = query["type"]
    if query_type == "A":
        return [template.format(r=constraints["reagent"]) for template in random.sample(TEMPLATES_A, 2)]
    if query_type == "B":
        return [template.format(i=constraints["instrument"]) for template in random.sample(TEMPLATES_B, 2)]
    if query_type == "C":
        return [template.format(p=constraints["property"]) for template in random.sample(TEMPLATES_C, 2)]

    has_r1 = "reagent1" in constraints
    has_r2 = "reagent2" in constraints
    has_reagent = "reagent" in constraints
    has_instrument = "instrument" in constraints
    has_property = "property" in constraints
    has_volume = "volume" in constraints
    volume = VOLUME_FULL.get(constraints.get("volume", ""), "")

    # D8 must precede D1; otherwise the three-constraint branch loses volume.
    if has_r1 and has_r2 and has_volume:
        return [
            template.format(
                v=volume,
                r1=constraints["reagent1"],
                r2=constraints["reagent2"],
            )
            for template in random.sample(TEMPLATES_D_VRR, 2)
        ]
    if has_r1 and has_r2:
        return [
            template.format(r1=constraints["reagent1"], r2=constraints["reagent2"])
            for template in random.sample(TEMPLATES_D_RR, 2)
        ]
    if has_volume and has_reagent:
        return [
            template.format(v=volume, r=constraints["reagent"])
            for template in random.sample(TEMPLATES_D_VR, 2)
        ]
    if has_volume and has_instrument:
        return [
            template.format(v=volume, i=constraints["instrument"])
            for template in random.sample(TEMPLATES_D_VI, 2)
        ]
    if has_property and has_instrument:
        return [
            template.format(p=constraints["property"], i=constraints["instrument"])
            for template in random.sample(TEMPLATES_D_PI, 2)
        ]
    if has_property and has_reagent:
        return [
            template.format(p=constraints["property"], r=constraints["reagent"])
            for template in random.sample(TEMPLATES_D_PR, 2)
        ]
    if has_instrument and has_reagent:
        return [
            template.format(i=constraints["instrument"], r=constraints["reagent"])
            for template in random.sample(TEMPLATES_D_IR, 2)
        ]
    if has_volume and has_property:
        return [
            template.format(v=volume, p=constraints["property"])
            for template in random.sample(TEMPLATES_D_VP, 2)
        ]
    raise ValueError(f"unsupported constraints for {query['id']}: {constraints}")


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description="Evaluate programmatic paraphrases.")
    parser.add_argument("--dataset-root", type=Path, default=script.parents[2] / "ChinesePharmaQA")
    parser.add_argument(
        "--eval-corpus",
        type=Path,
        default=script.parent / "resources" / "eval_corpus.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script.parent / "results" / "paraphrase_results.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, queries = load_release(args.dataset_root)
    records = load_eval_corpus(args.eval_corpus)
    vocab = build_vocab(queries)
    indexes = build_structured(records, vocab)
    by_type = {query_type: [] for query_type in "ABCD"}
    for query in queries:
        by_type[query["type"]].append(query)

    random.seed(42)
    sample_sizes = {"A": 10, "B": 5, "C": 10, "D": 75}
    sampled = []
    for query_type in "ABCD":
        sampled.extend(random.sample(by_type[query_type], sample_sizes[query_type]))

    items = []
    for query in sampled:
        for paraphrase in generate_two(query):
            parsed = parse_query(paraphrase, vocab)
            gold_constraints = gold_constraint_values(query)
            predicted_constraints = parsed_constraint_values(parsed)
            status = (
                "exact"
                if gold_constraints == predicted_constraints
                else "partial"
                if gold_constraints & predicted_constraints
                else "fail"
            )
            values = metric(
                e2e_predict(paraphrase, indexes, vocab),
                query["answer"],
            )
            items.append(
                {
                    "original_id": query["id"],
                    "type": query["type"],
                    "paraphrase": paraphrase,
                    "constraint_status": status,
                    **values,
                }
            )

    exact = sum(item["constraint_status"] == "exact" for item in items)
    partial = sum(item["constraint_status"] == "partial" for item in items)
    fail = sum(item["constraint_status"] == "fail" for item in items)
    overall = aggregate(items)
    output = {
        "seed": 42,
        "sampled_questions": len(sampled),
        "paraphrases": len(items),
        "constraint_accuracy": {
            "exact": exact,
            "partial": partial,
            "fail": fail,
            "exact_rate": round(exact / len(items), 4),
        },
        "e2e_metrics": overall,
        "items": items,
    }
    write_json(args.output, output)
    print(output["constraint_accuracy"])
    print({key: overall[key] for key in ("precision", "recall", "f1", "exact_match")})
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
