from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import aggregate, aggregate_by_type, load_release, metric, write_json


SYSTEM_PROMPT = (
    "你是一名中国药典专家。用户会问关于2020版《中国药典》的问题，请根据你的知识如实回答。"
    "如果问题要求列出药品名称，请以顿号或逗号分隔列出所有你知道的药品名称，"
    "每个名称单独一行或用逗号分隔，不要添加额外解释。"
    "如果你不确定或不知道，直接说不知道，不要编造。"
)

STOPWORDS = {
    "不知道",
    "抱歉",
    "以下",
    "根据",
    "药品",
    "以上",
    "如下",
    "共有",
    "包括",
    "这些",
    "其中",
    "另外",
    "此外",
    "以及",
    "还有",
    "具体",
    "品种",
    "药物",
    "以下是",
    "如下是",
    "主要",
    "部分",
    "其他",
    "注意",
    "说明",
    "但是",
    "然而",
    "因此",
    "总结",
    "综上",
    "经查",
    "确认",
    "目前",
    "暂时",
    "经过",
    "目录",
    "列表",
    "一部",
    "二部",
    "三部",
    "四部",
    "附录",
}


def normalize_name(value: str) -> str:
    return value.replace("　", "").replace(" ", "")


def parse_names(raw_text: str, name_set: set[str], normalized_names: dict[str, str]) -> tuple[set[str], int]:
    parts = re.split(r"[，,、\n；;\r]|\d+[\.、）)]\s*|-\s+|\*+\s*", raw_text)
    matched = set()
    false_positive_count = 0
    for part in parts:
        segment = re.sub(r"\*+|（[^）]*）|\([^)]*\)", "", part).strip()
        if not segment:
            continue
        if segment in name_set:
            matched.add(segment)
            continue
        normalized = normalize_name(segment)
        if normalized in normalized_names:
            matched.add(normalized_names[normalized])
        elif (
            2 <= len(segment) <= 20
            and segment not in STOPWORDS
            and re.search(r"[一-鿿]", segment)
            and not re.fullmatch(r"[\d\s\.\-\(\)（）,，]+", segment)
        ):
            false_positive_count += 1
    return matched, false_positive_count


def read_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_predictions(
    rows: list[dict],
    expected_query_ids: set[str],
    corpus_names: set[str],
) -> None:
    seen = set()
    for line_number, row in enumerate(rows, start=1):
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"prediction row {line_number} has an invalid query_id")
        if query_id in seen:
            raise ValueError(f"duplicate prediction for query_id {query_id}")
        seen.add(query_id)

        pred_names = row.get("pred_names")
        if not isinstance(pred_names, list) or any(not isinstance(name, str) for name in pred_names):
            raise ValueError(f"prediction row {line_number} has invalid pred_names")
        if len(pred_names) != len(set(pred_names)):
            raise ValueError(f"prediction row {line_number} contains duplicate pred_names")
        unknown_names = set(pred_names) - corpus_names
        if unknown_names:
            raise ValueError(
                f"prediction row {line_number} contains an unknown corpus name: "
                f"{sorted(unknown_names)[0]}"
            )

        unmapped_count = row.get("unmapped_count", 0)
        if not isinstance(unmapped_count, int) or isinstance(unmapped_count, bool) or unmapped_count < 0:
            raise ValueError(f"prediction row {line_number} has invalid unmapped_count")

    missing = expected_query_ids - seen
    extra = seen - expected_query_ids
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} queries; first: {sorted(missing)[0]}")
        if extra:
            details.append(f"unexpected {len(extra)} queries; first: {sorted(extra)[0]}")
        raise ValueError("prediction/query mismatch: " + "; ".join(details))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def generate_predictions(
    queries: list[dict],
    corpus: list[dict],
    model: str,
    workers: int,
    sanitized_output: Path,
    raw_output: Path | None,
) -> list[dict]:
    try:
        import dashscope
    except ImportError as error:
        raise SystemExit("Live generation requires the dashscope package.") from error
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY in the local environment before using --generate.")
    dashscope.api_key = api_key
    name_set = {record["title"] for record in corpus}
    normalized_names = {normalize_name(name): name for name in sorted(name_set)}

    def call(query: dict) -> tuple[dict, dict]:
        last_error = None
        for attempt in range(3):
            try:
                response = dashscope.Generation.call(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query["question"]},
                    ],
                    result_format="message",
                    max_tokens=4000,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"DashScope error {response.status_code}: {response.message}")
                raw = response.output.choices[0].message.content
                matched, unmapped_count = parse_names(raw, name_set, normalized_names)
                sanitized = {
                    "query_id": query["id"],
                    "pred_names": sorted(matched),
                    "unmapped_count": unmapped_count,
                }
                return sanitized, {**sanitized, "raw_answer": raw}
            except Exception as error:
                last_error = error
                if attempt < 2:
                    time.sleep(2**attempt)
        raise RuntimeError(f"failed query {query['id']}: {last_error}")

    sanitized_rows = []
    raw_rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, query): query["id"] for query in queries}
        for completed, future in enumerate(as_completed(futures), start=1):
            sanitized, raw = future.result()
            sanitized_rows.append(sanitized)
            raw_rows.append(raw)
            if completed % 100 == 0 or completed == len(futures):
                print(f"generated {completed}/{len(futures)}")
    sanitized_rows.sort(key=lambda item: item["query_id"])
    raw_rows.sort(key=lambda item: item["query_id"])
    write_jsonl(sanitized_output, sanitized_rows)
    if raw_output is not None:
        write_jsonl(raw_output, raw_rows)
    return sanitized_rows


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description="Score or generate the Direct Qwen-Plus baseline.")
    parser.add_argument("--dataset-root", type=Path, default=script.parents[2] / "ChinesePharmaQA")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=script.parent / "results" / "qwen_predictions.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script.parent / "results" / "qwen_results.json",
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow --generate to replace an existing prediction or raw-output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.raw_output is not None and not args.generate:
        raise SystemExit("--raw-output is only valid together with --generate")
    named_paths = [("predictions", args.predictions), ("output", args.output)]
    if args.raw_output is not None:
        named_paths.append(("raw-output", args.raw_output))
    for index, (left_name, left_path) in enumerate(named_paths):
        for right_name, right_path in named_paths[index + 1 :]:
            if left_path.resolve() == right_path.resolve():
                raise SystemExit(f"--{left_name} and --{right_name} must use different paths")

    corpus, queries = load_release(args.dataset_root)
    if args.generate:
        protected_paths = [args.predictions, args.output]
        if args.raw_output is not None:
            protected_paths.append(args.raw_output)
        existing_paths = [path for path in protected_paths if path.exists()]
        if existing_paths and not args.overwrite:
            joined = ", ".join(str(path) for path in existing_paths)
            raise SystemExit(
                "Refusing to overwrite existing generation output: "
                f"{joined}. Choose new paths or pass --overwrite explicitly."
            )
        prediction_rows = generate_predictions(
            queries,
            corpus,
            args.model,
            args.workers,
            args.predictions,
            args.raw_output,
        )
    else:
        prediction_rows = read_predictions(args.predictions)

    validate_predictions(
        prediction_rows,
        {query["id"] for query in queries},
        {record["title"] for record in corpus},
    )
    prediction_map = {row["query_id"]: row for row in prediction_rows}

    items = []
    for query in queries:
        prediction = prediction_map[query["id"]]
        values = metric(
            prediction.get("pred_names", []),
            query["answer"],
            int(prediction.get("unmapped_count", 0)),
        )
        items.append(
            {
                "id": query["id"],
                "type": query["type"],
                "gold_count": len(query["answer"]),
                "mapped_count": len(set(prediction.get("pred_names", []))),
                "unmapped_count": int(prediction.get("unmapped_count", 0)),
                **values,
            }
        )
    overall = aggregate(items)
    by_type = aggregate_by_type(items)
    write_json(
        args.output,
        {
            "model": args.model,
            "n_queries": len(items),
            "overall": {key: overall[key] for key in ("precision", "recall", "f1")},
            "by_type": {
                query_type: {
                    "n": result["n"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1": result["f1"],
                }
                for query_type, result in by_type.items()
            },
            "per_item": items,
        },
    )
    print({key: overall[key] for key in ("precision", "recall", "f1")})
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
