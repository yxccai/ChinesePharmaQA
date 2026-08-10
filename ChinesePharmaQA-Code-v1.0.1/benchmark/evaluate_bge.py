from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from common import aggregate, aggregate_by_type, load_eval_corpus, load_release, metric, sha256_file, write_json


K_VALUES = (10, 50, 100, 200, 500, 1000)
DEFAULT_REVISION = "79e7739b6ab944e86d6171e44d24c997fc1e0116"


def hash_texts(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def hash_local_model_manifest(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for item in sorted((item for item in path.rglob("*") if item.is_file()), key=str):
        stat = item.stat()
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description="Reproduce the BGE retrieval baseline.")
    parser.add_argument("--dataset-root", type=Path, default=script.parents[2] / "ChinesePharmaQA")
    parser.add_argument(
        "--eval-corpus",
        type=Path,
        default=script.parent / "resources" / "eval_corpus.json",
    )
    parser.add_argument("--model", default="BAAI/bge-large-zh-v1.5")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=script.parent / "cache")
    parser.add_argument(
        "--output",
        type=Path,
        default=script.parent / "results" / "bge_results.json",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import faiss
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "BGE evaluation requires numpy and faiss-cpu. Encoding also requires "
            "torch and sentence-transformers."
        ) from error

    public_corpus, queries = load_release(args.dataset_root)
    records = load_eval_corpus(args.eval_corpus)
    public_names = {record["title"] for record in public_corpus}
    eval_names = {record["名称"] for record in records}
    if public_names != eval_names:
        raise ValueError("public corpus and frozen evaluation view do not contain the same names")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    document_cache = args.cache_dir / "bge_documents.npy"
    query_cache = args.cache_dir / "bge_queries.npy"
    metadata_cache = args.cache_dir / "bge_cache_metadata.json"
    resolved_revision = None if Path(args.model).exists() else args.revision
    documents = [json.dumps(record, ensure_ascii=False) for record in records]
    query_texts = [query["question"] for query in queries]
    expected_metadata = {
        "format_version": 1,
        "model": str(args.model),
        "revision": resolved_revision,
        "local_model_manifest_sha256": hash_local_model_manifest(Path(args.model)),
        "eval_corpus_sha256": sha256_file(args.eval_corpus),
        "document_text_sha256": hash_texts(documents),
        "query_text_sha256": hash_texts(query_texts),
        "n_records": len(records),
        "n_queries": len(queries),
    }
    cached_metadata = None
    if metadata_cache.exists():
        try:
            cached_metadata = json.loads(metadata_cache.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached_metadata = None
    cache_valid = (
        not args.rebuild_cache
        and document_cache.exists()
        and query_cache.exists()
        and cached_metadata == expected_metadata
    )

    if cache_valid:
        document_vectors = np.load(document_cache)
        query_vectors = np.load(query_cache)
        print(f"loaded validated embedding cache from {args.cache_dir}")
    else:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise SystemExit(
                "Encoding requires torch and sentence-transformers. Install a "
                "hardware-appropriate PyTorch build, then install requirements.txt."
            ) from error
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"device": device}
        if resolved_revision:
            model_kwargs["revision"] = resolved_revision
        model = SentenceTransformer(args.model, **model_kwargs)

        def encode(texts: list[str]):
            return model.encode(
                texts,
                batch_size=args.batch_size if device == "cuda" else min(args.batch_size, 8),
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")

        document_vectors = encode(documents)
        query_vectors = encode(query_texts)
        np.save(document_cache, document_vectors)
        np.save(query_cache, query_vectors)
        write_json(metadata_cache, expected_metadata)

    if len(document_vectors) != len(records) or len(query_vectors) != len(queries):
        raise ValueError("embedding cache shape does not match the current corpus and queries")
    index = faiss.IndexFlatIP(document_vectors.shape[1])
    index.add(document_vectors)
    _, retrieved = index.search(query_vectors, max(K_VALUES))
    names = [record["名称"] for record in records]

    summary = {}
    per_item = {}
    for k in K_VALUES:
        items = []
        for query, row in zip(queries, retrieved):
            prediction = [names[int(index_value)] for index_value in row[:k] if int(index_value) >= 0]
            values = metric(prediction, query["answer"])
            items.append(
                {
                    "id": query["id"],
                    "type": query["type"],
                    "k": k,
                    "gold_count": len(query["answer"]),
                    **values,
                }
            )
        overall = aggregate(items)
        by_type = aggregate_by_type(items)
        summary[f"K={k}"] = {
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
        }
        per_item[str(k)] = items
        print(f"K={k}: {summary[f'K={k}']['overall']}")

    write_json(
        args.output,
        {
            "description": "Dense retrieval on the frozen 5,138-record evaluation view",
            "model": args.model,
            "revision": resolved_revision,
            "n_queries": len(queries),
            "n_records": len(records),
            "eval_corpus_sha256": sha256_file(args.eval_corpus),
            "k_values": list(K_VALUES),
            "summary": summary,
            "per_item": per_item,
        },
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
