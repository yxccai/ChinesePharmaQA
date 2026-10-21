from __future__ import annotations
import json, math, hashlib, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from data_utils import SECTION_MAP, write_jsonl, sha256_file, load_normalized_corpus
def query_tokens(text: str) -> list[str]:
    normalized = "".join(str(text).lower().split())
    chars = [char for char in normalized if "\u4e00" <= char <= "\u9fff" or char.isalnum()]
    return chars + [chars[index] + chars[index + 1] for index in range(len(chars) - 1)]

def chunk_text(text: str, max_length: int = 256, overlap: int = 64) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    if len(text) <= max_length:
        return [text]
    stride = max_length - overlap
    return [text[start:start + max_length] for start in range(0, len(text), stride) if text[start:start + max_length]]

def build_formal_retrieval_corpus() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    corpus_ids: list[str] = []
    with CORPUS_PATH.open(encoding="utf-8") as stream:
        for record in map(json.loads, stream):
            corpus_id = str(record["_id"])
            title = str(record.get("title", ""))
            corpus_ids.append(corpus_id)
            metadata = record.get("metadata") or {}
            sections = metadata.get("sections") or {}
            for normalized, source_name in SECTION_MAP.items():
                section_text = sections.get(source_name, "")
                for index, chunk in enumerate(chunk_text(section_text)):
                    rows.append(
                        {
                            "chunk_id": f"{corpus_id}:{normalized}:{index:04d}",
                            "corpus_id": corpus_id,
                            "drug_title": title,
                            "section": normalized,
                            "section_name": source_name,
                            "chunk_index": index,
                            "chunk_text": chunk,
                            "text": f"药品：{title}；章节：{source_name}；药典原文：{chunk}",
                        }
                    )
            volume = str(metadata.get("volume", "")).strip()
            if volume:
                rows.append(
                    {
                        "chunk_id": f"{corpus_id}:volume:0000",
                        "corpus_id": corpus_id,
                        "drug_title": title,
                        "section": "volume",
                        "section_name": "药典分部",
                        "chunk_index": 0,
                        "chunk_text": volume,
                        "text": f"药品：{title}；章节：药典分部；药典原文：{volume}",
                    }
                )
    if len(set(corpus_ids)) != 5138:
        raise RuntimeError(f"corpus ID count expected 5138, got {len(set(corpus_ids))}")
    write_jsonl(RETRIEVAL_CORPUS_PATH, rows)
    metadata = {
        "status": "FROZEN_FORMAL_RETRIEVAL_INDEX",
        "source": "corpus.jsonl only",
        "source_corpus_sha256": sha256_file(CORPUS_PATH),
        "corpus_entries": len(set(corpus_ids)),
        "chunk_count": len(rows),
        "chunking": {"max_chunk_length_chinese_characters": 256, "overlap": 64, "stride": 192},
        "representation_fields": ["drug_title", "section_name", "chunk_text"],
        "gold_independent": True,
        "ontology_used": False,
        "alias_resource_used": False,
        "formal_retrieval_corpus_sha256": sha256_file(RETRIEVAL_CORPUS_PATH),
    }
    return rows, metadata

class BM25Index:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        for index, row in enumerate(rows):
            counts = Counter(query_tokens(row["text"]))
            self.doc_lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                self.postings[token].append((index, frequency))
                document_frequency[token] += 1
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1.0
        count = len(rows)
        self.idf = {token: math.log(1.0 + (count - df + 0.5) / (df + 0.5)) for token, df in document_frequency.items()}
        self.k1 = 1.2
        self.b = 0.75

    def scores(self, question: str) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        query_counts = Counter(query_tokens(question))
        for token, qf in query_counts.items():
            idf = self.idf.get(token)
            if idf is None:
                continue
            for doc_index, tf in self.postings[token]:
                length = self.doc_lengths[doc_index]
                denom = tf + self.k1 * (1.0 - self.b + self.b * length / self.avgdl)
                scores[doc_index] += idf * (tf * (self.k1 + 1.0) / denom) * min(qf, 2)
        return scores

class RetrievalEngine:
    def __init__(self, rows: list[dict[str, Any]], embedding_path: Path, model_path: Path, device: str):
        self.rows = rows
        self.corpus_ids = sorted({row["corpus_id"] for row in rows})
        self.titles = {row["corpus_id"]: row["drug_title"] for row in rows}
        self.bm25 = BM25Index(rows)
        self.embedding_path = embedding_path
        self.model_path = model_path
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
        self.model.max_seq_length = 128
        self.embeddings = np.load(embedding_path, mmap_mode="r")
        if self.embeddings.shape[0] != len(rows):
            raise RuntimeError("formal retrieval embedding row mismatch")

    def rank_bm25(self, question: str, top_k: int = 600) -> list[dict[str, Any]]:
        scores = self.bm25.scores(question)
        best: dict[str, tuple[float, int]] = {}
        for index, score in scores.items():
            cid = self.rows[index]["corpus_id"]
            if cid not in best or score > best[cid][0] or (score == best[cid][0] and index < best[cid][1]):
                best[cid] = (float(score), index)
        ranked = [(best.get(cid, (0.0, -1))[0], cid, best.get(cid, (0.0, -1))[1]) for cid in self.corpus_ids]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [{"corpus_id": cid, "drug_title": self.titles[cid], "score": score, "chunk_index": index} for score, cid, index in ranked[:top_k]]

    def encode_query(self, question: str) -> np.ndarray:
        return self.model.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].astype("float32")

    def rank_bge(self, question: str, top_k: int = 600) -> list[dict[str, Any]]:
        vector = self.encode_query(question)
        scores = self.embeddings @ vector
        best: dict[str, tuple[float, int]] = {}
        for index, score in enumerate(scores.tolist()):
            cid = self.rows[index]["corpus_id"]
            score = float(score)
            if cid not in best or score > best[cid][0] or (score == best[cid][0] and index < best[cid][1]):
                best[cid] = (score, index)
        ranked = [(best.get(cid, (-1.0, -1))[0], cid, best.get(cid, (-1.0, -1))[1]) for cid in self.corpus_ids]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [{"corpus_id": cid, "drug_title": self.titles[cid], "score": score, "chunk_index": index} for score, cid, index in ranked[:top_k]]

    def rank_bge_many(self, questions: list[str], top_k: int = 600) -> list[list[dict[str, Any]]]:
        """Batch Dev query scoring; no Test text is accepted by the caller."""
        vectors = self.model.encode(questions, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        try:
            import torch
            device = next(self.model.parameters()).device
            if device.type == "cuda" and torch.cuda.is_available():
                emb = torch.from_numpy(np.asarray(self.embeddings)).to(device=device, dtype=torch.float32)
                scores_matrix = (torch.from_numpy(vectors).to(device=device, dtype=torch.float32) @ emb.T).detach().cpu().numpy()
                del emb
                torch.cuda.empty_cache()
            else:
                scores_matrix = vectors @ np.asarray(self.embeddings).T
        except Exception:
            scores_matrix = vectors @ np.asarray(self.embeddings).T
        output: list[list[dict[str, Any]]] = []
        for scores in scores_matrix:
            best: dict[str, tuple[float, int]] = {}
            for index, score in enumerate(scores.tolist()):
                cid = self.rows[index]["corpus_id"]
                score = float(score)
                if cid not in best or score > best[cid][0] or (score == best[cid][0] and index < best[cid][1]):
                    best[cid] = (score, index)
            ranked = [(best.get(cid, (-1.0, -1))[0], cid, best.get(cid, (-1.0, -1))[1]) for cid in self.corpus_ids]
            ranked.sort(key=lambda item: (-item[0], item[1]))
            output.append([{"corpus_id": cid, "drug_title": self.titles[cid], "score": score, "chunk_index": index} for score, cid, index in ranked[:top_k]])
        return output

    def top_chunk_for(self, ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in ranking:
            index = int(item["chunk_index"])
            chunk = self.rows[index]
            output.append({**item, "section": chunk["section"], "section_name": chunk["section_name"], "evidence": chunk["text"]})
        return output

SOURCE_FIELDS = {
    "description": ("description",),
    "identification": ("identification",),
    "tests": ("tests",),
    "assay": ("assay",),
    "extractives": ("extractives",),
}

def raw_evidence_id(corpus_id: str, field: str, text: str) -> str:
    digest = hashlib.sha256("\x1f".join((corpus_id, field, text)).encode("utf-8")).hexdigest()[:20]
    return f"raw-evidence::{digest}"

def build_rows(dataset_root: Path) -> tuple[list[dict], dict]:
    corpus = load_normalized_corpus(dataset_root)
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    before_dedup = Counter()
    for record in corpus:
        corpus_id = record["id"]
        drug_name = record["name"]
        for field, source_fields in SOURCE_FIELDS.items():
            for source_field in source_fields:
                text = str(record[source_field]).strip()
                if not text:
                    continue
                before_dedup[field] += 1
                key = (corpus_id, field, text)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "evidence_id": raw_evidence_id(corpus_id, field, text),
                        "corpus_id": corpus_id,
                        "drug_name": drug_name,
                        "field": field,
                        "text": text,
                    }
                )

        for rule in record["general_rules"]:
            text = " ".join(part for part in (rule["number"], rule["name"]) if part).strip()
            if not text:
                continue
            before_dedup["general_rule"] += 1
            key = (corpus_id, "general_rule", text)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "evidence_id": raw_evidence_id(corpus_id, "general_rule", text),
                    "corpus_id": corpus_id,
                    "drug_name": drug_name,
                    "field": "general_rule",
                    "text": text,
                }
            )

        volume = str(record["volume"]).strip()
        if volume:
            before_dedup["volume"] += 1
            key = (corpus_id, "volume", volume)
            if key not in seen:
                seen.add(key)
                rows.append(
                    {
                        "evidence_id": raw_evidence_id(corpus_id, "volume", volume),
                        "corpus_id": corpus_id,
                        "drug_name": drug_name,
                        "field": "volume",
                        "text": volume,
                    }
                )

    return rows, {
        "corpus_records": len(corpus),
        "records_before_dedup": dict(sorted(before_dedup.items())),
        "records_after_dedup": len(rows),
    }
