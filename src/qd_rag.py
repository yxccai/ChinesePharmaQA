from __future__ import annotations
import json, math, hashlib, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import copy
from methods.llm_parser import LLMParser, FIELD_DEFINITIONS
def validate_mentions(question: str, mentions: list[dict]) -> list[dict]:
    if not isinstance(mentions, list) or not mentions:
        raise ValueError("Missing model-generated condition plan")
    if any(not isinstance(item, dict) or set(item) != {"field", "text"}
           or item["field"] not in FIELD_DEFINITIONS
           or not isinstance(item["text"], str) or not item["text"]
           or item["text"] not in question for item in mentions):
        raise ValueError("Condition plan must contain only original question spans")
    if len({(item["field"], item["text"]) for item in mentions}) != len(mentions):
        raise ValueError("Duplicate condition plan entry")
    return [{"field": item["field"], "text": item["text"]} for item in mentions]

class EvidencePromptParser(LLMParser):
    PROMPT_VERSION = "raw-rag-parser-v1"
    MAX_PASSAGE_CHARACTERS = 256

    def __init__(self, *, model: str, output_root: Path, evidence: RawEvidenceResources,
                 top_k: int = 5, temperature: float = 0.0, top_p: float = 1.0):
        super().__init__(model=model, output_root=output_root,
                         temperature=temperature, top_p=top_p)
        if top_k != 10:
            raise ValueError("RAG+QD uses ten passages")
        self.evidence = evidence
        self.top_k = top_k
        self.base_system_prompt = self.system_prompt


    def prepare(self, question: str, passages: list[dict] | None = None) -> tuple[LLMParser, list[str]]:
        selected = passages
        if selected is None:
            raise ValueError("Supply the decomposed retrieval passages")
        if len(selected) != self.top_k:
            raise ValueError("Passage count differs from configuration")
        payload = []
        for row in selected:
            if set(row) != {"evidence_id", "corpus_id", "drug_name", "field", "text"}:
                raise ValueError("Only corpus-derived evidence fields are allowed")
            if len(row["text"]) > self.MAX_PASSAGE_CHARACTERS:
                raise ValueError("Passage length exceeds fixed budget")
            payload.append({"source_section": row["field"], "drug_name": row["drug_name"], "text": row["text"]})
        parser = copy.copy(self)
        parser.system_prompt = self.base_system_prompt + """

以下为依据当前问题检索的原始药典片段，可用于理解问题表述和规范术语。
它们不是问题的完整答案，也不是合法规范值词表。请根据原问题识别全部明确条件，药典片段仅供参考。
不要把片段中出现但问题未要求的条件加入输出，也不要因片段未覆盖某个条件而漏掉它。
输出仍为前述 constraints JSON，不返回药品名称或解释。

原始药典片段：
""" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return parser, [row["evidence_id"] for row in selected]


class QDRAGLLMParser(EvidencePromptParser):
    PROMPT_VERSION = "qd-rag-parser-v1"

    def retrieve_decomposed(self, question: str, mentions: list[dict]) -> list[dict]:
        mentions = validate_mentions(question, mentions)
        queries = [question] + [f"{FIELD_DEFINITIONS[item['field']]}：{item['text']}"
                                for item in mentions]
        vectors = self.evidence.model.encode(queries, normalize_embeddings=True,
                                            convert_to_numpy=True, show_progress_bar=False)
        similarities = np.asarray(self.evidence.embeddings) @ vectors.astype("float32").T
        candidates = set()
        for column in range(len(queries)):
            ranking = sorted(range(len(self.evidence.rows)), key=lambda position: (
                -float(similarities[position, column]),
                self.evidence.rows[position]["evidence_id"]))
            unique = set()
            for position in ranking:
                row = self.evidence.rows[position]
                key = (row["field"], row["text"][:self.MAX_PASSAGE_CHARACTERS])
                if key in unique:
                    continue
                unique.add(key)
                candidates.add(position)
                if len(unique) == self.top_k:
                    break
        # Published RAG+QD variant: merge each query's top-k and rank the pool
        # by retrieval similarity. Keep the maximum score for duplicate rows.
        ranking = sorted(candidates, key=lambda position: (
            -float(np.max(similarities[position])), self.evidence.rows[position]["evidence_id"]))
        selected, seen = [], set()
        for position in ranking:
            row = self.evidence.rows[position]
            if set(row) != {"evidence_id", "corpus_id", "drug_name", "field", "text"}:
                raise ValueError("Unexpected raw evidence fields")
            text = row["text"][:self.MAX_PASSAGE_CHARACTERS]
            key = (row["field"], text)
            if key in seen:
                continue
            seen.add(key)
            selected.append({**row, "text": text})
            if len(selected) == self.top_k:
                break
        if len(selected) != self.top_k:
            raise ValueError("Insufficient unique merged passages")
        return selected
