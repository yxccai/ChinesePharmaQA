from __future__ import annotations
import json, math, hashlib, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
FIELD_DESCRIPTIONS = {
    "reagent": "药典鉴别、检查、含量测定或浸出物中使用的检验试剂或对照物",
    "instrument": "药典检验仪器、分析方法、检测方法或通则方法",
    "property": "药品性状字段中的颜色、外观或形态",
    "volume": "中国药典一部或二部",
}

SOURCE_FIELD_DESCRIPTIONS = {
    "identification": "鉴别",
    "tests": "检查",
    "assay": "含量测定",
    "extractives": "浸出物",
    "description": "性状",
    "general_rule": "通则引用",
    "volume": "药典分部",
}

RETRIEVAL_FIELDS = {
    "reagent": ("identification", "tests", "assay", "extractives"),
    "instrument": ("identification", "tests", "assay", "extractives", "general_rule"),
    "property": ("description",),
    "volume": ("volume",),
}

MODEL_REVISION = "79e7739b6ab944e86d6171e44d24c997fc1e0116"

def evidence_representation(row: dict) -> str:
    """Encode only raw-index fields; no ontology or query-specific metadata."""
    source_field = SOURCE_FIELD_DESCRIPTIONS[row["field"]]
    return f"药典章节：{source_field}；药品：{row['drug_name']}；药典原文：{row['text']}"

def query_representation(
    *, question: str, mention: str, field: str, feedback: str | None = None
) -> str:
    parts = [
        f"字段：{FIELD_DESCRIPTIONS[field]}",
        f"用户表述：{mention}",
        f"问题上下文：{question}",
    ]
    if feedback:
        parts.append(f"重新检索提示：{feedback}")
    return "；".join(parts)


class EvidenceResources:
    """BGE retrieval over corpus-derived field records."""
    def __init__(self,root,model_path=None,device='cpu',global_fields=False):
        from build_indexes import load_encoder
        self.rows=[json.loads(s) for s in (root/'cache/evidence.jsonl').read_text(encoding='utf-8').splitlines()]
        meta=json.loads((root/'cache/evidence.meta.json').read_text(encoding='utf-8'))
        if hashlib.sha256((root/'cache/evidence.jsonl').read_bytes()).hexdigest()!=meta['index_sha256']:
            raise ValueError('Evidence index does not match its embeddings')
        self.embeddings=np.load(root/'cache/evidence.npy',mmap_mode='r')
        if len(self.embeddings)!=len(self.rows): raise ValueError('Evidence embedding row count differs')
        self.model=load_encoder(model_path,device,256)
        self.global_fields=global_fields
        self.positions_by_field=defaultdict(list)
        for i,row in enumerate(self.rows): self.positions_by_field[row['field']].append(i)

    def encode_one(self,text):
        return self.model.encode([text],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)[0].astype('float32')

    def retrieve(self,*,question,mention,field,top_k=5,feedback=None):
        if field not in RETRIEVAL_FIELDS: raise ValueError(f'Unknown field: {field}')
        positions=list(range(len(self.rows))) if self.global_fields else [p for f in RETRIEVAL_FIELDS[field] for p in self.positions_by_field[f]]
        if not positions:return []
        scores=np.asarray(self.embeddings[positions])@self.encode_one(query_representation(question=question,mention=mention,field=field,feedback=feedback))
        ranked=sorted(zip(positions,scores.tolist()),key=lambda x:(-float(x[1]),self.rows[x[0]]['evidence_id']))
        if field=='volume':ranked.sort(key=lambda x:(0 if self.rows[x[0]]['text']==mention.strip() else 1,-float(x[1]),self.rows[x[0]]['evidence_id']))
        return [{**self.rows[p],'score':round(float(s),8),'rank':rank} for rank,(p,s) in enumerate(ranked[:top_k],1)]
