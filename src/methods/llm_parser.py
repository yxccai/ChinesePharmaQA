from __future__ import annotations

import json
from pathlib import Path

from llm_client import qwen_json_call
from structured_executor import StructuredExecutor


FIELD_DEFINITIONS = {
    "reagent": "药典鉴别、检查、含量测定或浸出物中使用的检验试剂或对照物",
    "instrument": "药典检验仪器、分析方法、检测方法或通则方法",
    "property": "药品性状字段中的颜色、外观或形态",
    "volume": "中国药典一部或二部",
}


class LLMParser:
    PROMPT_VERSION = "llm-parser-v2"

    def __init__(self, *, model: str, output_root: Path, temperature: float = 0.0, top_p: float = 1.0):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        active_path = output_root / "resources" / "ontology" / "active_concepts.json"
        ontology_path = output_root / "resources" / "ontology" / "concepts.json"
        postings_path = output_root / "resources" / "ontology" / "postings.json"
        self.concepts = json.loads(active_path.read_text(encoding="utf-8"))["concepts"]
        self.concept_id_by_constraint = {
            (item["field"], item["canonical"]): item["concept_id"] for item in self.concepts
        }
        self.executor = StructuredExecutor(postings_path, ontology_path)
        self.system_prompt = """你是药典集合查询的结构化解析器。给定自然问题，请识别问题明确表达的全部规范药典约束。

规则：
1. 完整保留全部AND约束，不回答药品名称。
2. 合法字段仅为 reagent、instrument、property、volume。
3. 输出规范的药典 canonical 字符串；不得输出concept ID、药品答案、解释或多余字段。
4. 双试剂分别输出两个 reagent constraint；不得漏掉或合并约束。
5. 只输出JSON，格式为：{"constraints":[{"field":"<合法字段>","canonical":"<规范术语>"}]}。

字段定义：
""" + json.dumps(FIELD_DEFINITIONS, ensure_ascii=False)

    def run(self, question: str) -> dict:
        try:
            value, raw, usage = qwen_json_call(
                model=self.model,
                system_prompt=self.system_prompt,
                user_payload={"question": question},
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=600,
                retries=1,
            )
        except Exception as error:
            return {
                "parse_status": "api_or_format_failure",
                "concept_ids": [],
                "constraints": [],
                "invalid_constraints": [],
                "answer_ids": [],
                "llm_calls": 1,
                "usage": getattr(error, "audit", {}),
                "raw_response": "",
                "error": str(error),
            }
        values = value.get("constraints")
        if not isinstance(values, list) or not values:
            return {"parse_status": "MALFORMED_OUTPUT", "constraints": [], "invalid_constraints": [], "concept_ids": [], "answer_ids": [], "llm_calls": 1, "usage": usage, "raw_response": raw, "error": "constraints must be a non-empty list"}
        constraints = []
        invalid = []
        for item in values:
            if not isinstance(item, dict):
                return {"parse_status": "MALFORMED_OUTPUT", "constraints": constraints, "invalid_constraints": [], "concept_ids": [], "answer_ids": [], "llm_calls": 1, "usage": usage, "raw_response": raw, "error": "constraint must be an object"}
            field = str(item.get("field", "")).strip()
            canonical = str(item.get("canonical", "")).strip()
            if field not in FIELD_DEFINITIONS or not canonical:
                invalid.append({"field": field, "canonical": canonical, "error_type": "INVALID_FIELD_OR_SCHEMA"})
            else:
                constraints.append({"field": field, "canonical": canonical})
        if len({(item["field"], item["canonical"]) for item in constraints}) != len(constraints):
            return {"parse_status": "MALFORMED_OUTPUT", "constraints": constraints, "invalid_constraints": invalid, "concept_ids": [], "answer_ids": [], "llm_calls": 1, "usage": usage, "raw_response": raw, "error": "duplicate constraint"}
        for item in constraints:
            if (item["field"], item["canonical"]) not in self.concept_id_by_constraint:
                invalid.append({**item, "error_type": "INVALID_CANONICAL"})
        if invalid:
            return {"parse_status": "INVALID_CANONICAL", "constraints": constraints, "invalid_constraints": invalid, "concept_ids": [], "answer_ids": [], "llm_calls": 1, "usage": usage, "raw_response": raw, "error": "one or more constraints failed exact ontology validation"}
        concept_ids = [self.concept_id_by_constraint[(item["field"], item["canonical"])] for item in constraints]
        answer_ids = self.executor.execute_constraints(concept_ids)
        return {
            "parse_status": "OK",
            "constraints": constraints,
            "concept_ids": concept_ids,
            "invalid_constraints": [],
            "answer_ids": sorted(answer_ids),
            "llm_calls": 1,
            "usage": usage,
            "raw_response": raw,
        }
