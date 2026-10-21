from __future__ import annotations
import json, math, hashlib, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from llm_client import qwen_json_call
from methods.llm_parser import FIELD_DEFINITIONS, LLMParser
from agents.evidence_canonicalizer import FIRST_PROMPT, FINAL_PROMPT, EvidenceCanonicalizer
from structured_executor import StructuredExecutor
MODEL_NAME="qwen3.7-flash"
TEMPERATURE=0.0
TOP_P=1.0
RETRIES=3
RAG_BATCH=50
EVIDENCE_TOP_K=5
MAX_REGROUND=1

def numeric_usage(x):
    return {k:v for k,v in (x or {}).items() if isinstance(v,(int,float))}

def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()

def request_sha(*, model: str, system_prompt: str, user_payload: dict[str, Any], max_tokens: int) -> str:
    """Hash the exact qwen3.7-compatible request body (without API key)."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    return sha_bytes(json.dumps(body, ensure_ascii=False).encode("utf-8"))

def assert_retry_hashes(records: list[dict[str, Any]]) -> None:
    for record in records:
        attempts = int(record.get("attempts", 0))
        retry_count = int(record.get("retry_count", max(0, attempts - 1)))
        retry_sha = record.get("retry_request_sha256")
        if attempts > 1 or retry_count > 0:
            if retry_sha != record.get("serialized_request_sha256"):
                raise AssertionError(f"retry payload hash mismatch for {record.get('query_id')}")

def call_qwen_audited(
    *, query_id: str, stage: str, system_prompt: str, user_payload: dict[str, Any], max_tokens: int,
) -> tuple[dict[str, Any] | None, str, dict[str, int], dict[str, Any], str | None]:
    """Call the frozen endpoint and record request/response/retry hashes."""
    req = request_sha(model=MODEL_NAME, system_prompt=system_prompt, user_payload=user_payload, max_tokens=max_tokens)
    record: dict[str, Any] = {
        "query_id": query_id,
        "stage": stage,
        "model": MODEL_NAME,
        "prompt_version": stage,
        "payload_keys": sorted(user_payload),
        "serialized_request_sha256": req,
        "request_sha256": req,
        "retry_request_sha256": None,
        "response_sha256": None,
        "attempts": 0,
        "retry_count": 0,
        "status": "NOT_STARTED",
    }
    try:
        value, raw, usage = qwen_json_call(
            model=MODEL_NAME,
            system_prompt=system_prompt,
            user_payload=user_payload,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=max_tokens,
            retries=RETRIES,
        )
        audit = usage.get("_api_audit", {}) if isinstance(usage, dict) else {}
        attempts = int(audit.get("attempts", 1))
        record.update({
            "status": "SUCCESS",
            "attempts": attempts,
            "retry_count": max(0, attempts - 1),
            "retry_request_sha256": req if attempts > 1 else None,
            "response_sha256": sha_bytes(str(raw).encode("utf-8")) if raw else None,
            "response_present": bool(raw),
            "interface": audit.get("interface", "DashScope OpenAI-compatible /chat/completions"),
            "attempt_errors": list(audit.get("attempt_errors", [])),
        })
        assert_retry_hashes([record])
        return value, str(raw), numeric_usage(usage), record, None
    except Exception as exc:
        audit = getattr(exc, "audit", {})
        if not isinstance(audit, dict):
            audit = {}
        attempts = int(audit.get("attempts", 1))
        record.update({
            "status": "ERROR",
            "attempts": attempts,
            "retry_count": max(0, attempts - 1),
            "retry_request_sha256": req if attempts > 1 else None,
            "interface": audit.get("interface", "DashScope OpenAI-compatible /chat/completions"),
            "attempt_errors": list(audit.get("attempt_errors", [f"{type(exc).__name__}: {exc}"])),
            "error": str(exc),
        })
        assert_retry_hashes([record])
        return None, "", {}, record, str(exc)

DIRECT_PROMPT = """你是中国药典药品集合查询的直接回答器。只根据自然问题和你的语言知识，直接输出满足全部条件的药品标题集合。不要解释，不要输出不确定内容，不要使用工具、药典原文、候选列表、ontology、alias、Gold 或 qrels。只能输出 JSON：{\"drugs\":[\"药品标题1\",\"药品标题2\"]}。\n自然问题中的全部条件必须同时满足。"""

RAG_PROMPT = """你是一个候选药品过滤器。给定一个自然问题和一批由检索系统提供的候选药品；每个候选含 corpus_id、药品标题和一段原始药典证据。判断哪些候选药品同时满足自然问题全部显式条件。只能从 supplied candidates 中选择，不能发明或扩展候选，不能使用 Gold、ontology、alias、qrels、posting 或外部工具。只输出 JSON：{\"selected_corpus_ids\":[\"supplied-id\"]}。"""

def direct_predict(row: dict[str, str], title_to_ids: dict[str, set[str]]) -> dict[str, Any]:
    value, raw, usage, audit, error = call_qwen_audited(
        query_id=row["_id"], stage="direct-llm-v1", system_prompt=DIRECT_PROMPT,
        user_payload={"question": row["text"]}, max_tokens=1200,
    )
    if error or value is None:
        return {"status": "API_FAILURE", "answer_ids": [], "llm_calls": 1, "physical_api_attempts": audit.get("attempts", 1), "usage": usage, "api_calls": [audit], "error": error}
    titles = value.get("drugs")
    if not isinstance(titles, list):
        return {"status": "MALFORMED_OUTPUT", "answer_ids": [], "llm_calls": 1, "physical_api_attempts": audit.get("attempts", 1), "usage": usage, "api_calls": [audit], "error": "drugs must be a list"}
    answer: set[str] = set(); invalid: list[str] = []
    for title in titles:
        ids = title_to_ids.get(str(title))
        if ids:
            answer.update(ids)
        else:
            invalid.append(str(title))
    return {"status": "OK" if not invalid else "INVALID_TITLE", "answer_ids": sorted(answer), "invalid_titles": invalid[:50], "invalid_title_count": len(invalid), "llm_calls": 1, "physical_api_attempts": audit.get("attempts", 1), "usage": usage, "api_calls": [audit], "error": None}

def rag_predict(row: dict[str, str], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    selected: set[str] = set(); calls: list[dict[str, Any]] = []; usage_totals: Counter[str] = Counter(); errors: list[str] = []; invalid = 0
    batches = [candidates[i:i + RAG_BATCH] for i in range(0, len(candidates), RAG_BATCH)]

    def one(batch: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {"question": row["text"], "candidates": [{"corpus_id": x["corpus_id"], "drug_title": x["drug_title"], "evidence": x["evidence"]} for x in batch]}
        value, raw, usage, audit, error = call_qwen_audited(query_id=row["_id"], stage="rag-filter-v1", system_prompt=RAG_PROMPT, user_payload=payload, max_tokens=1800)
        supplied = {x["corpus_id"] for x in batch}
        selected_batch: list[str] = []
        invalid_batch = 0
        if error or value is None:
            return {"selected": [], "audit": audit, "usage": usage, "error": error or "empty response", "invalid": 0}
        values = value.get("selected_corpus_ids")
        if not isinstance(values, list):
            return {"selected": [], "audit": audit, "usage": usage, "error": "selected_corpus_ids must be a list", "invalid": 0}
        for cid in values:
            cid = str(cid)
            if cid in supplied:
                selected_batch.append(cid)
            else:
                invalid_batch += 1
        return {"selected": selected_batch, "audit": audit, "usage": usage, "error": None, "invalid": invalid_batch}

    # Two inner workers are the frozen Dev execution pattern; output order is
    # restored by ``map`` and no semantic retry is performed.
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one, batches))
    for result in results:
        selected.update(result["selected"]); calls.append(result["audit"]); usage_totals.update(result["usage"]); invalid += int(result["invalid"])
        if result["error"]:
            errors.append(str(result["error"]))
    assert_retry_hashes(calls)
    return {"status": "OK" if not errors and not invalid else "FILTER_FAILURE" if errors else "INVALID_CANDIDATE_ID", "answer_ids": sorted(selected), "candidate_count": len(candidates), "invalid_candidate_id_count": invalid, "llm_calls": len(calls), "physical_api_attempts": sum(int(x.get("attempts", 0)) for x in calls), "usage": dict(usage_totals), "api_calls": calls, "error": "; ".join(errors) if errors else None}

def parser_prediction(row: dict[str, str], parser: LLMParser) -> dict[str, Any]:
    started = time.perf_counter()
    result = parser.run(row["text"])
    usage = result.get("usage", {}) if isinstance(result, dict) else {}
    req = request_sha(model=MODEL_NAME, system_prompt=parser.system_prompt, user_payload={"question": row["text"]}, max_tokens=600)
    audit_info = usage.get("_api_audit", {}) if isinstance(usage, dict) else {}
    attempts = int(audit_info.get("attempts", 1))
    raw = str(result.get("raw_response", ""))
    audit = {
        "query_id": row["_id"], "stage": "llm-parser-v2", "model": MODEL_NAME,
        "prompt_version": "llm-parser-v2", "payload_keys": ["question"],
        "serialized_request_sha256": req, "request_sha256": req,
        "retry_request_sha256": req if attempts > 1 else None,
        "response_sha256": sha_bytes(raw.encode("utf-8")) if raw else None,
        "attempts": attempts, "retry_count": max(0, attempts - 1),
        "status": "SUCCESS" if raw else "ERROR", "response_present": bool(raw),
        "interface": audit_info.get("interface", "DashScope OpenAI-compatible /chat/completions"),
        "attempt_errors": list(audit_info.get("attempt_errors", [])),
    }
    assert_retry_hashes([audit])
    return {
        "status": result.get("parse_status", "UNKNOWN"),
        "answer_ids": result.get("answer_ids", []),
        "predicted_concept_ids": result.get("concept_ids", []),
        "predicted_constraints": result.get("constraints", []),
        "invalid_constraints": result.get("invalid_constraints", []),
        "llm_calls": int(result.get("llm_calls", 1)),
        "physical_api_attempts": attempts,
        "usage": numeric_usage(usage),
        "api_calls": [audit],
        "error": result.get("error"),
        "latency_seconds": round(time.perf_counter() - started, 3),
    }

def exact_constraint(value: Any, exact_map: dict[tuple[str, str], str]) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(value, dict):
        return None, "INVALID_FIELD_OR_SCHEMA"
    field = str(value.get("field", "")).strip(); canonical = str(value.get("canonical", "")).strip()
    if field not in FIELD_DEFINITIONS or not canonical:
        return None, "INVALID_FIELD_OR_SCHEMA"
    if (field, canonical) not in exact_map:
        return {"field": field, "canonical": canonical}, "INVALID_CANONICAL"
    return {"field": field, "canonical": canonical}, None

def evidence_call(
    *, query_id: str, question: str, mention: dict[str, str], evidence: list[dict[str, Any]], final: bool, canonicalizer: EvidenceCanonicalizer,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, int], str | None]:
    payload = {"question": question, "raw_mention": mention["text"], "predicted_field": mention["field"], "evidence": [{"corpus_id": x["corpus_id"], "drug_name": x["drug_name"], "field": x["field"], "text": x["text"], "score": x["score"]} for x in evidence]}
    prompt = FINAL_PROMPT if final else FIRST_PROMPT
    req = request_sha(model=MODEL_NAME, system_prompt=prompt, user_payload=payload, max_tokens=500)
    try:
        decision, log = canonicalizer.run(question=question, mention=mention, evidence=evidence, final=final)
        usage = dict(log.get("usage") or {}); audit_info = usage.get("_api_audit", {}) if isinstance(usage, dict) else {}; attempts = int(audit_info.get("attempts", 1)); raw = str(log.get("raw", ""))
        record = {"query_id": query_id, "stage": "final_evidence_canonicalize" if final else "evidence_canonicalize", "model": MODEL_NAME, "prompt_version": EvidenceCanonicalizer.PROMPT_VERSION, "payload_keys": sorted(payload), "serialized_request_sha256": req, "request_sha256": req, "retry_request_sha256": req if attempts > 1 else None, "response_sha256": sha_bytes(raw.encode("utf-8")) if raw else None, "attempts": attempts, "retry_count": max(0, attempts - 1), "status": "SUCCESS", "response_present": bool(raw), "interface": audit_info.get("interface", "DashScope OpenAI-compatible /chat/completions"), "attempt_errors": list(audit_info.get("attempt_errors", []))}
        assert_retry_hashes([record])
        return decision, record, numeric_usage(usage), None
    except Exception as exc:
        audit_info = getattr(exc, "audit", {}) if isinstance(getattr(exc, "audit", {}), dict) else {}; attempts = int(audit_info.get("attempts", 1))
        record = {"query_id": query_id, "stage": "final_evidence_canonicalize" if final else "evidence_canonicalize", "model": MODEL_NAME, "prompt_version": EvidenceCanonicalizer.PROMPT_VERSION, "payload_keys": sorted(payload), "serialized_request_sha256": req, "request_sha256": req, "retry_request_sha256": req if attempts > 1 else None, "response_sha256": None, "attempts": attempts, "retry_count": max(0, attempts - 1), "status": "ERROR", "interface": audit_info.get("interface", "DashScope OpenAI-compatible /chat/completions"), "attempt_errors": list(audit_info.get("attempt_errors", [f"{type(exc).__name__}: {exc}"])), "error": str(exc)}
        assert_retry_hashes([record])
        return None, record, {}, str(exc)

def adaptive_prediction(
    row: dict[str, str], parser_item: dict[str, Any], exact_map: dict[tuple[str, str], str], executor: StructuredExecutor,
    evidence: RawEvidenceResources, canonicalizer: EvidenceCanonicalizer,
) -> dict[str, Any]:
    raw_constraints = parser_item.get("predicted_constraints") if isinstance(parser_item.get("predicted_constraints"), list) else []
    valid: list[dict[str, str]] = []; invalid: list[dict[str, Any]] = []
    for index, value in enumerate(raw_constraints):
        normalized, err = exact_constraint(value, exact_map)
        if err is None and normalized is not None:
            valid.append(normalized)
        else:
            field = str(value.get("field", "")).strip() if isinstance(value, dict) else ""
            canonical = str(value.get("canonical", "")).strip() if isinstance(value, dict) else ""
            invalid.append({"index": index, "field": field, "canonical": canonical, "error_type": err or "INVALID_FIELD_OR_SCHEMA"})
    # Preserve schema-invalid entries retained by LLM-Parser in its separate list.
    for item in parser_item.get("invalid_constraints", []) or []:
        if isinstance(item, dict) and not any(x["field"] == str(item.get("field", "")).strip() and x["canonical"] == str(item.get("canonical", "")).strip() for x in invalid):
            invalid.append({"index": len(raw_constraints), "field": str(item.get("field", "")).strip(), "canonical": str(item.get("canonical", "")).strip(), "error_type": str(item.get("error_type", "INVALID_CANONICAL"))})
    if not invalid:
        cids = [exact_map[(x["field"], x["canonical"])] for x in valid]
        if not cids or len(cids) != len(set(cids)):
            return {"status": "INVALID_CANONICAL", "answer_ids": [], "predicted_concept_ids": [], "predicted_constraints": [], "routing": "all_valid_direct", "all_constraints_valid": False, "evidence_triggered": False, "failed_constraint_count": 0, "repaired_constraint_count": 0, "still_invalid_constraint_count": 0, "re_ground_count": 0, "llm_calls": 0, "physical_api_attempts": 0, "usage": {}, "api_calls": [], "parser_reused": True, "error": "duplicate_or_empty_valid_constraint_set"}
        answer_ids = sorted(executor.execute_constraints(cids))
        # The parser prediction is reused byte-for-byte; this assertion guards
        # against an implementation accidentally replacing the frozen parser.
        if parser_item.get("status") == "OK" and answer_ids != sorted(parser_item.get("answer_ids", [])):
            raise AssertionError(f"Adaptive all-valid parser/executor mismatch for {row['_id']}")
        return {"status": "OK", "answer_ids": answer_ids, "predicted_concept_ids": cids, "predicted_constraints": valid, "initial_predicted_constraints": raw_constraints, "routing": "all_valid_direct", "all_constraints_valid": True, "evidence_triggered": False, "failed_constraint_count": 0, "repaired_constraint_count": 0, "still_invalid_constraint_count": 0, "re_ground_count": 0, "llm_calls": 0, "physical_api_attempts": 0, "usage": {}, "api_calls": [], "parser_reused": True, "error": None}

    repaired: dict[int, dict[str, str]] = {}; traces: list[dict[str, Any]] = []; calls: list[dict[str, Any]] = []; totals: Counter[str] = Counter(); errors: list[str] = []; regrounds = 0
    for spec in invalid:
        index = int(spec["index"]); field = str(spec["field"]); canonical = str(spec["canonical"]); mention = {"field": field, "text": canonical}; trace = {"constraint_index": index, "failed_constraint": {"field": field, "canonical": canonical}, "initial_validation": spec.get("error_type"), "mention_source": "frozen LLM-Parser predicted canonical string"}
        if field not in FIELD_DEFINITIONS or not canonical:
            trace["status"] = "UNRESOLVED_NO_TYPED_MENTION"; traces.append(trace); continue
        try:
            initial_evidence = evidence.retrieve(question=row["text"], mention=canonical, field=field, top_k=EVIDENCE_TOP_K)
        except Exception as exc:
            trace.update({"status": "UNRESOLVED_RETRIEVAL", "error": str(exc)}); traces.append(trace); errors.append(str(exc)); continue
        trace["initial_evidence"] = initial_evidence
        decision, audit, usage, error = evidence_call(query_id=row["_id"], question=row["text"], mention=mention, evidence=initial_evidence, final=False, canonicalizer=canonicalizer)
        calls.append(audit); totals.update(usage); trace["initial_call"] = audit; trace["initial_decision"] = decision
        if error:
            trace["status"] = "UNRESOLVED_API"; traces.append(trace); errors.append(error); continue
        accepted: dict[str, str] | None = None
        if str((decision or {}).get("decision", "")).upper() == "ACCEPT":
            candidate, validation = exact_constraint(decision, exact_map); trace["initial_candidate_validation"] = validation or "VALID"
            if validation is None and candidate is not None:
                accepted = candidate; trace["status"] = "REPAIRED_INITIAL"
            else:
                trace["status"] = "STILL_INVALID_INITIAL_ACCEPT"
        elif str((decision or {}).get("decision", "")).upper() == "REGROUND" and regrounds < MAX_REGROUND:
            regrounds += 1; feedback = str((decision or {}).get("feedback", "")).strip(); trace["reground_triggered"] = True; trace["feedback"] = feedback
            try:
                reground_evidence = evidence.retrieve(question=row["text"], mention=canonical, field=field, top_k=EVIDENCE_TOP_K, feedback=feedback)
            except Exception as exc:
                trace.update({"status": "UNRESOLVED_REGROUND_RETRIEVAL", "error": str(exc)}); traces.append(trace); errors.append(str(exc)); continue
            trace["reground_evidence"] = reground_evidence
            final_decision, final_audit, final_usage, final_error = evidence_call(query_id=row["_id"], question=row["text"], mention=mention, evidence=reground_evidence, final=True, canonicalizer=canonicalizer)
            calls.append(final_audit); totals.update(final_usage); trace["final_call"] = final_audit; trace["final_decision"] = final_decision
            if final_error:
                trace["status"] = "UNRESOLVED_API"; errors.append(final_error)
            elif str((final_decision or {}).get("decision", "")).upper() == "ACCEPT":
                candidate, validation = exact_constraint(final_decision, exact_map); trace["final_candidate_validation"] = validation or "VALID"
                if validation is None and candidate is not None:
                    accepted = candidate; trace["status"] = "REPAIRED_REGROUND"
                else:
                    trace["status"] = "STILL_INVALID_REGROUND"
            else:
                trace["status"] = "UNRESOLVED"
        else:
            trace["status"] = "UNRESOLVED"
        if accepted is not None:
            repaired[index] = accepted; trace["repaired_constraint"] = accepted
        traces.append(trace)
    final_constraints: list[dict[str, str]] = []
    for index, value in enumerate(raw_constraints):
        normalized, err = exact_constraint(value, exact_map)
        if err is None and normalized is not None:
            final_constraints.append(normalized)
        elif index in repaired:
            final_constraints.append(repaired[index])
    all_repaired = len(repaired) == len(invalid) and not errors
    cids: list[str] = []; answer_ids: list[str] = []; status = "OK" if all_repaired else "UNRESOLVED"
    if all_repaired:
        try:
            cids = [exact_map[(x["field"], x["canonical"])] for x in final_constraints]
            if not cids or len(cids) != len(set(cids)):
                raise ValueError("duplicate_or_empty_constraint_set")
            answer_ids = sorted(executor.execute_constraints(cids))
        except Exception as exc:
            status = "UNRESOLVED_EXECUTION"; errors.append(str(exc)); cids = []; answer_ids = []
    if status != "OK":
        cids = []; answer_ids = []
    assert_retry_hashes(calls)
    return {"status": status, "answer_ids": answer_ids, "predicted_concept_ids": cids, "predicted_constraints": final_constraints if status == "OK" else [], "initial_predicted_constraints": raw_constraints, "invalid_constraints": invalid, "canonical_outputs": valid + list(repaired.values()), "evidence_trace": traces, "routing": "evidence_fallback", "all_constraints_valid": False, "evidence_triggered": True, "failed_constraint_count": len(invalid), "repaired_constraint_count": len(repaired), "still_invalid_constraint_count": max(0, len(invalid) - len(repaired)), "re_ground_count": regrounds, "llm_calls": len(calls), "physical_api_attempts": sum(int(x.get("attempts", 0)) for x in calls), "usage": dict(totals), "api_calls": calls, "parser_reused": True, "error": "; ".join(errors) if errors else None}
