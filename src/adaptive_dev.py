from __future__ import annotations
import hashlib,json
from collections import Counter
from typing import Any
from predictors import numeric_usage
from agents.evidence_canonicalizer import FIRST_PROMPT,FINAL_PROMPT,EvidenceCanonicalizer
from structured_executor import StructuredExecutor
MODEL="qwen3.7-flash"
TEMPERATURE=0.0
TOP_P=1.0
EVIDENCE_TOP_K=5
MAX_REGROUND=1
PROMPT_VERSION=EvidenceCanonicalizer.PROMPT_VERSION
PROTOCOL_VERSION="adaptive-evidence"
def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()

def request_hash(
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    max_tokens: int = 500,
) -> str:
    """Hash the exact serialized qwen3.7-compatible request body.

    This mirrors the qwen branch in ``llm_client.qwen_json_call``.  The API
    key is deliberately not part of the body or this audit digest.
    """
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    return sha256_bytes(json.dumps(body, ensure_ascii=False).encode("utf-8"))

def evidence_model_payload(
    *, question: str, mention: dict[str, str], evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    # Keep this projection byte-for-byte equivalent to EvidenceCanonicalizer.
    return {
        "question": question,
        "raw_mention": mention["text"],
        "predicted_field": mention["field"],
        "evidence": [
            {
                "corpus_id": item["corpus_id"],
                "drug_name": item["drug_name"],
                "field": item["field"],
                "text": item["text"],
                "score": item["score"],
            }
            for item in evidence
        ],
    }

def numeric_usage_without_audit(usage: dict[str, Any] | None) -> dict[str, int]:
    return numeric_usage(usage or {})

def exact_constraint(
    value: Any, exact_map: dict[tuple[str, str], str]
) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(value, dict):
        return None, "INVALID_FIELD_OR_SCHEMA"
    field = str(value.get("field", "")).strip()
    canonical = str(value.get("canonical", "")).strip()
    if field not in {"reagent", "instrument", "property", "volume"} or not canonical:
        return None, "INVALID_FIELD_OR_SCHEMA"
    if (field, canonical) not in exact_map:
        return {"field": field, "canonical": canonical}, "INVALID_CANONICAL"
    return {"field": field, "canonical": canonical}, None

def model_call_record(
    *,
    query_id: str,
    constraint_index: int,
    stage: str,
    question: str,
    mention: dict[str, str],
    evidence: list[dict[str, Any]],
    final: bool,
    canonicalizer: EvidenceCanonicalizer,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, int], str | None]:
    """Run one evidence canonicalization call and persist cryptographic audit."""
    payload = evidence_model_payload(question=question, mention=mention, evidence=evidence)
    system_prompt = FINAL_PROMPT if final else FIRST_PROMPT
    req_sha = request_hash(system_prompt=system_prompt, user_payload=payload)
    record: dict[str, Any] = {
        "query_id": query_id,
        "constraint_index": constraint_index,
        "stage": stage,
        "prompt_version": PROMPT_VERSION,
        "request_sha256": req_sha,
        "serialized_request_sha256": req_sha,
        "retry_request_sha256": None,
        "response_sha256": None,
        "attempts": 0,
        "retry_count": 0,
        "status": "NOT_STARTED",
    }
    try:
        decision, log = canonicalizer.run(
            question=question, mention=mention, evidence=evidence, final=final
        )
        usage = dict(log.get("usage") or {})
        audit = usage.get("_api_audit") if isinstance(usage.get("_api_audit"), dict) else {}
        attempts = int(audit.get("attempts", 1))
        raw = str(log.get("raw", ""))
        response_sha = sha256_bytes(raw.encode("utf-8")) if raw else None
        record.update(
            {
                "status": "SUCCESS",
                "attempts": attempts,
                "retry_count": max(0, attempts - 1),
                "retry_request_sha256": req_sha if attempts > 1 else None,
                "response_sha256": response_sha,
                "response_present": bool(raw),
                "interface": audit.get("interface", "DashScope OpenAI-compatible /chat/completions"),
                "attempt_errors": list(audit.get("attempt_errors", [])),
            }
        )
        return decision, record, numeric_usage_without_audit(usage), None
    except Exception as exc:
        audit = getattr(exc, "audit", {})
        if not isinstance(audit, dict):
            audit = {}
        attempts = int(audit.get("attempts", 1))
        record.update(
            {
                "status": "ERROR",
                "attempts": attempts,
                "retry_count": max(0, attempts - 1),
                "retry_request_sha256": req_sha if attempts > 1 else None,
                "interface": audit.get("interface", "DashScope OpenAI-compatible /chat/completions"),
                "attempt_errors": list(audit.get("attempt_errors", [f"{type(exc).__name__}: {exc}"])),
                "error": str(exc),
            }
        )
        return None, record, {}, str(exc)

def initial_invalid_specs(
    parser_item: dict[str, Any], exact_map: dict[tuple[str, str], str]
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    """Partition frozen parser constraints without using annotation metadata."""
    valid: list[dict[str, str]] = []
    invalid: list[dict[str, Any]] = []
    raw_constraints = parser_item.get("predicted_constraints")
    if not isinstance(raw_constraints, list):
        raw_constraints = []
    seen_invalid: set[tuple[int, str, str, str]] = set()
    for index, value in enumerate(raw_constraints):
        normalized, error_type = exact_constraint(value, exact_map)
        if error_type is None and normalized is not None:
            valid.append(normalized)
            continue
        if isinstance(value, dict):
            field = str(value.get("field", "")).strip()
            canonical = str(value.get("canonical", "")).strip()
        else:
            field, canonical = "", ""
        item = {
            "index": index,
            "field": field,
            "canonical": canonical,
            "error_type": error_type or "INVALID_FIELD_OR_SCHEMA",
        }
        invalid.append(item)
        seen_invalid.add((index, field, canonical, item["error_type"]))
    # The frozen parser keeps schema-invalid entries only in invalid_constraints.
    # Include such entries exactly once, but never invent a canonical value.
    for value in parser_item.get("invalid_constraints", []) or []:
        if not isinstance(value, dict):
            continue
        field = str(value.get("field", "")).strip()
        canonical = str(value.get("canonical", "")).strip()
        error_type = str(value.get("error_type", "INVALID_CANONICAL"))
        if not any(x["field"] == field and x["canonical"] == canonical for x in invalid):
            index = len(invalid) + len(valid)
            key = (index, field, canonical, error_type)
            if key not in seen_invalid:
                invalid.append(
                    {"index": index, "field": field, "canonical": canonical, "error_type": error_type}
                )
                seen_invalid.add(key)
    return invalid, valid, [dict(x) for x in raw_constraints if isinstance(x, dict)]

def process_query(
    *,
    dataset_row: dict[str, Any],
    parser_item: dict[str, Any],
    exact_map: dict[tuple[str, str], str],
    executor: StructuredExecutor,
    evidence: RawEvidenceResources,
    canonicalizer: EvidenceCanonicalizer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    qid = str(dataset_row["_id"])
    question = str(dataset_row["text"])
    invalid_specs, valid_constraints, raw_constraints = initial_invalid_specs(parser_item, exact_map)
    # A complete frozen OK parse must be executable and must be bit-for-bit
    # equivalent at the answer-set level.  This is the all-valid hard gate.
    if not invalid_specs:
        concept_ids = [exact_map[(x["field"], x["canonical"])] for x in valid_constraints]
        if not concept_ids:
            raise RuntimeError(f"all-valid query {qid} has no explicit constraints")
        if len(concept_ids) != len(set(concept_ids)):
            raise RuntimeError(f"duplicate valid concept IDs in {qid}")
        answer_ids = sorted(executor.execute_constraints(concept_ids))
        if parser_item.get("status") == "OK" and answer_ids != sorted(parser_item.get("answer_ids", [])):
            raise AssertionError(f"all-valid hard assertion failed for {qid}")
        result = {
            "query_id": qid,
            "split": "dev",
            "subtype": dataset_row.get("subtype", ""),
            "naturalization_type": dataset_row.get("naturalization_type", ""),
            "natural_question": question,
            "status": "OK",
            "answer_ids": answer_ids,
            "predicted_concept_ids": concept_ids,
            "predicted_constraints": valid_constraints,
            "initial_predicted_constraints": raw_constraints,
            "invalid_constraints": [],
            "canonical_outputs": [dict(x, source="llm_parser") for x in valid_constraints],
            "evidence_trace": [],
            "routing": "all_valid_direct",
            "all_constraints_valid": True,
            "evidence_triggered": False,
            "failed_constraint_count": 0,
            "repaired_constraint_count": 0,
            "still_invalid_constraint_count": 0,
            "re_ground_count": 0,
            "llm_calls": 0,
            "physical_api_attempts": 0,
            "usage": {},
            "api_calls": [],
            "error": None,
        }
        return result, []

    traces: list[dict[str, Any]] = []
    api_calls: list[dict[str, Any]] = []
    usage_totals: Counter[str] = Counter()
    physical_attempts = 0
    repaired_by_index: dict[int, dict[str, str]] = {}
    invalid_final: list[dict[str, Any]] = []
    errors: list[str] = []
    repaired_count = 0
    re_ground_count = 0

    for spec in invalid_specs:
        index = int(spec["index"])
        field = str(spec.get("field", ""))
        canonical = str(spec.get("canonical", ""))
        mention = {"field": field, "text": canonical}
        trace: dict[str, Any] = {
            "constraint_index": index,
            "failed_constraint": {"field": field, "canonical": canonical},
            "initial_validation": spec.get("error_type", "INVALID_CANONICAL"),
            "mention_source": "frozen LLM-Parser predicted canonical string",
        }
        # A malformed field/string has no legal typed raw-evidence route; it
        # fails closed without fabricating a new surface mention.
        if field not in {"reagent", "instrument", "property", "volume"} or not canonical:
            invalid_final.append({**spec, "final_status": "UNRESOLVED"})
            trace["status"] = "UNRESOLVED_NO_TYPED_MENTION"
            traces.append(trace)
            continue
        try:
            initial_evidence = evidence.retrieve(
                question=question,
                mention=canonical,
                field=field,
                top_k=EVIDENCE_TOP_K,
            )
        except Exception as exc:
            errors.append(f"{qid}[{index}] retrieval: {exc}")
            invalid_final.append({**spec, "final_status": "UNRESOLVED_RETRIEVAL"})
            trace["status"] = "UNRESOLVED_RETRIEVAL"
            trace["error"] = str(exc)
            traces.append(trace)
            continue
        trace["initial_evidence"] = initial_evidence
        decision, call_audit, usage, call_error = model_call_record(
            query_id=qid,
            constraint_index=index,
            stage="evidence_canonicalize",
            question=question,
            mention=mention,
            evidence=initial_evidence,
            final=False,
            canonicalizer=canonicalizer,
        )
        api_calls.append(call_audit)
        usage_totals.update(usage)
        physical_attempts += int(call_audit.get("attempts", 0))
        trace["initial_call"] = call_audit
        trace["initial_decision"] = decision
        if call_error:
            errors.append(f"{qid}[{index}] initial canonicalization: {call_error}")
            invalid_final.append({**spec, "final_status": "UNRESOLVED_API"})
            trace["status"] = "UNRESOLVED_API"
            traces.append(trace)
            continue

        accepted: dict[str, str] | None = None
        decision_name = str((decision or {}).get("decision", "")).upper()
        if decision_name == "ACCEPT":
            candidate, validation_error = exact_constraint(decision, exact_map)
            trace["initial_candidate_validation"] = validation_error or "VALID"
            if validation_error is None and candidate is not None:
                accepted = candidate
                trace["status"] = "REPAIRED_INITIAL"
            else:
                invalid_final.append(
                    {
                        **spec,
                        "repaired_field": str((decision or {}).get("field", "")),
                        "repaired_canonical": str((decision or {}).get("canonical", "")),
                        "final_status": validation_error or "INVALID_CANONICAL",
                    }
                )
                trace["status"] = "STILL_INVALID_INITIAL_ACCEPT"
        elif decision_name == "REGROUND":
            re_ground_count += 1
            feedback = str((decision or {}).get("feedback", "")).strip()
            trace["reground_triggered"] = True
            trace["feedback"] = feedback
            try:
                reground_evidence = evidence.retrieve(
                    question=question,
                    mention=canonical,
                    field=field,
                    top_k=EVIDENCE_TOP_K,
                    feedback=feedback,
                )
            except Exception as exc:
                errors.append(f"{qid}[{index}] re-ground retrieval: {exc}")
                invalid_final.append({**spec, "final_status": "UNRESOLVED_REGROUND_RETRIEVAL"})
                trace["status"] = "UNRESOLVED_REGROUND_RETRIEVAL"
                trace["error"] = str(exc)
                traces.append(trace)
                continue
            trace["reground_evidence"] = reground_evidence
            final_decision, final_audit, final_usage, final_error = model_call_record(
                query_id=qid,
                constraint_index=index,
                stage="final_evidence_canonicalize",
                question=question,
                mention=mention,
                evidence=reground_evidence,
                final=True,
                canonicalizer=canonicalizer,
            )
            api_calls.append(final_audit)
            usage_totals.update(final_usage)
            physical_attempts += int(final_audit.get("attempts", 0))
            trace["final_call"] = final_audit
            trace["final_decision"] = final_decision
            if final_error:
                errors.append(f"{qid}[{index}] final canonicalization: {final_error}")
                invalid_final.append({**spec, "final_status": "UNRESOLVED_API"})
                trace["status"] = "UNRESOLVED_API"
            elif str((final_decision or {}).get("decision", "")).upper() == "ACCEPT":
                candidate, validation_error = exact_constraint(final_decision, exact_map)
                trace["final_candidate_validation"] = validation_error or "VALID"
                if validation_error is None and candidate is not None:
                    accepted = candidate
                    trace["status"] = "REPAIRED_REGROUND"
                else:
                    invalid_final.append(
                        {
                            **spec,
                            "repaired_field": str((final_decision or {}).get("field", "")),
                            "repaired_canonical": str((final_decision or {}).get("canonical", "")),
                            "final_status": validation_error or "INVALID_CANONICAL",
                        }
                    )
                    trace["status"] = "STILL_INVALID_REGROUND"
            else:
                invalid_final.append({**spec, "final_status": "UNRESOLVED"})
                trace["status"] = "UNRESOLVED"
        else:
            invalid_final.append({**spec, "final_status": "UNRESOLVED_INVALID_DECISION"})
            trace["status"] = "UNRESOLVED_INVALID_DECISION"

        if accepted is not None:
            repaired_by_index[index] = accepted
            repaired_count += 1
            trace["repaired_constraint"] = accepted
        traces.append(trace)

    all_repaired = not invalid_final and len(repaired_by_index) == len(invalid_specs)
    final_constraints: list[dict[str, str]] = []
    # Preserve the frozen parser's original order and all valid constraints.
    for index, value in enumerate(raw_constraints):
        normalized, validation_error = exact_constraint(value, exact_map)
        if validation_error is None and normalized is not None:
            final_constraints.append(normalized)
        elif index in repaired_by_index:
            final_constraints.append(repaired_by_index[index])

    answer_ids: list[str] = []
    concept_ids: list[str] = []
    status = "OK" if all_repaired else "UNRESOLVED"
    if all_repaired:
        concept_ids = [exact_map[(x["field"], x["canonical"])] for x in final_constraints]
        if not concept_ids or len(concept_ids) != len(set(concept_ids)):
            status = "INVALID_CANONICAL"
            invalid_final.append({"final_status": "DUPLICATE_OR_EMPTY_CONSTRAINT_SET"})
            concept_ids = []
        else:
            try:
                answer_ids = sorted(executor.execute_constraints(concept_ids))
            except Exception as exc:
                status = "UNRESOLVED_EXECUTION"
                errors.append(f"{qid} executor: {exc}")
                concept_ids = []
                answer_ids = []

    if status != "OK":
        concept_ids = []
        answer_ids = []
    result = {
        "query_id": qid,
        "split": "dev",
        "subtype": dataset_row.get("subtype", ""),
        "naturalization_type": dataset_row.get("naturalization_type", ""),
        "natural_question": question,
        "status": status,
        "answer_ids": answer_ids,
        "predicted_concept_ids": concept_ids,
        "predicted_constraints": final_constraints if status == "OK" else [],
        "initial_predicted_constraints": raw_constraints,
        "invalid_constraints": invalid_final,
        "canonical_outputs": [dict(x, source="llm_parser") for x in valid_constraints]
        + [dict(x, source="adaptive_evidence") for x in repaired_by_index.values()],
        "evidence_trace": traces,
        "routing": "evidence_fallback",
        "all_constraints_valid": False,
        "evidence_triggered": True,
        "failed_constraint_count": len(invalid_specs),
        "repaired_constraint_count": repaired_count,
        "still_invalid_constraint_count": len(invalid_final),
        "re_ground_count": re_ground_count,
        "llm_calls": len(api_calls),
        "physical_api_attempts": physical_attempts,
        "usage": dict(usage_totals),
        "api_calls": api_calls,
        "error": "; ".join(errors) if errors else None,
    }
    return result, api_calls
