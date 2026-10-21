from __future__ import annotations

from llm_client import qwen_json_call


FIRST_PROMPT = """你是中国药典原始证据驱动的术语归一化器。根据原始自然问题、一个 raw surface mention、该 mention 的预测字段，以及检索到的 Top-5 原始药典证据，推断该 mention 对应的规范 canonical 字符串。

证据是药典原文片段，不是候选答案菜单。你必须自行从问题和原文证据归纳规范术语；不要假设证据中出现的药品就是答案。

只允许两种输出：
- ACCEPT：给出你从问题和原始证据归纳出的一个规范 canonical 字符串；
- REGROUND：当前证据不足以可靠归纳，给出下一轮只围绕同一 raw mention 语义的检索提示。

严格规则：
1. field 必须与预测字段完全相同。
2. canonical 只能是一个术语字符串，不要添加“法”等未被语义支持的后缀，不要输出 concept ID、药品答案、qrels 或 posting。
3. 不得调用或猜测 ontology 菜单、alias、Gold constraint；当前输入没有合法 canonical 候选列表。
4. 只输出 JSON：{"decision":"ACCEPT","field":"<预测字段>","canonical":"<规范术语>","reason":"<简短理由>"}；或 {"decision":"REGROUND","feedback":"<重新检索提示>","reason":"<简短理由>"}。
"""

FINAL_PROMPT = """你是中国药典原始证据驱动的最终术语归一化器。这是一次同一 raw mention 的重新检索后的最终判断。

你没有 ontology canonical 候选菜单。只能根据原始问题、raw surface mention、预测字段和 Top-5 原始药典证据自行归纳字符串。

只允许：
- ACCEPT：输出与预测字段相同的 canonical 字符串；
- UNRESOLVED：证据仍不足以可靠归纳。

不得输出 concept ID、药品答案、alias、Gold 信息或候选外菜单。只输出 JSON：{"decision":"ACCEPT","field":"<预测字段>","canonical":"<规范术语>","reason":"<简短理由>"}；或 {"decision":"UNRESOLVED","reason":"<简短理由>"}。
"""


class EvidenceCanonicalizer:
    PROMPT_VERSION = "evidence-canonicalizer-v1"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 500,
        retries: int = 1,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.retries = retries

    def run(
        self,
        *,
        question: str,
        mention: dict,
        evidence: list[dict],
        final: bool = False,
    ) -> tuple[dict, dict]:
        # The model-facing evidence projection is deliberately limited to the
        # five fields used by the prompt; rank/evidence IDs stay out.
        evidence_payload = [
            {
                "corpus_id": item["corpus_id"],
                "drug_name": item["drug_name"],
                "field": item["field"],
                "text": item["text"],
                "score": item["score"],
            }
            for item in evidence
        ]
        payload = {
            "question": question,
            "raw_mention": mention["text"],
            "predicted_field": mention["field"],
            "evidence": evidence_payload,
        }
        value, raw, usage = qwen_json_call(
            model=self.model,
            system_prompt=FINAL_PROMPT if final else FIRST_PROMPT,
            user_payload=payload,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            retries=self.retries,
        )
        decision = str(value.get("decision", "")).upper()
        allowed = {"ACCEPT", "UNRESOLVED"} if final else {"ACCEPT", "REGROUND"}
        if decision not in allowed:
            raise ValueError(f"invalid evidence canonicalization decision: {decision}")
        output = {"decision": decision, "reason": str(value.get("reason", "")).strip()}
        if decision == "ACCEPT":
            field = str(value.get("field", "")).strip()
            canonical = str(value.get("canonical", "")).strip()
            if field != mention["field"]:
                raise ValueError(f"canonicalizer changed field: {field}")
            if not canonical:
                raise ValueError("ACCEPT requires a non-empty canonical string")
            output.update({"field": field, "canonical": canonical})
        elif decision == "REGROUND":
            feedback = str(value.get("feedback", "")).strip()
            if not feedback:
                raise ValueError("REGROUND requires feedback")
            output["feedback"] = feedback
        return output, {
            "stage": "final_evidence_canonicalize" if final else "evidence_canonicalize",
            "system_prompt": FINAL_PROMPT if final else FIRST_PROMPT,
            "user_payload": payload,
            "raw": raw,
            "usage": usage,
        }
