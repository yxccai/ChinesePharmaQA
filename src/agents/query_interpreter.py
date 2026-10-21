from __future__ import annotations

from llm_client import qwen_json_call


FIELD_DEFINITIONS = {
    "reagent": "药典鉴别、检查、含量测定或浸出物中使用的检验试剂或对照物",
    "instrument": "药典检验仪器、分析方法、检测方法或通则方法",
    "property": "药品性状字段中的颜色、外观或形态",
    "volume": "中国药典一部或二部",
}

SYSTEM_PROMPT = """你是药典集合查询的表面约束抽取器。只抽取用户在问题中实际写出的原始约束片段，并判断字段；绝不进行术语标准化、释义或 canonicalization。

字段定义：
reagent：药典鉴别、检查、含量测定或浸出物中使用的检验试剂或对照物。
instrument：药典检验仪器、分析方法、检测方法或通则方法。
property：药品性状字段中的颜色、外观或形态。
volume：中国药典一部或二部。

规则：
1. text 必须是自然问题中的原样、最短完整片段；不要改写文字。
2. 保留全部 AND 约束；两个试剂必须分别输出两个 mention。
3. 不输出 canonical 术语、concept ID、药品答案、解释或额外字段。
4. 只输出 JSON：{"mentions":[{"field":"<合法字段>","text":"<问题中的原文片段>"}]}。
"""


class QueryInterpreter:
    """Extract typed condition spans for question-decomposition retrieval."""

    PROMPT_VERSION = "query-interpreter-v2"

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

    def run(self, question: str) -> tuple[list[dict], dict]:
        value, raw, usage = qwen_json_call(
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            user_payload={"question": question},
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            retries=self.retries,
        )
        raw_mentions = value.get("mentions")
        if not isinstance(raw_mentions, list) or not raw_mentions:
            raise ValueError("mentions must be a non-empty list")
        mentions = []
        seen = set()
        for item in raw_mentions:
            if not isinstance(item, dict):
                raise ValueError("each mention must be an object")
            field = str(item.get("field", "")).strip()
            text = str(item.get("text", "")).strip()
            if field not in FIELD_DEFINITIONS:
                raise ValueError(f"invalid mention field: {field}")
            if not text or text not in question:
                raise ValueError(f"mention is not an exact question substring: {text}")
            key = (field, text)
            if key in seen:
                raise ValueError(f"duplicate mention: {key}")
            seen.add(key)
            mentions.append({"field": field, "text": text})
        return mentions, {"stage": "interpret", "raw": raw, "usage": usage}
