from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request


class QwenCallError(RuntimeError):
    def __init__(self, message: str, audit: dict):
        super().__init__(message)
        self.audit = audit


def extract_json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise ValueError("LLM response contains no JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM response is not a JSON object")
    return value


def qwen_json_call(
    *,
    model: str,
    system_prompt: str,
    user_payload: dict,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1200,
    retries: int = 3,
) -> tuple[dict, str, dict]:
    import dashscope

    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    dashscope.api_key = api_key
    last_error: Exception | None = None
    attempt_errors = []
    for attempt in range(retries):
        try:
            if model.startswith("qwen3.7-"):
                base_url = os.environ.get(
                    "DASHSCOPE_COMPATIBLE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ).rstrip("/")
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    ],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                }
                request = urllib.request.Request(
                    f"{base_url}/chat/completions",
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        response_payload = json.loads(response.read().decode("utf-8"))
                except urllib.error.HTTPError as error:
                    body = error.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"DashScope compatible error {error.code}: {body}") from error
                choices = response_payload.get("choices", [])
                if not choices:
                    raise RuntimeError("DashScope compatible response contains no choices")
                raw = str(choices[0].get("message", {}).get("content", ""))
                usage = dict(response_payload.get("usage") or {})
                interface = "DashScope OpenAI-compatible /chat/completions"
            elif model.startswith("qwen3.8-"):
                response = dashscope.MultiModalConversation.call(
                    model=model,
                    messages=[
                        {"role": "system", "content": [{"text": system_prompt}]},
                        {
                            "role": "user",
                            "content": [{"text": json.dumps(user_payload, ensure_ascii=False)}],
                        },
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    enable_thinking=False,
                    result_format="message",
                )
                interface = "dashscope.MultiModalConversation"
            else:
                response = dashscope.Generation.call(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    ],
                    result_format="message",
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                interface = "dashscope.Generation"
            if not model.startswith("qwen3.7-"):
                if response.status_code != 200:
                    raise RuntimeError(f"DashScope error {response.status_code}: {response.message}")
                content = response.output.choices[0].message.content
                if isinstance(content, list):
                    raw = "".join(
                        str(item.get("text", "")) if isinstance(item, dict) else str(item)
                        for item in content
                    )
                else:
                    raw = str(content)
                usage_obj = getattr(response, "usage", None)
                usage = dict(usage_obj) if usage_obj is not None else {}
            usage["_api_audit"] = {
                "attempts": attempt + 1,
                "retries": attempt,
                "attempt_errors": attempt_errors,
                "interface": interface,
            }
            return extract_json_object(raw), raw, usage
        except Exception as error:  # network and model formatting retries share one audit path
            last_error = error
            attempt_errors.append(f"{type(error).__name__}: {error}")
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise QwenCallError(
        str(last_error),
        {"attempts": retries, "retries": max(0, retries - 1), "attempt_errors": attempt_errors},
    )
