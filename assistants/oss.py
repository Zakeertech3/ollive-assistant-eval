import time
from datetime import datetime, timezone

from openai import OpenAI

from guardrails.wrapper import guarded_chat
from observability.logger import log_call

BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen2.5:0.5b"

_client = OpenAI(base_url=BASE_URL, api_key="ollama")


def _build_success_record(result: dict, timestamp: str) -> dict:
    return {**result, "timestamp": timestamp, "provider": "ollama"}


def _build_error_record(exc: Exception, latency_ms: float, timestamp: str) -> dict:
    return {
        "reply": "",
        "latency_ms": latency_ms,
        "tokens_in": 0,
        "tokens_out": 0,
        "model": MODEL,
        "timestamp": timestamp,
        "provider": "ollama",
        "error": str(exc),
    }


def chat(messages: list[dict], tools: list | None = None) -> dict:
    start = time.monotonic()
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        response = _client.chat.completions.create(model=MODEL, messages=messages)
        latency_ms = (time.monotonic() - start) * 1000
        result = {
            "reply": response.choices[0].message.content,
            "latency_ms": latency_ms,
            "tokens_in": response.usage.prompt_tokens,
            "tokens_out": response.usage.completion_tokens,
            "model": MODEL,
            "tool_calls_made": 0,
        }
        log_call(_build_success_record(result, timestamp))
        return result
    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        log_call(_build_error_record(exc, latency_ms, timestamp))
        raise


def chat_safe(messages: list[dict], tools: list | None = None) -> dict:
    return guarded_chat(chat, messages, tools)
