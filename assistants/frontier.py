import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

from guardrails.wrapper import guarded_chat
from observability.logger import log_call
from tools import registry

BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"
PROVIDER = "groq"

load_dotenv()

_api_key = os.environ.get("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError("GROQ_API_KEY is not set")

_client = OpenAI(base_url=BASE_URL, api_key=_api_key)


def _build_success_record(result: dict, timestamp: str) -> dict:
    return {**result, "timestamp": timestamp, "provider": PROVIDER}


def _build_error_record(exc: Exception, latency_ms: float, timestamp: str) -> dict:
    return {
        "reply": "",
        "latency_ms": latency_ms,
        "tokens_in": 0,
        "tokens_out": 0,
        "model": MODEL,
        "timestamp": timestamp,
        "provider": PROVIDER,
        "error": str(exc),
    }


def _build_api_kwargs(messages: list, tools: list | None) -> dict:
    kwargs = {"model": MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return kwargs


def _call_once(messages: list, tools: list | None) -> tuple:
    t0 = time.monotonic()
    response = _client.chat.completions.create(**_build_api_kwargs(messages, tools))
    return response, (time.monotonic() - t0) * 1000


def _append_tool_results(messages: list, assistant_msg, tool_calls) -> int:
    messages.append(assistant_msg)
    for tc in tool_calls:
        args = json.loads(tc.function.arguments)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": registry.dispatch(tc.function.name, args),
        })
    return len(tool_calls)


def _handle_tool_path(r1, lat1: float, work_messages: list, tools: list) -> tuple:
    n = _append_tool_results(work_messages, r1.choices[0].message, r1.choices[0].message.tool_calls)
    r2, lat2 = _call_once(work_messages, tools)
    return (
        r2.choices[0].message.content,
        lat1 + lat2,
        r1.usage.prompt_tokens + r2.usage.prompt_tokens,
        r1.usage.completion_tokens + r2.usage.completion_tokens,
        n,
    )


def _resolve_response(r1, lat1: float, work_messages: list, tools: list | None) -> tuple:
    tool_calls = r1.choices[0].message.tool_calls
    if tools and tool_calls:
        return _handle_tool_path(r1, lat1, work_messages, tools)
    return (
        r1.choices[0].message.content,
        lat1,
        r1.usage.prompt_tokens,
        r1.usage.completion_tokens,
        0,
    )


def chat(messages: list[dict], tools: list | None = None) -> dict:
    work_messages = list(messages)
    start_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    try:
        r1, lat1 = _call_once(work_messages, tools)
        reply, latency_ms, tokens_in, tokens_out, n_tools = _resolve_response(
            r1, lat1, work_messages, tools
        )
        result = {
            "reply": reply,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "model": MODEL,
            "tool_calls_made": n_tools,
        }
        log_call(_build_success_record(result, start_ts))
        return result
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        log_call(_build_error_record(exc, latency_ms, start_ts))
        raise


def chat_safe(messages: list[dict], tools: list | None = None) -> dict:
    return guarded_chat(chat, messages, tools)
