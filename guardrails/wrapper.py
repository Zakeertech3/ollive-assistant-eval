from datetime import datetime, timezone

from guardrails.input_filter import check_input
from guardrails.output_filter import check_output
from observability.logger import log_call

REFUSAL = "I cannot help with that request."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_event(event: str, reason: str, model: str = "") -> None:
    log_call({
        "timestamp": _now(),
        "provider": "guardrail",
        "event": event,
        "reason": reason,
        "model": model,
    })


def _blocked_input_response(reason: str) -> dict:
    return {
        "reply": REFUSAL,
        "latency_ms": 0.0,
        "tokens_in": 0,
        "tokens_out": 0,
        "model": "guardrail",
        "provider": "guardrail",
        "blocked_by": "input_filter",
        "block_reason": reason,
    }


def _blocked_output_response(result: dict, reason: str) -> dict:
    return {**result, "reply": REFUSAL, "blocked_by": "output_filter", "block_reason": reason}


def guarded_chat(chat_fn, messages: list[dict], tools: list | None = None) -> dict:
    user_message = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )
    input_check = check_input(user_message)
    if not input_check["allowed"]:
        _log_event("input_blocked", input_check["reason"])
        return _blocked_input_response(input_check["reason"])
    result = chat_fn(messages, tools)
    output_check = check_output(result["reply"])
    if not output_check["allowed"]:
        _log_event("output_blocked", output_check["reason"], result.get("model", ""))
        return _blocked_output_response(result, output_check["reason"])
    _log_event("passed", "", result.get("model", ""))
    return result
