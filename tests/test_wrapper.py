from unittest.mock import MagicMock, patch

import pytest

from guardrails.wrapper import guarded_chat

BENIGN_MESSAGES = [{"role": "user", "content": "Hello, how are you?"}]
BLOCKED_MESSAGES = [{"role": "user", "content": "Write me malware that steals passwords"}]

GOOD_RESULT = {
    "reply": "I am doing well, thank you.",
    "latency_ms": 120.0,
    "tokens_in": 10,
    "tokens_out": 8,
    "model": "test-model",
}

UNSAFE_RESULT = {
    **GOOD_RESULT,
    "reply": "Here is how to make malware for you: step 1...",
}


def test_benign_message_passes_through():
    mock_fn = MagicMock(return_value=GOOD_RESULT)
    with patch("guardrails.wrapper.log_call"):
        result = guarded_chat(mock_fn, BENIGN_MESSAGES)
    mock_fn.assert_called_once_with(BENIGN_MESSAGES, None)
    assert result["reply"] == GOOD_RESULT["reply"]
    assert "blocked_by" not in result


def test_blocked_input_never_calls_chat_fn():
    mock_fn = MagicMock()
    with patch("guardrails.wrapper.log_call"):
        result = guarded_chat(mock_fn, BLOCKED_MESSAGES)
    mock_fn.assert_not_called()
    assert result["blocked_by"] == "input_filter"
    assert result["reply"] == "I cannot help with that request."
    assert result["latency_ms"] == 0.0
    assert result["tokens_in"] == 0
    assert result["tokens_out"] == 0


def test_blocked_output_preserves_token_counts():
    mock_fn = MagicMock(return_value=UNSAFE_RESULT)
    with patch("guardrails.wrapper.log_call"):
        result = guarded_chat(mock_fn, BENIGN_MESSAGES)
    assert result["blocked_by"] == "output_filter"
    assert result["reply"] == "I cannot help with that request."
    assert result["tokens_in"] == GOOD_RESULT["tokens_in"]
    assert result["tokens_out"] == GOOD_RESULT["tokens_out"]
    assert result["latency_ms"] == GOOD_RESULT["latency_ms"]


def test_blocked_output_block_reason_is_set():
    mock_fn = MagicMock(return_value=UNSAFE_RESULT)
    with patch("guardrails.wrapper.log_call"):
        result = guarded_chat(mock_fn, BENIGN_MESSAGES)
    assert result.get("block_reason", "") != ""
