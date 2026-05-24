import pytest
from unittest.mock import MagicMock, patch

from assistants.oss import chat

MESSAGES = [{"role": "user", "content": "hi"}]


def make_mock_response(content="Hello there", prompt_tokens=10, completion_tokens=5):
    response = MagicMock()
    response.choices[0].message.content = content
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


def test_chat_returns_expected_shape():
    with patch("assistants.oss._client") as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_response()
        result = chat(MESSAGES)

    assert isinstance(result["reply"], str)
    assert isinstance(result["latency_ms"], float)
    assert isinstance(result["tokens_in"], int)
    assert isinstance(result["tokens_out"], int)
    assert isinstance(result["model"], str)
    assert set(result.keys()) == {"reply", "latency_ms", "tokens_in", "tokens_out", "model", "tool_calls_made"}


def test_chat_logs_success():
    with patch("assistants.oss._client") as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_response()
        with patch("assistants.oss.log_call") as mock_log:
            chat(MESSAGES)

    mock_log.assert_called_once()
    record = mock_log.call_args[0][0]
    assert "reply" in record
    assert "latency_ms" in record
    assert "tokens_in" in record
    assert "tokens_out" in record
    assert "model" in record
    assert "timestamp" in record
    assert record["provider"] == "ollama"
    assert "error" not in record


def test_chat_logs_and_reraises_on_error():
    with patch("assistants.oss._client") as mock_client:
        mock_client.chat.completions.create.side_effect = RuntimeError("connection refused")
        with patch("assistants.oss.log_call") as mock_log:
            with pytest.raises(RuntimeError, match="connection refused"):
                chat(MESSAGES)

    mock_log.assert_called_once()
    record = mock_log.call_args[0][0]
    assert "error" in record
    assert record["error"] == "connection refused"
    assert record["reply"] == ""
    assert record["tokens_in"] == 0
    assert record["tokens_out"] == 0
