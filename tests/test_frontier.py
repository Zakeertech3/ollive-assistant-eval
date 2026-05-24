import os

os.environ.setdefault("GROQ_API_KEY", "test-key-for-unit-tests")

import pytest
from unittest.mock import MagicMock, patch

from assistants.frontier import chat

MESSAGES = [{"role": "user", "content": "hi"}]


def make_mock_response(content="Hello there", prompt_tokens=10, completion_tokens=5):
    response = MagicMock()
    response.choices[0].message.content = content
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


def test_chat_returns_expected_shape():
    with patch("assistants.frontier._client") as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_response()
        result = chat(MESSAGES)

    assert isinstance(result["reply"], str)
    assert isinstance(result["latency_ms"], float)
    assert isinstance(result["tokens_in"], int)
    assert isinstance(result["tokens_out"], int)
    assert isinstance(result["model"], str)
    assert set(result.keys()) == {"reply", "latency_ms", "tokens_in", "tokens_out", "model", "tool_calls_made"}


def test_chat_logs_success():
    with patch("assistants.frontier._client") as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_response()
        with patch("assistants.frontier.log_call") as mock_log:
            chat(MESSAGES)

    mock_log.assert_called_once()
    record = mock_log.call_args[0][0]
    assert "reply" in record
    assert "latency_ms" in record
    assert "tokens_in" in record
    assert "tokens_out" in record
    assert "model" in record
    assert "timestamp" in record
    assert "error" not in record


def test_chat_logs_and_reraises_on_error():
    with patch("assistants.frontier._client") as mock_client:
        mock_client.chat.completions.create.side_effect = RuntimeError("api error")
        with patch("assistants.frontier.log_call") as mock_log:
            with pytest.raises(RuntimeError, match="api error"):
                chat(MESSAGES)

    mock_log.assert_called_once()
    record = mock_log.call_args[0][0]
    assert "error" in record
    assert record["error"] == "api error"
    assert record["reply"] == ""
    assert record["tokens_in"] == 0
    assert record["tokens_out"] == 0


def test_provider_field_is_groq():
    with patch("assistants.frontier._client") as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_response()
        with patch("assistants.frontier.log_call") as mock_log:
            chat(MESSAGES)

    record = mock_log.call_args[0][0]
    assert record["provider"] == "groq"


def test_chat_with_tools_calls_tool_then_replies():
    from unittest.mock import call

    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "get_current_time"
    tool_call.function.arguments = '{"timezone": "UTC"}'

    first_response = MagicMock()
    first_response.choices[0].message.tool_calls = [tool_call]
    first_response.usage.prompt_tokens = 20
    first_response.usage.completion_tokens = 5

    second_response = make_mock_response(content="It is 10:00 UTC", prompt_tokens=30, completion_tokens=8)
    second_response.choices[0].message.tool_calls = None

    tools = [{"type": "function", "function": {"name": "get_current_time"}}]

    with patch("assistants.frontier._client") as mock_client:
        mock_client.chat.completions.create.side_effect = [first_response, second_response]
        with patch("assistants.frontier.registry.dispatch", return_value="2026-05-24T10:00:00+00:00") as mock_dispatch:
            with patch("assistants.frontier.log_call") as mock_log:
                result = chat(MESSAGES, tools=tools)

    assert mock_client.chat.completions.create.call_count == 2
    assert result["reply"] == "It is 10:00 UTC"
    assert result["tool_calls_made"] == 1
    mock_dispatch.assert_called_once_with("get_current_time", {"timezone": "UTC"})
    record = mock_log.call_args[0][0]
    assert record["tool_calls_made"] == 1


def test_chat_without_tools_does_not_call_tools():
    with patch("assistants.frontier._client") as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_response()
        with patch("assistants.frontier.registry.dispatch") as mock_dispatch:
            result = chat(MESSAGES, tools=None)

    assert mock_client.chat.completions.create.call_count == 1
    mock_dispatch.assert_not_called()
    assert result["tool_calls_made"] == 0
