import os

os.environ.setdefault("GROQ_API_KEY", "test-key-for-tests")

from unittest.mock import patch

import app

OSS_CHOICE = "OSS (Qwen2.5-0.5B via Ollama)"
FRONTIER_CHOICE = "Frontier (GPT-OSS-120B via Groq)"

MOCK_RESULT = {
    "reply": "Hi there!",
    "latency_ms": 42.0,
    "tokens_in": 5,
    "tokens_out": 3,
    "model": "test-model",
}


def test_handle_message_routes_to_oss():
    with patch("assistants.oss.chat_safe", return_value=MOCK_RESULT) as mock_oss, \
         patch("assistants.frontier.chat_safe", return_value=MOCK_RESULT) as mock_frontier:
        chatbot, history, status = app.handle_message("hello", [], OSS_CHOICE)

    mock_oss.assert_called_once()
    mock_frontier.assert_not_called()
    assert any(m["role"] == "user" and m["content"] == "hello" for m in history)
    assert any(m["role"] == "assistant" and m["content"] == MOCK_RESULT["reply"] for m in history)
    assert len(chatbot) == len([m for m in history if m["role"] != "system"])


def test_handle_message_routes_to_frontier():
    with patch("assistants.oss.chat_safe", return_value=MOCK_RESULT) as mock_oss, \
         patch("assistants.frontier.chat_safe", return_value=MOCK_RESULT) as mock_frontier:
        chatbot, history, status = app.handle_message("hello", [], FRONTIER_CHOICE)

    mock_frontier.assert_called_once()
    mock_oss.assert_not_called()
    assert any(m["role"] == "user" and m["content"] == "hello" for m in history)
    assert any(m["role"] == "assistant" and m["content"] == MOCK_RESULT["reply"] for m in history)
