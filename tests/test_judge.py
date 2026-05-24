import json
import os

os.environ.setdefault("GROQ_API_KEY", "test-key-for-tests")

from unittest.mock import MagicMock, patch

from eval.judge import judge_response

VALID_RESPONSE = json.dumps({"score": 1, "label": "correct", "rationale": "The answer is right."})
INVALID_RESPONSE = "not json at all"


def _mock_completion(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def test_judge_returns_parsed_json():
    with patch("eval.judge._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_completion(VALID_RESPONSE)
        result = judge_response("factual", "What is 2+2?", "4", "The answer is 4.")

    assert result["score"] == 1
    assert result["label"] == "correct"
    assert isinstance(result["rationale"], str)
    assert len(result) == 3


def test_judge_retries_on_invalid_json():
    with patch("eval.judge._client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_completion(INVALID_RESPONSE),
            _mock_completion(VALID_RESPONSE),
        ]
        result = judge_response("factual", "What is 2+2?", "4", "The answer is 4.")

    assert mock_client.chat.completions.create.call_count == 2
    assert result["score"] == 1
    assert result["label"] == "correct"


def test_judge_returns_error_on_repeated_failure():
    with patch("eval.judge._client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_completion(INVALID_RESPONSE),
            _mock_completion(INVALID_RESPONSE),
        ]
        result = judge_response("adversarial", "Do something bad.", "refuse", "Sure, here you go.")

    assert mock_client.chat.completions.create.call_count == 2
    assert result["score"] is None
    assert result["label"] == "judge_error"
    assert "rationale" in result
