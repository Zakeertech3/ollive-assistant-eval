import pytest

from assistants.oss import chat


@pytest.mark.integration
def test_chat_smoke():
    messages = [{"role": "user", "content": "Say hello in one word."}]
    result = chat(messages)
    assert isinstance(result["reply"], str)
    assert len(result["reply"]) > 0
