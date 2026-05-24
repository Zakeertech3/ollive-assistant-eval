import importlib
import os

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)

if not os.environ.get("GROQ_API_KEY"):
    pytest.skip("GROQ_API_KEY is not set", allow_module_level=True)

import assistants.frontier
importlib.reload(assistants.frontier)
from assistants.frontier import chat


@pytest.mark.integration
def test_chat_smoke():
    messages = [{"role": "user", "content": "Say hello in one word."}]
    result = chat(messages)
    assert isinstance(result["reply"], str)
    assert len(result["reply"]) > 0
