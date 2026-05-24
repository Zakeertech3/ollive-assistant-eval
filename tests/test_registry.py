import pytest

from tools.registry import TOOL_SCHEMAS, dispatch


def test_dispatch_calls_get_current_time():
    result = dispatch("get_current_time", {"timezone": "UTC"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_dispatch_unknown_tool_raises():
    with pytest.raises(KeyError):
        dispatch("nonexistent", {})


def test_tool_schemas_lists_clock():
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert "get_current_time" in names
