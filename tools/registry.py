from tools.clock import TOOL_SCHEMA, get_current_time

TOOLS = {
    "get_current_time": get_current_time,
}

TOOL_SCHEMAS = [TOOL_SCHEMA]


def dispatch(name: str, arguments: dict) -> str:
    return TOOLS[name](**arguments)
