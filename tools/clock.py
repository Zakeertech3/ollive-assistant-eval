from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Return the current date and time in a given timezone as an ISO 8601 string.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. 'UTC', 'Asia/Kolkata'. Defaults to UTC.",
                }
            },
            "required": [],
        },
    },
}


def get_current_time(timezone: str = "UTC") -> str:
    try:
        tz = ZoneInfo(timezone)
        return datetime.now(tz).isoformat()
    except ZoneInfoNotFoundError:
        return datetime.now(ZoneInfo("UTC")).isoformat() + " (UTC fallback)"
