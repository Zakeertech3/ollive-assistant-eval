import json
import os
from pathlib import Path

LOG_PATH = Path("logs/calls.jsonl")


def log_call(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
