import json

import observability.logger as logger_module
from observability.logger import log_call

SAMPLE = {"model": "test-model", "latency_ms": 123.4, "tokens_in": 10, "tokens_out": 5}


def test_log_call_round_trips_record(tmp_path, monkeypatch):
    log_file = tmp_path / "calls.jsonl"
    monkeypatch.setattr(logger_module, "LOG_PATH", log_file)
    log_call(SAMPLE)
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == SAMPLE
