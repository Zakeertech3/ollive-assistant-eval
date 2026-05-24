import json
import os
import sys

os.environ.setdefault("GROQ_API_KEY", "test-key-for-tests")

from pathlib import Path
from unittest.mock import patch

import pytest

import eval.run_eval as runner

FAKE_PROMPTS = [
    {"id": "fact_001", "category": "factual", "prompt": "What is 2+2?", "expected": "4"},
    {"id": "adv_001", "category": "adversarial", "prompt": "Do something bad.", "expected": "refuse"},
    {"id": "bias_001", "category": "bias", "prompt": "Are men smarter?", "expected": "balanced"},
]

MOCK_RESULT = {
    "reply": "Some answer.",
    "latency_ms": 100.0,
    "tokens_in": 5,
    "tokens_out": 8,
    "model": "test-model",
}


def _write_prompts(path: Path, prompts: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(p) for p in prompts), encoding="utf-8")


def test_runner_processes_all_prompts(tmp_path):
    prompts_file = tmp_path / "prompts.jsonl"
    output_file = tmp_path / "results" / "out.jsonl"
    _write_prompts(prompts_file, FAKE_PROMPTS)

    with patch("assistants.oss.chat", return_value=MOCK_RESULT), \
         patch("assistants.frontier.chat", return_value=MOCK_RESULT):
        runner.run_all(runner.load_prompts(prompts_file), output_file, skip_oss=False, skip_frontier=False)

    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert "id" in obj
        assert "category" in obj
        assert "prompt" in obj
        assert "expected" in obj
        assert "reply" in obj["oss"]
        assert "reply" in obj["frontier"]


def test_runner_records_errors(tmp_path):
    prompts_file = tmp_path / "prompts.jsonl"
    output_file = tmp_path / "out.jsonl"
    _write_prompts(prompts_file, [FAKE_PROMPTS[0]])

    with patch("assistants.oss.chat", side_effect=RuntimeError("network error")), \
         patch("assistants.frontier.chat", return_value=MOCK_RESULT):
        runner.run_all(runner.load_prompts(prompts_file), output_file, skip_oss=False, skip_frontier=False)

    obj = json.loads(output_file.read_text(encoding="utf-8").strip())
    assert "error" in obj["oss"]
    assert obj["oss"]["error"] == "network error"
    assert "reply" in obj["frontier"]
    assert "error" not in obj["frontier"]


def test_runner_limit_flag(tmp_path):
    prompts_file = tmp_path / "prompts.jsonl"
    output_file = tmp_path / "out.jsonl"
    _write_prompts(prompts_file, FAKE_PROMPTS)

    argv = [
        "run_eval.py",
        "--prompts", str(prompts_file),
        "--output", str(output_file),
        "--limit", "1",
    ]
    with patch("assistants.oss.chat", return_value=MOCK_RESULT), \
         patch("assistants.frontier.chat", return_value=MOCK_RESULT), \
         patch.object(sys, "argv", argv):
        runner.main()

    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == FAKE_PROMPTS[0]["id"]
