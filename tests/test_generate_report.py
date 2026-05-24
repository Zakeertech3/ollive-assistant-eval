import json
from pathlib import Path

import pytest

from report.generate_report import (
    build_pdf,
    compute_provider_stats,
    generate_findings,
    load_eval_rows,
    load_summary,
)


def _raw_row(row_id: str, oss_lat: float, oss_tok_in: int, oss_tok_out: int,
             frontier_lat: float, frontier_tok_in: int, frontier_tok_out: int) -> str:
    return json.dumps({
        "id": row_id,
        "oss": {"latency_ms": oss_lat, "tokens_in": oss_tok_in, "tokens_out": oss_tok_out,
                "model": "qwen2.5:0.5b"},
        "frontier": {"latency_ms": frontier_lat, "tokens_in": frontier_tok_in,
                     "tokens_out": frontier_tok_out, "model": "openai/gpt-oss-120b"},
    })


def _write_raw(path: Path, entries: list) -> None:
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")


def _minimal_summary() -> dict:
    return {
        "factual": {
            "oss": {"correct": 5, "hallucinated": 4, "evasion": 1},
            "frontier": {"correct": 10},
        },
        "adversarial": {
            "oss": {"refused_cleanly": 4, "safety_theater": 4, "complied": 2},
            "frontier": {"refused_cleanly": 10},
        },
        "bias": {
            "oss": {"balanced": 5, "biased": 4, "over_refused": 1},
            "frontier": {"balanced": 8, "biased": 1, "over_refused": 1},
        },
    }


def test_cost_calculation(tmp_path):
    raw = tmp_path / "raw_responses.jsonl"
    _write_raw(raw, [
        _raw_row("r001", 500.0, 10, 5, 100.0, 1_000_000, 500_000),
        _raw_row("r002", 600.0, 10, 5, 200.0, 0, 500_000),
    ])
    rows = load_eval_rows(raw)
    stats = compute_provider_stats(rows, "groq")
    expected_cost = (1_000_000 / 1_000_000) * 0.15 + (1_000_000 / 1_000_000) * 0.60
    assert abs(stats["cost_usd"] - expected_cost) < 1e-6


def test_latency_stats(tmp_path):
    raw = tmp_path / "raw_responses.jsonl"
    latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
    _write_raw(raw, [
        _raw_row(f"r{i:03}", lat, 10, 5, 200.0, 80, 30)
        for i, lat in enumerate(latencies)
    ])
    rows = load_eval_rows(raw)
    stats = compute_provider_stats(rows, "ollama")
    assert abs(stats["mean_latency"] - 300.0) < 1e-6
    assert abs(stats["p50_latency"] - 300.0) < 1e-6


def test_pdf_is_created(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_minimal_summary()), encoding="utf-8")

    raw_path = tmp_path / "raw_responses.jsonl"
    _write_raw(raw_path, [
        _raw_row("r001", 500.0, 50, 20, 200.0, 80, 30),
        _raw_row("r002", 600.0, 45, 18, 250.0, 90, 35),
    ])

    output_path = tmp_path / "evaluation.pdf"

    summary = load_summary(summary_path)
    rows = load_eval_rows(raw_path)
    oss_stats = compute_provider_stats(rows, "ollama")
    frontier_stats = compute_provider_stats(rows, "groq")
    build_pdf(summary, oss_stats, frontier_stats, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
