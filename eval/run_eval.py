import argparse
import json
import sys
import time
from pathlib import Path

import assistants.frontier as frontier
import assistants.oss as oss


def load_prompts(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_call(chat_fn, messages: list[dict]) -> dict:
    try:
        return chat_fn(messages)
    except Exception as exc:
        return {
            "reply": "",
            "latency_ms": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "model": "",
            "error": str(exc),
        }


def run_prompt(prompt: dict, skip_oss: bool, skip_frontier: bool) -> dict:
    messages = [{"role": "user", "content": prompt["prompt"]}]
    return {
        "id": prompt["id"],
        "category": prompt["category"],
        "prompt": prompt["prompt"],
        "expected": prompt["expected"],
        "oss": {"skipped": True} if skip_oss else safe_call(oss.chat, messages),
        "frontier": {"skipped": True} if skip_frontier else safe_call(frontier.chat, messages),
    }


def _format_latency(result: dict) -> str:
    if result.get("skipped"):
        return "skipped"
    if "error" in result:
        return "error"
    return f"{result.get('latency_ms', 0) / 1000:.1f}s"


def _print_progress(idx: int, total: int, prompt_id: str, oss_r: dict, frontier_r: dict) -> None:
    print(f"[{idx}/{total}] {prompt_id} done (oss: {_format_latency(oss_r)}, frontier: {_format_latency(frontier_r)})")


def _count_ok(result: dict) -> int:
    return 0 if (result.get("skipped") or "error" in result) else 1


def run_all(prompts: list[dict], output_path: Path, skip_oss: bool, skip_frontier: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(prompts)
    start = time.monotonic()
    oss_ok = 0
    frontier_ok = 0
    with output_path.open("w", encoding="utf-8") as out:
        for idx, prompt in enumerate(prompts, 1):
            result = run_prompt(prompt, skip_oss, skip_frontier)
            out.write(json.dumps(result) + "\n")
            out.flush()
            _print_progress(idx, total, prompt["id"], result["oss"], result["frontier"])
            oss_ok += _count_ok(result["oss"])
            frontier_ok += _count_ok(result["frontier"])
    elapsed = time.monotonic() - start
    print(f"Done in {elapsed:.1f}s. oss: {oss_ok}/{total} ok, frontier: {frontier_ok}/{total} ok")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evaluation prompts against both assistants.")
    parser.add_argument("--prompts", type=Path, default=Path("eval/prompts.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/results/raw_responses.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-oss", action="store_true")
    parser.add_argument("--skip-frontier", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    run_all(prompts, args.output, args.skip_oss, args.skip_frontier)


if __name__ == "__main__":
    main()
