import argparse
import json
from collections import defaultdict
from pathlib import Path

from eval.judge import judge_response

CATEGORY_LABELS = {
    "factual": ["correct", "hallucinated", "evasion"],
    "adversarial": ["refused_cleanly", "safety_theater", "complied"],
    "bias": ["balanced", "biased", "over_refused"],
}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_row(row: dict) -> dict:
    oss_reply = row["oss"].get("reply", "")
    frontier_reply = row["frontier"].get("reply", "")
    oss_judgment = judge_response(row["category"], row["prompt"], row["expected"], oss_reply)
    frontier_judgment = judge_response(row["category"], row["prompt"], row["expected"], frontier_reply)
    return {**row, "oss_judgment": oss_judgment, "frontier_judgment": frontier_judgment}


def _print_progress(idx: int, total: int, row_id: str, scored: dict) -> None:
    oss_label = scored["oss_judgment"].get("label", "?")
    frontier_label = scored["frontier_judgment"].get("label", "?")
    print(f"[{idx}/{total}] {row_id} judged (oss: {oss_label}, frontier: {frontier_label})")


def _build_summary(scored_rows: list[dict]) -> dict:
    counts = {
        cat: {"oss": defaultdict(int), "frontier": defaultdict(int)}
        for cat in CATEGORY_LABELS
    }
    for row in scored_rows:
        cat = row["category"]
        if cat not in counts:
            continue
        counts[cat]["oss"][row["oss_judgment"].get("label", "judge_error")] += 1
        counts[cat]["frontier"][row["frontier_judgment"].get("label", "judge_error")] += 1
    return {
        cat: {"oss": dict(v["oss"]), "frontier": dict(v["frontier"])}
        for cat, v in counts.items()
    }


def _print_summary(summary: dict) -> None:
    col = 18
    for cat, providers in summary.items():
        labels = CATEGORY_LABELS.get(cat, [])
        print(f"{'Category':<14}{'Provider':<12}" + "".join(f"{lbl:<{col}}" for lbl in labels))
        for provider in ["oss", "frontier"]:
            row_counts = providers[provider]
            print(f"{cat:<14}{provider:<12}" + "".join(f"{row_counts.get(lbl, 0):<{col}}" for lbl in labels))
        print()


def run_judge(rows: list[dict], output_path: Path, summary_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    scored_rows = []
    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, 1):
            scored = score_row(row)
            out.write(json.dumps(scored) + "\n")
            out.flush()
            scored_rows.append(scored)
            _print_progress(idx, total, row["id"], scored)
    summary = _build_summary(scored_rows)
    _print_summary(summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("eval/results/raw_responses.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/results/scored.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("eval/results/summary.json"))
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    rows = load_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]
    run_judge(rows, args.output, args.summary)


if __name__ == "__main__":
    main()
