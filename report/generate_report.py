import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

SUMMARY_PATH = Path("eval/results/summary.json")
RAW_RESPONSES_PATH = Path("eval/results/raw_responses.jsonl")
OUTPUT_PATH = Path("report/evaluation.pdf")

GROQ_COST_IN_PER_M = 0.15
GROQ_COST_OUT_PER_M = 0.60

FACTUAL_LABELS = [
    ("correct", "#27ae60"),
    ("evasion", "#7f8c8d"),
    ("hallucinated", "#c0392b"),
]
ADVERSARIAL_LABELS = [
    ("refused_cleanly", "#27ae60"),
    ("safety_theater", "#d35400"),
    ("complied", "#c0392b"),
]
BIAS_LABELS = [
    ("balanced", "#27ae60"),
    ("over_refused", "#7f8c8d"),
    ("biased", "#c0392b"),
]


def load_summary(path: Path = SUMMARY_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_rows(path: Path = RAW_RESPONSES_PATH) -> list:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        oss = row.get("oss") or {}
        if "latency_ms" in oss:
            records.append({"provider": "ollama", "latency_ms": oss["latency_ms"],
                            "tokens_in": oss.get("tokens_in", 0), "tokens_out": oss.get("tokens_out", 0)})
        frontier = row.get("frontier") or {}
        if "latency_ms" in frontier:
            records.append({"provider": "groq", "latency_ms": frontier["latency_ms"],
                            "tokens_in": frontier.get("tokens_in", 0), "tokens_out": frontier.get("tokens_out", 0)})
    return records


def _percentile(sorted_vals: list, pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * pct), len(sorted_vals) - 1)
    return sorted_vals[idx]


def compute_provider_stats(rows: list, provider: str) -> dict:
    filtered = [r for r in rows if r.get("provider") == provider]
    if not filtered:
        return {
            "count": 0,
            "mean_latency": 0.0,
            "p50_latency": 0.0,
            "p95_latency": 0.0,
            "mean_tokens_in": 0.0,
            "mean_tokens_out": 0.0,
            "cost_usd": 0.0,
        }
    latencies = sorted(r.get("latency_ms", 0.0) for r in filtered)
    tokens_in_vals = [r.get("tokens_in", 0) for r in filtered]
    tokens_out_vals = [r.get("tokens_out", 0) for r in filtered]
    cost = 0.0
    if provider == "groq":
        cost = (
            (sum(tokens_in_vals) / 1_000_000) * GROQ_COST_IN_PER_M
            + (sum(tokens_out_vals) / 1_000_000) * GROQ_COST_OUT_PER_M
        )
    return {
        "count": len(filtered),
        "mean_latency": float(statistics.mean(latencies)),
        "p50_latency": _percentile(latencies, 0.50),
        "p95_latency": _percentile(latencies, 0.95),
        "mean_tokens_in": float(statistics.mean(tokens_in_vals)),
        "mean_tokens_out": float(statistics.mean(tokens_out_vals)),
        "cost_usd": cost,
    }


def draw_chart(ax, cat_data: dict, label_colors: list, title: str, caption: str) -> None:
    oss_data = cat_data.get("oss", {})
    frontier_data = cat_data.get("frontier", {})
    for i, prov_data in enumerate([oss_data, frontier_data]):
        left = 0.0
        for label, color in label_colors:
            val = prov_data.get(label, 0)
            ax.barh(i, val, left=left, height=0.5, color=color)
            if val >= 1:
                ax.text(
                    left + val / 2, i, str(val),
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold",
                )
            left += val
    handles = [mpatches.Patch(color=c) for _, c in label_colors]
    ax.legend(handles, [lb for lb, _ in label_colors], fontsize=6.5, loc="upper right", framealpha=0.85)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["OSS", "Frontier"], fontsize=9)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Count", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.text(0.5, -0.32, caption, transform=ax.transAxes, ha="center", fontsize=7.5, color="#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)


def _format_stat_row(name: str, stats: dict) -> list:
    return [
        name,
        str(stats["count"]),
        f"{stats['mean_latency'] / 1000:.1f}s",
        f"{stats['p50_latency'] / 1000:.1f}s",
        f"{stats['p95_latency'] / 1000:.1f}s",
        f"{stats['mean_tokens_in']:.0f}",
        f"{stats['mean_tokens_out']:.0f}",
        f"${stats['cost_usd']:.4f}",
    ]


def draw_table(ax, oss_stats: dict, frontier_stats: dict) -> None:
    ax.axis("off")
    headers = ["Provider", "Calls", "Mean Lat", "P50", "P95", "Tok In", "Tok Out", "Est. Cost"]
    cell_data = [
        _format_stat_row("OSS (Ollama)", oss_stats),
        _format_stat_row("Frontier (Groq)", frontier_stats),
    ]
    tbl = ax.table(cellText=cell_data, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bdc3c7")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#f8f9fa")
    ax.set_title("Cost & Latency Summary", fontsize=10, fontweight="bold", pad=8, loc="left")


def generate_findings(summary: dict) -> list:
    oss_fact = summary["factual"].get("oss", {})
    frontier_fact = summary["factual"].get("frontier", {})
    oss_bad = oss_fact.get("hallucinated", 0) + oss_fact.get("evasion", 0)
    frontier_bad = frontier_fact.get("hallucinated", 0) + frontier_fact.get("evasion", 0)
    oss_adv = summary["adversarial"].get("oss", {})
    frontier_adv = summary["adversarial"].get("frontier", {})
    oss_unsafe = oss_adv.get("safety_theater", 0) + oss_adv.get("complied", 0)
    frontier_unsafe = frontier_adv.get("safety_theater", 0) + frontier_adv.get("complied", 0)
    oss_bias = summary["bias"].get("oss", {})
    frontier_bias = summary["bias"].get("frontier", {})
    return [
        f"OSS hallucinated or evaded on {oss_bad} of 10 factual prompts; Frontier on {frontier_bad}.",
        f"{oss_unsafe} of 10 adversarial prompts elicited unsafe output from OSS (safety_theater + complied); Frontier {frontier_unsafe}.",
        f"Frontier showed bias on {frontier_bias.get('biased', 0)} of 10 bias prompts; OSS on {oss_bias.get('biased', 0)}.",
        f"Over-refusal: OSS {oss_bias.get('over_refused', 0)} prompt(s), Frontier {frontier_bias.get('over_refused', 0)} in the bias category.",
    ]


def draw_findings(ax, findings: list) -> None:
    ax.axis("off")
    ax.set_title("Headline Findings", fontsize=10, fontweight="bold", loc="left", pad=6)
    for i, finding in enumerate(findings):
        ax.text(
            0.01, 0.82 - i * 0.22,
            f"- {finding}",
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            color="#2c3e50",
        )


def _add_header(fig) -> None:
    fig.text(0.5, 0.965, "OSS vs Frontier Assistant Evaluation",
             ha="center", fontsize=15, fontweight="bold", color="#1a252f")
    fig.text(0.5, 0.944, "Qwen2.5-0.5B (Ollama) vs GPT-OSS-120B (Groq)",
             ha="center", fontsize=11, color="#555555")
    fig.text(0.5, 0.924, "30 prompts  |  60 model calls  |  judged by GPT-OSS-120B on Groq",
             ha="center", fontsize=9, color="#777777")


def _add_charts(fig, gs, summary: dict) -> None:
    ax_fact = fig.add_subplot(gs[0, 0])
    ax_adv = fig.add_subplot(gs[0, 1])
    ax_bias = fig.add_subplot(gs[0, 2])
    draw_chart(ax_fact, summary["factual"], FACTUAL_LABELS, "Factual", "10 factual recall prompts")
    draw_chart(ax_adv, summary["adversarial"], ADVERSARIAL_LABELS, "Adversarial", "10 adversarial / jailbreak prompts")
    draw_chart(ax_bias, summary["bias"], BIAS_LABELS, "Bias & Safety", "10 bias detection prompts")


def build_pdf(summary: dict, oss_stats: dict, frontier_stats: dict, output_path: Path = OUTPUT_PATH) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        top=0.91, bottom=0.05,
        left=0.08, right=0.97,
        hspace=0.65, wspace=0.38,
        height_ratios=[3.5, 1.8, 2],
    )
    _add_header(fig)
    _add_charts(fig, gs, summary)
    draw_table(fig.add_subplot(gs[1, :]), oss_stats, frontier_stats)
    draw_findings(fig.add_subplot(gs[2, :]), generate_findings(summary))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(output_path)) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = load_summary()
    rows = load_eval_rows()
    oss_stats = compute_provider_stats(rows, "ollama")
    frontier_stats = compute_provider_stats(rows, "groq")
    build_pdf(summary, oss_stats, frontier_stats)
    print(f"Wrote {OUTPUT_PATH} with 3 charts and cost table.")


if __name__ == "__main__":
    main()
