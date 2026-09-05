#!/usr/bin/env python3
"""
Plot load vs. slowdown from scripts/experiments/run_load_vs_slowdown.py's output.

Reads the per-job CSV (system,load,rep,job_id,...,slowdown), pools per-job
slowdowns across reps for each (system, load) pair, and plots mean and P99
slowdown vs. offered load -- one line per system. Styling (colors, markers,
the y=1.0 "ideal" reference line) matches the load-vs-slowdown figures in the
research repo (plot/rail.ipynb, plot/push_plot.py).

Usage:
    python3 scripts/experiments/plot_load_vs_slowdown.py
    python3 scripts/experiments/plot_load_vs_slowdown.py --input results/load_vs_slowdown/per_job.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent.parent

# Plot order, colors, markers, and display names -- matching the research
# repo's load-vs-slowdown plots (plot/rail.ipynb / plot/push_plot.py).
SYSTEMS = ["ecmp", "crux", "sglb", "monkeytree_perfect"]
COLORS = {
    "ecmp": "#e74c3c",
    "crux": "#3498db",
    "sglb": "#f39c12",
    "monkeytree_perfect": "#2ecc71",
}
MARKERS = {
    "ecmp": "o",
    "crux": "s",
    "sglb": "v",
    "monkeytree_perfect": "^",
}
LABELS = {
    "ecmp": "ECMP",
    "crux": "Crux",
    "sglb": "SGLB",
    "monkeytree_perfect": "MonkeyTree",
}


def percentile(values, p):
    """Nearest-rank percentile, no numpy dependency."""
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[idx]


def load_per_job(csv_path: Path):
    """(system, load) -> list of slowdown values, pooled across reps."""
    data = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["system"], float(row["load"]))
            data[key].append(float(row["slowdown"]))
    return data


def plot_metric(data, loads, systems, metric_fn, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for system in systems:
        values = [metric_fn(data.get((system, load), [1.0])) for load in loads]
        ax.plot(
            loads, values,
            color=COLORS.get(system, "#333333"),
            marker=MARKERS.get(system, "o"),
            markersize=8,
            linewidth=2,
            label=LABELS.get(system, system),
        )

    ax.axhline(y=1.0, linestyle="--", color="gray", alpha=0.5, label="Ideal")
    ax.set_xlabel("Cluster Load", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(loads)
    ax.set_xlim(min(loads) - 0.05, max(loads) + 0.05)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "results" / "load_vs_slowdown" / "per_job.csv")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Defaults to the input file's directory")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"No such file: {args.input}\n"
                          f"Run scripts/experiments/run_load_vs_slowdown.py first.")

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_per_job(args.input)
    loads = sorted({load for (_system, load) in data.keys()})
    systems = [s for s in SYSTEMS if any((s, load) in data for load in loads)]

    if not data:
        raise SystemExit(f"No rows found in {args.input}")

    plot_metric(
        data, loads, systems,
        metric_fn=lambda v: sum(v) / len(v),
        ylabel="Mean Slowdown",
        title="Load vs. Mean Slowdown (1024 GPUs, 16 spines)",
        output_path=output_dir / "avg_slowdown.png",
    )
    plot_metric(
        data, loads, systems,
        metric_fn=lambda v: percentile(v, 99),
        ylabel="P99 Slowdown",
        title="Load vs. P99 Slowdown (1024 GPUs, 16 spines)",
        output_path=output_dir / "p99_slowdown.png",
    )

    print("\nSummary (mean slowdown per system/load):")
    header = "load".rjust(6) + "".join(LABELS.get(s, s).rjust(16) for s in systems)
    print(header)
    for load in loads:
        row = f"{load:6.1f}"
        for system in systems:
            values = data.get((system, load))
            cell = f"{sum(values)/len(values):.3f}" if values else "-"
            row += cell.rjust(16)
        print(row)


if __name__ == "__main__":
    main()
