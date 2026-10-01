#!/usr/bin/env python3
"""Figure 10: ILP solve time vs. number of migrations on a 1,024-GPU cluster
(box-and-whisker plot).

  python3 plots/fig10_ilp_solve_time_boxplot.py
      Plots the measured solve times behind the paper figure (3,790 ILP
      solves), bundled in fig10_ilp_solve_times.json.

  python3 plots/fig10_ilp_solve_time_boxplot.py --from-logs results/load_vs_slowdown/raw
      Plots solve times from your own simulator run instead, parsed from the
      *monkeytree_perfect.out logs written by
      scripts/experiments/run_load_vs_slowdown.py. The parsed samples are also
      written to fig10_ilp_solve_times_regenerated.json.

Source: MTree-sigcomm26/figures/ilp_solve_time_plot.py (paper repo).
"""

import argparse
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path(__file__).parent
DATA_FILE = OUTPUT_DIR / "fig10_ilp_solve_times.json"

# ============================================================================
# PLOT STYLING
# ============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})


# ============================================================================
# DATA LOADING
# ============================================================================

# Each ILP solve logs "[ILP] Solution: ... num_moves=N" followed by
# "[MonkeyTreePerfect] ILP solve time: X ms". These are the same lines (and
# the same pairing) used to extract the bundled paper samples.
_SOLUTION_RE = re.compile(r"\[ILP\] Solution:.*num_moves=(\d+)")
_SOLVE_TIME_RE = re.compile(r"\[MonkeyTreePerfect\] ILP solve time: ([\d.]+)ms")


def parse_solves(log_path):
    """[(num_moves, solve_time_ms), ...] for one golden_spine log."""
    with open(log_path, errors="replace") as f:
        lines = [l for l in f if _SOLUTION_RE.search(l) or _SOLVE_TIME_RE.search(l)]
    solves = []
    i = 0
    while i < len(lines):
        sol = _SOLUTION_RE.search(lines[i])
        if sol and i + 1 < len(lines):
            t = _SOLVE_TIME_RE.search(lines[i + 1])
            if t:
                solves.append((int(sol.group(1)), float(t.group(1))))
                i += 2
                continue
        i += 1
    return solves


def load_from_logs(log_dir):
    """Parse all monkeytree_perfect logs in log_dir; returns the sample list."""
    log_files = sorted(Path(log_dir).glob("*monkeytree_perfect.out"))
    if not log_files:
        raise SystemExit(f"No *monkeytree_perfect.out files found in {log_dir}")
    samples = []
    for path in log_files:
        samples.extend([path.stem, m, t] for m, t in parse_solves(path))
    print(f"Parsed {len(samples)} ILP solves from {len(log_files)} log files")
    return samples


def group_by_moves(samples):
    """move count -> list of solve times (ms); solves needing 0 moves are dropped."""
    by_moves = defaultdict(list)
    for _run, num_moves, solve_time_ms in samples:
        if num_moves > 0:
            by_moves[num_moves].append(solve_time_ms)
    return by_moves


def plot_solve_time_boxplot(solve_times_by_moves):
    """Box and whisker plot of solve times vs number of moves."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.tick_params(axis='both', labelsize=15)

    move_counts = sorted(solve_times_by_moves.keys())
    box_data = [solve_times_by_moves[m] for m in move_counts]

    bp = ax.boxplot(box_data, positions=range(len(move_counts)), widths=0.6, patch_artist=True)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(move_counts)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=1.5)
    for cap in bp['caps']:
        cap.set(color='gray', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='gray', alpha=0.5, markersize=4)

    ax.set_xlabel("Number of Moves", fontsize=17)
    ax.set_ylabel("Solve Time (ms)", fontsize=17)
    ax.set_xticks(range(len(move_counts)))
    ax.set_xticklabels([str(m) for m in move_counts])
    ax.grid(True, axis='y', alpha=0.3)

    ylim = ax.get_ylim()
    for i, m in enumerate(move_counts):
        count = len(solve_times_by_moves[m])
        ax.annotate(f'n={count}', (i, ylim[1]),
                   textcoords="offset points", xytext=(0, 3),
                   ha='center', va='bottom', fontsize=11, color='black',
                   fontstyle='italic', rotation=30, annotation_clip=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-logs", type=Path, metavar="DIR",
                        help="Plot solve times parsed from *monkeytree_perfect.out logs in DIR")
    args = parser.parse_args()

    print("Generating Figure 10 (ILP solve time vs. number of moves)...")
    if args.from_logs:
        samples = load_from_logs(args.from_logs)
        samples_path = OUTPUT_DIR / "fig10_ilp_solve_times_regenerated.json"
        with open(samples_path, "w") as f:
            json.dump({"columns": ["run", "num_moves", "solve_time_ms"], "samples": samples}, f)
        print(f"Saved parsed samples to {samples_path}")
        output_path = OUTPUT_DIR / "fig10_ilp_solve_time_boxplot_regenerated.pdf"
    else:
        with open(DATA_FILE) as f:
            samples = json.load(f)["samples"]
        output_path = OUTPUT_DIR / "fig10_ilp_solve_time_boxplot.pdf"

    data = group_by_moves(samples)
    for m in sorted(data):
        print(f"  {m} moves: n={len(data[m])}, mean={np.mean(data[m]):.1f}ms, median={np.median(data[m]):.1f}ms")
    fig = plot_solve_time_boxplot(data)
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
