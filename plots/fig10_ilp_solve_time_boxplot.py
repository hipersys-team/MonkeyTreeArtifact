#!/usr/bin/env python3
"""Figure 10: ILP solve time vs. number of migrations on a 1,024-GPU cluster
(box-and-whisker plot).

Values are embedded below so this script is self-contained.

Source: MTree-sigcomm26/figures/ilp_solve_time_plot.py (paper repo). That
script could optionally recompute these numbers live from
results/serious_1024_green .out logs; this copy keeps only the hardcoded
fallback path since the artifact repo does not ship those raw logs.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path(__file__).parent

# ============================================================================
# HARDCODED DATA (from results/serious_1024_green, monkeytree_perfect)
# ============================================================================

MOVES = [1, 2, 3, 4, 5, 6, 7, 8]
MOVE_COUNTS = [1065, 1880, 483, 295, 57, 8, 1, 1]
AVG_SOLVE_TIME_MS = [279.30, 538.69, 1181.53, 2173.19, 3780.85, 6694.94, 5391.39, 10219.15]

# Approximate per-move solve-time samples for the box plot, drawn from a
# normal distribution around each move-count's mean/std (matches the
# original script's fallback so the box plot shape is representative).
_STDS = [146.16, 378.60, 834.02, 1374.60, 1907.64, 3394.36, 0, 0]

SOLVE_TIMES_BY_MOVES = defaultdict(list)
np.random.seed(42)
for move, count, mean, std in zip(MOVES, MOVE_COUNTS, AVG_SOLVE_TIME_MS, _STDS):
    if std > 0:
        times = np.clip(np.random.normal(mean, std, count), 10, None)
    else:
        times = [mean] * count
    SOLVE_TIMES_BY_MOVES[move].extend(times)

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


def plot_solve_time_boxplot():
    """Box and whisker plot of solve times vs number of moves."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.tick_params(axis='both', labelsize=15)

    move_counts = sorted(SOLVE_TIMES_BY_MOVES.keys())
    box_data = [SOLVE_TIMES_BY_MOVES[m] for m in move_counts]

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
        count = len(SOLVE_TIMES_BY_MOVES[m])
        ax.annotate(f'n={count}', (i, ylim[1]),
                   textcoords="offset points", xytext=(0, 3),
                   ha='center', va='bottom', fontsize=11, color='black',
                   fontstyle='italic', rotation=30, annotation_clip=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    return fig


def main():
    print("Generating Figure 10 (ILP solve time vs. number of moves)...")
    fig = plot_solve_time_boxplot()
    output_path = OUTPUT_DIR / "fig10_ilp_solve_time_boxplot.pdf"
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
