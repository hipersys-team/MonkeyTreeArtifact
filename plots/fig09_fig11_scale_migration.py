#!/usr/bin/env python3
"""Figure 9: distribution of job migrations required to restore low
fragmentation across cluster sizes (128/256/512/1024/2048 GPUs).

Figure 11: ILP solve time (average, P90) across the same cluster sizes.

Values are embedded below so this script is self-contained.

Source: MTree-sigcomm26/figures/scale_migration_plot.py (paper repo),
using results/scale_migrations_5rep.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# ============================================================================
# HARDCODED DATA (from results/scale_migrations_5rep)
# ============================================================================

CLUSTER_SIZES = [128, 256, 512, 1024, 2048]
AVG_SOLVE_TIME_MS = [19.84, 43.01, 127.08, 439.73, 2393.94]
P90_SOLVE_TIME_MS = [26.01, 103.69, 370.62, 888.83, 4038.59]

MOVES_128 = [1, 2]
MOVE_PCTS_128 = [22.5, 77.5]

MOVES_256 = [1, 2, 3, 4, 5]
MOVE_PCTS_256 = [35.381, 59.534, 2.119, 2.754, 0.212]

MOVES_512 = [1, 2, 3, 4, 5, 6]
MOVE_PCTS_512 = [36.602, 52.21, 8.84, 1.934, 0.276, 0.138]

MOVES_1024 = [1, 2, 3, 4, 5, 7]
MOVE_PCTS_1024 = [42.883, 42.523, 10.09, 3.604, 0.721, 0.18]

MOVES_2048 = [1, 2, 3, 4, 5]
MOVE_PCTS_2048 = [43.402, 44.575, 7.625, 4.106, 0.293]

MOVES_BY_SIZE = {
    128: (MOVES_128, MOVE_PCTS_128),
    256: (MOVES_256, MOVE_PCTS_256),
    512: (MOVES_512, MOVE_PCTS_512),
    1024: (MOVES_1024, MOVE_PCTS_1024),
    2048: (MOVES_2048, MOVE_PCTS_2048),
}

# ============================================================================
# PLOT STYLING
# ============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})

AVG_COLOR = '#1f77b4'
P90_COLOR = '#d62728'


def plot_solve_time_vs_cluster_size():
    """Figure 11: line plot of Average and P90 solve time vs cluster size."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.tick_params(axis='both', labelsize=15)

    ax.plot(CLUSTER_SIZES, AVG_SOLVE_TIME_MS, marker='o', linewidth=2, markersize=8,
            label='Average', color=AVG_COLOR)
    ax.plot(CLUSTER_SIZES, P90_SOLVE_TIME_MS, marker='s', linewidth=2, markersize=8,
            label='P90', color=P90_COLOR)

    ax.set_xlabel("Cluster Size (GPUs)", fontsize=17)
    ax.set_ylabel("Solve Time (ms)", fontsize=17)
    ax.set_xscale('log', base=2)
    ax.set_xticks(CLUSTER_SIZES)
    ax.set_xticklabels([str(s) for s in CLUSTER_SIZES])
    ax.legend(loc='upper left', fontsize=13)
    ax.grid(True, alpha=0.3)

    ax.set_ylim(-500, None)
    ax.set_yticks([t for t in ax.get_yticks() if t >= 0])

    for i, (s, avg, p90) in enumerate(zip(CLUSTER_SIZES, AVG_SOLVE_TIME_MS, P90_SOLVE_TIME_MS)):
        if i == len(CLUSTER_SIZES) - 1:
            ax.annotate(f'{avg:.1f}', (s, avg), textcoords="offset points",
                       xytext=(0, 8), ha='center', fontsize=11, color=AVG_COLOR)
        else:
            ax.annotate(f'{avg:.1f}', (s, avg), textcoords="offset points",
                       xytext=(0, -12), ha='center', fontsize=11, color=AVG_COLOR)
        ax.annotate(f'{p90:.1f}', (s, p90), textcoords="offset points",
                   xytext=(0, 8), ha='center', fontsize=11, color=P90_COLOR)

    plt.tight_layout()
    return fig


def plot_moves_distribution():
    """Figure 9: bar chart showing distribution of moves for each cluster size."""
    all_moves = set()
    for size in CLUSTER_SIZES:
        moves, _ = MOVES_BY_SIZE[size]
        all_moves.update(moves)
    max_moves = max(all_moves)

    fig, axes = plt.subplots(1, len(CLUSTER_SIZES), figsize=(16, 2.5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(CLUSTER_SIZES)))

    for idx, size in enumerate(CLUSTER_SIZES):
        ax = axes[idx]
        moves, pcts = MOVES_BY_SIZE[size]

        full_pcts = []
        for m in range(1, max_moves + 1):
            if m in moves:
                full_pcts.append(pcts[moves.index(m)])
            else:
                full_pcts.append(0)

        bars = ax.bar(range(1, max_moves + 1), full_pcts, color=colors[idx],
                     edgecolor='black', alpha=0.8)

        ax.set_xlabel("Number of Moves")
        if idx == 0:
            ax.set_ylabel("Percentage (%)")
        ax.set_title(f"{size} GPUs", fontweight='bold')
        ax.set_xticks(range(1, max_moves + 1))
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.grid(True, axis='y', alpha=0.3)

        for i, (bar, pct) in enumerate(zip(bars, full_pcts)):
            move_num = i + 1
            if pct > 0:
                label = f'{pct:.1f}%' if pct < 1 else f'{pct:.0f}%'
                if move_num <= 2:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                           label, ha='center', va='center', fontsize=12, fontweight='bold',
                           color='white', rotation=90)
                else:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           label, ha='center', va='bottom', fontsize=12, fontweight='bold',
                           color='black', rotation=90)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, 1,
                       '0.0%', ha='center', va='bottom', fontsize=12, fontweight='bold',
                       color='black', rotation=90)

        ax.set_ylim(0, 80)

    plt.tight_layout()
    return fig


def main():
    print("Generating Figure 9 (moves distribution) and Figure 11 (ILP solve time vs cluster size)...")

    fig = plot_solve_time_vs_cluster_size()
    output_path = OUTPUT_DIR / "fig11_ilp_solve_time_vs_cluster_size.pdf"
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close(fig)

    fig = plot_moves_distribution()
    output_path = OUTPUT_DIR / "fig09_moves_distribution.pdf"
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
