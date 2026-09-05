#!/usr/bin/env python3
"""Figure 15: impact of front-end schedulers -- number of migrations
performed by MonkeyTree+Perfect vs cluster load on the 2-tier spine-leaf
topology (512 GPUs, 16 spines), under three placement schedulers (best-fit
block, first-fit FIFO, random block).

Values are embedded below so this script is self-contained. Migration
counts are averaged over the completed reps at each load.

All cells use 5 reps except random at load 0.6 and 0.8 (4 reps; one rep
each hit a pre-existing migration-planner panic and was excluded).

Source: MonkeyTree-Final/scripts/scheduler_migrations_push_plot.py (paper repo).
"""

import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "fig15_scheduler_ablation.pdf")

LOADS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

MIGRATIONS = {
    "block":  [0.0, 0.0, 0.6, 2.0, 6.2, 18.6, 56.0, 95.0, 173.4, 212.0],
    "fifo":   [12.8, 78.0, 150.6, 254.8, 369.2, 429.4, 524.4, 530.0, 560.4, 551.8],
    "random": [417.2, 765.6, 1019.2, 1219.0, 1299.2, 1348.2, 1321.0, 1230.8, 1151.0, 1098.2],
}

SCHEDULERS = ["block", "fifo", "random"]

SCHEDULER_COLORS = {
    "block":  "#2ca02c",
    "fifo":   "#1f77b4",
    "random": "#d62728",
}

SCHEDULER_NAMES = {
    "block":  "Best-fit (block)",
    "fifo":   "FIFO (first-fit)",
    "random": "Random",
}


def main():
    print("Generating Figure 15 (front-end scheduler ablation)...")
    fig, ax = plt.subplots(figsize=(10, 6))

    for sched in SCHEDULERS:
        ax.plot(
            LOADS,
            MIGRATIONS[sched],
            label=SCHEDULER_NAMES[sched],
            color=SCHEDULER_COLORS[sched],
            marker="o",
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Cluster Load", fontsize=12)
    ax.set_ylabel("Migrations Performed", fontsize=12)
    ax.set_xticks(LOADS)
    ax.set_xlim(0.05, 1.05)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
