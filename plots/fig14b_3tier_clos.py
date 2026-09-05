#!/usr/bin/env python3
"""Figure 14(b): P99 network slowdown vs cluster load on the 3-tier fat-tree
topology (4 pods x 4 ToRs/pod x 64 GPUs/ToR = 1024 GPUs, 16 aggs/pod, 16
cores), pooled over 5 reps.

Values are embedded below so this script is self-contained.

Source: MonkeyTree-Final/scripts/fat_push_plot.py (paper repo).
"""

import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "fig14b_3tier_clos.pdf")

# load01 (0.1) not included; flat at ~1.0 in this regime.
LOADS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# P99 real slowdown (real_runtime / ideal_runtime), pooled over 5 reps.
P99 = {
    "ecmp":               [1.0000, 1.0000, 1.0000, 1.0000, 1.0522, 1.6210, 2.9866, 2.9936, 3.0629],
    "crux":               [1.0000, 1.0000, 1.0000, 1.0000, 1.0010, 1.1354, 2.2472, 2.1540, 2.4923],
    "perfect":            [1.0000, 1.0000, 1.0000, 1.0000, 1.0004, 1.0840, 1.5726, 1.6255, 1.9255],
    "sglb":               [1.0000, 1.0000, 1.0000, 1.0000, 1.0134, 1.3383, 1.9478, 1.9662, 2.0738],
    "monkeytree_perfect": [1.0000, 1.0000, 1.0000, 1.0000, 1.0002, 1.0321, 1.3113, 1.3258, 1.4861],
    "monkeytree3":        [1.0010, 1.0018, 1.0022, 1.0030, 1.0035, 1.0056, 1.0118, 1.0128, 1.0154],
}

SYSTEMS = ["ecmp", "crux", "perfect", "sglb", "monkeytree_perfect", "monkeytree3"]

SYSTEM_COLORS = {
    "ecmp":               "#1f77b4",
    "crux":               "#ff7f0e",
    "perfect":            "#2ca02c",
    "sglb":               "#bcbd22",
    "monkeytree_perfect": "#d62728",
    "monkeytree3":        "#9467bd",
}

SYSTEM_NAMES = {
    "ecmp":               "ECMP",
    "crux":               "Crux",
    "perfect":            "Perfect",
    "sglb":               "SGLB",
    "monkeytree_perfect": "MonkeyTree",
    "monkeytree3":        "MonkeyTree3",
}


def main():
    print("Generating Figure 14b (3-tier Clos topology)...")
    fig, ax = plt.subplots(figsize=(9, 4.2))

    for system in SYSTEMS:
        ax.plot(
            LOADS,
            P99[system],
            label=SYSTEM_NAMES[system],
            color=SYSTEM_COLORS[system],
            marker="o",
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Cluster Load", fontsize=28)
    ax.set_ylabel("P99 Slowdown", fontsize=28)
    ax.set_xticks(LOADS)
    ax.set_xlim(0.05, 1.05)
    ax.tick_params(axis="both", labelsize=24)
    ax.legend(loc="upper left", fontsize=18)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
