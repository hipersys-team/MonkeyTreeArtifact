#!/usr/bin/env python3
"""Figure 14(a): P99 network slowdown vs cluster load on the rail-optimized
topology (2 spines, 1024 GPUs), pooled over 5 reps.

Values are embedded below so this script is self-contained.

Source: MonkeyTree-Final/scripts/push_plot.py (paper repo).
"""

import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "fig14a_rail_optimized.pdf")

LOADS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# P99 real slowdown (real_runtime / ideal_runtime), pooled over 5 reps.
# Rail topology: 8 pods x 16 blocks/pod x 8 GPUs/block = 1024 GPUs, 2 spines.
P99 = {
    "ecmp":               [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 2.2158, 2.7836, 2.8337, 3.1168],
    "crux":               [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.1510, 1.5653, 1.8485, 1.9208],
    "perfect":            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0879, 1.4351, 1.6935, 1.7584],
    "sglb":               [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0144, 1.1344, 1.3937, 1.5174],
    "monkeytree_perfect": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0065, 1.0100, 1.0161, 1.0174],
}

SYSTEMS = ["ecmp", "crux", "perfect", "sglb", "monkeytree_perfect"]

SYSTEM_COLORS = {
    "ecmp":               "#1f77b4",
    "crux":               "#ff7f0e",
    "perfect":            "#2ca02c",
    "sglb":               "#bcbd22",
    "monkeytree_perfect": "#d62728",
}

SYSTEM_NAMES = {
    "ecmp":               "ECMP",
    "crux":               "Crux",
    "perfect":            "Perfect",
    "sglb":               "SGLB",
    "monkeytree_perfect": "MonkeyTree",
}


def main():
    print("Generating Figure 14a (rail-optimized topology)...")
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
