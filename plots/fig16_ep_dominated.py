#!/usr/bin/env python3
"""Figure 16: EP-dominated workloads -- mean and P99 slowdown vs
oversubscription ratio for the in-distribution spine sweep (seq16k, 1024
GPUs, threshold 2, 100% load, 5 reps).

Slowdown = real_runtime / ideal_runtime, pooled over all jobs (preloaded
seed + trace jobs). Non-migrating systems (ECMP, Perfect, SGLB) start from
the harvested ECMP fragmented seeds; MonkeyTree systems inherit the
in-distribution MonkeyTree seeds (no FIFO compaction).

Values are embedded below so this script is self-contained.

Source: MonkeyTree-Final/scripts/indist_push_plot.py (paper repo).
"""

import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "fig16_ep_dominated.pdf")

SPINES = [2, 4, 6, 8, 10, 12, 14, 16]

# Mean real slowdown (real_runtime / ideal_runtime), over all jobs, pooled over 5 reps.
MEAN = {
    "ecmp":               [7.3528, 4.0936, 3.1140, 2.6377, 2.3648, 2.1606, 2.0024, 1.8959],
    "crux":               [6.8875, 3.7424, 3.1095, 2.4040, 2.3132, 2.1208, 1.9953, 1.9429],
    "perfect":            [6.8316, 3.6731, 3.0114, 2.1177, 2.4318, 1.9629, 1.8132, 1.6291],
    "sglb":               [7.4027, 3.6892, 2.6717, 2.0970, 1.8365, 1.6183, 1.5134, 1.4157],
    "monkeytree_perfect": [1.6258, 1.3502, 1.2918, 1.1589, 1.2535, 1.1656, 1.1228, 1.0837],
    "monkeytree_sglb":    [1.6150, 1.3125, 1.1952, 1.1387, 1.0951, 1.0747, 1.0634, 1.0502],
}

# P99 real slowdown (real_runtime / ideal_runtime), over all jobs, pooled over 5 reps.
P99 = {
    "ecmp":               [14.4501, 8.2249, 6.0146, 4.8805, 4.2268, 3.7960, 3.4275, 3.1398],
    "crux":               [14.2935, 7.4916, 6.1037, 4.3614, 4.3478, 3.7391, 3.4739, 3.2107],
    "perfect":            [13.8152, 7.2535, 5.7854, 3.9367, 4.9280, 3.3854, 3.0466, 2.7415],
    "sglb":               [13.9530, 6.9819, 4.7769, 3.7478, 3.0519, 2.6323, 2.3425, 2.1501],
    "monkeytree_perfect": [5.2031, 3.1388, 2.6184, 1.9980, 2.7209, 1.8540, 1.7662, 1.6192],
    "monkeytree_sglb":    [5.1194, 2.9336, 2.1658, 1.8415, 1.6157, 1.4759, 1.3929, 1.3595],
}

SYSTEMS = ["ecmp", "crux", "perfect", "sglb", "monkeytree_perfect", "monkeytree_sglb"]

# Only show spine counts that give clean (power-of-two) oversubscription ratios.
# Oversub = num_spines : gpus_per_tor(64), so 2->1:32, 4->1:16, 8->1:8, 16->1:4.
PLOT_SPINES = [2, 4, 8, 16]
OVERSUB_LABELS = ["1:32", "1:16", "1:8", "1:4"]

SYSTEM_NAMES = {
    "ecmp":               "ECMP",
    "crux":               "Crux",
    "perfect":            "Perfect",
    "sglb":               "SGLB",
    "monkeytree_perfect": "MonkeyTree",
    "monkeytree_sglb":    "MonkeyTree-SGLB",
}

SYSTEM_COLORS = {
    "ecmp":               "#1f77b4",
    "crux":               "#ff7f0e",
    "perfect":            "#2ca02c",
    "sglb":               "#9467bd",
    "monkeytree_perfect": "#d62728",
    "monkeytree_sglb":    "#e377c2",
}

SYSTEM_MARKERS = {
    "ecmp":               "o",
    "crux":               "X",
    "perfect":            "s",
    "sglb":               "^",
    "monkeytree_perfect": "D",
    "monkeytree_sglb":    "v",
}


def _draw(ax, data, ylabel):
    x = list(range(len(PLOT_SPINES)))
    idx = [SPINES.index(s) for s in PLOT_SPINES]
    for system in SYSTEMS:
        y = [data[system][i] for i in idx]
        ax.plot(
            x,
            y,
            label=SYSTEM_NAMES[system],
            color=SYSTEM_COLORS[system],
            marker=SYSTEM_MARKERS[system],
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel("Oversubscription Ratio", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(OVERSUB_LABELS)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, alpha=0.3)


def main():
    print("Generating Figure 16 (EP-dominated workloads)...")
    fig, (ax_mean, ax_p99) = plt.subplots(1, 2, figsize=(15, 3.8))

    _draw(ax_mean, MEAN, "Mean Slowdown")
    _draw(ax_p99, P99, "P99 Slowdown")

    ax_mean.legend(loc="upper right", fontsize=14)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
