#!/usr/bin/env python3
"""Figure 6: network slowdown vs cluster load on the 2-tier spine-leaf
topology (16 ToRs x 64 GPUs/ToR = 1024 GPUs, 16 spines, block size 8), pooled
over 5 reps. Row of 3 panels: Average, P90, and P99 slowdown.

Values are embedded below so this script is self-contained -- run it to
(re)produce the figure with no simulator/results dependency.

Source: MonkeyTree-Final/scripts/new_spine_push_plot.py (paper repo).
"""

import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "fig06_main_slowdown.pdf")

# ============================================================================
# HARDCODED DATA (from green.ipynb / serious_1024_green analysis)
# ============================================================================

LOADS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Plot order (controls legend + draw order).
SYSTEMS = ["full_bisection", "ecmp", "crux", "perfect", "sglb",
           "monkeytree_perfect", "monkeytree", "monkeytree_crux", "monkeytree_sglb"]

SYSTEM_NAMES = {
    "full_bisection": "Full-bisection packet spraying",
    "ecmp": "ECMP",
    "crux": "Crux",
    "perfect": "Perfect-routing",
    "sglb": "SGLB",
    "monkeytree_perfect": "MonkeyTree",
    "monkeytree": "MonkeyTree-ECMP",
    "monkeytree_crux": "MonkeyTree-Crux",
    "monkeytree_sglb": "MonkeyTree-SGLB",
}

SYSTEM_COLORS = {
    "full_bisection": "#000000",
    "ecmp": "#1f77b4",
    "crux": "#ff7f0e",
    "perfect": "#2ca02c",
    "sglb": "#bcbd22",
    "monkeytree_perfect": "#d62728",
    "monkeytree": "#8c564b",
    "monkeytree_crux": "#7f7f7f",
    "monkeytree_sglb": "#e377c2",
}

# Average slowdown
AVG_SLOWDOWN = {
    "full_bisection": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
    "ecmp": [1.000, 1.000, 1.000, 1.000, 1.000, 1.003, 1.075, 1.564, 1.532, 1.606],
    "crux": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.005, 1.051, 1.193, 1.259],
    "perfect": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.002, 1.029, 1.136, 1.202],
    "sglb": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.002, 1.019, 1.073, 1.148],
    "monkeytree_perfect": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.001, 1.002, 1.004],
    "monkeytree": [1.000, 1.000, 1.000, 1.000, 1.000, 1.003, 1.025, 1.115, 1.169, 1.195],
    "monkeytree_crux": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.003, 1.015, 1.036, 1.058],
    "monkeytree_sglb": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.001, 1.008, 1.017, 1.032],
}

# P90 slowdown
P90_SLOWDOWN = {
    "full_bisection": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
    "ecmp": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.218, 2.653, 2.605, 2.716],
    "crux": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.186, 1.632, 1.746],
    "perfect": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.085, 1.482, 1.589],
    "sglb": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.048, 1.273, 1.423],
    "monkeytree_perfect": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.001, 1.004, 1.008],
    "monkeytree": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.001, 1.429, 1.703, 1.788],
    "monkeytree_crux": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.014, 1.109, 1.217],
    "monkeytree_sglb": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.009, 1.057, 1.127],
}

# P99 slowdown
P99_SLOWDOWN = {
    "full_bisection": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
    "ecmp": [1.000, 1.000, 1.000, 1.000, 1.000, 1.013, 2.150, 3.709, 3.620, 3.743],
    "crux": [1.000, 1.000, 1.000, 1.000, 1.000, 1.001, 1.187, 1.683, 2.054, 2.114],
    "perfect": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.070, 1.485, 1.837, 1.911],
    "sglb": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.070, 1.327, 1.510, 1.654],
    "monkeytree_perfect": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.003, 1.009, 1.015, 1.044],
    "monkeytree": [1.000, 1.000, 1.000, 1.000, 1.000, 1.013, 1.732, 2.404, 2.451, 2.498],
    "monkeytree_crux": [1.000, 1.000, 1.000, 1.000, 1.000, 1.001, 1.088, 1.379, 1.518, 1.597],
    "monkeytree_sglb": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.042, 1.177, 1.222, 1.265],
}

# ============================================================================
# PLOT STYLING
# ============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 14,
})


def plot_slowdown_combined():
    """Combined view: Average, P90, and P99 Slowdown in a row of 3."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 2.8))

    # Left: Average Slowdown
    lines = []
    labels = []
    for system in SYSTEMS:
        label = SYSTEM_NAMES.get(system, system)
        color = SYSTEM_COLORS.get(system)
        line, = ax1.plot(LOADS, AVG_SLOWDOWN[system], label=label, color=color,
                         marker='o', linewidth=2, markersize=6)
        lines.append(line)
        labels.append(label)

    ax1.set_xlabel("Cluster Load")
    ax1.set_ylabel("Average Slowdown")
    ax1.set_xticks(LOADS)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.45, 1.05)
    ax1.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8])

    # Middle: P90 Slowdown
    for system in SYSTEMS:
        color = SYSTEM_COLORS.get(system)
        ax2.plot(LOADS, P90_SLOWDOWN[system], color=color,
                 marker='o', linewidth=2, markersize=6)

    ax2.set_xlabel("Cluster Load")
    ax2.set_ylabel("P90 Slowdown")
    ax2.set_xticks(LOADS)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.45, 1.05)
    ax2.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])

    # Right: P99 Slowdown
    for system in SYSTEMS:
        color = SYSTEM_COLORS.get(system)
        ax3.plot(LOADS, P99_SLOWDOWN[system], color=color,
                 marker='o', linewidth=2, markersize=6)

    ax3.set_xlabel("Cluster Load")
    ax3.set_ylabel("P99 Slowdown")
    ax3.set_xticks(LOADS)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.45, 1.05)
    ax3.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    # Unified legend on top
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12),
               ncol=5, frameon=False)

    plt.subplots_adjust(top=0.78, wspace=0.3)
    return fig


def main():
    print("Generating Figure 6 (main load vs. slowdown)...")
    fig = plot_slowdown_combined()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
