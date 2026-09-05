#!/usr/bin/env python3
"""Figure 13: number of migrations and job slowdown vs. fragmentation
threshold on a 1,024-GPU cluster (combined bar + line view).

Values are embedded below so this script is self-contained.

Source: MTree-sigcomm26/figures/lambda_plot2.py (paper repo), using
results/lambda_experiment_fix.
"""

import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# ============================================================================
# HARDCODED DATA (from results/lambda_experiment_fix analysis)
# ============================================================================

THRESHOLDS = [2, 3, 4, 5, 6, 7, 8]
AVG_MIGRATIONS = [266.0, 205.0, 135.0, 69.0, 34.0, 5.0, 0.0]
AVG_SLOWDOWN = [1.007, 1.007, 1.008, 1.127, 1.255, 1.267, 1.281]
P90_SLOWDOWN = [1.022, 1.021, 1.021, 1.427, 1.696, 1.686, 1.754]
P99_SLOWDOWN = [1.056, 1.062, 1.060, 1.742, 1.826, 1.833, 1.831]

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
    'legend.fontsize': 15,
})
MAIN_COLOR = '#2E86AB'
SECONDARY_COLOR = '#E94F37'
TERTIARY_COLOR = '#9B59B6'


def plot_combined():
    """Combined view: Migrations + Slowdown side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.2))

    ax1.bar(THRESHOLDS, AVG_MIGRATIONS, color=MAIN_COLOR, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel("Fragmentation Threshold")
    ax1.set_ylabel("Average Migrations\nper Experiment")
    ax1.set_xticks(THRESHOLDS)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0)

    for th, mig in zip(THRESHOLDS, AVG_MIGRATIONS):
        if mig > 0:
            ax1.annotate(f'{mig:.0f}', xy=(th, mig), ha='center', va='bottom',
                        fontsize=14, fontweight='bold')

    ax2.plot(THRESHOLDS, AVG_SLOWDOWN, label='Average Slowdown', color=MAIN_COLOR,
             marker='o', linewidth=2, markersize=8)
    ax2.plot(THRESHOLDS, P90_SLOWDOWN, label='P90 Slowdown', color=SECONDARY_COLOR,
             marker='s', linewidth=2, markersize=8)
    ax2.plot(THRESHOLDS, P99_SLOWDOWN, label='P99 Slowdown', color=TERTIARY_COLOR,
             marker='^', linewidth=2, markersize=8)
    ax2.set_xlabel("Fragmentation Threshold")
    ax2.set_ylabel("Slowdown")
    ax2.set_xticks(THRESHOLDS)
    ax2.legend(loc='best', fontsize=13, labelspacing=0.3, handletextpad=0.4)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0.95)

    plt.tight_layout()
    return fig


def main():
    print("Generating Figure 13 (migrations & slowdown vs. fragmentation threshold)...")
    fig = plot_combined()
    output_path = OUTPUT_DIR / "fig13_lambda_threshold.pdf"
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
