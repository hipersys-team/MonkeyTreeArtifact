#!/usr/bin/env python3
"""Figure 8: heatmaps of average and p99 slowdown across oversubscription
ratios for a 2,048-GPU cluster (ECMP, Crux, SGLB, Perfect-routing-only,
MonkeyTree). Y-axis = GPUs per ToR, X-axis = number of spines.

Values are embedded below so this script is self-contained.

Source: MTree-sigcomm26/figures/spine_sweep_heatmap_plot.py (paper repo).
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# ============================================================================
# HARDCODED DATA (from spine_sweep_heatmaps.ipynb analysis)
# ============================================================================

GPUS_PER_TOR = [16, 32, 64, 128, 256]  # Y-axis (rows)
NUM_SPINES = [16, 32, 64, 128, 256]    # X-axis (columns)

SYSTEMS = ["ecmp", "crux", "sglb", "perfect", "mtree"]

SYSTEM_NAMES = {
    "ecmp": "ECMP",
    "crux": "Crux",
    "perfect": "Perfect-routing-only",
    "mtree": "MTree",
    "sglb": "SGLB",
}

# Data structure: data[system][gpus_per_tor_idx][spine_idx]
# N/A values represented as np.nan

AVG_SLOWDOWN = {
    "ecmp": [
        [1.498, np.nan, np.nan, np.nan, np.nan],
        [1.666, 1.443, np.nan, np.nan, np.nan],
        [1.781, 1.465, 1.305, np.nan, np.nan],
        [1.894, 1.543, 1.332, 1.224, np.nan],
        [1.886, 1.532, 1.313, 1.173, 1.094],
    ],
    "crux": [
        [1.002, np.nan, np.nan, np.nan, np.nan],
        [1.023, 1.002, np.nan, np.nan, np.nan],
        [1.229, 1.003, 1.002, np.nan, np.nan],
        [1.330, 1.010, 1.003, 1.003, np.nan],
        [1.333, 1.051, 1.005, 1.004, 1.004],
    ],
    "perfect": [
        [1.000, np.nan, np.nan, np.nan, np.nan],
        [1.026, 1.000, np.nan, np.nan, np.nan],
        [1.197, 1.001, 1.001, np.nan, np.nan],
        [1.312, 1.006, 1.001, 1.001, np.nan],
        [1.347, 1.057, 1.001, 1.001, 1.001],
    ],
    "mtree": [
        [1.000, np.nan, np.nan, np.nan, np.nan],
        [1.001, 1.000, np.nan, np.nan, np.nan],
        [1.001, 1.001, 1.001, np.nan, np.nan],
        [1.002, 1.001, 1.001, 1.001, np.nan],
        [1.002, 1.002, 1.001, 1.001, 1.001],
    ],
    "sglb": [
        [1.005, np.nan, np.nan, np.nan, np.nan],
        [1.020, 1.002, np.nan, np.nan, np.nan],
        [1.132, 1.005, 1.000, np.nan, np.nan],
        [1.224, 1.015, 1.000, 1.000, np.nan],
        [1.249, 1.053, 1.003, 1.000, 1.000],
    ],
}

P90_SLOWDOWN = {
    "ecmp": [
        [2.308, np.nan, np.nan, np.nan, np.nan],
        [2.731, 2.203, np.nan, np.nan, np.nan],
        [3.112, 2.311, 1.924, np.nan, np.nan],
        [3.651, 2.624, 2.053, 1.723, np.nan],
        [4.013, 2.826, 2.143, 1.697, 1.449],
    ],
    "crux": [
        [1.001, np.nan, np.nan, np.nan, np.nan],
        [1.016, 1.003, np.nan, np.nan, np.nan],
        [1.705, 1.000, 1.000, np.nan, np.nan],
        [2.020, 1.001, 1.000, 1.000, np.nan],
        [2.234, 1.196, 1.002, 1.001, 1.001],
    ],
    "perfect": [
        [1.001, np.nan, np.nan, np.nan, np.nan],
        [1.019, 1.000, np.nan, np.nan, np.nan],
        [1.652, 1.001, 1.000, np.nan, np.nan],
        [1.991, 1.001, 1.000, 1.000, np.nan],
        [2.230, 1.275, 1.002, 1.002, 1.002],
    ],
    "mtree": [
        [1.001, np.nan, np.nan, np.nan, np.nan],
        [1.001, 1.000, np.nan, np.nan, np.nan],
        [1.001, 1.000, 1.000, np.nan, np.nan],
        [1.002, 1.000, 1.000, 1.000, np.nan],
        [1.003, 1.002, 1.002, 1.002, 1.002],
    ],
    "sglb": [
        [1.003, np.nan, np.nan, np.nan, np.nan],
        [1.036, 1.000, np.nan, np.nan, np.nan],
        [1.465, 1.001, 1.000, np.nan, np.nan],
        [1.708, 1.032, 1.000, 1.000, np.nan],
        [1.940, 1.256, 1.000, 1.000, 1.000],
    ],
}

P99_SLOWDOWN = {
    "ecmp": [
        [2.962, np.nan, np.nan, np.nan, np.nan],
        [3.666, 2.909, np.nan, np.nan, np.nan],
        [4.240, 3.082, 2.514, np.nan, np.nan],
        [5.091, 3.489, 2.823, 2.243, np.nan],
        [5.683, 3.931, 2.920, 2.232, 1.864],
    ],
    "crux": [
        [1.067, np.nan, np.nan, np.nan, np.nan],
        [1.515, 1.069, np.nan, np.nan, np.nan],
        [2.096, 1.095, 1.066, np.nan, np.nan],
        [2.478, 1.271, 1.083, 1.086, np.nan],
        [2.912, 1.669, 1.105, 1.088, 1.088],
    ],
    "perfect": [
        [1.003, np.nan, np.nan, np.nan, np.nan],
        [1.539, 1.012, np.nan, np.nan, np.nan],
        [2.039, 1.023, 1.014, np.nan, np.nan],
        [2.430, 1.224, 1.016, 1.017, np.nan],
        [2.802, 1.668, 1.022, 1.026, 1.026],
    ],
    "mtree": [
        [1.003, np.nan, np.nan, np.nan, np.nan],
        [1.019, 1.013, np.nan, np.nan, np.nan],
        [1.031, 1.014, 1.015, np.nan, np.nan],
        [1.036, 1.018, 1.017, 1.019, np.nan],
        [1.047, 1.033, 1.026, 1.023, 1.023],
    ],
    "sglb": [
        [1.133, np.nan, np.nan, np.nan, np.nan],
        [1.353, 1.064, np.nan, np.nan, np.nan],
        [1.767, 1.137, 1.001, np.nan, np.nan],
        [2.082, 1.274, 1.005, 1.000, np.nan],
        [2.466, 1.502, 1.085, 1.000, 1.000],
    ],
}

# ============================================================================
# PLOT STYLING
# ============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})


def plot_heatmap(ax, data_matrix, title, y_labels, x_labels, vmin=None, vmax=None, cmap='YlOrRd'):
    masked_data = np.ma.masked_invalid(data_matrix)
    im = ax.imshow(masked_data, aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_facecolor('#e0e0e0')

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    ax.set_xlabel('Number of Spines')
    ax.set_ylabel('GPUs per ToR')
    ax.set_title(title, fontweight='bold')
    ax.grid(False)

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            value = data_matrix[i, j]
            if not np.isnan(value):
                if vmin is not None and vmax is not None:
                    text_color = 'white' if value > (vmax - vmin) * 0.6 + vmin else 'black'
                else:
                    text_color = 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=text_color, fontsize=14)
            else:
                ax.text(j, i, 'N/A', ha='center', va='center',
                       color='gray', fontsize=13)

    return im


def get_data_matrix(data_dict, system):
    return np.array(data_dict[system])


def plot_combined_heatmaps():
    """Combined view: 2 rows (Average, P99) x N cols (systems)."""
    metric_order = [("Average", AVG_SLOWDOWN), ("P99", P99_SLOWDOWN)]

    fig, axes = plt.subplots(len(metric_order), len(SYSTEMS),
                             figsize=(4.5*len(SYSTEMS), 3.5*len(metric_order)))

    all_values = []
    for _, data_dict in metric_order:
        for system in SYSTEMS:
            matrix = get_data_matrix(data_dict, system)
            all_values.extend(matrix.flatten())
    all_values = np.array(all_values)
    all_values = all_values[~np.isnan(all_values)]
    vmin, vmax = 1.0, np.percentile(all_values, 95) if len(all_values) > 0 else 2.0

    for row, (metric_title, data_dict) in enumerate(metric_order):
        for col, system in enumerate(SYSTEMS):
            ax = axes[row, col]
            title = SYSTEM_NAMES.get(system, system) if row == 0 else ""
            matrix = get_data_matrix(data_dict, system)
            im = plot_heatmap(ax, matrix, title, GPUS_PER_TOR, NUM_SPINES, vmin=vmin, vmax=vmax)

            if row == 0:
                ax.set_xlabel('')
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f'{metric_title} Slowdown\n\nGPUs per ToR')
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])

    fig.subplots_adjust(right=0.88, wspace=0.15, hspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Slowdown')

    return fig


def main():
    print("Generating Figure 8 (spine sweep heatmaps)...")
    fig = plot_combined_heatmaps()
    output_path = OUTPUT_DIR / "fig08_spine_sweep_heatmaps.pdf"
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
