#!/usr/bin/env python3
"""Figure 7: cluster-wide and per-rack fragmentation over time at 70-100%
load in a 1,024-GPU cluster (ECMP, Crux, MonkeyTree).

Reads time-series data from fig07_fragmentation_data.json (bundled alongside
this script). No simulator/results dependency.

Source: MTree-sigcomm26/figures/fragmentation_plot.py (paper repo), reading
fragmentation_data.json produced by extract_fragmentation_data.py there.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "fig07_fragmentation_data.json"
OUTPUT_DIR = SCRIPT_DIR

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})


def load_data():
    """Load time-series data from JSON."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    with open(DATA_FILE, 'r') as f:
        return json.load(f)


def plot_traffic_combined_grid(data):
    """
    Plot both total rings and max rings per ToR in a combined 2x4 grid.

    Top row: Total rings for loads 0.7, 0.8, 0.9, 1.0
    Bottom row: Max rings per ToR for same loads
    """
    loads = data["loads"]
    systems = data["systems"]
    system_names = data["system_names"]
    system_colors = data["system_colors"]
    time_series = data["time_series"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 4.5))

    metrics = [("total_rings", "Total DP-Flows \n(cross-ToR)"),
               ("max_rings_per_tor", "Max DP-flows\nper ToR")]

    lines = []
    labels = []

    for row_idx, (metric_key, ylabel) in enumerate(metrics):
        for col_idx, load in enumerate(loads):
            ax = axes[row_idx, col_idx]
            load_key = str(load)

            if load_key not in time_series:
                continue

            for system in systems:
                if system not in time_series[load_key]:
                    continue

                series = time_series[load_key][system]
                x_values = series["time_hours"]
                y_values = series[metric_key]

                color = system_colors.get(system, "#95a5a6")
                label = system_names.get(system, system)
                line, = ax.plot(x_values, y_values, color=color, linewidth=1.2, label=label, alpha=0.8)

                if row_idx == 0 and col_idx == 0:
                    lines.append(line)
                    labels.append(label)

            if row_idx == 1:
                ax.set_xlabel("Time (hours)")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(f"Load {load}", fontweight='bold')

            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

    fig.legend(lines, labels, loc='upper center', ncol=len(systems),
               bbox_to_anchor=(0.5, 1.01), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.94], pad=0.5, h_pad=0.8, w_pad=0.4)
    return fig


def main():
    print("Generating Figure 7 (fragmentation over time)...")
    data = load_data()
    print(f"  Loaded data for loads: {data['loads']}")
    print(f"  Systems: {data['systems']}")

    fig = plot_traffic_combined_grid(data)
    output_path = OUTPUT_DIR / "fig07_fragmentation.pdf"
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
