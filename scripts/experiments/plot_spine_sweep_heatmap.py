#!/usr/bin/env python3
"""
Plot slowdown heatmaps from run_spine_sweep_heatmap.py's output.

Reads the per-job CSV (system,gpus_per_tor,num_spines,rep,...,slowdown),
pools per-job slowdowns across reps for each (system, gpus_per_tor,
num_spines) point, and renders average/p99 slowdown heatmaps -- one column
per system, y-axis = GPUs per ToR, x-axis = number of spines. Invalid grid
points (more spines than GPUs per ToR) are shown as "N/A".

Styling matches plots/fig08_spine_sweep_heatmaps.py (the paper's Figure 8
reproduction) so a regenerated run and the paper figure are visually
comparable.

Usage:
    python3 scripts/experiments/plot_spine_sweep_heatmap.py
    python3 scripts/experiments/plot_spine_sweep_heatmap.py --input results/spine_sweep_heatmap/per_job.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent.parent

# Plot order and display names -- matching plots/fig08_spine_sweep_heatmaps.py.
SYSTEMS = ["ecmp", "crux", "sglb", "perfect", "monkeytree_perfect"]
SYSTEM_NAMES = {
    "ecmp": "ECMP",
    "crux": "Crux",
    "sglb": "SGLB",
    "perfect": "Perfect-routing-only",
    "monkeytree_perfect": "MonkeyTree",
}


def percentile(values, p):
    """Nearest-rank percentile, no numpy dependency."""
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[idx]


def load_per_job(csv_path: Path):
    """(system, gpus_per_tor, num_spines) -> list of slowdown values, pooled across reps."""
    data = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["system"], int(row["gpus_per_tor"]), int(row["num_spines"]))
            data[key].append(float(row["slowdown"]))
    return data


def build_matrix(data, system, gpus_per_tor_values, num_spines_values, metric_fn):
    """Rows = gpus_per_tor, cols = num_spines. NaN where num_spines > gpus_per_tor
    or no data was collected for that combination."""
    matrix = np.full((len(gpus_per_tor_values), len(num_spines_values)), np.nan)
    for i, tor in enumerate(gpus_per_tor_values):
        for j, spines in enumerate(num_spines_values):
            if spines > tor:
                continue
            values = data.get((system, tor, spines))
            if values:
                matrix[i, j] = metric_fn(values)
    return matrix


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
                text_color = 'black'
                if vmin is not None and vmax is not None and vmax > vmin:
                    text_color = 'white' if value > (vmax - vmin) * 0.6 + vmin else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color, fontsize=14)
            else:
                ax.text(j, i, 'N/A', ha='center', va='center', color='gray', fontsize=13)

    return im


def plot_combined_heatmaps(data, systems, gpus_per_tor_values, num_spines_values):
    """2 rows (Average, P99) x N cols (systems)."""
    metric_order = [
        ("Average", lambda v: sum(v) / len(v)),
        ("P99", lambda v: percentile(v, 99)),
    ]

    matrices = {
        metric_title: {s: build_matrix(data, s, gpus_per_tor_values, num_spines_values, metric_fn)
                       for s in systems}
        for metric_title, metric_fn in metric_order
    }

    all_values = np.concatenate([
        matrices[metric_title][s].flatten()
        for metric_title, _ in metric_order for s in systems
    ])
    all_values = all_values[~np.isnan(all_values)]
    vmin, vmax = 1.0, (np.percentile(all_values, 95) if len(all_values) > 0 else 2.0)

    fig, axes = plt.subplots(len(metric_order), len(systems),
                             figsize=(4.5 * len(systems), 3.5 * len(metric_order)))
    if len(systems) == 1:
        axes = axes.reshape(len(metric_order), 1)

    for row, (metric_title, _) in enumerate(metric_order):
        for col, system in enumerate(systems):
            ax = axes[row, col]
            title = SYSTEM_NAMES.get(system, system) if row == 0 else ""
            im = plot_heatmap(ax, matrices[metric_title][system], title,
                               gpus_per_tor_values, num_spines_values, vmin=vmin, vmax=vmax)

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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "results" / "spine_sweep_heatmap" / "per_job.csv")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Defaults to the input file's directory")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"No such file: {args.input}\n"
                          f"Run scripts/experiments/run_spine_sweep_heatmap.py first.")

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_per_job(args.input)
    if not data:
        raise SystemExit(f"No rows found in {args.input}")

    gpus_per_tor_values = sorted({k[1] for k in data.keys()})
    num_spines_values = sorted({k[2] for k in data.keys()})
    systems = [s for s in SYSTEMS if any(k[0] == s for k in data.keys())]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })

    fig = plot_combined_heatmaps(data, systems, gpus_per_tor_values, num_spines_values)
    output_path = output_dir / "spine_sweep_heatmap.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
