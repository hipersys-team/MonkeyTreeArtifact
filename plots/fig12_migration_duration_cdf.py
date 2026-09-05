#!/usr/bin/env python3
"""Figure 12: CDF of migration duration at different loads (70/80/90/100%).

Reads fig12_migration_duration_data.json (bundled alongside this script).
No simulator/results dependency.

Source: MTree-sigcomm26/figures/migration_duration_cdf_plot.py (paper repo),
reading migration_duration_data.json produced by
extract_migration_duration_data.py there.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "fig12_migration_duration_data.json"
OUTPUT_PDF = SCRIPT_DIR / "fig12_migration_duration_cdf.pdf"

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    with open(DATA_FILE) as f:
        return json.load(f)


def cdf_step_vals(durations):
    """Return (x_vals, y_vals) for a step CDF with 0 prepended."""
    n = len(durations)
    if n == 0:
        return [], []
    unique_t = sorted(set(durations))
    x_vals = [0] + unique_t
    y_vals = [0] + [sum(1 for t in durations if t <= x) / n for x in unique_t]
    return x_vals, y_vals


def main():
    print("Generating Figure 12 (CDF of migration duration)...")
    data = load_data()
    loads = data["loads"]
    durations_with_delay = data["durations_with_delay"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for load in loads:
        key = str(load)
        if key not in durations_with_delay:
            continue
        durs = durations_with_delay[key]
        if len(durs) == 0:
            continue
        x_vals, y_vals = cdf_step_vals(durs)
        pct = int(load * 100)
        ax.step(x_vals, y_vals, where="post", label=f"{pct}% load")

    ax.set_xlabel("Migration duration (s)")
    ax.set_ylabel("CDF")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
