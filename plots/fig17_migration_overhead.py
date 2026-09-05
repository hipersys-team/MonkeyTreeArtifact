#!/usr/bin/env python3
"""Figure 17: timing breakdown of migrating a single Llama3-8B worker on the
testbed (horizontal stacked bar chart, Source/Destination rows).

Values are embedded below so this script is self-contained. These are
physical-testbed measurements (5-node prototype, A100 GPUs, 100 Gbps
ConnectX-5 NIC), not simulator output.

Source: MTree-sigcomm26/figures/migration_overhead_plot.py (paper repo).
"""

import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# ============================================================================
# HARDCODED DATA (migration overhead in ms, from testbed measurement)
# ============================================================================

SRC_COMPONENTS = [
    ("Ckpt. in GPU memory", 3935.07),
    ("GPUDirect RDMA transfer", 3188.65),
    ("Tearing down training", 1239.11),
]

DST_COMPONENTS = [
    ("Startup overhead", 1167.94),
    ("NCCL initialization", 646.72),
    ("Recreate model", 272.06),
    ("Load ckpt. into GPU memory", 879.43),
]

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
    'legend.fontsize': 16,
})

SRC_COLORS = ['#f9a825', '#fbc02d', '#ffeb3b']  # Dark to light yellow
DST_COLORS = ['#6a1b9a', '#8e24aa', '#ab47bc', '#ce93d8']  # Dark to light purple


def plot_migration_overhead():
    """Horizontal stacked bar chart for migration overhead breakdown."""
    fig, ax = plt.subplots(figsize=(10, 4))

    bar_height = 0.2
    positions = [0.25, 0]  # Source on top, Destination below

    src_left = 0
    for i, (label, value) in enumerate(SRC_COMPONENTS):
        ax.barh(positions[0], value, bar_height, left=src_left, color=SRC_COLORS[i], linewidth=0)
        if value > 500:
            ax.text(src_left + value/2, positions[0], f'{value:.0f}',
                   ha='center', va='center', fontsize=16, fontweight='bold', color='black')
        src_left += value

    dst_left = 0
    for i, (label, value) in enumerate(DST_COMPONENTS):
        ax.barh(positions[1], value, bar_height, left=dst_left, color=DST_COLORS[i], linewidth=0)
        if value > 400:
            ax.text(dst_left + value/2, positions[1], f'{value:.0f}',
                   ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        dst_left += value

    src_total = sum(v for _, v in SRC_COMPONENTS)
    dst_total = sum(v for _, v in DST_COMPONENTS)
    ax.text(src_total + 100, positions[0], f'{src_total:.0f} ms', ha='left', va='center', fontsize=18, fontweight='bold')
    ax.text(dst_total + 100, positions[1], f'{dst_total:.0f} ms', ha='left', va='center', fontsize=18, fontweight='bold')

    ax.set_xlabel("Time (ms)", fontsize=22)
    ax.set_yticks(positions)
    ax.set_yticklabels(['Source', 'Destination'], fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', labelsize=20)
    ax.set_xlim(0, max(src_total, dst_total) * 1.15)
    ax.set_ylim(-0.15, 0.4)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)

    from matplotlib.patches import Patch
    src_handles = [Patch(facecolor=c, label=l) for c, (l, _) in zip(SRC_COLORS, SRC_COMPONENTS)]
    empty_handle = Patch(facecolor='none', edgecolor='none', label=' ')
    dst_handles = [Patch(facecolor=c, label=l) for c, (l, _) in zip(DST_COLORS, DST_COMPONENTS)]
    all_handles = src_handles + [empty_handle] + dst_handles

    ax.legend(handles=all_handles, loc='upper left', bbox_to_anchor=(0, 1.85),
             fontsize=16, frameon=False, ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.62)
    return fig


def main():
    print("Generating Figure 17 (testbed migration overhead breakdown)...")
    fig = plot_migration_overhead()
    output_path = OUTPUT_DIR / "fig17_migration_overhead.pdf"
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close(fig)

    src_total = sum(v for _, v in SRC_COMPONENTS)
    dst_total = sum(v for _, v in DST_COMPONENTS)
    print(f"Source total: {src_total:.2f} ms | Destination total: {dst_total:.2f} ms "
          f"| Combined: {(src_total + dst_total)/1000:.2f} s")


if __name__ == "__main__":
    main()
