# Paper figures

Each script below reproduces one figure from the paper. All values are
embedded directly in the script (or in a small bundled JSON file sitting
next to it) -- none of them require the simulator's raw `results/` logs.
Run any script directly, e.g.:

```
pip install -r plots/requirements.txt
python3 plots/fig06_main_slowdown.py
```

Each writes its output PDF into `plots/` next to the script.

| Figure | Script | Paper section | Notes |
|---|---|---|---|
| 6  | `fig06_main_slowdown.py` | 6.2 | Main result: avg/p90/p99 slowdown vs. load, 1024 GPUs, 9 systems |
| 7  | `fig07_fragmentation.py` (+ `fig07_fragmentation_data.json`) | 6.2 | Cluster-wide & per-rack fragmentation over time, 70-100% load |
| 8  | `fig08_spine_sweep_heatmaps.py` | 6.3 | Avg/p99 slowdown heatmaps across oversubscription ratios, 2048 GPUs |
| 9  | `fig09_fig11_scale_migration.py` | 6.4 | Distribution of migrations needed, cluster sizes 128-2048 GPUs |
| 10 | `fig10_ilp_solve_time_boxplot.py` (+ `fig10_ilp_solve_times.json`) | 6.4 | ILP solve time vs. number of migrations, 1024 GPUs. Plots the 3,790 measured solve times behind the paper figure; `--from-logs` plots your own run (see below) |
| 11 | `fig09_fig11_scale_migration.py` | 6.4 | ILP solve time vs. cluster size, 128-2048 GPUs (same script as Fig. 9) |
| 12 | `fig12_migration_duration_cdf.py` (+ `fig12_migration_duration_data.json`) | 6.5 | CDF of migration duration at 70/80/90/100% load |
| 13 | `fig13_lambda_threshold.py` | 6.6 | Migrations & slowdown vs. fragmentation threshold, 1024 GPUs |
| 14a | `fig14a_rail_optimized.py` | 6.7 | P99 slowdown vs. load, rail-optimized topology |
| 14b | `fig14b_3tier_clos.py` | 6.7 | P99 slowdown vs. load, 3-tier Clos topology |
| 15 | `fig15_scheduler_ablation.py` | 6.8 | Migrations performed vs. load, front-end scheduler ablation |
| 16 | `fig16_ep_dominated.py` | 6.9 | Mean/p99 slowdown vs. oversubscription ratio, EP-dominated workloads |
| 17 | `fig17_migration_overhead.py` | 7 | Testbed migration timing breakdown (Llama3-8B worker) |

Figures 1-5 are architecture/illustration diagrams, not experimental
results, and have no generating script.

## Regenerating Figure 10

`fig10_ilp_solve_times.json` holds every measured ILP solve behind the paper's
Figure 10: one `[run, num_moves, solve_time_ms]` row per solve, 3,790 in all.
They come from the `monkeytree_perfect` runs at 1024 GPUs (loads 0.1-1.0, 5
reps). Only loads 0.7 and above triggered migrations.

To measure solve times yourself, run the load-vs-slowdown experiment. Its
`monkeytree_perfect` runs log every ILP solve. Then plot from those logs:

```
python3 scripts/experiments/run_load_vs_slowdown.py --reps 5
python3 plots/fig10_ilp_solve_time_boxplot.py --from-logs results/load_vs_slowdown/raw
```

This writes `fig10_ilp_solve_time_boxplot_regenerated.pdf` and the parsed
samples to `fig10_ilp_solve_times_regenerated.json`. The paper runs used
longer traces (~2,400 jobs vs. the default 1,000), so expect fewer solves
(one 1,000-job rep gives ~170). Absolute solve times depend on CPU and CBC
version; compare how solve time grows with the number of migrations.

## Provenance

Ported from two paper-source repos (not this codebase): `MonkeyTree-Final/scripts/`
(Figs. 6, 14a, 14b, 15, 16) and `MTree-sigcomm26/figures/` (Figs. 7-13, 17).
Each script's docstring names its exact source file. A handful of these
(Figs. 7, 8, 9, 11, 13) come from an earlier paper draft and showed minor
numeric drift against the current camera-ready PDF when spot-checked --
worth a final numeric check against the paper before relying on them for
anything beyond reproducing the shape of each result.
