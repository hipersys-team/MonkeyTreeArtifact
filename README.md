# MonkeyTree Artifact

## Compile
Install Rust & Cargo. The repository does not require anything unusual other than a sufficiently up-to-date version of COIN-OR. It can be installed via Conda:

```
conda install -c conda-forge coin-or-cbc
```

Compiling the project:
```
cargo build --release --bin golden_spine
```

## Load vs. Slowdown Experiment
The first main result varies the load and measures the slowdown values of traces of jobs across different systems. Results are written to `results/load_vs_slowdown/`.
```
pip install -r scripts/requirements.txt

python3 scripts/experiments/run_load_vs_slowdown.py --num-jobs 1000
python3 scripts/experiments/run_load_vs_slowdown.py

python3 scripts/experiments/plot_load_vs_slowdown.py
```

## Spine Sweep Heatmap Experiment
This experiment varies the oversubscription ratio by iterating over the number of spine switches and the number of GPUs per ToR. Results are written to `results/spine_sweep_heatmap/`.

```
pip install -r scripts/requirements.txt

python3 scripts/experiments/run_spine_sweep_heatmap.py --num-jobs 1000
python3 scripts/experiments/run_spine_sweep_heatmap.py

python3 scripts/experiments/plot_spine_sweep_heatmap.py
```

## Paper Figures
Each script reproduces one figure from the paper, and writes its image file into `plots/` alongside the script.

```
pip install -r plots/requirements.txt

for f in plots/fig*.py; do python3 "$f"; done
```
