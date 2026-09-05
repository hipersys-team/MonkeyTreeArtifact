#!/usr/bin/env python3
"""
Experiment 2: slowdown vs. oversubscription ratio on a fixed 2048-GPU cluster.

Reproduces the structure behind Figure 8: fixes total cluster size at 2048
GPUs and sweeps GPUs-per-ToR (16..256) x number-of-spines (16..256),
skipping invalid combinations where spine count exceeds GPUs-per-ToR (a rack
can't have more uplinks than it has downlinks). At each valid (GPUs/ToR,
spines) point, replays the same job trace through five systems (ecmp, crux,
sglb, perfect, monkeytree_perfect) and records each job's slowdown.

Same orchestration approach as run_load_vs_slowdown.py -- this only drives
existing pieces of the repo, no core simulator/scheduler code is touched:
  - scripts/generate_trace.py     synthesizes the job-arrival traces
  - target/release/golden_spine   runs one trace through one system/topology
  - scripts/job_loader.py         supplies the per-job-type formulas used
                                   to recompute each job's ideal duration

The trace itself does not depend on GPUs-per-ToR or spine count (only on
total GPU count and load), so one trace is generated per rep and reused
across every (GPUs/ToR, spines) combination -- matching how the research
repo's generate_spine_sweep_traces.sh does it.

Output (under --output-dir, default results/spine_sweep_heatmap/):
  traces/rep{N}.trace                                  generated traces
  raw/tor{G}_spines{S}_rep{N}_{system}.out              raw golden_spine stdout
  per_job.csv                                           one row per completed job:
      system,gpus_per_tor,num_spines,rep,job_id,job_type,num_workers,
      num_iterations,ideal_us,real_runtime_us,slowdown

Run scripts/experiments/plot_spine_sweep_heatmap.py afterwards to chart it.

Usage:
    python3 scripts/experiments/run_spine_sweep_heatmap.py
    python3 scripts/experiments/run_spine_sweep_heatmap.py --num-jobs 10   # quick smoke test
    python3 scripts/experiments/run_spine_sweep_heatmap.py --num-jobs 1000 --reps 3 --workers 4
"""

import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = EXPERIMENTS_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent

sys.path.insert(0, str(SCRIPTS_DIR))
from job_loader import load_job_definitions          # noqa: E402
from generate_trace import compute_job_duration_us    # noqa: E402

# ---------------------------------------------------------------------------
# Fixed cluster size; topology varies per grid point below.
# ---------------------------------------------------------------------------
NUM_GPUS = 2048
BLOCK_SIZE = 8
BANDWIDTH_BPS = 400e9  # 400 Gbps; hard-coded inside golden_spine too
LOAD = 0.8

GPUS_PER_TOR_VALUES = [16, 32, 64, 128, 256]
NUM_SPINES_VALUES = [16, 32, 64, 128, 256]
SYSTEMS = ["ecmp", "crux", "sglb", "perfect", "monkeytree_perfect"]

# Realistic 10-job-type mix + per-type GPU counts, matching
# scripts/generate_worker_traces.sh in the research repo.
JOB_TYPES = [
    "llama2_70b_fsdp", "gpt_oss_20b_dp_moe", "llama2_70b_tp_dp",
    "llama2_70b_tp_dp_pp_2", "llama2_70b_tp_dp_pp_4", "llama3_70b_fsdp",
    "llama3_70b_tp_dp", "gpt_oss_120b_dp_moe", "gpt_7b_dp", "gpt_13b_dp",
]
JOB_WEIGHTS = [25.0, 1.0, 100.0, 25.0, 25.0, 25.0, 100.0, 1.0, 25.0, 25.0]
DEFAULT_GPU_COUNTS = [16, 24, 32]
JOB_GPU_COUNTS = {
    "llama2_70b_fsdp": [16, 24, 32],
    "gpt_oss_20b_dp_moe": [16],
    "llama2_70b_tp_dp": [16, 32],
    "llama2_70b_tp_dp_pp_2": [32, 48, 64],
    "llama2_70b_tp_dp_pp_4": [64],
    "llama3_70b_tp_dp": [16, 32],
    "gpt_oss_120b_dp_moe": [16],
}
MIN_ITERATIONS = 100
MAX_ITERATIONS = 1000

JOB_COMPLETE_RE = re.compile(
    r"JobComplete (\d+) submit=(\d+) schedule=(\d+) complete=(\d+) "
    r"wait=(\d+)us real_runtime=(\d+)us total_runtime=(\d+)us"
)


def valid_combos():
    """(gpus_per_tor, num_spines) pairs where a rack can't have more uplinks
    (spines) than downlinks (GPUs beneath it)."""
    return [
        (tor, spines)
        for tor in GPUS_PER_TOR_VALUES
        for spines in NUM_SPINES_VALUES
        if spines <= tor
    ]


def build_binary() -> str:
    binary = REPO_ROOT / "target" / "release" / "golden_spine"
    print("Building golden_spine (cargo build --release --bin golden_spine)...")
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "golden_spine"],
        cwd=REPO_ROOT, check=True,
    )
    if not binary.exists():
        raise RuntimeError(f"Build finished but binary not found at {binary}")
    return str(binary)


def generate_trace(rep: int, num_jobs: int, seed: int, out_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "generate_trace.py"),
        "--output", str(out_path),
        "--load", str(LOAD),
        "--num-jobs", str(num_jobs),
        "--num-gpus", str(NUM_GPUS),
        "--bandwidth", str(BANDWIDTH_BPS),
        "--job-types", *JOB_TYPES,
        "--job-weights", *[str(w) for w in JOB_WEIGHTS],
        "--gpu-counts", *[str(g) for g in DEFAULT_GPU_COUNTS],
        "--job-gpu-counts", *[f"{t}:{','.join(str(c) for c in cs)}" for t, cs in JOB_GPU_COUNTS.items()],
        "--min-iterations", str(MIN_ITERATIONS),
        "--max-iterations", str(MAX_ITERATIONS),
        "--seed", str(seed),
        "--quiet",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_simulation(binary: str, trace_path: str, system: str, gpus_per_tor: int,
                    num_spines: int, out_path: str) -> dict:
    """Run one (trace, system, topology) combination through golden_spine."""
    # golden_spine <TRACE> <SYSTEM> <MAX_JOBS> <BLOCK_SIZE> <MAX_SLOWDOWN>
    #              <NUM_SPINES> <NUM_GPUS> <THRESHOLD> <POST_MIGRATION_DELAY_US>
    #              <GPUS_PER_TOR> [INITIAL_SNAPSHOT] [SCHEDULER]
    cmd = [
        binary, trace_path, system,
        "0",                    # max_jobs: load all
        str(BLOCK_SIZE),
        "0",                    # max_slowdown: no cap
        str(num_spines),
        str(NUM_GPUS),
        "0",                    # threshold: default to num_spines
        "0",                    # post_migration_delay_us: none
        str(gpus_per_tor),
    ]
    with open(out_path, "w") as f:
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
    return {"trace_path": trace_path, "system": system, "success": result.returncode == 0}


def parse_trace_jobs(trace_path: Path):
    """job_id -> (job_type, num_workers, num_iterations), 0-indexed by line order."""
    lines = trace_path.read_text().splitlines()
    n = int(lines[0])
    jobs = {}
    for job_id, line in enumerate(lines[1:1 + n]):
        _arrival, job_type, num_workers, num_iterations = line.split()
        jobs[job_id] = (job_type, int(num_workers), int(num_iterations))
    return jobs


def parse_completions(out_path: Path):
    """job_id -> real_runtime_us, from JobComplete lines in golden_spine's stdout."""
    completions = {}
    for line in out_path.read_text(errors="replace").splitlines():
        m = JOB_COMPLETE_RE.search(line)
        if m:
            job_id = int(m.group(1))
            real_runtime_us = int(m.group(6))
            completions[job_id] = real_runtime_us
    return completions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-jobs", type=int, default=1000, help="Jobs per trace (default: 1000)")
    parser.add_argument("--reps", type=int, default=1, help="Repetitions (default: 1)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel simulation runs (default: 4)")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "spine_sweep_heatmap")
    parser.add_argument("--skip-build", action="store_true", help="Reuse the existing target/release binary")
    args = parser.parse_args()

    trace_dir = args.output_dir / "traces"
    raw_dir = args.output_dir / "raw"
    trace_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    binary = str(REPO_ROOT / "target" / "release" / "golden_spine") if args.skip_build else build_binary()

    combos = valid_combos()

    # ---- 1. Generate traces (one per rep, reused across all topology combos) ----
    print(f"\nGenerating traces: {args.reps} rep(s), {args.num_jobs} jobs each, load={LOAD}...")
    trace_paths = {}  # rep -> Path
    for rep in range(1, args.reps + 1):
        seed = rep * 1000
        trace_path = trace_dir / f"rep{rep}.trace"
        generate_trace(rep, args.num_jobs, seed, trace_path)
        trace_paths[rep] = trace_path
        print(f"  {trace_path.name} (seed={seed})")

    # ---- 2. Run all (trace, system, topology) combinations ------------------
    jobs = []
    for rep, trace_path in trace_paths.items():
        for gpus_per_tor, num_spines in combos:
            for system in SYSTEMS:
                out_path = raw_dir / f"tor{gpus_per_tor}_spines{num_spines}_rep{rep}_{system}.out"
                jobs.append((trace_path, system, gpus_per_tor, num_spines, rep, out_path))

    print(f"\nRunning {len(jobs)} simulations ({len(combos)} topology combos x {len(SYSTEMS)} systems x "
          f"{args.reps} rep(s)) with {args.workers} worker(s)...")
    completed, failed = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_simulation, binary, str(t), s, tor, spines, str(o)): (t, s, tor, spines, rep)
            for t, s, tor, spines, rep, o in jobs
        }
        for future in as_completed(futures):
            trace_path, system, tor, spines, rep = futures[future]
            completed += 1
            result = future.result()
            status = "OK" if result["success"] else "FAILED"
            if not result["success"]:
                failed += 1
            print(f"  [{completed}/{len(jobs)}] {status} tor={tor} spines={spines} rep={rep} + {system}")

    if failed:
        print(f"\nWARNING: {failed}/{len(jobs)} simulation runs failed. Check raw/*.out for details.")

    # ---- 3. Parse results and compute per-job slowdown -----------------------
    print("\nParsing results and computing per-job slowdown...")
    job_definitions = load_job_definitions()

    csv_path = args.output_dir / "per_job.csv"
    rows_written = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "system", "gpus_per_tor", "num_spines", "rep", "job_id", "job_type",
            "num_workers", "num_iterations", "ideal_us", "real_runtime_us", "slowdown",
        ])
        for rep, trace_path in trace_paths.items():
            trace_jobs = parse_trace_jobs(trace_path)
            for gpus_per_tor, num_spines in combos:
                for system in SYSTEMS:
                    out_path = raw_dir / f"tor{gpus_per_tor}_spines{num_spines}_rep{rep}_{system}.out"
                    if not out_path.exists():
                        continue
                    completions = parse_completions(out_path)
                    for job_id, real_runtime_us in completions.items():
                        if job_id not in trace_jobs:
                            continue  # shouldn't happen with no initial snapshot
                        job_type, num_workers, num_iterations = trace_jobs[job_id]
                        job_def = job_definitions.get(job_type)
                        if job_def is None:
                            continue
                        ideal_us = compute_job_duration_us(job_def, num_workers, num_iterations, BANDWIDTH_BPS)
                        if ideal_us <= 0:
                            continue
                        slowdown = real_runtime_us / ideal_us
                        writer.writerow([
                            system, gpus_per_tor, num_spines, rep, job_id, job_type, num_workers,
                            num_iterations, f"{ideal_us:.2f}", real_runtime_us, f"{slowdown:.6f}",
                        ])
                        rows_written += 1

    print(f"Wrote {rows_written} per-job rows to {csv_path}")
    print(f"\nDone. Next: python3 {EXPERIMENTS_DIR / 'plot_spine_sweep_heatmap.py'} "
          f"--input {csv_path}")


if __name__ == "__main__":
    main()
