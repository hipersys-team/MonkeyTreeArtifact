#!/usr/bin/env python3
"""
Generate synthetic initial cluster states for the golden_spine simulator.

Produces a snapshot file that can be passed as the INITIAL_SNAPSHOT arg to
golden_spine, pre-filling the cluster with jobs at the desired occupancy.

The algorithm:
  1. Pack jobs contiguously (block-aligned) until target occupancy is reached.
  2. Remove a fraction of jobs to create holes (slight fragmentation).
  3. Write the snapshot in the format golden_spine expects.

Job sizes are sampled uniformly from --gpu-counts. Iteration counts are
sampled uniformly from [--min-iterations, --max-iterations].

Usage:
  python build_initial_state.py \\
      --num-gpus 1024 --block-size 8 \\
      --gpu-counts 8 16 24 32 \\
      --target-occupancy 0.8 \\
      --job-type one_layer_moe \\
      --min-iterations 10 --max-iterations 500 \\
      --seed 42 --out initial_states/state_load08_rep1.snapshot
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple


def generate_initial_state(
    num_gpus: int,
    block_size: int,
    gpu_counts: List[int],
    target_occupancy: float,
    job_type: str,
    min_iterations: int,
    max_iterations: int,
    hole_fraction: float,
    seed: int,
    job_specs: List[Tuple[str, int]] = None,
    job_specs_weighted: List[Tuple[str, int, float]] = None,
) -> Tuple[List[Tuple[int, int, str, int]], dict]:
    """
    Returns (jobs, placements) where:
      jobs = [(job_id, num_workers, job_type, num_iterations), ...]
      placements = {job_id: [host_indices...]}

    Sampling modes (highest priority first):
      1. *job_specs_weighted*: list of (job_type, gpu_count, weight) —
         sampling uses ``random.choices`` with the given weights.
      2. *job_specs*: list of (job_type, gpu_count) — uniform sampling.
      3. Legacy: single *job_type* + uniform from *gpu_counts*.
    """
    rng = random.Random(seed)

    num_blocks = num_gpus // block_size
    fill_target = min(1.0, target_occupancy + hole_fraction)
    target_hosts = int(num_gpus * fill_target)

    # Pre-compute sampling helpers
    if job_specs_weighted:
        _specs = [(t, s) for t, s, _ in job_specs_weighted]
        _weights = [w for _, _, w in job_specs_weighted]
    elif job_specs:
        _specs = job_specs
        _weights = None
    else:
        _specs = None
        _weights = None

    block_used = [False] * num_blocks
    jobs = []
    placements = {}
    job_id = 0
    used_hosts = 0

    while used_hosts < target_hosts:
        if _specs is not None:
            if _weights is not None:
                (jtype, size), = rng.choices(_specs, weights=_weights, k=1)
            else:
                jtype, size = rng.choice(_specs)
        else:
            size = rng.choice(gpu_counts)
            jtype = job_type
        blocks_needed = size // block_size

        start = _find_contiguous_free(block_used, blocks_needed)
        if start is None:
            break

        hosts = []
        for b in range(start, start + blocks_needed):
            block_used[b] = True
            for h in range(b * block_size, (b + 1) * block_size):
                hosts.append(h)

        iters = rng.randint(min_iterations, max_iterations)
        jobs.append((job_id, size, jtype, iters))
        placements[job_id] = hosts
        used_hosts += size
        job_id += 1

    # Phase 2: punch holes by removing random jobs
    if hole_fraction > 0 and len(jobs) > 1:
        num_to_remove = max(1, int(len(jobs) * hole_fraction))
        to_remove = set(rng.sample(range(len(jobs)), min(num_to_remove, len(jobs) - 1)))
        new_jobs = []
        for i, j in enumerate(jobs):
            if i in to_remove:
                del placements[j[0]]
            else:
                new_jobs.append(j)
        jobs = new_jobs

    return jobs, placements


def _find_contiguous_free(block_used: List[bool], count: int):
    """Find the first run of `count` consecutive False entries."""
    run = 0
    for i, used in enumerate(block_used):
        if not used:
            run += 1
            if run >= count:
                return i - count + 1
        else:
            run = 0
    return None


def write_snapshot(
    path: Path,
    jobs: List[Tuple[int, int, str, int]],
    placements: dict,
):
    lines = []
    lines.append(str(len(jobs)))
    for job_id, num_workers, jtype, iters in jobs:
        lines.append(f"{job_id}: {num_workers} {jtype} {iters}")
    lines.append("")
    lines.append("Placement")

    host_to_job = {}
    for job_id, hosts in placements.items():
        for h in hosts:
            host_to_job[h] = job_id
    for h in sorted(host_to_job.keys()):
        lines.append(f"{h}: {host_to_job[h]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic initial cluster state snapshot"
    )
    parser.add_argument("--num-gpus", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument(
        "--gpu-counts", type=int, nargs="+", default=[8, 16, 24, 32],
        help="Allowed job sizes in GPUs (must be multiples of block-size)"
    )
    parser.add_argument(
        "--target-occupancy", type=float, default=0.8,
        help="Fraction of GPUs occupied after hole-punching (0.0-1.0)"
    )
    parser.add_argument("--job-type", type=str, default="one_layer_moe")
    parser.add_argument(
        "--job-specs", type=str, nargs="+", default=None,
        help="Per-type fixed sizes, e.g. 'seq16k/gpt_120b_32gpu:32 seq16k/gpt_20b_16gpu:16'. "
             "Overrides --job-type and --gpu-counts when provided."
    )
    parser.add_argument(
        "--job-specs-weighted", type=str, nargs="+", default=None,
        help="Weighted per-type fixed sizes, e.g. 'gpt_7b_dp:16:25.0 seq16k/gpt_20b_16gpu:16:70.0'. "
             "Format: type:gpu_count:weight. Overrides --job-specs when provided."
    )
    parser.add_argument("--min-iterations", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument(
        "--hole-fraction", type=float, default=0.15,
        help="Fraction of packed jobs to remove (creates fragmentation)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", "-o", type=Path, required=True)

    args = parser.parse_args()

    # Parse --job-specs-weighted into [(type, size, weight), ...] if provided
    parsed_weighted = None
    if args.job_specs_weighted:
        parsed_weighted = []
        for spec in args.job_specs_weighted:
            parts = spec.split(":")
            if len(parts) != 3:
                parser.error(f"Invalid --job-specs-weighted format: '{spec}'. Expected 'type:gpu_count:weight'")
            jtype, cnt, wt = parts[0], int(parts[1]), float(parts[2])
            assert cnt % args.block_size == 0, \
                f"GPU count {cnt} for {jtype} not a multiple of block_size {args.block_size}"
            parsed_weighted.append((jtype, cnt, wt))

    # Parse --job-specs into [(type, size), ...] if provided
    parsed_specs = None
    if args.job_specs and not parsed_weighted:
        parsed_specs = []
        for spec in args.job_specs:
            if ":" not in spec:
                parser.error(f"Invalid --job-specs format: '{spec}'. Expected 'type:gpu_count'")
            jtype, cnt = spec.rsplit(":", 1)
            cnt = int(cnt)
            assert cnt % args.block_size == 0, \
                f"GPU count {cnt} for {jtype} not a multiple of block_size {args.block_size}"
            parsed_specs.append((jtype, cnt))
    elif not parsed_weighted:
        for c in args.gpu_counts:
            assert c % args.block_size == 0, \
                f"GPU count {c} not a multiple of block_size {args.block_size}"

    jobs, placements = generate_initial_state(
        num_gpus=args.num_gpus,
        block_size=args.block_size,
        gpu_counts=args.gpu_counts,
        target_occupancy=args.target_occupancy,
        job_type=args.job_type,
        min_iterations=args.min_iterations,
        max_iterations=args.max_iterations,
        hole_fraction=args.hole_fraction,
        seed=args.seed,
        job_specs=parsed_specs,
        job_specs_weighted=parsed_weighted,
    )

    write_snapshot(args.out, jobs, placements)

    occupied = sum(len(h) for h in placements.values())
    print(f"Generated initial state: {args.out}")
    print(f"  Jobs: {len(jobs)}")
    print(f"  Occupied: {occupied}/{args.num_gpus} ({100*occupied/args.num_gpus:.1f}%)")
    sizes = {}
    types = {}
    for _, nw, jt, _ in jobs:
        sizes[nw] = sizes.get(nw, 0) + 1
        types[jt] = types.get(jt, 0) + 1
    print(f"  Size distribution: {dict(sorted(sizes.items()))}")
    if len(types) > 1:
        print(f"  Type distribution: {dict(sorted(types.items()))}")


if __name__ == "__main__":
    main()
