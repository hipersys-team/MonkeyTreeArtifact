#!/usr/bin/env python3
"""
Generate trace files for the golden_spine simulator.

Traces are generated using a Poisson arrival process based on target load.
Job types are loaded from YAML files in the jobs/ directory.

GPU counts must be multiples of 8 (one block = 8 GPUs = 1 physical server).
"""

import argparse
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from job_loader import load_job_definitions, get_job_definitions_dict

# Block size: 8 GPUs per physical server
BLOCK_SIZE = 8


def compute_allreduce_traffic_bytes(num_workers: int, model_size_bytes: int) -> float:
    """
    Compute the total data transferred per worker in a ring allreduce.
    
    Formula: 2 * (N - 1) / N * D
    
    where:
    - N = number of workers
    - D = model size in bytes
    
    This accounts for both the reduce-scatter and all-gather phases.
    """
    n = num_workers
    return 2.0 * (n - 1) / n * model_size_bytes


def compute_strided_ring_traffic_bytes(model_size: int, num_workers: int, stride: int) -> float:
    """
    Compute the traffic per worker for a strided ring AllReduce.
    
    The ring size is N/stride workers.
    Data per worker = 2 * (ring_size - 1) / ring_size * model_size
    
    If ring_size <= 1, there's no communication.
    """
    ring_size = num_workers // stride if stride > 0 else 1
    if ring_size <= 1:
        return 0
    # Apply AllReduce scaling for the ring size
    return 2.0 * (ring_size - 1) / ring_size * model_size


def compute_job_duration_us(
    job_def: Dict[str, Any],
    num_workers: int,
    num_iterations: int,
    bandwidth_bps: float
) -> float:
    """
    Compute the ideal job duration (no network congestion) in microseconds.
    
    Each iteration consists of:
    - Compute phase: compute_time_us
    - Network phases: sum of all collective operations
    
    Total duration = num_iterations * (compute_time + network_time)
    """
    compute_time_us = job_def["compute_time_us"]
    model_size_bytes = job_def["model_size_bytes"]
    
    # Calculate network time for each collective
    network_time_us = 0.0
    
    # AllReduce: 2 * (n-1) / n * model_size
    if job_def.get("has_all_reduce", False) or job_def.get("strategy") == "all_reduce":
        allreduce_traffic = compute_allreduce_traffic_bytes(num_workers, model_size_bytes)
        network_time_us += (allreduce_traffic * 8) / bandwidth_bps * 1_000_000
    
    # All-to-All: total data is divided across n*(n-1) flows
    # Each worker sends total / n bytes
    all_to_all_size = job_def.get("all_to_all_size_bytes", 0)
    if all_to_all_size > 0 or (job_def.get("strategy") == "all_to_all" and all_to_all_size == 0):
        # For backwards compat: if strategy is all_to_all but no explicit size, use model_size
        if all_to_all_size == 0:
            all_to_all_size = model_size_bytes
        per_worker_traffic = all_to_all_size / num_workers if num_workers > 1 else 0
        network_time_us += (per_worker_traffic * 8) / bandwidth_bps * 1_000_000
    
    # Strided Ring AllReduce: data = 2 * (ring_size - 1) / ring_size * model_size
    strided_ring_steps = job_def.get("strided_ring_steps", [])
    for step in strided_ring_steps:
        model_size = step.get("model_size_bytes", 0)
        stride = step.get("stride", 1)
        traffic = compute_strided_ring_traffic_bytes(model_size, num_workers, stride)
        network_time_us += (traffic * 8) / bandwidth_bps * 1_000_000
    
    # Pipeline parallel: activations/gradients between stages + optional TP/DP AllReduce
    pipeline_steps = job_def.get("pipeline_steps", [])
    for step in pipeline_steps:
        num_stages = step.get("num_stages", 1)
        tp_size = step.get("tp_size", 1)
        microbatches = step.get("microbatches", 1)
        activation_size = step.get("activation_size_bytes", 1_000_000_000)
        model_shard_bytes = step.get("model_shard_bytes", 0)
        
        # Compute dp_replicas from num_workers if not specified
        workers_per_replica = num_stages * tp_size
        dp_replicas = step.get("dp_replicas")
        if dp_replicas is None:
            dp_replicas = num_workers // workers_per_replica if workers_per_replica > 0 else 1
        
        # PP: Each microbatch sends activations forward and gradients backward
        # Per-worker traffic: 2 * microbatches * activation_size for intermediate stages
        # (first stage doesn't receive, last stage doesn't send forward, etc.)
        if num_stages > 1:
            pp_traffic = 2 * microbatches * activation_size  # simplified: send + recv
            network_time_us += (pp_traffic * 8) / bandwidth_bps * 1_000_000
        
        # DP: AllReduce at end of iteration
        if dp_replicas > 1 and model_shard_bytes > 0:
            d = dp_replicas
            dp_allreduce_traffic = 2 * (d - 1) / d * model_shard_bytes
            network_time_us += (dp_allreduce_traffic * 8) / bandwidth_bps * 1_000_000
    
    iteration_time_us = compute_time_us + network_time_us
    total_duration_us = num_iterations * iteration_time_us
    
    return total_duration_us


def parse_gpu_counts(gpu_counts_str: str) -> List[int]:
    """
    Parse a comma-separated string of GPU counts into a list of integers.
    
    Example: "8,16,24,32" -> [8, 16, 24, 32]
    """
    counts = []
    for part in gpu_counts_str.split(","):
        count = int(part.strip())
        if count % BLOCK_SIZE != 0:
            raise ValueError(f"GPU count {count} is not a multiple of {BLOCK_SIZE}")
        if count < BLOCK_SIZE:
            raise ValueError(f"GPU count {count} must be at least {BLOCK_SIZE}")
        counts.append(count)
    return sorted(set(counts))


def parse_job_gpu_counts(job_gpu_counts_args: List[str]) -> Dict[str, List[int]]:
    """
    Parse per-job-type GPU counts from command-line arguments.
    
    Format: ["s1:8,16,24", "s2:8,16,32,64"]
    Returns: {"s1": [8, 16, 24], "s2": [8, 16, 32, 64]}
    """
    result = {}
    for arg in job_gpu_counts_args:
        if ":" not in arg:
            raise ValueError(f"Invalid job GPU counts format: {arg}. Expected 'job_type:count1,count2,...'")
        job_type, counts_str = arg.split(":", 1)
        result[job_type.strip()] = parse_gpu_counts(counts_str)
    return result


def get_gpu_counts_for_job(
    job_type: str,
    job_gpu_counts: Dict[str, List[int]],
    default_gpu_counts: List[int],
    job_def: Optional[Dict[str, Any]] = None
) -> List[int]:
    """
    Get the allowed GPU counts for a specific job type.
    
    For pipeline jobs with variable size, validates that counts are valid multiples
    of workers_per_replica. Raises an error if no valid counts are found.
    
    Uses job-specific counts if defined, otherwise falls back to default.
    """
    # Start with explicit counts or defaults
    counts = job_gpu_counts.get(job_type, default_gpu_counts)
    
    # For variable-size pipeline jobs, validate counts
    if job_def is not None and job_def.get("pipeline_variable_size", False):
        workers_per_replica = job_def.get("pipeline_workers_per_replica", 1)
        pipeline_step = job_def.get("pipeline_steps", [{}])[0]
        num_stages = pipeline_step.get("num_stages", 1)
        tp_size = pipeline_step.get("tp_size", 1)
        
        # GPU counts must satisfy two constraints:
        # 1. Must be a multiple of workers_per_replica (so dp_replicas is an integer)
        # 2. workers_per_stage must be a multiple of BLOCK_SIZE (for block-aligned scheduling)
        #    workers_per_stage = num_gpus / num_stages, so num_gpus must be multiple of (num_stages * BLOCK_SIZE)
        min_gpus = num_stages * BLOCK_SIZE
        
        def is_valid_gpu_count(c):
            if c % workers_per_replica != 0 or c < workers_per_replica:
                return False
            # workers_per_stage = c / num_stages (since dp_replicas = c / workers_per_replica
            # and workers_per_stage = tp_size * dp_replicas = tp_size * c / (num_stages * tp_size) = c / num_stages)
            workers_per_stage = c // num_stages
            return workers_per_stage % BLOCK_SIZE == 0
        
        valid_counts = [c for c in counts if is_valid_gpu_count(c)]
        invalid_counts = [c for c in counts if c not in valid_counts]
        
        if not valid_counts:
            raise ValueError(
                f"Job type '{job_type}' requires GPU counts that are multiples of {min_gpus} "
                f"(num_stages={num_stages} × BLOCK_SIZE={BLOCK_SIZE}) to ensure block-aligned scheduling. "
                f"Requested counts {counts} have no valid options. "
                f"Try: {[min_gpus * i for i in range(1, 5)]}"
            )
        
        if invalid_counts:
            raise ValueError(
                f"Job type '{job_type}' requires GPU counts that are multiples of {min_gpus} "
                f"(num_stages={num_stages} × BLOCK_SIZE={BLOCK_SIZE}). "
                f"Invalid counts: {invalid_counts}. Valid from requested: {valid_counts}."
            )
        
        counts = valid_counts
    
    # For fixed-size pipeline jobs, validate the fixed worker count is in the list
    elif job_def is not None and job_def.get("pipeline_workers") is not None:
        fixed_workers = job_def["pipeline_workers"]
        if fixed_workers not in counts:
            raise ValueError(
                f"Job type '{job_type}' has fixed size of {fixed_workers} workers, "
                f"but this is not in the allowed GPU counts: {counts}"
            )
        counts = [fixed_workers]
    
    return counts


def compute_expected_server_us(
    job_definitions: Dict[str, Dict[str, Any]],
    job_types: List[str],
    job_weights: List[float],
    job_gpu_counts: Dict[str, List[int]],
    default_gpu_counts: List[int],
    iteration_range: Tuple[int, int],
    bandwidth_bps: float
) -> float:
    """
    Compute the expected server-microseconds per job.
    
    E[server_us] = E[duration(job_type, num_gpus, num_iterations) * num_gpus]
    
    Each job type can have its own set of allowed GPU counts.
    """
    total_weight = sum(job_weights)
    normalized_weights = [w / total_weight for w in job_weights]
    
    min_iters, max_iters = iteration_range
    # For uniform distribution, E[iterations] = (min + max) / 2
    avg_iterations = (min_iters + max_iters) / 2.0
    
    expected_server_us = 0.0
    
    for jt, jw in zip(job_types, normalized_weights):
        job_def = job_definitions[jt]
        gpu_counts = get_gpu_counts_for_job(jt, job_gpu_counts, default_gpu_counts, job_def)
        num_gpu_values = len(gpu_counts)
        
        for num_gpus in gpu_counts:
            # Probability of this (job_type, num_gpus) combination
            prob = jw * (1.0 / num_gpu_values)
            
            # Duration for this specific combination (using average iterations)
            duration = compute_job_duration_us(job_def, num_gpus, int(avg_iterations), bandwidth_bps)
            
            # Server-us = duration * num_gpus (GPU-us actually)
            server_us = duration * num_gpus
            
            expected_server_us += prob * server_us
    
    return expected_server_us


def generate_trace(
    job_definitions: Dict[str, Dict[str, Any]],
    num_jobs: int,
    target_load: float,
    num_gpus: int,
    bandwidth_bps: float,
    job_types: List[str],
    job_weights: List[float],
    job_gpu_counts: Dict[str, List[int]],
    default_gpu_counts: List[int],
    iteration_range: Tuple[int, int],
    seed: int
) -> List[Tuple[int, str, int, int]]:
    """
    Generate a trace with Poisson arrivals based on target load.
    
    Load is defined as the ratio of GPU utilization:
    ρ = λ * E[gpu_us_per_job] / num_gpus
    
    where λ is the arrival rate (jobs per us).
    
    Solving for λ:
    λ = ρ * num_gpus / E[gpu_us_per_job]
    
    Returns list of (arrival_time_us, job_type, num_gpus, num_iterations).
    GPU counts are always multiples of BLOCK_SIZE (8).
    """
    random.seed(seed)
    
    # Compute expected GPU-us per job (using average iterations)
    expected_gpu_us = compute_expected_server_us(
        job_definitions, job_types, job_weights, job_gpu_counts, default_gpu_counts, iteration_range, bandwidth_bps
    )
    
    # Compute arrival rate (jobs per us)
    if expected_gpu_us <= 0:
        raise ValueError("Expected GPU-us must be positive")
    
    arrival_rate = target_load * num_gpus / expected_gpu_us
    
    # Mean inter-arrival time (us)
    mean_interarrival = 1.0 / arrival_rate if arrival_rate > 0 else float('inf')
    
    # Compute expected duration for display
    min_iters, max_iters = iteration_range
    avg_iters = (min_iters + max_iters) / 2.0
    
    # Compute average GPU count across all job types
    total_gpu_count = 0
    total_count = 0
    for jt in job_types:
        job_def = job_definitions[jt]
        gpu_counts = get_gpu_counts_for_job(jt, job_gpu_counts, default_gpu_counts, job_def)
        total_gpu_count += sum(gpu_counts)
        total_count += len(gpu_counts)
    avg_gpus = total_gpu_count / total_count if total_count > 0 else BLOCK_SIZE
    
    expected_duration = expected_gpu_us / avg_gpus
    
    print(f"Load calculation:")
    print(f"  Average iterations: {avg_iters:.0f}")
    print(f"  Average GPUs per job: {avg_gpus:.1f}")
    print(f"  Expected job duration: {expected_duration:.2f} us ({expected_duration / 1_000_000:.2f} s)")
    print(f"  Expected GPU-us per job: {expected_gpu_us:.2f}")
    print(f"  Arrival rate: {arrival_rate:.9f} jobs/us ({arrival_rate * 1_000_000:.4f} jobs/s)")
    print(f"  Mean inter-arrival time: {mean_interarrival:.2f} us ({mean_interarrival / 1_000_000:.2f} s)")
    
    jobs = []
    current_time_us = 0.0
    
    for _ in range(num_jobs):
        # Sample job type based on weights
        job_type = random.choices(job_types, weights=job_weights, k=1)[0]
        job_def = job_definitions[job_type]
        
        # Sample number of GPUs from allowed counts for this job type
        gpu_counts = get_gpu_counts_for_job(job_type, job_gpu_counts, default_gpu_counts, job_def)
        job_num_gpus = random.choice(gpu_counts)
        
        # Sample number of iterations uniformly
        num_iterations = random.randint(min_iters, max_iters)
        
        # Record job at current time
        jobs.append((int(current_time_us), job_type, job_num_gpus, num_iterations))
        
        # Sample inter-arrival time from exponential distribution
        interarrival = random.expovariate(arrival_rate) if arrival_rate > 0 else 0
        current_time_us += interarrival
    
    return jobs


def write_trace(jobs: List[Tuple[int, str, int, int]], output_path: Path) -> None:
    """Write trace to file in the format expected by golden_spine."""
    with output_path.open("w", encoding="utf-8") as f:
        # First line: number of jobs
        f.write(f"{len(jobs)}\n")
        
        # Subsequent lines: arrival_time job_type num_workers num_iterations
        for arrival_time, job_type, num_workers, num_iterations in jobs:
            f.write(f"{arrival_time} {job_type} {num_workers} {num_iterations}\n")


def main() -> None:
    # Load job definitions from YAML
    try:
        job_definitions = load_job_definitions()
        available_job_types = sorted(job_definitions.keys())
    except FileNotFoundError:
        print("Warning: Could not load job definitions from YAML files.")
        print("Using hardcoded fallback definitions.")
        job_definitions = {
            "s1": {"compute_time_us": 500, "model_size_bytes": 6_875_000_000},
            "s2": {"compute_time_us": 1000, "model_size_bytes": 3_750_000_000},
            "s3": {"compute_time_us": 800, "model_size_bytes": 5_000_000_000},
        }
        available_job_types = ["s1", "s2", "s3"]
    
    parser = argparse.ArgumentParser(
        description="Generate trace files for golden_spine simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GPU Count Specification:
  GPU counts must be multiples of 8 (one block = 8 GPUs = 1 physical server).
  
  Use --gpu-counts to set the default allowed GPU counts for all job types:
    --gpu-counts 8 16 24 32
  
  Use --job-gpu-counts to override GPU counts for specific job types:
    --job-gpu-counts s1:8,16,24 s2:8,16,32,64
  
  Job types not specified in --job-gpu-counts will use --gpu-counts.

Examples:
  # All jobs use 8, 16, or 24 GPUs
  python generate_trace.py -o trace.txt --gpu-counts 8 16 24
  
  # s1 jobs use 8-32 GPUs, s2 jobs use only 64 GPUs, s3 uses default
  python generate_trace.py -o trace.txt --gpu-counts 8 16 \\
      --job-gpu-counts s1:8,16,24,32 s2:64
"""
    )
    
    # Required arguments
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output trace file path"
    )
    
    # Load parameters
    parser.add_argument(
        "--load",
        type=float,
        default=0.5,
        help="Target average load (0.0 to 1.0+, where 1.0 means 100%% GPU utilization)"
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=100,
        help="Number of jobs to generate"
    )
    
    # Cluster parameters
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=768,
        help="Total number of GPUs in the cluster (must be multiple of 8)"
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=50e9,
        help="Network bandwidth in bits per second (default: 50 Gbps)"
    )
    
    # Job parameters
    parser.add_argument(
        "--job-types",
        type=str,
        nargs="+",
        default=["s1", "s2", "s3"],
        help=f"Job types to include in the trace. Available: {available_job_types}"
    )
    parser.add_argument(
        "--job-weights",
        type=float,
        nargs="+",
        default=None,
        help="Relative weights for job types (default: equal weights)"
    )
    parser.add_argument(
        "--gpu-counts",
        type=int,
        nargs="+",
        default=[8, 16, 24, 32],
        help="Default allowed GPU counts for jobs (must be multiples of 8)"
    )
    parser.add_argument(
        "--job-gpu-counts",
        type=str,
        nargs="+",
        default=[],
        help="Per-job-type GPU counts. Format: 'job_type:count1,count2,...' (e.g., 's1:8,16,24')"
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=100,
        help="Minimum number of iterations per job"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum number of iterations per job"
    )
    
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Verbosity
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress informational output"
    )
    
    # List available job types
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="List available job types and exit"
    )
    
    args = parser.parse_args()
    
    # Handle --list-jobs
    if args.list_jobs:
        print("Available job types:")
        for job_type in available_job_types:
            job_def = job_definitions[job_type]
            print(f"  {job_type}: compute={job_def['compute_time_us']}us, "
                  f"model={job_def['model_size_bytes']/1e9:.2f}GB")
        return
    
    # Validate arguments
    if args.load <= 0:
        parser.error("Load must be positive")
    if args.num_gpus <= 0:
        parser.error("Number of GPUs must be positive")
    if args.num_gpus % BLOCK_SIZE != 0:
        parser.error(f"Number of GPUs ({args.num_gpus}) must be a multiple of {BLOCK_SIZE}")
    if args.bandwidth <= 0:
        parser.error("Bandwidth must be positive")
    if args.min_iterations < 1:
        parser.error("Minimum iterations must be at least 1")
    if args.max_iterations < args.min_iterations:
        parser.error("Maximum iterations must be >= minimum iterations")
    
    # Validate and parse default GPU counts
    for count in args.gpu_counts:
        if count % BLOCK_SIZE != 0:
            parser.error(f"GPU count {count} is not a multiple of {BLOCK_SIZE}")
        if count < BLOCK_SIZE:
            parser.error(f"GPU count {count} must be at least {BLOCK_SIZE}")
        if count > args.num_gpus:
            parser.error(f"GPU count {count} exceeds total cluster GPUs ({args.num_gpus})")
    
    default_gpu_counts = sorted(set(args.gpu_counts))
    
    # Parse per-job-type GPU counts
    try:
        job_gpu_counts = parse_job_gpu_counts(args.job_gpu_counts)
    except ValueError as e:
        parser.error(str(e))
    
    # Validate per-job-type GPU counts
    for job_type, counts in job_gpu_counts.items():
        if job_type not in available_job_types:
            parser.error(f"Unknown job type in --job-gpu-counts: {job_type}. Available: {available_job_types}")
        for count in counts:
            if count > args.num_gpus:
                parser.error(f"GPU count {count} for {job_type} exceeds total cluster GPUs ({args.num_gpus})")
    
    # Validate job types exist
    for jt in args.job_types:
        if jt not in job_definitions:
            parser.error(f"Unknown job type: {jt}. Available: {available_job_types}")
    
    # Default job weights if not specified
    job_weights = args.job_weights
    if job_weights is None:
        job_weights = [1.0] * len(args.job_types)
    elif len(job_weights) != len(args.job_types):
        parser.error(f"Number of job weights ({len(job_weights)}) must match job types ({len(args.job_types)})")
    
    if not args.quiet:
        print(f"Generating trace with:")
        print(f"  Target load: {args.load:.2%}")
        print(f"  Number of jobs: {args.num_jobs}")
        print(f"  Total GPUs: {args.num_gpus} ({args.num_gpus // BLOCK_SIZE} servers)")
        print(f"  Bandwidth: {args.bandwidth / 1e9:.1f} Gbps")
        print(f"  Job types: {args.job_types} (weights: {job_weights})")
        print(f"  Default GPU counts: {default_gpu_counts}")
        if job_gpu_counts:
            print(f"  Per-job GPU counts: {job_gpu_counts}")
        print(f"  Iterations per job: {args.min_iterations}-{args.max_iterations}")
        print(f"  Seed: {args.seed}")
        print()
    
    # Generate trace
    jobs = generate_trace(
        job_definitions=job_definitions,
        num_jobs=args.num_jobs,
        target_load=args.load,
        num_gpus=args.num_gpus,
        bandwidth_bps=args.bandwidth,
        job_types=args.job_types,
        job_weights=job_weights,
        job_gpu_counts=job_gpu_counts,
        default_gpu_counts=default_gpu_counts,
        iteration_range=(args.min_iterations, args.max_iterations),
        seed=args.seed
    )
    
    # Write trace
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_trace(jobs, args.output)
    
    if not args.quiet:
        print()
        print(f"Trace written to: {args.output}")
        print(f"  Total jobs: {len(jobs)}")
        if jobs:
            print(f"  Time span: {jobs[0][0]} - {jobs[-1][0]} us")
            
            # Job type distribution
            type_counts = {}
            gpu_counts_dist = {}
            iteration_stats = []
            for _, jt, ng, ni in jobs:
                type_counts[jt] = type_counts.get(jt, 0) + 1
                gpu_counts_dist[ng] = gpu_counts_dist.get(ng, 0) + 1
                iteration_stats.append(ni)
            
            print(f"  Job type distribution: {type_counts}")
            print(f"  GPU count distribution: {dict(sorted(gpu_counts_dist.items()))}")
            if iteration_stats:
                avg_iters = sum(iteration_stats) / len(iteration_stats)
                min_iters = min(iteration_stats)
                max_iters = max(iteration_stats)
                print(f"  Iteration stats: min={min_iters}, max={max_iters}, avg={avg_iters:.1f}")


if __name__ == "__main__":
    main()
