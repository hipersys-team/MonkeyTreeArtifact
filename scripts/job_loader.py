#!/usr/bin/env python3
"""
Job definition loader from YAML files.

This module provides functionality to load job type definitions from YAML files
in the `jobs/` directory. It serves as a centralized source of job definitions
for trace generation, analysis, and plotting scripts.

Usage:
    from job_loader import load_job_definitions, get_job_definition
    
    # Load all job definitions
    jobs = load_job_definitions()
    
    # Get a specific job definition
    job_def = get_job_definition("s1")
    print(job_def["model_size_bytes"])  # 6875000000
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

# Default jobs directory relative to this file
DEFAULT_JOBS_DIR = Path(__file__).parent.parent / "jobs"


class JobStepLoader(yaml.SafeLoader):
    """Custom YAML loader that handles job step tags."""
    pass


def compute_constructor(loader, node):
    """Handle !compute tag."""
    if isinstance(node, yaml.MappingNode):
        data = loader.construct_mapping(node)
        return {"compute": data}
    return {"compute": {}}


def all_reduce_constructor(loader, node):
    """Handle !all_reduce tag."""
    if isinstance(node, yaml.MappingNode):
        data = loader.construct_mapping(node)
        return {"all_reduce": data}
    return {"all_reduce": {}}


def all_to_all_constructor(loader, node):
    """Handle !all_to_all tag."""
    if isinstance(node, yaml.MappingNode):
        data = loader.construct_mapping(node)
        return {"all_to_all": data}
    return {"all_to_all": {}}


def strided_ring_constructor(loader, node):
    """Handle !strided_ring tag."""
    if isinstance(node, yaml.MappingNode):
        data = loader.construct_mapping(node)
        return {"strided_ring": data}
    return {"strided_ring": {}}


def all_to_all_ring_constructor(loader, node):
    """Handle !all_to_all_ring tag."""
    if isinstance(node, yaml.MappingNode):
        data = loader.construct_mapping(node)
        return {"all_to_all_ring": data}
    return {"all_to_all_ring": {}}


def pipeline_constructor(loader, node):
    """Handle !pipeline tag for pipeline parallel jobs."""
    if isinstance(node, yaml.MappingNode):
        data = loader.construct_mapping(node)
        return {"pipeline": data}
    return {"pipeline": {}}


# Register custom constructors
JobStepLoader.add_constructor('!compute', compute_constructor)
JobStepLoader.add_constructor('!all_reduce', all_reduce_constructor)
JobStepLoader.add_constructor('!all_to_all', all_to_all_constructor)
JobStepLoader.add_constructor('!all_to_all_ring', all_to_all_ring_constructor)
JobStepLoader.add_constructor('!strided_ring', strided_ring_constructor)
JobStepLoader.add_constructor('!pipeline', pipeline_constructor)


def parse_yaml_int(value: Any) -> int:
    """Parse a YAML value that might contain underscores as an integer."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # Remove underscores and parse
        return int(value.replace("_", ""))
    return int(value)


def load_job_definition(yaml_path: Path) -> Dict[str, Any]:
    """
    Load a single job definition from a YAML file.
    
    Returns:
        Dict with keys: name, model_size_bytes, compute_time_us, steps
    """
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader=JobStepLoader)
    
    # Normalize the data
    job_def = {
        "name": data.get("name", yaml_path.stem),
        "model_size_bytes": parse_yaml_int(data.get("model_size_bytes", 0)),
        "steps": data.get("steps", []),
    }
    
    # Compute total compute time from steps
    compute_time_us = 0
    for step in job_def["steps"]:
        if "compute" in step:
            compute_time_us += step["compute"].get("duration_us", 0)
        if "pipeline" in step:
            # Pipeline compute: (forward + backward) * microbatches per stage
            pp = step["pipeline"]
            fwd = pp.get("forward_us", 50000)
            bwd = pp.get("backward_us", 100000)
            microbatches = pp.get("microbatches", 1)
            compute_time_us += (fwd + bwd) * microbatches
    job_def["compute_time_us"] = compute_time_us
    
    # Extract collective information from steps
    has_allreduce = any("all_reduce" in s for s in job_def["steps"])
    has_alltoall = any("all_to_all" in s for s in job_def["steps"])
    has_alltoall_ring = any("all_to_all_ring" in s for s in job_def["steps"])
    has_strided_ring = any("strided_ring" in s for s in job_def["steps"])
    has_pipeline = any("pipeline" in s for s in job_def["steps"])
    
    # Extract all_to_all size if present (include both simultaneous and ring variants)
    all_to_all_size = 0
    for step in job_def["steps"]:
        if "all_to_all" in step:
            all_to_all_size += step["all_to_all"].get("total_size_bytes", 0)
        if "all_to_all_ring" in step:
            all_to_all_size += step["all_to_all_ring"].get("total_size_bytes", 0)
    job_def["all_to_all_size_bytes"] = all_to_all_size
    
    # Extract strided_ring info if present
    strided_ring_steps = []
    for step in job_def["steps"]:
        if "strided_ring" in step:
            strided_ring_steps.append({
                "stride": step["strided_ring"].get("stride", 1),
                "model_size_bytes": step["strided_ring"].get("model_size_bytes", 0),
            })
    job_def["strided_ring_steps"] = strided_ring_steps
    
    # Extract pipeline info if present
    pipeline_steps = []
    for step in job_def["steps"]:
        if "pipeline" in step:
            pp = step["pipeline"]
            pipeline_steps.append({
                "num_stages": pp.get("num_stages", 1),
                "tp_size": pp.get("tp_size", 1),
                "dp_replicas": pp.get("dp_replicas"),  # None if not specified (variable)
                "microbatches": pp.get("microbatches", 1),
                "forward_us": pp.get("forward_us", 50000),
                "backward_us": pp.get("backward_us", 100000),
                "activation_size_bytes": pp.get("activation_size_bytes", 1_000_000_000),
                "model_shard_bytes": pp.get("model_shard_bytes", 0),
            })
    job_def["pipeline_steps"] = pipeline_steps
    
    # For pipeline jobs, calculate workers_per_replica and fixed/variable worker count
    GPU_BLOCK_SIZE = 8
    if pipeline_steps:
        pp = pipeline_steps[0]  # Use first pipeline step
        workers_per_replica = pp["num_stages"] * pp["tp_size"]
        job_def["pipeline_workers_per_replica"] = workers_per_replica
        
        if pp["dp_replicas"] is not None:
            # Fixed worker count
            job_def["pipeline_workers"] = workers_per_replica * pp["dp_replicas"]
            job_def["pipeline_variable_size"] = False
            
            # Validate workers_per_stage is a multiple of GPU_BLOCK_SIZE
            workers_per_stage = pp["tp_size"] * pp["dp_replicas"]
            if workers_per_stage % GPU_BLOCK_SIZE != 0:
                raise ValueError(
                    f"Pipeline job '{job_def['name']}': workers_per_stage ({workers_per_stage} = "
                    f"{pp['tp_size']} tp_size * {pp['dp_replicas']} dp_replicas) must be a multiple of "
                    f"{GPU_BLOCK_SIZE} (GPU block size)"
                )
        else:
            # Variable worker count - valid counts are multiples of workers_per_replica
            job_def["pipeline_workers"] = None
            job_def["pipeline_variable_size"] = True
            
            # For variable-size jobs, store the minimum valid GPU count
            # (must be multiple of both workers_per_replica and have workers_per_stage % 8 == 0)
            # workers_per_stage = tp_size * dp_replicas, and dp_replicas = total / workers_per_replica
            # So workers_per_stage = tp_size * total / (num_stages * tp_size) = total / num_stages
            # For workers_per_stage % 8 == 0: (total / num_stages) % 8 == 0
            # So total must be a multiple of num_stages * 8
            min_gpus = pp["num_stages"] * GPU_BLOCK_SIZE
            job_def["pipeline_min_gpus"] = min_gpus
            job_def["pipeline_gpu_step"] = min_gpus  # Valid GPU counts: min_gpus, 2*min_gpus, ...
    
    # Determine the primary strategy (for backwards compatibility)
    # Jobs can have multiple collective types
    job_def["has_all_reduce"] = has_allreduce
    job_def["has_all_to_all"] = has_alltoall or has_alltoall_ring
    job_def["has_all_to_all_ring"] = has_alltoall_ring
    job_def["has_strided_ring"] = has_strided_ring
    job_def["has_pipeline"] = has_pipeline
    
    # Count number of collective types (treat all_to_all and all_to_all_ring as the same category)
    collective_types = sum([has_allreduce, has_alltoall or has_alltoall_ring, has_strided_ring, has_pipeline])
    
    if collective_types > 1:
        job_def["strategy"] = "mixed"
    elif has_pipeline:
        job_def["strategy"] = "pipeline"
    elif has_allreduce:
        job_def["strategy"] = "all_reduce"
    elif has_alltoall or has_alltoall_ring:
        job_def["strategy"] = "all_to_all"
    elif has_strided_ring:
        job_def["strategy"] = "strided_ring"
    else:
        job_def["strategy"] = "none"
    
    return job_def


def load_job_definitions(jobs_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load all job definitions from the jobs directory (recursively).
    
    Top-level files get keys equal to their stem (e.g. ``s1.yaml`` → ``"s1"``).
    Files in subdirectories use the relative path from *jobs_dir* without the
    extension (e.g. ``seq16k/gpt_120b_32gpu.yaml`` → ``"seq16k/gpt_120b_32gpu"``).
    
    Args:
        jobs_dir: Path to the jobs directory. Defaults to the ``jobs/`` directory
                  relative to the project root.
    
    Returns:
        Dict mapping job type name to job definition dict.
    """
    if jobs_dir is None:
        jobs_dir = DEFAULT_JOBS_DIR
    
    jobs_dir = Path(jobs_dir)
    
    if not jobs_dir.exists():
        raise FileNotFoundError(f"Jobs directory not found: {jobs_dir}")
    
    definitions = {}
    
    for yaml_path in jobs_dir.rglob("*.yaml"):
        job_type = str(yaml_path.relative_to(jobs_dir).with_suffix(""))
        try:
            definitions[job_type] = load_job_definition(yaml_path)
        except Exception as e:
            print(f"Warning: Failed to load {yaml_path}: {e}")
    
    for yaml_path in jobs_dir.rglob("*.yml"):
        job_type = str(yaml_path.relative_to(jobs_dir).with_suffix(""))
        if job_type not in definitions:
            try:
                definitions[job_type] = load_job_definition(yaml_path)
            except Exception as e:
                print(f"Warning: Failed to load {yaml_path}: {e}")
    
    return definitions


def get_job_definition(job_type: str, jobs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get a single job definition by type name.
    
    Args:
        job_type: The job type name (e.g., "s1", "s2", "s3")
        jobs_dir: Optional path to the jobs directory
    
    Returns:
        Job definition dict with keys: name, model_size_bytes, compute_time_us, steps
    
    Raises:
        KeyError: If the job type is not found
    """
    definitions = load_job_definitions(jobs_dir)
    
    if job_type not in definitions:
        raise KeyError(f"Unknown job type: {job_type}. Available: {list(definitions.keys())}")
    
    return definitions[job_type]


def list_job_types(jobs_dir: Optional[Path] = None) -> List[str]:
    """
    List all available job types.
    
    Returns:
        Sorted list of job type names
    """
    definitions = load_job_definitions(jobs_dir)
    return sorted(definitions.keys())


# Compatibility functions for existing code

def get_job_definitions_dict(jobs_dir: Optional[Path] = None) -> Dict[str, tuple]:
    """
    Get job definitions in the legacy format used by generate_trace.py.
    
    Returns:
        Dict mapping job type to (compute_time_us, model_size_bytes)
    """
    definitions = load_job_definitions(jobs_dir)
    return {
        job_type: (
            job_def["compute_time_us"],
            job_def["model_size_bytes"]
        )
        for job_type, job_def in definitions.items()
    }


def get_analysis_job_definitions(jobs_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get job definitions in the format used by analysis.py.
    
    Returns:
        Dict mapping job type to analysis-compatible dict
    """
    definitions = load_job_definitions(jobs_dir)
    
    # Import network time functions from analysis if available
    try:
        from plot.analysis import network_time_simple_flow
        default_strategy = network_time_simple_flow
    except ImportError:
        default_strategy = None
    
    result = {}
    for job_type, job_def in definitions.items():
        result[job_type] = {
            "compute_time_us": job_def["compute_time_us"],
            "data_size_bytes": job_def["model_size_bytes"],
            "num_iterations": 100,  # Default, can be overridden
            "strategy": default_strategy,
            "description": f"{job_def['name']} job - {job_def['strategy']}",
        }
    
    return result


if __name__ == "__main__":
    # Demo: print all job definitions
    print("Job Definitions:")
    print("=" * 60)
    
    try:
        definitions = load_job_definitions()
        
        for job_type, job_def in sorted(definitions.items()):
            print(f"\n{job_type}:")
            print(f"  Name: {job_def['name']}")
            print(f"  Model Size: {job_def['model_size_bytes']:,} bytes ({job_def['model_size_bytes'] / 1e9:.2f} GB)")
            print(f"  Compute Time: {job_def['compute_time_us']} us ({job_def['compute_time_us'] / 1000:.1f} ms)")
            print(f"  Strategy: {job_def['strategy']}")
            print(f"  Steps: {len(job_def['steps'])}")
            for i, step in enumerate(job_def['steps']):
                step_type = list(step.keys())[0]
                print(f"    {i+1}. {step_type}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you're running from the project root directory.")
