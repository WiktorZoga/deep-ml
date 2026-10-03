def gpu_ops_byte_ratio(gpu_specs: dict) -> dict:
    """
    Compute the ops:byte ratio (ridge point) for each precision
    format from GPU hardware specifications.

    Args:
        gpu_specs: Dictionary with keys:
            - 'compute_tflops': dict mapping precision name -> peak TFLOPS
            - 'memory_bandwidth_gbps': float, peak memory bandwidth in GB/s

    Returns:
        Dictionary mapping each precision name to its ops:byte ratio
        (FLOPs per byte), rounded to 2 decimal places.
    """
    precision_mapping = gpu_specs["compute_tflops"]
    memory_bandwidth_gbps = gpu_specs["memory_bandwidth_gbps"]

    return {key: val * 1000.0 / memory_bandwidth_gbps for key, val in precision_mapping.items()}