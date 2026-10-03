def compute_arithmetic_intensity(flops: float, bytes_accessed: float, peak_performance: float, peak_bandwidth: float) -> dict:
    """
    Analyze a computational kernel using the Roofline Model.
    
    Args:
        flops: Total floating-point operations of the kernel
        bytes_accessed: Total bytes transferred to/from memory
        peak_performance: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)
    
    Returns:
        Dictionary with arithmetic_intensity, ridge_point, bottleneck,
        achieved_performance, and utilization_percent
    """
    arithmetic_intensity = flops / bytes_accessed

    # ridge_point - ai takie, że T_math = T_comm
    
    # T_math = flops / peak_performance
    # T_comm = bytes_accessed / peak_bandwidth

    # T_math = T_comm <=>
    # flops / bytes_accessed = peak_performance / peak_bandwidth

    ridge_point = peak_performance / peak_bandwidth

    # bottleneck - czy jesteśmy math bound czy comm bound

    # T_math > T_comm <=> ai >= ridge_point

    bottleneck = "compute-bound" if arithmetic_intensity >= ridge_point else "memory-bound"

    achieved_performance = min(peak_performance, arithmetic_intensity * peak_bandwidth)

    utilization_percent = (achieved_performance / peak_performance) * 100.0

    return {
        "arithmetic_intensity": arithmetic_intensity,
        "ridge_point": ridge_point,
        "bottleneck": bottleneck,
        "achieved_performance": achieved_performance,
        "utilization_percent": utilization_percent,
    }
