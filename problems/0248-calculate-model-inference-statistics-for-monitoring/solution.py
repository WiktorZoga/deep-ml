import numpy as np

def calculate_inference_stats(latencies_ms: list) -> dict:
    """
    Calculate inference statistics for model monitoring.
    
    Args:
        latencies_ms: list of latency measurements in milliseconds
    
    Returns:
        dict with keys: 'throughput_per_sec', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms'
        All values rounded to 2 decimal places.
    """

    if len(latencies_ms) == 0:
        return {}

    latencies_ms = np.asarray(latencies_ms)

    mean = np.mean(latencies_ms)
    throughput = 1000.0 / mean
    quantiles = np.quantile(latencies_ms, [0.5, 0.95, 0.99])

    stats = {
        "throughput_per_sec": throughput,
        "avg_latency_ms": mean,
        "p50_ms": quantiles[0],
        "p95_ms": quantiles[1],
        "p99_ms": quantiles[2],
    }

    for key in stats:
        stats[key] = round(stats[key], 2)
    
    return stats