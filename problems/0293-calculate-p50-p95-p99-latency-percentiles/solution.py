import numpy as np

def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """
    Calculate P50, P95, and P99 latency percentiles.
    
    Args:
        latencies: List of latency measurements
    
    Returns:
        Dictionary with keys 'P50', 'P95', 'P99' containing
        the respective percentile values rounded to 4 decimal places
    """
    # Your code here

    if len(latencies) == 0:
        return {
            "P50": 0.0,
            "P95": 0.0,
            "P99": 0.0,
        }

    x = np.asarray(latencies)

    quantiles = np.quantile(x, [0.5, 0.95, 0.99]).round(4)

    return {
        "P50": quantiles[0],
        "P95": quantiles[1],
        "P99": quantiles[2],
    }