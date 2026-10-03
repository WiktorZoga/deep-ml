import numpy as np

def compute_inference_metrics(timestamps: list[float]) -> dict:
    """
    Compute LLM inference performance metrics from token timestamps.
    
    Args:
        timestamps: List of floats where timestamps[0] is the request start time
                    and timestamps[1:] are the times when each output token was generated.
    
    Returns:
        Dictionary with keys 'ttft', 'tps', 'itl' containing the metric values.
    """
    if len(timestamps) < 2:
        return {}

    ts = np.asarray(timestamps)
    token_generated = len(ts) - 1

    ttft = float(ts[1] - ts[0])

    total_time = ts[-1] - ts[0]
    tps = float(token_generated / total_time) if total_time > 0 else 0.0

    if token_generated > 1:
        # itl = float(np.mean(np.diff(ts[1:])))
        itl = float(np.mean(ts[2:] - ts[1:-1]))
    else:
        itl = 0.0

    return {
        "ttft": ttft,
        "tps": tps,
        "itl": itl,
    }