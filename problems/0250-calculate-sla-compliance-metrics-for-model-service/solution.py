def calculate_sla_metrics(requests: list, latency_sla_ms: float = 100.0) -> dict:
    """
    Calculate SLA compliance metrics for a model serving endpoint.
    
    Args:
        requests: list of request results, each a dict with 'latency_ms' and 'status'
        latency_sla_ms: maximum acceptable latency in ms for SLA compliance
    
    Returns:
        dict with keys: 'latency_sla_compliance', 'error_rate', 'overall_sla_compliance'
        All values as percentages (0-100), rounded to 2 decimal places.
    """

    reqs = len(requests)

    if reqs == 0:
        return {}
    
    successes = 0
    success_and_lat_tresh = 0
    error_or_timeout = 0
    
    for req in requests:
        if req["status"] == "success":
            successes += 1
            if req["latency_ms"] < latency_sla_ms:
                success_and_lat_tresh += 1
        else:
            error_or_timeout += 1

    latency_sla_compliance = success_and_lat_tresh / successes if successes > 0 else 0.0

    error_rate = error_or_timeout / reqs
    succsess_rate = 1 - error_rate

    overall_sla_compliance = succsess_rate * latency_sla_compliance

    stats = {
        "latency_sla_compliance": latency_sla_compliance,
        "error_rate": error_rate,
        "overall_sla_compliance": overall_sla_compliance
    }

    for key, val in stats.items():
        stats[key] = round(100.0 * val, 2)
    
    return stats