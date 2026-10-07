import numpy as np

def track_gpu_utilization(timestamps_ms: list, gpu_util_pct: list, phase_labels: list) -> dict:
    """
    Analyze GPU utilization during RL training and compute per-phase statistics.
    
    Args:
        timestamps_ms: List of N sorted timestamps (ms) marking interval boundaries.
        gpu_util_pct: List of N-1 GPU utilization percentages (0-100) per interval.
        phase_labels: List of N-1 phase labels per interval.
    
    Returns:
        dict with keys: total_time_ms, avg_gpu_util_pct, phase_stats,
                        bottleneck_phase, gpu_idle_fraction_pct
    """

    durations = np.diff(np.asarray(timestamps_ms))
    gpu_util = np.asarray(gpu_util_pct)
    phases = np.asarray(phase_labels)

    total_time_ms = np.sum(durations)
    avg_gpu_util_pct = np.sum(durations * gpu_util) / total_time_ms

    phase_stats = {}
    phase_idle_time = {}

    unique_phases = np.unique(phases)

    for phase in unique_phases:
        mask = (phases == phase)
        p_durations = durations[mask]
        p_util = gpu_util[mask]
        
        p_total_time = np.sum(p_durations)

        p_avg_util = np.sum(p_durations * p_util) / p_total_time
        p_fraction = 100.0 * p_total_time / total_time_ms

        phase_stats[phase] = {
            'total_time_ms': p_total_time.round(2),
            'avg_util_pct': p_avg_util.round(2),
            'time_fraction_pct': p_fraction.round(2),
        }

        phase_idle_time[phase] = np.sum(p_durations * (100.0 - p_util)).round(2)

    bottleneck_phase = max(phase_idle_time, key=phase_idle_time.get)

    return {
        'total_time_ms': total_time_ms.round(2),
        'avg_gpu_util_pct': avg_gpu_util_pct.round(2),
        'phase_stats': phase_stats,
        'bottleneck_phase': bottleneck_phase,
        'gpu_idle_fraction_pct': (100.0 - avg_gpu_util_pct).round(2),
    }