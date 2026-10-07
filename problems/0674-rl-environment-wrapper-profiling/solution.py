import numpy as np

def profile_env_wrappers(wrapper_names: list, cumulative_step_times: list) -> dict:
    """
    Profile RL environment wrappers to identify per-wrapper overhead.
    
    Args:
        wrapper_names: List of wrapper names from outermost to innermost.
        cumulative_step_times: List of lists of timing measurements (ms) per wrapper level.
    
    Returns:
        dict with keys: 'wrappers' (list of per-wrapper stats), 'bottleneck' (str),
                        'total_overhead_ms' (float)
    """
    
    cumulative_step_times = np.asarray(cumulative_step_times)

    mean_cumulative = np.mean(cumulative_step_times, axis=1)[::-1]

    means = np.concatenate(([0], mean_cumulative))

    overhead = np.diff(means)[::-1]

    total_overhead = np.sum(overhead)

    overhead_pct = (100.0 * overhead / total_overhead)

    return {
        'wrappers': [{'name': name, 'mean_cumulative_ms': mean_cumulative[i].round(2), 'overhead_ms': overhead[i].round(2), 'overhead_pct': overhead_pct[i].round(2)} for i, name in enumerate(wrapper_names)],

        'bottleneck': wrapper_names[np.argmax(overhead)],
        'total_overhead_ms': total_overhead.round(2),
    }
        
