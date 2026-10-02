import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    
    x = np.asarray(data)

    def mode(x):
        values, counts = np.unique(x, return_counts=True)
        return values[np.argmax(counts)]

    quantiles = np.quantile(x, [0.25, 0.5, 0.75])

    stats = {
        "mean": np.mean(x),
        "median": np.median(x),
        "mode": mode(x),
        "variance": np.var(x),
        "standard_deviation": np.std(x),
        "25th_percentile": quantiles[0],
        "50th_percentile": quantiles[1],
        "75th_percentile": quantiles[2],
        "interquartile_range": quantiles[2] - quantiles[0],
    }

    return stats