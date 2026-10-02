import numpy as np

def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here

    n = len(x)

    x = np.asarray(x)

    mini = np.min(x, axis=0)
    maxi = np.max(x, axis=0)

    interval = maxi - mini

    if interval == 0:
        return np.zeros(n).tolist()
    
    return ((x - mini) / interval).tolist()
