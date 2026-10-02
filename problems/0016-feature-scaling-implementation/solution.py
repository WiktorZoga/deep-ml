import numpy as np

def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, n = data.shape

    mean = np.mean(data, axis=0)
    var = np.sum((data - mean) ** 2, axis=0) / (m) + 1e-8
    standardized_data = (data - mean) / np.sqrt(var)

    mini = np.min(data, axis=0)
    maxi = np.max(data, axis=0)

    range_val = maxi - mini
    range_val[range_val == 0] = 1.0 

    normalized_data = (data - mini) / range_val

    return standardized_data, normalized_data
