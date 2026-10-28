import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:

    B, C, H, W = X.shape

    group_size = C // num_groups

    x = X.reshape(B, num_groups, group_size, H, W)

    mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    var = np.var(x, axis=(2, 3, 4), ddof=0, keepdims=True)
    
    z = (x - mean) / np.sqrt(var + epsilon)

    z = z.reshape(B, C, H, W)
    gamma = gamma.reshape(1, C, 1, 1)
    beta = beta.reshape(1, C, 1, 1)

    y = gamma * z + beta

    return y