import numpy as np

def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in', distribution: str = 'normal', seed: int = None) -> np.ndarray:
    """
    Implement He (Kaiming) weight initialization.
    """
    if seed is not None:
        np.random.seed(seed)

    if mode == 'fan_in':
        fan = n_in
    elif mode == 'fan_out':
        fan = n_out
    else:
        raise ValueError("Mode must be 'fan_in' or 'fan_out'")

    if distribution == "normal":
        scale = np.sqrt(2.0 / fan)
        w = np.random.normal(loc=0.0, scale=scale, size=(n_in, n_out))
        
    elif distribution == "uniform":
        limit = np.sqrt(6.0 / fan)
        w = np.random.uniform(-limit, limit, size=(n_in, n_out))
        
    else:
        raise ValueError("Not known distribution")

    return w
