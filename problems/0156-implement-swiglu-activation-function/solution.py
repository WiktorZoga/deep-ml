import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    d = x.shape[1] // 2

    a, b = x[:, :d], x[:, d:]

    swish_b = b / (1 + np.exp(-b))

    scores = swish_b * a

    return np.round(scores, 4)
