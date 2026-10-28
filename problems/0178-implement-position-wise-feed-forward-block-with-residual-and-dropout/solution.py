import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float=0.1, seed: int=42) -> list[float]:
    """
    Implement a position-wise feed-forward block with residual and dropout.
    """
    np.random.seed(seed)

    x = np.asarray(x)
    W1 = np.asarray(W1)
    b1 = np.asarray(b1)
    W2 = np.asarray(W2)
    b2 = np.asarray(b2)

    def relu(z):
        return np.maximum(0, z)

    h = relu(W1 @ x + b1)
    y = W2 @ h + b2

    if dropout_p > 0.0:
        keep_prob = 1.0 - dropout_p
        mask = (np.random.rand(*y.shape) > dropout_p)
        y = (mask * y) / keep_prob

    return np.round(x + y, 4).tolist()
