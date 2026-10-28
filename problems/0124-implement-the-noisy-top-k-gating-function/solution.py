import numpy as np

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """

    H_base = X @ W_g
    H_noise = X @ W_noise

    H = H_base + N * np.log(1 + np.exp(H_noise))
    
    batch_size, num_experts = H.shape

    top_ind = np.argsort(H, axis=-1)[:, -k:]

    H_prim = np.full_like(H, -np.inf)

    row_indices = np.arange(batch_size)[:, None] 

    H_prim[row_indices, top_ind] = H[row_indices, top_ind]

    def softmax(z):
        maxes = np.max(z, axis=-1, keepdims=True)
        z = np.exp(z - maxes)
        sums = np.sum(z, axis=-1, keepdims=True)
        return z / sums

    G = softmax(H_prim)

    return G
