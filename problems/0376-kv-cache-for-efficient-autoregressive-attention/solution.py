import numpy as np
import math

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """

    q, k, v = x_new @ W_Q, x_new @ W_K, x_new @ W_V

    d = q.shape[-1]

    if cache is None:
        K = k.reshape(1, -1)
        V = v.reshape(1, -1)
    else:
        K_cache, V_cache = cache
        K = np.vstack([K_cache, k.reshape(1, -1)])
        V = np.vstack([V_cache, v.reshape(1, -1)])

    scores = (q @ K.T) / math.sqrt(d)

    def softmax(z):
        maxes = np.max(z, axis=-1, keepdims=True)
        scores_exp = np.exp(z - maxes)
        sums = np.sum(scores_exp, axis=-1, keepdims=True)
        return scores_exp / sums
    
    attn = softmax(scores)

    output = attn @ V

    return output, (K, V)
