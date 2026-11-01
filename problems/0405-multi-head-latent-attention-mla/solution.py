import numpy as np
import math

def multi_head_latent_attention(
    X: np.ndarray,
    W_dkv: np.ndarray,
    W_uk: np.ndarray,
    W_uv: np.ndarray,
    W_dq: np.ndarray,
    W_uq: np.ndarray,
    W_o: np.ndarray,
    n_heads: int
) -> tuple:
    """
    Perform Multi-Head Latent Attention (MLA).
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_dkv: Down-projection for KV compression (d_model, d_c_kv)
        W_uk: Up-projection for keys (d_c_kv, d_model)
        W_uv: Up-projection for values (d_c_kv, d_model)
        W_dq: Down-projection for query compression (d_model, d_c_q)
        W_uq: Up-projection for queries (d_c_q, d_model)
        W_o: Output projection (d_model, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Tuple of (output, c_kv) where:
        - output: shape (seq_len, d_model)
        - c_kv: compressed KV latent of shape (seq_len, d_c_kv)
    """

    c_kv = X @ W_dkv

    K = c_kv @ W_uk
    V = c_kv @ W_uv

    c_q = X @ W_dq

    Q = c_q @ W_uq

    seq_len, d_model = Q.shape

    d_h = d_model // n_heads

    Q = Q.reshape(seq_len, n_heads, d_h).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_h).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_h).transpose(1, 0, 2)

    scores = (Q @ np.transpose(K, (0, 2, 1))) * (1.0 / math.sqrt(d_h))

    def softmax(z):
        maxes = np.max(z, axis=-1, keepdims=True)
        z = np.exp(z - maxes)
        sums = np.sum(z, axis=-1, keepdims=True)
        return z / sums
    
    attn = softmax(scores) @ V

    output = attn.transpose(1, 0, 2).reshape(seq_len, d_model) @ W_o

    return output, c_kv

