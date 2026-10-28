import numpy as np
from typing import Tuple
import math

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    return X @ W_q, X @ W_k, X @ W_v

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """

    def softmax(z):
        maxes = np.max(z, axis=-1, keepdims=True)
        scores = np.exp(z - maxes)
        sums = np.sum(scores, axis=-1, keepdims=True)
        return scores/sums
    
    d = Q.shape[-1]

    scores = (Q @ K.swapaxes(-2, -1)) * (1.0 / math.sqrt(d))
    
    # mask = np.zeros()

    attn = softmax(scores)

    return attn @ V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """

    seq_len, d = Q.shape

    head_dim = d // n_heads

    Q = Q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)

    attn = self_attention(Q, K, V).transpose(1, 0, 2)

    attn = attn.reshape(seq_len, d)

    return attn

