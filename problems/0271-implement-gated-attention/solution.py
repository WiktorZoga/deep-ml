import numpy as np
import math

def gated_attention(
    X: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_g: np.ndarray
) -> np.ndarray:
    """
    Compute Gated Attention output.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_q: Query projection of shape (d_model, d_k)
        W_k: Key projection of shape (d_model, d_k)
        W_v: Value projection of shape (d_model, d_v)
        W_g: Gate projection of shape (d_model, d_v)
    
    Returns:
        Gated attention output of shape (seq_len, d_v), rounded to 4 decimal places
    
    Hint: First compute standard scaled dot-product attention, then apply
    a sigmoid gate to modulate the output.
    """

    d_model = X.shape[-1]

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    G = X @ W_g

    scores = (Q @ K.T) * (1.0 / math.sqrt(d_model))

    def softmax(z):
        z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return z / np.sum(z, axis=-1, keepdims=True)
    
    att = softmax(scores)

    y = att @ V

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    gated = sigmoid(G) * y

    return gated