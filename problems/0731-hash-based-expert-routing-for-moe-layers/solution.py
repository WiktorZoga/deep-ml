import numpy as np

def hash_moe_forward(token_ids, embeddings, expert_weights, num_experts: int) -> np.ndarray:
    """
    Hash-based routing forward pass for an MoE layer.

    Args:
        token_ids: 1D array-like of N integer token IDs
        embeddings: 2D array-like of shape (N, d) of token embeddings
        expert_weights: list of E numpy arrays, each of shape (d, d_out)
        num_experts: number of experts E

    Returns:
        numpy array of shape (N, d_out) with per-token expert outputs.
    """

    N = len(token_ids)
    
    embeddings = np.asarray(embeddings)
    expert_weights = np.asarray(expert_weights)
    token_ids = np.asarray(token_ids, dtype=int)

    expert_ids = token_ids % num_experts

    return np.asarray([embeddings[i] @ expert_weights[token_ids[i] % num_experts] for i in range(N)])


