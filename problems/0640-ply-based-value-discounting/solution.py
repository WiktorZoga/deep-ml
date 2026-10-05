import numpy as np

def ply_based_discount(evaluations: list, ply_depths: list, gamma: float) -> tuple:
    """
    Apply ply-based discounting to terminal evaluations at various search depths.
    
    Args:
        evaluations: Terminal evaluation scores for each candidate move sequence
        ply_depths: Number of plies to reach each terminal evaluation
        gamma: Per-ply discount factor (0 < gamma <= 1)
    
    Returns:
        Tuple of (discounted_values, best_index)
    """
    
    evaluations = np.asarray(evaluations)
    ply_depths = np.asarray(ply_depths)

    v = evaluations * (gamma ** ply_depths)

    return v.round(4).tolist(), np.argmax(v)