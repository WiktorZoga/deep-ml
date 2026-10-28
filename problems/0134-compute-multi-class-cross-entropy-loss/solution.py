import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:

    clipped_probs = np.clip(predicted_probs, a_min=epsilon, a_max=1.0 - epsilon)
    
    loss_per_sample = -np.sum(true_labels * np.log(clipped_probs), axis=-1)

    return float(np.mean(loss_per_sample))