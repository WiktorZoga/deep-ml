import numpy as np

def information_bottleneck_loss(task_loss: float, mu: np.ndarray, log_var: np.ndarray, beta: float) -> tuple:
    """
    Compute the Information Bottleneck regularized loss.
    
    Args:
        task_loss: Precomputed task loss (scalar)
        mu: Encoder means, shape (batch_size, latent_dim)
        log_var: Encoder log-variances, shape (batch_size, latent_dim)
        beta: Trade-off parameter for compression regularization
    
    Returns:
        Tuple of (total_loss, mean_kl_divergence), both rounded to 4 decimal places
    """
    dkl = -0.5 * np.sum(1.0 + log_var - mu**2 - np.exp(log_var), axis=-1)

    return task_loss + beta * np.mean(dkl), np.mean(dkl)