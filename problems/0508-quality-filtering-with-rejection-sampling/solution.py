import numpy as np

def quality_filter_rejection_sampling(scores: list, threshold: float, n_select: int = None) -> dict:
    """
    Filter generated samples using quality-based rejection sampling.
    
    Args:
        scores: list of float quality scores for generated candidate samples
        threshold: minimum quality score required for acceptance
        n_select: optional maximum number of samples to return (top by score)
    
    Returns:
        dict with 'accepted_indices', 'acceptance_rate', 'mean_quality'
    """

    scores = np.asarray(scores)
    accepted_samples = scores >= threshold

    accepted_count = np.sum(accepted_samples)

    if accepted_count == 0:
        return {
            "accepted_indices": [],
            "acceptance_rate": 0.0,
            "mean_quality": 0.0,
        }

    indices  = np.where(accepted_samples)[0]
    
    accepted_scores = scores[indices]

    sorted_accepted_scores = np.argsort(accepted_scores)[::-1]

    if n_select is not None:
        sorted_accepted_scores = sorted_accepted_scores[:n_select]
    
    accepted_indices = indices[sorted_accepted_scores]

    acceptance_rate = round(accepted_count / scores.size, 4)

    mean_quality = np.mean(scores[accepted_indices]).round(4)

    return {
        "accepted_indices": accepted_indices.tolist(),
        "acceptance_rate": acceptance_rate,
        "mean_quality": mean_quality,
    }
