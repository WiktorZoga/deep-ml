import numpy as np

def reward_model_validation(
    chosen_scores: np.ndarray,
    rejected_scores: np.ndarray,
    margin_thresholds: list[float]
) -> dict:
    """
    Compute validation metrics for a reward model on held-out preference pairs.

    Args:
        chosen_scores: 1D array of reward scores for preferred responses.
        rejected_scores: 1D array of reward scores for rejected responses.
        margin_thresholds: List of thresholds for margin-based accuracy.

    Returns:
        Dictionary with 'accuracy', 'mean_margin', 'concordance', 'margin_accuracy'.
    """

    margins = chosen_scores - rejected_scores

    wins = margins > 0.0

    accuracy = np.mean(wins).round(4)

    ties = margins == 0.0

    mean_margin = np.mean(margins).round(4)

    concordance = np.mean(wins + 0.5 * ties).round(4)

    margin_accuracy = {
        margin: np.mean(margins >= margin).round(4) for margin in margin_thresholds
    }

    return {
        "accuracy": accuracy,
        "mean_margin": mean_margin,
        "concordance": concordance,
        "margin_accuracy": margin_accuracy,
    }
