import numpy as np

def hard_voting_classifier(predictions: list[list[int]]) -> list[int]:
    """
    Implement a hard voting classifier using majority vote.
    
    Args:
        predictions: 2D list where predictions[i][j] is classifier i's prediction for sample j
        
    Returns:
        List of final predictions using majority vote
    """
    predictions = np.asarray(predictions)

    if predictions.size == 0:
        return []

    num_samples = predictions.shape[1]
    final_predictions = []

    for col in range(num_samples):
        values, counts = np.unique(predictions[:, col], return_counts=True)

        winner = values[np.argmax(counts)]
        final_predictions.append(winner)

    return final_predictions