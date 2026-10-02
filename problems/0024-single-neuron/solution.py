import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    X = torch.tensor(features, dtype=torch.float32)
    W = torch.tensor(weights, dtype=torch.float32)
    b = torch.tensor(bias, dtype=torch.float32)
    target = torch.tensor(labels, dtype=torch.float32) # Musi być float32 do mse_loss

    z = torch.matmul(X, W) + b
    probabilities = torch.sigmoid(z)

    mse = F.mse_loss(probabilities, target)

    return probabilities.tolist(), mse.item()
