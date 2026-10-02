import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    x = torch.tensor(scores, dtype=torch.float32)
    
    x_stable = x - torch.max(x, dim=0).values

    log_sum_exp = torch.log(torch.sum(torch.exp(x_stable)))

    return x_stable - log_sum_exp
