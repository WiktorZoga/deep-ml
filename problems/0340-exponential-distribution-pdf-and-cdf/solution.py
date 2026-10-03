import numpy as np

def exponential_distribution(x: list, lam: float) -> dict:
    """
    Compute exponential distribution properties.
    
    Args:
        x: Points at which to evaluate PDF and CDF
        lam: Rate parameter (lambda) of the distribution
        
    Returns:
        Dictionary with 'pdf', 'cdf', 'mean', and 'variance' keys
    """
    # Your code here

    if lam <= 0.0:
        return {
            "pdf": None,
            "cdf": None,
            "mean": None,
            "variance": None,
        }

    def pdf(x):
        return round(lam * np.exp(-lam * x) if x >= 0.0 else 0.0, 4)

    def cdf(x):
        return round(1.0 - np.exp(-lam * x) if x >= 0.0 else 0.0, 4)
    
    mean = round(lam ** -1.0, 4)
    variance = round(lam ** -2.0, 4)

    return {
        "pdf": [pdf(x_i) for x_i in x],
        "cdf": [cdf(x_i) for x_i in x],
        "mean": mean,
        "variance": variance,
    }