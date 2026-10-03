import numpy as np

def calculate_portfolio_variance(cov_matrix: list[list[float]], weights: list[float]) -> float:
    """
    Calculate the variance of a portfolio.

    Args:
        cov_matrix (list[list[float]]): Covariance matrix of asset returns.
        weights (list[float]): Portfolio weights.

    Returns:
        float: Portfolio variance.
    """
    # Your code here
    cov_matrix = np.asarray(cov_matrix)
    weights = np.asarray(weights)

    return np.dot(weights, np.matmul(cov_matrix, weights))