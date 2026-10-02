import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros

    # Your code here: implement gradient descent

    for _ in range(iterations):
        preds = np.matmul(X, theta)

        errors = preds - y

        grad = np.matmul(X.T, errors) / m

        theta -= alpha * grad

    return theta.flatten()

    """
        L(theta, X, Y) = 1/2m sum_{i=1}^{m} (h_theta(x^{i}) - y^{i})^2

        dL/dtheta(theta, X, Y) = 1/m sum_{i=1}^{m} (h_theta(x^{i}) - y^{i}) * dh_theta/dtheta(x^{i})

        dh_theta/dtheta (x) = d/dtheta(dot(x, theat))(x) = x

        dL/dtheta(theta, X, Y) = 1/m sum_{i=1}^{m} (h_theta(x^{i}) - y^{i}) * x^{i})

        = 1/m sum_{i=1}^{m} error_i * x^{i}


    """