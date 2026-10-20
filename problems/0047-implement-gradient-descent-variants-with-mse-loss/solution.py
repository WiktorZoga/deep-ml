import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """

    if method not in ['batch', 'stochastic', 'mini_batch']:
        raise ValueError()
    
    w = weights.copy()
    m = X.shape[0]

    step = 1 if method == 'stochastic' else (batch_size if method == 'mini_batch' else m)

    for _ in range(n_epochs):
        for i in range(0, m, step):
            x_batch = X[i:i+step]
            y_batch = y[i:i+step]

            pred = x_batch @ w
            error = pred - y_batch

            grad = 2 * np.mean(x_batch * error[:, np.newaxis], axis=0)

            w = w - learning_rate * grad

    return w