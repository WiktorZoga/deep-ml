import torch
import torch.linalg


def linear_regression_normal_equation(X, y) -> torch.Tensor:
    """
    Solve linear regression via the normal equation using PyTorch.
    X: Tensor or convertible of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1,1)
    # Your implementation here
    pass

    xT = X_t.mT

    xTx = xT @ X_t

    xTx_inv = torch.linalg.inv(xTx)

    return xTx_inv @ xT @ y

    # y = X @ W
    # X^T @ y = X^T @ X @  W
    # ((X^T @  X)^{-1} @ X^T) @ y = W


