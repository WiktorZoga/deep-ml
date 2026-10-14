import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:

    _, m = A.shape

    x = np.zeros(m)

    D = np.diag(np.diag(A))
    C = A - D
    
    D_inv = np.diag(1.0 / np.diag(A))

    for _ in range(n):
        x = D_inv @ (b - C @ x)

    return x
