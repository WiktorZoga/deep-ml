import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    M = np.concatenate([A, b[:, np.newaxis]], axis=-1)
    n, m = M.shape

    for k in range(n - 1):
        row = k + np.argmax(np.abs(M[k:, k]))

        if row != k:
            M[[k, row]] = M[[row, k]]

        for i in range(k + 1, n):
            c = M[i, k] / M[k, k]
            for j in range(k, m):
                M[i, j] -= c * M[k, j]

    x = np.zeros_like(b, dtype=float)

    b = M[:, -1].copy()

    for k in range(n - 1, -1, -1):
        for i in range(k + 1, m - 1):
            b[k] -= M[k, i] * x[i]
        x[k] = b[k] / M[k, k]
    
    return x
