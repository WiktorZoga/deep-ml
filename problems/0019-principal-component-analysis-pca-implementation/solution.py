import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """

    x = data.copy()

    x_mean = np.mean(x, axis=0, keepdims=True)
    x_std = np.std(x, axis=0, keepdims=True)

    x_std[x_std == 0] = 1e-10

    x_norm = (x - x_mean) / x_std

    n, d = x.shape
    covariance_matrix = (x_norm.T @ x_norm) / (n - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    for i in range(eigenvectors.shape[1]):
        ev = eigenvectors[:, i]
        valid_indices = np.where(np.abs(ev) > 1e-10)[0]
        if len(valid_indices) > 0:
            first_valid_element = ev[valid_indices[0]]
            if first_valid_element < 0:
                eigenvectors[:, i] *= -1

    v_top = eigenvectors[:, :k]

    x_pca = x_norm @ v_top

    # return x_pca, v_top
    return v_top