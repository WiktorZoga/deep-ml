import numpy as np

def map_latent_to_real(latents_labeled: np.ndarray, reals_labeled: np.ndarray, latents_query: np.ndarray) -> list:
    """Fit a linear map from latent actions to real actions using the labeled set,
    then apply it to latents_query. Return predictions as a nested list."""
    pass

    # y = XW
    # X^T y = X^T X W
    # (X^T X)^-1 X^T y = W

    x = latents_labeled
    y = reals_labeled

    xTx = x.T @ x

    xTx_inv = np.linalg.inv(xTx)

    W = xTx_inv @ x.T @ y

    return latents_query @ W