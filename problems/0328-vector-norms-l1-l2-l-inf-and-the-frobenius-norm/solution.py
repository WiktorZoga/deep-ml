import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    # Your code here

    if norm_type == "l1":
        return np.sum(np.abs(arr)).astype(np.float32)
    if norm_type == "l2" or (norm_type == "frobenius" and arr.ndim == 2):
        return np.linalg.norm(arr).astype(np.float32)
    if norm_type == "linf":
        return np.max(np.abs(arr)).astype(np.float32)
    raise ValueError("Not know norm_type")
