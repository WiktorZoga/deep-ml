import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """

    samples = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(samples)

    fold_len = n_samples // k
    fold_end = fold_len + (n_samples % k)

    folds = [samples[:fold_end]]

    for i in range(k-1):
        l = fold_end + i * fold_len
        r = l + fold_len
        folds.append(samples[l:r])

    k_fold = []
    for i in range(k):
        k_fold.append((np.concatenate(folds[:i] + folds[i+1:], axis=0).tolist(), folds[i].tolist()))

    return k_fold


