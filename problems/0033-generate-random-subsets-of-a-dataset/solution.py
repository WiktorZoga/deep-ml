import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True): 
    n = X.shape[0]

    if replacements:
        subset_size = n
    else:
        subset_size = n // 2

    subsets = []

    for _ in range(n_subsets):
        sampled_indices = np.random.choice(n, subset_size, replace=replacements)

        subsets.append((X[sampled_indices], y[sampled_indices]))
    
    return subsets