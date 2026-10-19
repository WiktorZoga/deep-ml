import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    n, d = X.shape

    combinations = list(combinations_with_replacement(range(d + 1), degree))

    ones = np.ones((n, 1))
    x = np.concatenate([ones, X], axis=-1)

    res = []

    for i in range(n):
        row = []
        for comb in combinations:
            p = 1.0
            for j in comb:
                p *= x[i, j]
            row.append(p)
        res.append(sorted(row))
    
    return np.asarray(res)
                 
            