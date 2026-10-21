import numpy as np

def rref(matrix):
    matrix = np.asarray(matrix, dtype=float)
    m, n = matrix.shape

    row = 0

    for col in range(n):
        nonzero = np.nonzero(matrix[row:, col])[0]

        if len(nonzero) == 0:
            continue
        
        pivot = nonzero[0] + row
        
        if row != pivot:
            matrix[[row, pivot]] = matrix[[pivot, row]]

        pivot_value = matrix[row, col]
        matrix[row, :] = matrix[row, :] / pivot_value

        for i in range(m):
            if i != row:
                factor = matrix[i, col]
                matrix[i, :] -= matrix[row, :] * factor
        
        row += 1
        if row >= m:
            break
    
    return matrix