import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    
    num_vars = A.shape[0]
    
    if x_ini is None:
        x = np.zeros(num_vars, dtype=float)
    else:
        x = np.asarray(x_ini, dtype=float).copy()
        
    for _ in range(n):
        for i in range(num_vars):
            suma = 0.0
            for j in range(num_vars):
                if i != j:
                    suma += A[i, j] * x[j]
            
            x[i] = (b[i] - suma) / A[i, i]
            
    return x
