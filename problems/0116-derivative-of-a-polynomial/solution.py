import math

def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Your code here
    if n == 0:
        return 0.0

    return c * n * math.pow(x, n - 1)   