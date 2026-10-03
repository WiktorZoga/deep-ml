import numpy as np

def taylor_approximation(func_name: str, x: float, n_terms: int) -> float:
    """
    Compute Taylor series approximation for common functions.
    
    Args:
        func_name: Name of function ('exp', 'sin', 'cos')
        x: Point at which to evaluate
        n_terms: Number of non-zero terms in the series
    
    Returns:
        Taylor series approximation rounded to 6 decimal places
    """
    if n_terms <= 0:
        return 0.0

    total = 0.0

    if func_name == "exp":
        term = 1.0
        total = term
        for n in range(1, n_terms):
            term *= x / n
            total += term

    elif func_name == "sin":
        term = float(x)
        total = term
        for n in range(1, n_terms):
            term *= - (x ** 2) / ((2 * n) * (2 * n + 1))
            total += term

    elif func_name == "cos":
        term = 1.0
        total = term
        for n in range(1, n_terms):
            term *= - (x ** 2) / ((2 * n - 1) * (2 * n))
            total += term

    else:
        raise ValueError(f"Unknown func_name: {func_name}")

    return round(float(total), 6)