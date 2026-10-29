import numpy as np

def asymmetric_quantize(x: np.ndarray, bits: int = 8):
    """
    Asymmetric uniform quantization.

    Returns q (int array), scale (float), zero_point (int), x_hat (float array).
    Unsigned grid: qmin=0, qmax=2^bits-1.
    """

    x_min = np.min(x)
    x_max = np.max(x)

    if x_max - x_min < 1e-12:
        return (np.zeros_like(x, dtype=np.int32), 1.0, 0, x)

    qmin = 0
    qmax = 2**bits - 1

    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = np.clip((qmin - x_min/scale).round(0), qmin, qmax).astype(int)
    q = np.clip((x/scale + zero_point).round(0), qmin, qmax)

    x_hat = scale * (q - zero_point)

    return (q.astype(int), scale, zero_point, x_hat)