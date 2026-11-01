import numpy as np
import math

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """

    N, C, H, W = x.shape

    dtype = x.dtype

    out_h = (H - kernel_size + stride - 1) // stride + 1
    out_w = (W - kernel_size + stride - 1) // stride + 1

    out = np.zeros((N, C, out_h, out_w), dtype=dtype)

    for i in range(out_h):
        row_start = i * stride
        row_end = row_start + kernel_size
        for j in range(out_w):
            col_start = j * stride
            col_end = col_start + kernel_size

            window = x[:, :, row_start:row_end, col_start:col_end]

            out[:, :, i, j] = np.max(window, axis=(-2, -1))

    return out

