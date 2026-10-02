import torch
import torch.linalg
from typing import List


def transform_basis(B: List[List[float]], C: List[List[float]]) -> List[List[float]]:
    """Return the change-of-basis matrix **P = C⁻¹ B**.

    - *B*, *C* may be 2×2 or 3×3 nested lists.
    - Result is rounded to 4 decimals and returned as a nested list.
    """
    # Your implementation here
    pass

    b = torch.tensor(B, dtype=torch.float32)
    c = torch.tensor(C, dtype=torch.float32)

    return (torch.linalg.inv(c) @ b).tolist()


