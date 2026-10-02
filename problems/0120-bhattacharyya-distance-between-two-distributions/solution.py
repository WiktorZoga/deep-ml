import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:

    if len(p) != len(q):
        return 0.0
    
    # Your code here
    p = np.asarray(p)
    q = np.asarray(q)

    bc = np.sum(np.sqrt(p * q))

    return -np.log(bc)