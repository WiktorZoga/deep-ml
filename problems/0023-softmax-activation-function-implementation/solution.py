import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    
    maxi = np.max(scores)

    w = [math.exp(z - maxi) for z in scores]
    sum_w = sum(w)

    return [x / sum_w for x in w]
    