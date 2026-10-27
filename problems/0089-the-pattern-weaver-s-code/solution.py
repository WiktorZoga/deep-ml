import numpy as np

def softmax(values):
    maxes = np.max(values, axis=-1, keepdims=True)
    values = values - maxes
    scores = np.exp(values)
    sums = np.sum(scores, axis=-1, keepdims=True)
    scores = scores / sums
    return scores

def pattern_weaver(n, crystal_values, dimension):
    v = np.asarray(crystal_values)
    scores = np.outer(v, v) * (1.0 / np.sqrt(dimension))
    x = softmax(scores) @ v
    return np.round(x, 5)

