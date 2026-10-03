import numpy as np

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    samples = np.asarray(samples)

    values, counts = np.unique(samples, return_counts=True)

    return list(zip(values.tolist(), (counts / np.sum(counts)).tolist()))