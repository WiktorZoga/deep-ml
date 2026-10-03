import numpy as np

def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> float:
    """
    Compute the mean of sample means to demonstrate the sampling distribution.

    Args:
        num_samples: Number of independent samples to draw
        sample_size: Size of each sample
        distribution: 'uniform' (0,1) or 'exponential' (scale=1)

    Returns:
        Mean of the sample means (float)
    """
    # Your code here
    if distribution == 'uniform':
        samples = np.random.random((num_samples, sample_size))
    elif distribution == 'exponential':
        samples = np.random.standard_exponential((num_samples, sample_size))
    else:
        raise Exception("Not known distribution")
    
    return np.mean(np.mean(samples, axis=1), axis=0)