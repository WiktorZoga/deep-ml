import numpy as np

def create_bandit_testbed(k: int, num_pulls: int, seed: int = 42) -> tuple:
    """
    Build a k-armed bandit testbed and simulate pulling each arm.
    
    Args:
        k: Number of arms
        num_pulls: Number of times to pull each arm
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (true_values, sample_means, optimal_arm)
    """
    
    np.random.seed(seed)

    q = np.random.normal(loc=0.0, scale=1.0, size=k)

    rewards = np.random.normal(loc=q, scale=1.0, size=(num_pulls, k))

    Q = np.mean(rewards, axis=0)

    return q.round(4).tolist(), Q.round(4).tolist(), np.argmax(q)

