import numpy as np

def optimistic_greedy_bandit(
    true_rewards: list,
    initial_q: float,
    n_steps: int,
    step_size: float
) -> tuple:
    """
    Simulate a greedy bandit agent with optimistic initialization.
    
    Args:
        true_rewards: List of true deterministic rewards for each arm
        initial_q: Optimistic initial Q-value for all arms
        n_steps: Number of steps to simulate
        step_size: Constant step-size (alpha) for Q-value updates
    
    Returns:
        Tuple of (Q_values, action_counts) where Q_values is a list of
        floats rounded to 4 decimal places, and action_counts is a list of ints.
    """

    true_rewards = np.asarray(true_rewards)

    Q = initial_q * np.ones_like(true_rewards)

    counts = np.zeros_like(Q)

    for i in range(n_steps):
        idx = np.argmax(Q)

        reward = true_rewards[idx]
        counts[idx] += 1

        Q[idx] += step_size * (reward - Q[idx])

    return Q, counts
