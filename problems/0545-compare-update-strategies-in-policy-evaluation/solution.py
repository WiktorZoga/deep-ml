import numpy as np

def compare_policy_evaluation(V: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float, n_sweeps: int, method: str) -> np.ndarray:
    """
    Perform policy evaluation sweeps using the specified update method.
    
    Args:
        V: np.ndarray of shape (n_states,), initial value function
        P: np.ndarray of shape (n_states, n_states), transition probability matrix under the policy
        R: np.ndarray of shape (n_states,), expected immediate reward for each state
        gamma: float, discount factor
        n_sweeps: int, number of sweeps to perform
        method: str, either 'synchronous' or 'in_place'
    
    Returns:
        np.ndarray: updated value function after n_sweeps
    """

    n = V.size

    for _ in range(n_sweeps):
        if method == "synchronous":
            V = R + gamma * np.sum(P * V, axis=1)
        elif method == "in_place":
            for j in range(n):
                V[j] = R[j] + gamma * np.sum(P[j, :] * V)

    return V