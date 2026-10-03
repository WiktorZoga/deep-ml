import numpy as np

def extract_optimal_policy(Q: np.ndarray) -> dict:
    """
    Extract the optimal policy, state-value function, and advantage
    function from a Q-value table.
    
    Args:
        Q: Q-value table of shape (num_states, num_actions)
    
    Returns:
        Dictionary with keys:
        - 'optimal_actions': list of int (optimal action per state)
        - 'state_values': list of float (V*(s) per state)
        - 'advantages': nested list of float (A(s,a) for all pairs)
    """

    s, a = Q.shape

    optimal_actions = np.argmax(Q, axis=1)

    state_values = np.asarray([Q[i, optimal_actions[i]] for i in range(s)])

    advantages = Q - state_values[:, np.newaxis]

    return {
        "optimal_actions": optimal_actions.tolist(),
        "state_values": state_values.round(4).tolist(),
        "advantages": advantages.round(4).tolist(),
    }
