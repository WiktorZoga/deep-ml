import numpy as np

def unified_task_handler(state_features: np.ndarray, task_weights: np.ndarray, task_id: int, epsilon: float = 0.0) -> dict:
    """
    Compute task-conditioned action values and epsilon-greedy action probabilities.

    Args:
        state_features: (num_actions, feature_dim) feature vectors for each action
        task_weights: (num_tasks, feature_dim) weight vectors for each task
        task_id: integer index of the current task
        epsilon: exploration parameter for epsilon-greedy (0 <= epsilon <= 1)

    Returns:
        dict with keys:
          - 'q_values': numpy array of shape (num_actions,)
          - 'greedy_action': int
          - 'action_probs': numpy array of shape (num_actions,)
    """

    num_actions, feature_dim = state_features.shape

    q_values = np.matmul(state_features, task_weights[task_id])

    greedy_action = np.argmax(q_values)

    action_probs = (epsilon / num_actions) * np.ones(num_actions)

    action_probs[greedy_action] += 1 - epsilon

    return {
        "q_values": q_values,
        "greedy_action": greedy_action,
        "action_probs": action_probs,
    }