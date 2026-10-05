import numpy as np

def gibbs_softmax_action_selection(q_values: list, temperature: float, seed: int) -> tuple:
    """
    Perform Gibbs softmax (Boltzmann) action selection.

    Args:
        q_values: list of floats, estimated action values
        temperature: float, temperature parameter (tau > 0)
        seed: int, random seed for reproducibility

    Returns:
        tuple: (probabilities as list of floats, selected action as int)
    """

    np.random.seed(seed)

    n = len(q_values)

    q = np.asarray(q_values)

    scores = np.exp(q / temperature)

    p = scores / np.sum(scores)

    sample = np.random.choice(np.arange(n), size=1, p=p)

    return p.round(4).tolist(), sample.tolist()[0]