import numpy as np


def bernoulli_mask_schedule(
    seq_len: int, t: float, schedule: str, seed: int
) -> list:
    """Generate a Bernoulli mask whose probability is set by a schedule.

    Args:
        seq_len: length of the token sequence
        t: schedule parameter in [0, 1]
        schedule: 'linear' or 'cosine'
        seed: random seed

    Returns:
        A list of 0/1 integers of length seq_len (1 = masked).
    """
    np.random.seed(seed)

    if schedule == "linear":
        p = float(t)
    elif schedule == "cosine":
        p = float(1.0 - np.cos(np.pi * t / 2.0))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    uniform_samples = np.random.rand(seq_len)
    mask = uniform_samples < p

    return mask.astype(int).tolist()