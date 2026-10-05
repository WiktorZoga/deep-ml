import numpy as np

def canonicalize_observations(observations: np.ndarray, agent_ids: np.ndarray) -> np.ndarray:
    """
    Canonicalize multi-agent observations for experience sharing.
    
    Args:
        observations: (B, N, F) array of observations
        agent_ids: (B,) array indicating the self-agent index per observation
    
    Returns:
        (B, N, F) array of canonicalized observations, rounded to 4 decimals
    """

    B, N, F = observations.shape

    obs_self = observations[np.arange(B), agent_ids][:, np.newaxis, :]

    mask = np.ones((B, N), dtype=bool)
    mask[np.arange(B), agent_ids] = False

    obs_others = observations[mask].reshape(B, N - 1, F)

    keys = tuple(obs_others[:, :, f] for f in reversed(range(F)))
    
    order = np.lexsort(keys, axis=1)

    obs_others_sorted = np.take_along_axis(obs_others, order[..., np.newaxis], axis=1)

    return np.concatenate([obs_self, obs_others_sorted], axis=1)