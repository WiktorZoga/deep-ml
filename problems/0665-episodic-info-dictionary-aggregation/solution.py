import numpy as np

def aggregate_episodic_info(infos: list) -> dict:
    """
    Aggregate episodic statistics from a list of step-level info dictionaries.
    
    Args:
        infos: List of info dictionaries from environment steps.
               Each dict may contain an 'episode' key with sub-dict
               having 'r' (total reward) and 'l' (length) keys.
    
    Returns:
        Dictionary with aggregated episode statistics.
    """
    eps = np.asarray([(info["episode"]["r"], info["episode"]["l"]) for info in infos if "episode" in info])       

    if eps.size == 0:
        return {'num_episodes': 0, 'mean_reward': 0.0, 'mean_length': 0.0, 'min_reward': 0.0, 'max_reward': 0.0, 'min_length': 0, 'max_length': 0}

    num_episodes = eps.shape[0]

    means = np.mean(eps, axis=0)

    mean_reward = means[0]
    mean_length = means[1]

    mini = np.min(eps, axis=0)
    maxi = np.max(eps, axis=0)

    min_reward = mini[0]
    min_length = mini[1]

    max_reward = maxi[0]
    max_length = maxi[1]

    return {
        "num_episodes": num_episodes,
        "mean_reward": mean_reward,
        "mean_length": mean_length,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "min_length": min_length,
        "max_length": max_length,   
    }
    

        