import numpy as np

def track_rl_experiment(episode_rewards: list, episode_lengths: list, window_size: int = 10) -> dict:
    """
    Summarize metrics from an RL training run.
    
    Args:
        episode_rewards: Total reward per episode
        episode_lengths: Number of timesteps per episode
        window_size: Window size for moving average and improvement calculation
    
    Returns:
        Dictionary with training summary statistics
    """

    rewards = np.asarray(episode_rewards)
    lengths = np.asarray(episode_lengths)

    total_episodes = len(episode_rewards)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    best_episode = np.argmax(rewards)
    mean_length = np.mean(lengths)

    first_moving_avg = np.mean(rewards[:window_size])
    final_moving_avg = np.mean(rewards[-window_size:])

    reward_improvement = final_moving_avg - first_moving_avg

    return {
        'total_episodes': total_episodes,
        'mean_reward': mean_reward.round(4),
        'std_reward': std_reward.round(4),
        'max_reward': max_reward.round(4),
        'min_reward': min_reward.round(4),
        'best_episode': best_episode,
        'mean_length': mean_length.round(4),
        'final_moving_avg': final_moving_avg.round(4),
        'reward_improvement': reward_improvement.round(4),
    }
