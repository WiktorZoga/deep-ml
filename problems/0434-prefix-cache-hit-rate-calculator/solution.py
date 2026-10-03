def prefix_cache_hit_rate(prompts: list, cache: list) -> dict:
    """
    Calculate the prefix cache hit rate for a batch of tokenized prompts.

    Args:
        prompts: List of tokenized prompts (list of list of ints).
        cache: List of cached prefixes (list of list of ints).

    Returns:
        Dictionary with 'hit_rate', 'cached_tokens', and 'total_tokens'.
    """
    total_tokens = sum(len(p) for p in prompts)
    
    if not cache or total_tokens == 0:
        return {
            "hit_rate": 0.0,
            "cached_tokens": 0,
            "total_tokens": total_tokens,
        }

    cached_tokens = 0
    for prompt in prompts:
        matching_lengths = [
            len(c) for c in cache 
            if len(c) <= len(prompt) and prompt[:len(c)] == c
        ]
        cached_tokens += max(matching_lengths, default=0)

    hit_rate = round(cached_tokens / total_tokens, 4) if total_tokens > 0 else 0.0

    return {
        "hit_rate": hit_rate,
        "cached_tokens": cached_tokens,
        "total_tokens": total_tokens,
    }