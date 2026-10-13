from collections import defaultdict

def most_frequent_pair(sequences):
    """
    Args:
        sequences: list[list[int]] - list of token ID sequences
    Returns:
        tuple(int, int) or None - the most frequent adjacent pair, with ties
        broken by first appearance. Returns None if no pair exists.
    """

    counter = defaultdict(int)

    for seq in sequences:
        if len(seq) > 1:
            for a, b in zip(seq, seq[1:]):
                counter[(a, b)] += 1
    
    if not counter:
        return None
    
    arg_max = None
    max_occ = 0

    for key, value in counter.items():
        if value > max_occ:
            max_occ = value
            arg_max = key

    return arg_max