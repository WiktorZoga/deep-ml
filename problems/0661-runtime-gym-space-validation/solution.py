import numpy as np

def validate_space(space: dict, samples: list) -> list:
    """
    Validate whether each sample conforms to the given space definition.
    
    Args:
        space: A dictionary defining the space with a 'type' key and type-specific parameters.
        samples: A list of samples to validate against the space.
    
    Returns:
        A list of booleans indicating whether each sample is valid for the space.
    """
    def check_space(sample, space):
        if space["type"] == "Discrete":
            return isinstance(sample, int) and 0 <= sample < space["n"]
        elif space["type"] == "MultiBinary":
            return len(sample) == space["n"] and all([x == 0 or x == 1 for x in sample])
        elif space["type"] == "Box":
            if len(sample) != space["shape"][0]:
                return False
            return all([low <= x <= hi for low, x, hi in zip(space["low"], sample, space["high"])])
        elif space["type"] == "MultiDiscrete": 
            if len(space["nvec"]) == len(sample):
                return all([0 <= x < y for x, y in zip(sample, space["nvec"])])

    return [check_space(sample, space) for sample in samples]