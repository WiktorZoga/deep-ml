import string

def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """
    Calculate the exact match score between predictions and references.
    
    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings
    
    Returns:
        Exact match score as a float between 0 and 1
    """
    if not predictions:
        return 0.0

    def clean(s: str) -> str:
        no_punct = ''.join(char.lower() for char in s if char not in string.punctuation)
        return ' '.join(no_punct.split())

    matches = sum(clean(pred) == clean(ref) for pred, ref in zip(predictions, references))
    
    return matches / len(predictions)