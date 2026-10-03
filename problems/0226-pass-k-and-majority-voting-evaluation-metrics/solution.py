import numpy as np
from collections import Counter
import math

def pass_at_1(responses_correct: np.ndarray) -> float:
	"""
	Compute pass@1 by averaging correctness.
	
	Args:
		responses_correct: Boolean array for each response
		
	Returns:
		pass@1 score
	"""
	# Your code here
	return np.mean(responses_correct)

def majority_voting(responses: list[str]) -> str:
	"""
	Return the most common response.
	
	Args:
		responses: List of response strings
		
	Returns:
		Most frequent response
	"""
	# Your code here
	responses = np.asarray(responses)

	values, counts = np.unique(responses, return_counts=True)

	return values[np.argmax(counts)]

def pass_at_k(n: int, c: int, k: int) -> float:
	"""
	Compute unbiased pass@k from n samples with c correct.
	
	Formula: pass@k = 1 - C(n-c, k) / C(n, k)
	
	Args:
		n: Total samples
		c: Correct samples
		k: k in pass@k
		
	Returns:
		Estimated pass@k
	"""

	return 1.0 - math.comb(n - c, k) / math.comb(n, k)