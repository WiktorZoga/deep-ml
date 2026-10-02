import math

def softsign(x: float) -> float:
	"""
	Implements the Softsign activation function.

	Args:
		x (float): Input value

	Returns:
		float: The Softsign of the input
	"""

	result = x / (1 + (x if x > 0 else -x))

	return round(result, 4)