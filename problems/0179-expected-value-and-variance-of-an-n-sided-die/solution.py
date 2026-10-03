def dice_statistics(n: int) -> tuple[float, float]:
	"""
	Compute the expected value and variance of a fair n-sided die roll.

	Args:
		n (int): Number of sides of the die

	Returns:
		tuple: (expected_value, variance)
	"""
	# Your code here

	mean = (n + 1.0) / 2.0
	var = (n ** 2.0 - 1.0) / 12.0

	return (mean, var)
