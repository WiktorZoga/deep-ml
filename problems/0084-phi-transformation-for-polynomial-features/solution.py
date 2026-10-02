import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
	"""
	Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

	Args:
		data (list[float]): A list of numerical values to transform.
		degree (int): The degree of the polynomial expansion.

	"""
	# Your code here
	n = len(data)

	if n == 0 or degree < 0:
		return []

	x = np.asarray(data)

	res = np.ones(n)
	for i in range(1, degree + 1):
		res = np.vstack([res, res[-1] * x])
	
	return res.T.tolist()