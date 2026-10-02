import numpy as np

def phi_corr(X: list[int], Y: list[int]) -> float:
	"""
	Calculate the Phi coefficient between two binary variables.
	"""
	if len(X) == 0:
		return 0.0

	a = np.zeros((2, 2))
	np.add.at(a, (X, Y), 1)
	
	phi = np.linalg.det(a)

	denominator = np.sqrt(np.prod(np.sum(a, axis=0)) * np.prod(np.sum(a, axis=1)))

	if denominator == 0:
		return 0.0

	return round(float(phi / denominator), 4)
