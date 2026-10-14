import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	
	matrix = np.asarray(matrix)

	return np.linalg.eigvals(matrix).real