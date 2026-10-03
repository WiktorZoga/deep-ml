import numpy as np

def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	"""
	Compute the determinant and trace of a square matrix.
	
	Args:
		matrix: A square matrix (n x n) represented as list of lists
	
	Returns:
		Tuple of (determinant, trace)
	"""
	# Your code her
	n = len(matrix)

	matrix = np.asarray(matrix)

	det = float(np.linalg.det(matrix))
    trace = float(np.linalg.trace(matrix))

	return (det, trace)