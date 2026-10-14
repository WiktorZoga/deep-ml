import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	A = np.asarray(A)
	T = np.asarray(T)
	S = np.asarray(S)

	detT, detS = np.linalg.det(T), np.linalg.det(S)

	if detT == 0.0 or detS == 0.0:
		return -1

	return np.linalg.inv(T) @ A @ S