import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	flatten = []
	for row in a:
		flatten.extend(row)

	nxm = len(flatten)

	n = new_shape[0]
	m = new_shape[1]

	reshaped_matrix = []

	if n * m != nxm:
		return []

	for i in range(0, nxm, m):
		reshaped_matrix.append(flatten[i:i+m])

	return reshaped_matrix