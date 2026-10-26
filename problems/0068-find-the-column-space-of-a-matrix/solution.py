
import numpy as np

def matrix_image(A):
	
	n, m = A.shape

	cur_rank = 0

	chosen_columns = []

	for i in range(m):
		B = A[:, chosen_columns + [i]]
		new_rank = np.linalg.matrix_rank(B)

		if cur_rank < new_rank:
			cur_rank = new_rank
			chosen_columns.append(i)
	
	return A[:, chosen_columns]

