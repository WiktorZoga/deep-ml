import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	dense_matrix = np.asarray(dense_matrix)	
	n, m = dense_matrix.shape

	vals, cols_idx, row_ptr = [], [], [0]

	for i in range(n):
		for j in range(m):
			if dense_matrix[i, j] != 0:
				vals.append(dense_matrix[i, j].tolist())
				cols_idx.append(j)
		row_ptr.append(len(vals))
		

	return vals, cols_idx, row_ptr
