import numpy as np

def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	dense_matrix = np.asarray(dense_matrix)

	vals, row_idx, col_ptr = [], [], [0]

	n, m = dense_matrix.shape

	for j in range(m):
		for i in range(n):
			if dense_matrix[i, j] != 0:
				vals.append(dense_matrix[i, j].item())
				row_idx.append(i)
		col_ptr.append(len(vals))
	
	return vals, row_idx, col_ptr


