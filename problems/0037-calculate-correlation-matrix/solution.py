import numpy as np

def calculate_correlation_matrix(X, Y=None):

	if Y is None:
		Y = X.copy()

	n = X.shape[0]

	x = X - np.mean(X, axis=0, keepdims=True)
	y = Y - np.mean(Y, axis=0, keepdims=True)

	cov = x.T @ y

	std_x = np.std(x, axis=0, ddof=0)
	std_y = np.std(y, axis=0, ddof=0)

	std_outer = np.outer(std_x, std_y)

	eps = 1e-12

	return cov / (n * std_outer + eps)

