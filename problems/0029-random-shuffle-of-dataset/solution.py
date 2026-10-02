import numpy as np

def shuffle_data(X, y, seed=42):
	# Your code here

	np.random.seed(seed)

	perm = np.random.permutation(len(y))

	return X[perm], y[perm]
