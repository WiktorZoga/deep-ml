import numpy as np

def make_diagonal(x):
	# Your code here
	n = x.shape[0]

	output = np.eye(n)

	for i in range(n):
		output[i, i] = x[i]
	return output.tolist()