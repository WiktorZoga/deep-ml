import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here

	n = X.shape[0]

	X_batch = []

	for i in range(0, n, batch_size):
		x_batch = X[i:i+batch_size]

		if y is not None:
			y_batch = y[i:i+batch_size]
			yield [x_batch, y_batch]
		else:
			yield x_batch
	