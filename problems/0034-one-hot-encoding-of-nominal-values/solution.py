import numpy as np

def to_categorical(x, n_col=None):
	# Your code here

	if n_col is None:
		categories = {}
		size = 0

		for i in range(x.shape[0]):
			if not x[i] in categories:
				categories[x[i]] = size
				size += 1

		output = []

		for i in range(x.shape[0]):
			cat = np.zeros(size)
			cat[categories[x[i]]] = 1

			output.append(cat.tolist())
		
		return output

	else:

		output = []

		for i in range(x.shape[0]):
			cat = np.zeros(n_col)
			cat[x[i]] = 1

			output.append(cat.tolist())
		
		return output
	
		