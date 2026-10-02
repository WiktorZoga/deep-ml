import numpy as np

def disorder(apples: list) -> float:
	"""
	Compute the disorder in a basket of apples.
	"""
	# Your code here

	def gini(x):
		n = x.shape[0]
		_, counts = np.unique(x, return_counts=True)
		p = counts / n
		return 1.0 - np.sum(p ** 2)
	
	x = np.array(apples)

	return gini(x)