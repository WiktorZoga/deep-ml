
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	n = len(y)

	y = np.asarray(y, dtype=np.float32)

	_, counts = np.unique(y, return_counts=True)
	
	probs = counts / n

	gini = 1 - np.sum(probs**2)

	val = gini.round(3)

	return round(val,3)