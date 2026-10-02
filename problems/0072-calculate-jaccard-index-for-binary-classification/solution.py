
import numpy as np

def jaccard_index(y_true, y_pred):
	# Write your code here

	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	intersection = np.sum((y_true == 1) & (y_true == y_pred))
	union = np.sum(y_pred | y_true)

	result = intersection / union

	return round(result, 3)
