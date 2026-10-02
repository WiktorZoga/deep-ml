import numpy as np

def r_squared(y_true, y_pred):

	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	# Write your code here
	ssr = np.sum((y_true - y_pred) ** 2)

	mean = np.mean(y_true)

	sst = np.sum((y_true - mean) ** 2)

	r2 = 1 - (ssr / sst)

	return r2.round(3)
	