import numpy as np

def f_score(y_true, y_pred, beta=1.0):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	tp = np.sum((y_pred == y_true) & (y_true == 1))
	t = np.sum(y_true == 1)
	p = np.sum(y_pred == 1)

	recall = tp / t
	precision = tp / p

	f_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

	return f_score.round(3)