import numpy as np

def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	# Your code here

	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	tp = np.sum((y_pred == 1) & (y_true == 1))
	p = np.sum(y_pred)
	t = np.sum(y_true)

	precision = tp / p if p > 0 else 0
	recall = tp / t if t > 0 else 1

	f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

	return round(f1,3)