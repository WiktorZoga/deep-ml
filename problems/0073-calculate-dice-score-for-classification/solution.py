
import numpy as np

def dice_score(y_true, y_pred):

	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	intersection = np.sum((y_pred == 1) & (y_true == 1))
	cnt_true = np.sum(y_true)
	cnt_pred = np.sum(y_pred)

	cnt_all = (cnt_true + cnt_pred)

	if cnt_all == 0:
		return 0.0

	res = 2 * intersection / (cnt_true + cnt_pred)

	return round(res, 3)
