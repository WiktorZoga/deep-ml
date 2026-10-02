
from collections import Counter
import numpy as np

def confusion_matrix(data):
	# Implement the function here
	
	conf = np.zeros((2, 2), dtype=np.int32)

	for (y, pred) in data:
		conf[pred, y] += 1

	conf[0, 0], conf[1, 1] = conf[1, 1], conf[0, 0]

	return conf.tolist()
