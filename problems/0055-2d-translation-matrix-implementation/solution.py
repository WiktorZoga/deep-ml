import numpy as np

def translate_object(points, tx, ty):
	points = np.asarray(points, dtype=np.float32)

	t = np.array([tx, ty], dtype=np.float32)

	return points + t[np.newaxis, :]
