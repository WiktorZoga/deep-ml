import numpy as np

def cross_product(a, b):
	a = np.asarray(a).reshape(3)
	b = np.asarray(b).reshape(3)

	m = np.vstack([a, b])

	c = np.zeros(3)

	c[0] = np.linalg.det(m[:, [1, 2]])
	c[1] = -np.linalg.det(m[:, [0, 2]])
	c[2] = np.linalg.det(m[:, [0, 1]])

	return c
