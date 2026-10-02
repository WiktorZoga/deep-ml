import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here

	def relu(x):
		return np.maximum(0.0, x)
	
	o1 = relu(np.matmul(x, w1))
	o2 = np.matmul(o1, w2)

	return relu(x + o2)